import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "triton", "numpy", "pandas", "tabulate", "torch_geometric",
        "jax[cuda12]"
    )
)

app = modal.App("benchmark-segmented-softmax")

@app.function(image=image, gpu="A10", timeout=600)
def run_benchmark():
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    import torch
    import triton
    import triton.language as tl
    from torch_geometric.utils import softmax as pyg_softmax
    import jax
    import jax.numpy as jnp
    import numpy as np

    # ==========================================
    # TRITON KERNEL
    # ==========================================
    @triton.jit
    def segmented_softmax_aligned_kernel(
        scores_ptr, out_ptr, ptr_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start_part = tl.load(ptr_ptr + pid)
        end_part = tl.load(ptr_ptr + pid + 1)
        start_part = tl.multiple_of(start_part, 4)
        offset = start_part + tl.arange(0, BLOCK_SIZE)
        mask = offset < end_part
        scores = tl.load(scores_ptr + offset, mask=mask, other=-1e20)
        max_val = tl.max(scores, axis=0)
        scores = scores - max_val
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        softmax = exp_scores / sum_exp
        tl.store(out_ptr + offset, softmax, mask=mask)

    def triton_softmax_wrapper(scores, ptr):
        out = torch.empty_like(scores)
        grid = (ptr.shape[0] - 1,)
        segmented_softmax_aligned_kernel[grid](
            scores, out, ptr, 
            BLOCK_SIZE=256, num_warps=8
        )
        return out

    # ==========================================
    # JAX KERNEL (HLO dump)
    # ==========================================
    def jax_segmented_softmax_impl(scores, segment_ids, num_segments):
        max_val = jax.ops.segment_max(scores, segment_ids, num_segments=num_segments)
        scores_shifted = scores - max_val[segment_ids]
        exp_scores = jnp.exp(scores_shifted)
        sum_exp = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)
        return exp_scores / sum_exp[segment_ids]

    # JIT compile with static args
    jax_segmented_softmax = jax.jit(
        jax_segmented_softmax_impl, 
        static_argnames=['num_segments']
    )

    # ==========================================
    # HLO DUMP + PROFILING SETUP
    # ==========================================
    
    # Config: 10K segments, 128 length (middle case)
    num_segments = 10000
    seg_len = 128
    total_elements = num_segments * seg_len

    # Data prep
    scores_torch = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    scores_np = scores_torch.cpu().numpy()
    jax_scores = jnp.array(scores_np)
    jax_segment_ids = jnp.repeat(jnp.arange(num_segments), seg_len)

    # Warmup first (compilation)
    _ = jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments=num_segments).block_until_ready()

    # === HLO DUMP ===
    print("\n" + "="*80)
    print(" JAX HLO (High Level Optimized) DUMP ")
    print("="*80)
    
    lowered = jax_segmented_softmax.lower(jax_scores, jax_segment_ids, num_segments=num_segments)
    hlo_text = lowered.as_text()
    print(hlo_text[:5000])  # First 5000 chars, enough to see fusion
    print("... [truncated for brevity] ...")
    
    # Check for fusion in HLO
    if "fusion" in hlo_text.lower():
        print("\n✅ XLA IS FUSING operations")
    else:
        print("\n❌ XLA NOT FUSING — separate kernels for segment_max, exp, segment_sum")
    
    # Count custom calls (separate GPU kernels)
    custom_calls = hlo_text.count("custom-call")
    print(f"   Number of custom-call (separate kernels): {custom_calls}")

    # === JAX PROFILER TRACE ===
    print("\n" + "="*80)
    print(" JAX PROFILER TRACE (10 iterations) ")
    print("="*80)
    
    import tempfile
    trace_dir = tempfile.mkdtemp()
    
    with jax.profiler.trace(trace_dir):
        for i in range(10):
            out = jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments=num_segments).block_until_ready()
    
    print(f"   Trace saved to: {trace_dir}")
    print("   (Download karke chrome://tracing mein dekho)")

    # === BENCHMARK (teri existing table) ===
    print("\n" + "="*80)
    print(" FULL BENCHMARK ")
    print("="*80)
    
    configs = [
        {"num_segments": 10000, "seg_len": 64},
        {"num_segments": 10000, "seg_len": 128},
        {"num_segments": 10000, "seg_len": 256},
        {"num_segments": 50000, "seg_len": 128},
    ]

    results = []
    for c in configs:
        num_segments = c["num_segments"]
        seg_len = c["seg_len"]
        total_elements = num_segments * seg_len

        ptr = torch.arange(0, total_elements + 1, seg_len, dtype=torch.int64, device="cuda")
        scores = torch.randn(total_elements, device="cuda", dtype=torch.float32)
        
        np_scores = scores.cpu().numpy()
        jax_scores = jnp.array(np_scores)
        jax_segment_ids = jnp.repeat(jnp.arange(num_segments), seg_len)

        # Correctness
        pyg_out = pyg_softmax(scores, ptr=ptr)
        triton_out = triton_softmax_wrapper(scores, ptr)
        is_correct = torch.max(torch.abs(pyg_out - triton_out)).item() < 1e-4

        # Warmup
        _ = jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments=num_segments).block_until_ready()

        quantiles = [0.5, 0.2, 0.8]
        
        pyg_ms, _, _ = triton.testing.do_bench(lambda: pyg_softmax(scores, ptr=ptr), quantiles=quantiles)
        triton_ms, _, _ = triton.testing.do_bench(lambda: triton_softmax_wrapper(scores, ptr), quantiles=quantiles)
        jax_ms, _, _ = triton.testing.do_bench(
            lambda: jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments=num_segments).block_until_ready(), 
            quantiles=quantiles
        )

        results.append({
            "Segments": num_segments,
            "Length": seg_len,
            "PyG (ms)": round(pyg_ms, 3),
            "JAX (ms)": round(jax_ms, 3),
            "Triton (ms)": round(triton_ms, 3),
            "Triton vs PyG": f"{pyg_ms / triton_ms:.2f}x Faster",
            "Triton vs JAX": f"{jax_ms / triton_ms:.2f}x Faster",
            "Correct": "✅" if is_correct else "❌"
        })

    return results

@app.local_entrypoint()
def main():
    import pandas as pd
    from tabulate import tabulate
    
    results = run_benchmark.remote()
    
    print("\n" + "="*85)
    print(" 🏆 FINAL SHOWDOWN: PyG vs Google JAX vs Umang's Triton ")
    print("="*85)
    
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("="*85 + "\n")
