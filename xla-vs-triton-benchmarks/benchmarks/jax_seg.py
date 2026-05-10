import modal
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "triton", "numpy", "pandas", "tabulate", "torch_geometric",
        "jax[cuda12]" # JAX with CUDA
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
    # 1. UMANG'S TRITON KERNEL
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
    # 2. GOOGLE JAX (XLA) KERNEL
    # ==========================================
    @jax.jit(static_argnames=['num_segments'])
    def jax_segmented_softmax(scores, segment_ids, num_segments):
        # Find max per segment for numerical stability
        max_val = jax.ops.segment_max(scores, segment_ids, num_segments=num_segments)
        scores_shifted = scores - max_val[segment_ids]
        
        # Exponentiate
        exp_scores = jnp.exp(scores_shifted)
        
        # Sum exponentials per segment
        sum_exp = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)
        
        # Normalize
        return exp_scores / sum_exp[segment_ids]

    # ==========================================
    # 3. BENCHMARKING CONFIGURATION
    # ==========================================
    print("🚀 Running Benchmark: PyG vs JAX vs Umang's Triton on A10G...")
    
    configs =[
        {"num_segments": 10000, "seg_len": 64},
        {"num_segments": 10000, "seg_len": 128},
        {"num_segments": 10000, "seg_len": 256},
        {"num_segments": 50000, "seg_len": 128}, # Heavy GNN Workload
    ]

    results =[]

    for c in configs:
        num_segments = c["num_segments"]
        seg_len = c["seg_len"]
        total_elements = num_segments * seg_len

        # PyTorch Data
        ptr = torch.arange(0, total_elements + 1, seg_len, dtype=torch.int64, device="cuda")
        scores = torch.randn(total_elements, device="cuda", dtype=torch.float32)

        # JAX Data (Moving data to JAX format)
        np_scores = scores.cpu().numpy()
        jax_scores = jnp.array(np_scores)
        jax_segment_ids = jnp.repeat(jnp.arange(num_segments), seg_len)

        # Correctness Check
        pyg_out = pyg_softmax(scores, ptr=ptr)
        triton_out = triton_softmax_wrapper(scores, ptr)
        
        max_diff = torch.max(torch.abs(pyg_out - triton_out)).item()
        is_correct = max_diff < 1e-4

        # WARMUP JAX (XLA compilation takes time initially)
        _ = jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments).block_until_ready()

        quantiles =[0.5, 0.2, 0.8]
        
        # Time PyG
        pyg_ms, _, _ = triton.testing.do_bench(lambda: pyg_softmax(scores, ptr=ptr), quantiles=quantiles)
        
        # Time Triton
        triton_ms, _, _ = triton.testing.do_bench(lambda: triton_softmax_wrapper(scores, ptr), quantiles=quantiles)
        
        # Time JAX (block_until_ready is mandatory for JAX timing!)
        jax_ms, _, _ = triton.testing.do_bench(
            lambda: jax_segmented_softmax(jax_scores, jax_segment_ids, num_segments).block_until_ready(), 
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
            "Correct": "✅" if is_correct else f"❌"
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
