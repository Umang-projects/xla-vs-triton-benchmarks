# xla-vs-triton-benchmarks
- Systems: Profiled XLA/HLO fusion boundaries, implemented fused Triton kernels    for segmented ops (8x-19x vs XLA), contributions to NVIDIA, OpenAI Triton

# XLA vs Triton: Segmented Softmax Benchmark

Identified critical XLA compiler fusion gaps in JAX segmented softmax via HLO analysis. 
Hand-fused Triton kernel achieves **19x speedup over XLA** and **14x over PyTorch Geometric**.

## Benchmark Results (A100 / A10G)

| Segments | Length | JAX (ms) | PyG (ms) | Triton (ms) | Speedup vs JAX |
|----------|--------|----------|----------|-------------|----------------|
| 10,000   | 64     | 0.288    | 0.488    | 0.033       | **8.78x**      |
| 10,000   | 128    | 0.431    | 0.629    | 0.032       | **13.58x**     |
| 10,000   | 256    | 0.869    | 0.478    | 0.035       | **24.97x**     |
| 50,000   | 128    | 1.455    | 0.880    | 0.132       | **11.01x**     |

## Key Finding: XLA Fusion Gap

Dumped HLO shows `stablehlo.scatter` creates hard fusion boundaries, forcing 
3 separate GPU kernels + expensive HBM round-trips. Triton kernel fuses 
max → exp → sum → divide into a single SRAM pass.

See [HLO Analysis](analysis/xla_hlo_dump.md)

## JAX Issue

Reported to Google JAX team: [Issue #37511](https://github.com/google/jax/issues/37511)

## Triton Kernel

Fused segmented softmax in Triton using CSR `ptr` format and online softmax.

See [kernels/segmented_softmax.py](kernels/segmented_softmax.py)

## Run Benchmarks

```bash
pip install -r requirements.txt
python benchmarks/jax_seg2.py

```
## ⚠️ Hardware Requirements

Benchmarks require **NVIDIA GPU (A100/A10/T4)** and are designed to run on 
[Modal](https://modal.com) cloud infrastructure.
