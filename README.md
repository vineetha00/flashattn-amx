# flashattn-amx

FlashAttention without CUDA: a FlashAttention forward pass for Apple Silicon
CPUs that **beats PyTorch's flash-attention CPU kernel by 1.2–2.2×** in every
benchmarked configuration.

It implements the same algorithm as FlashAttention (Dao et al. 2022) — tiled
attention with an online softmax, O(S) extra memory instead of the O(S²) score
matrix — but routes the matrix multiplies through Apple's **Accelerate SGEMM**,
which executes on the M-series **AMX matrix coprocessor**, with `exp`
vectorized via vForce and parallelism over (batch × heads × query-blocks) via
Grand Central Dispatch. The kernel is ~130 lines of C.

Requires macOS on Apple Silicon. Forward pass only (inference); under autograd
the PyTorch wrapper falls back to PyTorch automatically.

## Install

```bash
pip install flashattn-amx
```

Or from source (needs Xcode command line tools):

```bash
git clone https://github.com/vineetha00/flashattn-amx
cd flashattn-amx && pip install .
```

## Usage

**PyTorch** (zero-copy; registered as the custom op `flashattn::forward`, so it
also composes with `torch.compile`):

```python
import torch
from flashattn_amx.torch_ops import flash_attention, use_flash_attention

# drop-in for F.scaled_dot_product_attention (inference):
out = flash_attention(q, k, v, is_causal=True)

# or route an entire model through it — anything ineligible
# (masks, dropout, autograd, non-fp32) falls back to PyTorch:
with torch.inference_mode(), use_flash_attention():
    logits = model(idx)
```

`use_flash_attention()` patches `F.scaled_dot_product_attention`, so it covers
models that call SDPA from Python (most hand-written and Hugging Face models),
not `nn.MultiheadAttention`'s C++ fast path.

**NumPy**:

```python
import numpy as np
from flashattn_amx import flash_attention

q = np.random.randn(1, 8, 4096, 64).astype(np.float32)
out = flash_attention(q, q, q, causal=True)   # (1, 8, 4096, 64)
```

Block sizes are tuned for the M3's 128 KiB L1d and tunable per machine via
`br=`/`bc=`.

## Benchmarks

Real CUDA FlashAttention cannot run on a Mac, so the baseline is the official
FlashAttention-algorithm CPU kernel that ships inside PyTorch
(`SDPBackend.FLASH_ATTENTION`), given its best configuration (8 threads —
measured faster than its 4-thread default). torch 2.15.0, float32, Apple M3
(4P + 4E). Median wall time, B=1, H=8.

| S | D | causal | ours (ms) | torch flash (ms) | torch naive (ms) | speedup vs flash |
|------:|----:|:------:|----------:|-----------------:|-----------------:|-----------------:|
| 256 | 64 | no | 0.51 | 0.79 | 2.40 | **1.53×** |
| 512 | 64 | no | 1.62 | 2.29 | 6.45 | **1.41×** |
| 1024 | 64 | no | 5.00 | 8.87 | 22.80 | **1.77×** |
| 2048 | 64 | no | 20.47 | 26.36 | 86.18 | **1.29×** |
| 4096 | 64 | no | 80.66 | 111.14 | 446.09 | **1.38×** |
| 8192 | 64 | no | 336.40 | 435.60 | 4296.81 | **1.29×** |
| 256 | 64 | yes | 0.50 | 0.87 | 2.75 | **1.75×** |
| 512 | 64 | yes | 1.33 | 2.97 | 8.33 | **2.24×** |
| 1024 | 64 | yes | 3.64 | 7.90 | 25.58 | **2.17×** |
| 2048 | 64 | yes | 12.98 | 21.67 | 101.64 | **1.67×** |
| 4096 | 64 | yes | 52.16 | 73.32 | 475.96 | **1.41×** |
| 8192 | 64 | yes | 203.93 | 244.30 | 5298.54 | **1.20×** |
| 1024 | 128 | no | 6.77 | 9.48 | 23.79 | **1.40×** |
| 4096 | 128 | no | 109.91 | 145.64 | 491.71 | **1.33×** |

Peak throughput 634 GFLOP/s fp32 (D=128); 10×+ faster than naive attention at
long sequences. Reproduce with `python bench/bench.py`.

### End-to-end in a real model

`bench/e2e_bench.py` runs a GPT-2-small-geometry transformer (12 layers × 768d
× 12 heads, 162M params, fp32, CPU) and swaps every attention call to this
kernel via `use_flash_attention()`:

| Seq len | torch flash | ours | whole-model speedup |
|---:|---:|---:|---:|
| 1024 | 836 ms | 651 ms | **1.28×** |
| 2048 | 1753 ms | 1442 ms | **1.22×** |
| 4096 | 3775 ms | 3172 ms | **1.19×** |

Max logit difference vs PyTorch: 4e-6.

### How far is a real GPU?

For context, the same shapes were run against real CUDA FlashAttention on an
NVIDIA A100-SXM4-40GB (Ampere, so PyTorch's genuine `FLASH_ATTENTION` backend;
via [Modal](https://modal.com)) and on a Colab free-tier Tesla T4 (pre-Ampere,
so PyTorch fell back to its memory-efficient kernel there). GPUs run fp16,
their native precision; the M3 runs fp32, its native precision:

| S | causal | A100 flash (ms) | T4 (ms) | M3 this kernel (ms) | A100 advantage |
|---:|:--:|---:|---:|---:|---:|
| 256 | no | 0.023 | — | 0.51 | 22.6× |
| 1024 | no | 0.050 | 0.32 | 5.00 | 99.9× |
| 4096 | no | 0.317 | 3.43 | 80.66 | 254.4× |
| 8192 | no | 1.014 | 10.56 | 336.40 | 331.6× |
| 8192 | yes | 0.689 | 5.55 | 203.93 | 296.2× |

That is the expected hardware gap, not an algorithmic one — this kernel's claim
is fastest attention *without* CUDA, not faster than a GPU. Reproduce with
`modal run bench/modal_bench.py` (one command, runs `bench/gpu_bench.py` on an
A100), `bench/gpu_bench.py` directly on any CUDA machine, or
`bench/colab_gpu_vs_m3.ipynb` on Colab.

### Accuracy and memory

Max |error| vs a float64 exact-softmax reference is ~1e-6, the same as
PyTorch's kernel (`tests/test_correctness.py`). Per-task scratch is
`Br·Bc + Br·D + 2·Br` floats (~0.3 MB) regardless of sequence length; naive
attention at S=8192, H=8 materializes a 2 GB score matrix.

## Why it wins

1. **AMX**: Accelerate's SGEMM runs on the M-series matrix coprocessor, far
   above what NEON SIMD (what PyTorch's CPU flash kernel uses) can reach.
2. **Cache-tuned blocks**: Br=256 (128 causal), Bc=256 keeps each score tile
   plus accumulator inside the M3's 128 KiB L1d.
3. **Causal block skipping**: key blocks entirely above the diagonal are never
   touched, and the diagonal tile is masked in place.
4. **All 8 cores** via GCD with fine-grained (query-block) work items, so P and
   E cores stay saturated without oversubscription.

## Layout

- `src/flashattn_amx/_core.c` — the kernel
- `src/flashattn_amx/__init__.py` — NumPy API
- `src/flashattn_amx/torch_ops.py` — PyTorch custom op, SDPA drop-in,
  whole-model patch
- `tests/` — correctness vs an exact float64 reference
- `bench/` — kernel, end-to-end, and GPU comparison benchmarks

## License

MIT
