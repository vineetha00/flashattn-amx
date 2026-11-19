"""Real FlashAttention on an NVIDIA GPU vs the CUDA-free M3 kernel, same shapes.

Run on any CUDA machine (e.g. an HPC GPU node). On compute capability >= 8.0
(Ampere+: A100, A40, L4, ...) PyTorch's genuine FLASH_ATTENTION backend runs;
older GPUs use the labeled memory-efficient fallback. If the official
`flash-attn` package is installed, it benchmarks that too. Needs only torch.

    python gpu_bench.py
"""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

assert torch.cuda.is_available(), "run on a GPU node"
CAP = torch.cuda.get_device_capability(0)
GPU = torch.cuda.get_device_name(0)
HAS_FLASH = CAP >= (8, 0)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {GPU} (sm{CAP[0]}{CAP[1]})")
print(f"real FlashAttention backend available: {HAS_FLASH}")

try:
    from flash_attn import flash_attn_func  # official Dao kernel, optional
    HAS_FA_PKG = True
except ImportError:
    HAS_FA_PKG = False
print(f"flash-attn package installed: {HAS_FA_PKG}\n")

SHAPES = [(S, c) for c in (False, True) for S in (256, 512, 1024, 2048, 4096, 8192)]

# Apple M3 results (fp32, B=1, H=8, D=64), measured 2026-11-19.
# "ours" = AMX/Accelerate FlashAttention kernel (github.com/vineetha00/flashattn-amx);
# "torch_cpu" = PyTorch's flash-attention CPU kernel at its best config.
M3_OURS = {(256, False): 0.51, (512, False): 1.62, (1024, False): 5.00, (2048, False): 20.47,
           (4096, False): 80.66, (8192, False): 336.40, (256, True): 0.50, (512, True): 1.33,
           (1024, True): 3.64, (2048, True): 12.98, (4096, True): 52.16, (8192, True): 203.93}
M3_TORCH = {(256, False): 0.79, (512, False): 2.29, (1024, False): 8.87, (2048, False): 26.36,
            (4096, False): 111.14, (8192, False): 435.60, (256, True): 0.87, (512, True): 2.97,
            (1024, True): 7.90, (2048, True): 21.67, (4096, True): 73.32, (8192, True): 244.30}


def bench(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


backend = SDPBackend.FLASH_ATTENTION if HAS_FLASH else SDPBackend.EFFICIENT_ATTENTION
label = "FlashAttention (CUDA)" if HAS_FLASH else "mem-efficient fallback (pre-Ampere GPU)"

sdpa_ms, fa_ms = {}, {}
for S, causal in SHAPES:
    q, k, v = (torch.randn(1, 8, S, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    with sdpa_kernel([backend]):
        sdpa_ms[(S, causal)] = bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    if HAS_FA_PKG:
        # flash_attn_func wants (B, S, H, D)
        qf, kf, vf = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        fa_ms[(S, causal)] = bench(lambda: flash_attn_func(qf, kf, vf, causal=causal))

cols = f"{'S':>6} {'causal':>6} | {'GPU sdpa ms':>11}"
if HAS_FA_PKG:
    cols += f" {'flash-attn ms':>13}"
cols += f" {'M3 ours ms':>11} {'M3 torch ms':>11} | {'GPU vs M3-ours':>14}"
print(f"{GPU} | torch backend: {label}")
print(cols)
print("-" * len(cols))
for S, causal in SHAPES:
    g = sdpa_ms[(S, causal)]
    line = f"{S:>6} {int(causal):>6} | {g:>11.3f}"
    if HAS_FA_PKG:
        line += f" {fa_ms[(S, causal)]:>13.3f}"
        g = min(g, fa_ms[(S, causal)])
    o, t = M3_OURS[(S, causal)], M3_TORCH[(S, causal)]
    line += f" {o:>11.2f} {t:>11.2f} | {o / g:>13.1f}x"
    print(line)

print("\nLast column: how many times faster this GPU's best flash kernel is than")
print("the M3 CPU kernel (fp16 GPU vs fp32 CPU — each side's native precision).")
