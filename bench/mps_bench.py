"""The M3's own GPU via PyTorch MPS, same shapes as bench.py.

Answers the obvious question about a CPU attention kernel on a Mac:
what does the integrated GPU do on the same chip? fp32 (the CPU kernel's
precision) and fp16 (the GPU's fastest practical precision).

    python bench/mps_bench.py
"""
import time

import torch
import torch.nn.functional as F

assert torch.backends.mps.is_available(), "needs an Apple Silicon Mac"
print(f"torch {torch.__version__} | MPS\n")

SHAPES = [(S, c) for c in (False, True) for S in (256, 512, 1024, 2048, 4096, 8192)]

# Apple M3 CPU results (fp32, B=1, H=8, D=64), measured 2026-11-19.
M3_OURS = {(256, False): 0.51, (512, False): 1.62, (1024, False): 5.00, (2048, False): 20.47,
           (4096, False): 80.66, (8192, False): 336.40, (256, True): 0.50, (512, True): 1.33,
           (1024, True): 3.64, (2048, True): 12.98, (4096, True): 52.16, (8192, True): 203.93}


def bench(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


header = (f"{'S':>6} {'causal':>6} | {'MPS fp32 ms':>11} {'MPS fp16 ms':>11} "
          f"{'ours CPU ms':>11} | {'CPU/MPS32':>9}")
print(header)
print("-" * len(header))
for S, causal in SHAPES:
    row = {}
    for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        try:
            q, k, v = (torch.randn(1, 8, S, 64, device="mps", dtype=dt) for _ in range(3))
            row[name] = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        except RuntimeError as e:
            row[name] = float("nan")
            print(f"  (S={S} {name}: {str(e).splitlines()[0][:80]})")
    o = M3_OURS[(S, causal)]
    print(f"{S:>6} {int(causal):>6} | {row['fp32']:>11.3f} {row['fp16']:>11.3f} "
          f"{o:>11.2f} | {o / row['fp32']:>8.1f}x")

print("\nLast column: this CPU kernel's time over MPS fp32 (>1 means the GPU is faster).")
