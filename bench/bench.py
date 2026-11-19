"""Benchmark: our CUDA-free FlashAttention vs PyTorch's flash-attention CPU kernel.

Both run float32 on the same CPU. PyTorch's kernel is the official FlashAttention
algorithm implementation that ships in torch (SDPBackend.FLASH_ATTENTION); the
naive math backend is included to show what flash-style tiling buys at all.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from flashattn_amx import flash_attention

BR = int(sys.argv[1]) if len(sys.argv) > 1 else None
BC = int(sys.argv[2]) if len(sys.argv) > 2 else 256


def timeit(fn, min_time=0.4, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    t_total = 0.0
    while t_total < min_time or len(times) < 5:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        t_total += dt
    return float(np.median(times))


def run(configs, causal):
    rows = []
    for (B, H, S, D) in configs:
        rng = np.random.default_rng(42)
        q = rng.standard_normal((B, H, S, D), dtype=np.float32)
        k = rng.standard_normal((B, H, S, D), dtype=np.float32)
        v = rng.standard_normal((B, H, S, D), dtype=np.float32)
        tq, tk, tv = (torch.from_numpy(x) for x in (q, k, v))

        t_ours = timeit(lambda: flash_attention(q, k, v, causal=causal, br=BR, bc=BC))

        def torch_flash():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
        t_flash = timeit(torch_flash)

        def torch_math():
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
        t_math = timeit(torch_math) if S <= 8192 else float("nan")

        # FLOPs: 4 * B*H*S^2*D (2 gemms), halved for causal.
        flops = 4 * B * H * S * S * D * (0.5 if causal else 1.0)
        rows.append({
            "B": B, "H": H, "S": S, "D": D, "causal": causal,
            "ours_ms": t_ours * 1e3, "torch_flash_ms": t_flash * 1e3,
            "torch_math_ms": t_math * 1e3,
            "ours_gflops": flops / t_ours / 1e9,
            "flash_gflops": flops / t_flash / 1e9,
            "speedup_vs_flash": t_flash / t_ours,
        })
        r = rows[-1]
        print(f"S={S:6d} D={D:3d} causal={int(causal)}  "
              f"ours {r['ours_ms']:9.2f} ms ({r['ours_gflops']:7.1f} GF/s)   "
              f"torch-flash {r['torch_flash_ms']:9.2f} ms ({r['flash_gflops']:7.1f} GF/s)   "
              f"naive {r['torch_math_ms']:9.2f} ms   "
              f"speedup {r['speedup_vs_flash']:5.2f}x", flush=True)
    return rows


if __name__ == "__main__":
    torch.set_num_threads(8)  # all M3 cores — measured faster than torch's default of 4
    print(f"torch {torch.__version__}, torch threads={torch.get_num_threads()}, "
          f"ours: Br={BR} Bc={BC}\n")
    configs = [(1, 8, s, 64) for s in (256, 512, 1024, 2048, 4096, 8192)]
    configs_d128 = [(1, 8, s, 128) for s in (1024, 4096)]
    all_rows = []
    print("=== non-causal, D=64, B=1 H=8 ===")
    all_rows += run(configs, causal=False)
    print("\n=== causal, D=64, B=1 H=8 ===")
    all_rows += run(configs, causal=True)
    print("\n=== non-causal, D=128 ===")
    all_rows += run(configs_d128, causal=False)
    with open("bench_results.json", "w") as f:
        json.dump(all_rows, f, indent=1)
    print("\nwrote bench_results.json")
