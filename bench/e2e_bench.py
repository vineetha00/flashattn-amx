"""End-to-end benchmark: a GPT-style transformer with PyTorch's flash attention
vs the same model routed through our AMX kernel via use_flash_attention().

GPT-2-small-ish geometry: 12 layers, d_model 768, 12 heads (D=64), fp32, CPU.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from flashattn_amx.torch_ops import use_flash_attention


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, S, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, d // self.h) \
                      .permute(2, 0, 3, 1, 4).unbind(0)          # (B, H, S, D)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, layers=12, d=768, h=12, vocab=50257):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def timeit(fn, reps=5):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


if __name__ == "__main__":
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = GPT().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2-small geometry: 12L x 768d x 12h, {n_params / 1e6:.0f}M params, fp32, CPU\n")

    for S in (1024, 2048, 4096):
        idx = torch.randint(0, 50257, (1, S))

        with torch.inference_mode():
            def run_torch():
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    model(idx)
            t_torch = timeit(run_torch)

            def run_ours():
                with use_flash_attention():
                    model(idx)
            t_ours = timeit(run_ours)

            # sanity: same logits
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                ref = model(idx)
            with use_flash_attention():
                out = model(idx)
            err = (out - ref).abs().max().item()

        print(f"S={S:5d}  torch-flash {t_torch * 1e3:8.0f} ms   ours {t_ours * 1e3:8.0f} ms   "
              f"end-to-end speedup {t_torch / t_ours:4.2f}x   "
              f"({S / t_ours:6.0f} vs {S / t_torch:6.0f} tok/s)   max|logit diff| {err:.1e}",
              flush=True)
