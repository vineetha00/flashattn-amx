"""Correctness: flashattn_amx vs a float64 exact-softmax reference (and PyTorch)."""
import numpy as np

from flashattn_amx import flash_attention

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

SHAPES = [(1, 2, 128, 64), (2, 4, 513, 64), (1, 8, 1000, 128), (1, 1, 77, 32)]


def reference(q, k, v, causal):
    """Exact attention in float64."""
    q64, k64, v64 = (x.astype(np.float64) for x in (q, k, v))
    s = q64 @ k64.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
    if causal:
        S = q.shape[2]
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
        s[..., mask] = -np.inf
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v64


def test_matches_fp64_reference():
    rng = np.random.default_rng(0)
    for (B, H, S, D) in SHAPES:
        for causal in (False, True):
            q, k, v = (rng.standard_normal((B, H, S, D), dtype=np.float32) for _ in range(3))
            err = np.abs(flash_attention(q, k, v, causal=causal) - reference(q, k, v, causal)).max()
            assert err < 5e-5, f"S={S} D={D} causal={causal}: err={err:.2e}"


def test_torch_ops():
    if torch is None:
        return
    from flashattn_amx.torch_ops import flash_attention as fa, use_flash_attention

    torch.manual_seed(0)
    for S, causal in [(513, False), (1024, True), (77, True)]:
        q, k, v = (torch.randn(2, 4, S, 64) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert (fa(q, k, v, is_causal=causal) - ref).abs().max().item() < 5e-5

    # fallback paths: mask and autograd must route to torch, not crash
    q = torch.randn(1, 2, 64, 32)
    assert fa(q, q, q, attn_mask=torch.zeros(64, 64, dtype=torch.bool)).shape == q.shape
    qg = q.clone().requires_grad_(True)
    fa(qg, qg, qg).sum().backward()
    assert qg.grad is not None

    # whole-model patch context
    with use_flash_attention():
        o1 = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    o2 = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert (o1 - o2).abs().max().item() < 5e-5


if __name__ == "__main__":
    test_matches_fp64_reference()
    test_torch_ops()
    print("All correctness checks passed.")
