"""PyTorch integration for flashattn-amx.

Three ways to use it:

1. Direct call (drop-in for F.scaled_dot_product_attention, inference):

       from flashattn_amx.torch_ops import flash_attention
       out = flash_attention(q, k, v, is_causal=True)

2. As a registered torch custom op (composes with torch.compile):

       torch.ops.flashattn.forward(q, k, v, True)

3. Patch a whole model: every F.scaled_dot_product_attention call inside the
   context routes through this kernel when eligible (CPU float32 inference),
   and falls back to PyTorch otherwise:

       with use_flash_attention():
           model(x)

Eligibility: CPU, float32, 4-D (B, H, S, D), no attn_mask, no dropout, no
gradients required. Anything else falls back to PyTorch's SDPA transparently.
"""
import ctypes
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from . import _fp, _lib

__all__ = ["flash_attention", "use_flash_attention"]


def _run_kernel(q, k, v, causal):
    """Zero-copy call: hands torch's own buffers to the C kernel."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    br = 128 if causal else 256  # tuned for Apple M3 L1d
    _lib.flash_attention_f32(
        ctypes.cast(q.data_ptr(), _fp), ctypes.cast(k.data_ptr(), _fp),
        ctypes.cast(v.data_ptr(), _fp), ctypes.cast(o.data_ptr(), _fp),
        B, H, S, D, int(causal), br, 256,
    )
    return o


# --- torch custom op registration (torch >= 2.4) -------------------------------

@torch.library.custom_op("flashattn::forward", mutates_args=())
def _op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    return _run_kernel(q, k, v, is_causal)


@_op.register_fake
def _(q, k, v, is_causal):
    return torch.empty_like(q)


def _eligible(q, k, v, attn_mask, dropout_p):
    return (
        q.device.type == "cpu" and q.dtype == torch.float32
        and q.dim() == 4 and k.shape == q.shape and v.shape == q.shape
        and attn_mask is None and dropout_p == 0.0
        and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    )


def flash_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                    enable_gqa=False):
    """Drop-in replacement for F.scaled_dot_product_attention (inference path).

    Routes to the AMX kernel when eligible, otherwise falls back to PyTorch.
    """
    if scale is None and not enable_gqa and _eligible(q, k, v, attn_mask, dropout_p):
        return torch.ops.flashattn.forward(q, k, v, is_causal)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
        scale=scale, enable_gqa=enable_gqa)


@contextmanager
def use_flash_attention():
    """Route all F.scaled_dot_product_attention calls through this kernel."""
    orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = flash_attention
    torch.nn.functional.scaled_dot_product_attention = flash_attention
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig
        torch.nn.functional.scaled_dot_product_attention = orig
