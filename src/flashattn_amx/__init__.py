"""flashattn-amx: FlashAttention forward pass for Apple Silicon CPUs. No CUDA.

Same algorithm as FlashAttention (Dao et al. 2022) — tiled attention with an
online softmax, O(S) extra memory — with the matrix multiplies routed through
Apple Accelerate's SGEMM, which executes on the M-series AMX matrix
coprocessor.

NumPy API:  flash_attention(q, k, v, causal=...)
PyTorch:    import flashattn_amx.torch_ops
"""
import ctypes

import numpy as np

from . import _core

__version__ = "0.1.0"
__all__ = ["flash_attention"]

_lib = ctypes.CDLL(_core.__file__)
_lib.flash_attention_f32.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int] * 7
_lib.flash_attention_f32.restype = None
_fp = ctypes.POINTER(ctypes.c_float)


def flash_attention(q, k, v, causal=False, br=None, bc=256):
    """FlashAttention forward. q, k, v: float32 arrays of shape (B, H, S, D).

    Default block sizes were tuned on an Apple M3 (128 KiB L1d, AMX sgemm);
    override with br=/bc= for other chips.
    """
    if br is None:
        br = 128 if causal else 256
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    assert q.shape == k.shape == v.shape and q.ndim == 4, "expected (B, H, S, D)"
    B, H, S, D = q.shape
    o = np.empty_like(q)
    _lib.flash_attention_f32(
        q.ctypes.data_as(_fp), k.ctypes.data_as(_fp), v.ctypes.data_as(_fp),
        o.ctypes.data_as(_fp), B, H, S, D, int(causal), br, bc,
    )
    return o
