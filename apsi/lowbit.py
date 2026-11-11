"""Low-bit training: fit(..., precision="fp8" | "fp6" | "fp6_e2m3" | "fp4" | "int8" | "int4").

The weights of every layer are rounded to the format in each forward pass - one scale per 32 values along the
input side, exactly the layout and rounding the GPU uses to run a model in that format (hub_gpu.qnumpy) - while the
optimizer keeps a float32 master copy and the gradient passes straight through the rounding. A model trained in fp4
therefore runs in fp4 afterwards with the very weights it was trained with, on any card.

On the processor and on every GPU the arithmetic itself is float32 / bfloat16 (the format shapes the weights). On
GPUs with 8-bit float tensor cores (NVIDIA Ada, Hopper, Blackwell: compute capability 8.9 and up) with a PyTorch that
has torch._scaled_mm, precision="fp8" also runs the products of Linear and attention layers in fp8 on those tensor
cores (APSI_FP8_MATMUL=0 turns that off); anything it cannot take falls back to the path above.
"""
import numpy as np

KINDS = ("fp8", "fp6", "fp6_e2m3", "fp4", "int8", "int4")
ALIASES = {"mxfp8": "fp8", "e4m3": "fp8", "fp8_e4m3": "fp8", "mxfp6": "fp6_e2m3", "e2m3": "fp6_e2m3",
           "fp6_e3m2": "fp6", "e3m2": "fp6", "mxfp4": "fp4", "e2m1": "fp4", "fp4_e2m1": "fp4", "q8": "int8",
           "q4": "int4", "mxint8": "int8"}
BLOCK = 32


def parse(precision):
    """The low-bit format named by `precision`, or None for the ordinary precisions (fp32, bf16, ...)."""
    if not isinstance(precision, str):
        return None
    k = precision.strip().lower()
    k = ALIASES.get(k, k)
    return k if k in KINDS else None


def _as_rows(shape):
    """[..., in, out] weights as [out, in*...] rows (the GPU's layout: blocks run along the input side)."""
    return int(np.prod(shape[:-1])), int(shape[-1])


# ------------------------------------------------------------------------------------------------ NumPy
def round_weights(a, kind):
    """A float array rounded as the format stores it (float32 values; 1-D arrays - biases, norms - unchanged)."""
    from .hub_gpu import qnumpy
    a = np.asarray(a)
    if a.ndim < 2 or a.dtype.kind != "f":
        return a
    n_in, n_out = _as_rows(a.shape)
    rows = np.asarray(a, np.float32).reshape(n_in, n_out).T
    pad = (-n_in) % BLOCK
    if pad:
        rows = np.concatenate([rows, np.zeros((n_out, pad), np.float32)], axis=1)
    q, sc = qnumpy(rows, kind)
    d = (q.reshape(n_out, -1, BLOCK) * sc[:, :, None]).reshape(n_out, -1)[:, :n_in]
    return d.T.reshape(a.shape).astype(a.dtype)


def round_tree(params, kind):
    from .values import tree_map
    return tree_map(lambda v: round_weights(v, kind) if isinstance(v, np.ndarray) else v, params)


# ------------------------------------------------------------------------------------------------ PyTorch
FMT = {"fp8": (4, 3, 7, 448.0), "fp6": (3, 2, 3, 28.0), "fp6_e2m3": (2, 3, 1, 7.5), "fp4": (2, 1, 1, 6.0)}


def _round_minifloat_torch(t, kind):
    """t (|t| <= the format's top) to the nearest value of the minifloat, ties to even: the same as fp_round."""
    import torch
    eb, mbits, bias, top = FMT[kind]
    a = t.abs()
    _, ex = torch.frexp(a)                                 # a = m * 2^ex, m in [0.5, 1): floor(log2 a) = ex - 1
    e = torch.clamp(ex - 1, min=1 - bias)
    step = torch.ldexp(torch.ones_like(a), (e - mbits).to(torch.int32))
    q = torch.round(a / step) * step
    return torch.where(t < 0, -q, q)


def round_weights_torch(w, kind):
    """The rounded weights (float32), computed without gradient."""
    import torch
    n_in, n_out = _as_rows(tuple(w.shape))
    rows = w.detach().float().reshape(n_in, n_out).t()
    pad = (-n_in) % BLOCK
    if pad:
        rows = torch.cat([rows, rows.new_zeros((n_out, pad))], dim=1)
    blk = rows.reshape(n_out, -1, BLOCK)
    a = blk.abs()
    if kind in FMT:
        top = FMT[kind][3]
        sc = a.amax(dim=2, keepdim=True) / top
        inv = torch.where(sc != 0, 1.0 / torch.where(sc != 0, sc, torch.ones_like(sc)), torch.zeros_like(sc))
        q = _round_minifloat_torch(torch.clamp(blk * inv, -top, top), kind)
    elif kind == "int8":
        sc = a.amax(dim=2, keepdim=True) / 127.0
        inv = torch.where(sc != 0, 1.0 / torch.where(sc != 0, sc, torch.ones_like(sc)), torch.zeros_like(sc))
        q = torch.clamp(torch.round(blk * inv), -127, 127)
    else:                                                   # int4 as ggml's Q4_0: the extreme value maps to -8
        idx = a.argmax(dim=2, keepdim=True)
        sc = torch.gather(blk, 2, idx) / -8.0
        inv = torch.where(sc != 0, 1.0 / torch.where(sc != 0, sc, torch.ones_like(sc)), torch.zeros_like(sc))
        q = torch.clamp(torch.round(blk * inv), -8, 7)
    d = (q * sc).reshape(n_out, -1)[:, :n_in]
    return d.t().reshape(w.shape).to(w.dtype)


def fake_quant_torch(w, kind):
    """w in the format for the forward pass; the gradient goes straight through to the float32 master."""
    if w.dim() < 2 or not w.is_floating_point():
        return w
    return w + (round_weights_torch(w, kind) - w).detach()


def fake_quant_tree_torch(tp, kind):
    from .values import tree_map
    import torch
    return tree_map(lambda v: fake_quant_torch(v, kind) if isinstance(v, torch.Tensor) else v, tp)


# ---------------------------------------------------------------------- products on 8-bit float tensor cores
_HW = {"on": False, "why": None}


def fp8_matmul_available(dev):
    """True when this GPU and PyTorch can multiply in fp8 on tensor cores (compute capability 8.9 and up)."""
    import os
    import torch
    if os.environ.get("APSI_FP8_MATMUL", "1").strip().lower() in ("0", "false", "no"):
        return False
    if dev.type != "cuda" or not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return False
    try:
        return tuple(torch.cuda.get_device_capability(dev)) >= (8, 9)
    except Exception:
        return False


def _fp8_fn():
    import torch

    class F8(torch.autograd.Function):
        """y = x @ w with x and w in fp8 E4M3 (one scale each) on the tensor cores, bfloat16 out; the backward
        pass in bfloat16 (as Transformer Engine's default recipe keeps it for stability)."""

        @staticmethod
        def forward(ctx, x2, w):
            xs = (x2.detach().abs().amax().float() / 448.0).clamp(min=1e-12)
            ws = (w.detach().abs().amax().float() / 448.0).clamp(min=1e-12)
            a = (x2.float() / xs).to(torch.float8_e4m3fn)
            b = (w.float() / ws).to(torch.float8_e4m3fn).t().contiguous().t()      # column-major, as required
            y = torch._scaled_mm(a, b, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
            if isinstance(y, tuple):                                            # older PyTorch: (out, amax)
                y = y[0]
            ctx.save_for_backward(x2, w)
            return y

        @staticmethod
        def backward(ctx, g):
            x2, w = ctx.saved_tensors
            g = g.to(torch.bfloat16)
            gx = g @ w.to(torch.bfloat16).t()
            gw = x2.to(torch.bfloat16).t() @ g
            return gx.to(x2.dtype), gw.to(w.dtype)
    return F8


def matmul(x, w):
    """x @ w: on fp8 tensor cores when that is on and the shapes allow it, else the ordinary product."""
    if not _HW["on"] or not x.is_cuda or w.dim() != 2 or x.shape[-1] % 16 or w.shape[1] % 16:
        return x @ w
    import torch
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    pad = (-M) % 16
    if pad:
        x2 = torch.cat([x2, x2.new_zeros((pad, K))])
    try:
        if "fn" not in _HW:
            _HW["fn"] = _fp8_fn()
        y = _HW["fn"].apply(x2.contiguous(), w)
    except Exception as e:                                  # noqa: BLE001 - never fail a run over the fast path
        _HW["on"], _HW["why"] = False, f"{type(e).__name__}: {e}"
        print(f"fp8 tensor cores unavailable ({_HW['why'][:160]}); continuing in bfloat16 with fp8 weights")
        return x @ w
    return y[:M].reshape(*x.shape[:-1], w.shape[1])
