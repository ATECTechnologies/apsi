"""PyTorch backend for `use layers`.

The same model value runs on NumPy (default, exact float64, always available) or on PyTorch — CPU,
CUDA (NVIDIA / AMD ROCm), Apple MPS or Intel XPU — with torch autograd, torch.optim, mixed precision,
torch.compile, cuDNN LSTMs and fused (flash) attention:

    let r = fit(model, params, X, y, loss="cross_entropy", backend="torch", device="cuda", precision="bf16")
    let m = torch_module(model, r.params)      # a real torch.nn.Module (state_dict, DDP, ONNX export, …)
    let p2 = params_from_torch(model, m)       # back to an Apsi parameter tree

Every layer of `use layers` has a torch implementation below that follows the NumPy definition exactly,
so `check_torch()` can compare both backends layer by layer on your machine.
"""
import math

import os

import numpy as np

from . import ops
from .errors import RuntimeErr
from .values import Record, tree_map, tree_leaves

_torch = None


def available():
    try:
        _import()
        return True
    except RuntimeErr:
        return False


def _import():
    global _torch
    if _torch is None:
        try:
            import torch
        except ImportError:
            raise RuntimeErr("backend=\"torch\" needs PyTorch: install it with  pip install torch  "
                             "(see https://pytorch.org for the CUDA/ROCm builds)") from None
        _torch = torch
    return _torch


def _F():
    return _import().nn.functional


# =============================================================== devices
def devices():
    """Every compute device this machine offers, as records(name, kind, detail)."""
    out = [Record(name="numpy", kind="cpu", detail=f"NumPy {np.__version__}, float64, {_threads()} threads")]
    try:
        torch = _import()
    except RuntimeErr:
        return out
    out.append(Record(name="cpu", kind="cpu", detail=f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads"))
    if torch.cuda.is_available():
        kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(Record(name=f"cuda:{i}", kind=kind,
                              detail=f"{p.name}, {p.total_memory / 2 ** 30:.1f} GiB, bf16 "
                                     f"{'yes' if torch.cuda.is_bf16_supported() else 'no'}"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append(Record(name="mps", kind="mps", detail="Apple Metal (float32)"))
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        for i in range(xpu.device_count()):
            out.append(Record(name=f"xpu:{i}", kind="xpu", detail=str(xpu.get_device_name(i))))
    return out


def _threads():
    import os
    return os.cpu_count() or 1


def pick_device(device="auto"):
    torch = _import()
    d = str(device or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeErr("device=\"cuda\": PyTorch sees no CUDA/ROCm GPU (check the driver and that the GPU build of "
                         "PyTorch is installed); use device=\"auto\" to fall back to the CPU")
    if d == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeErr("device=\"mps\" is only available on Apple-silicon Macs")
    return torch.device(d)


_PRECISION = {"fp64": ("float64", None), "float64": ("float64", None), "f64": ("float64", None),
              "fp32": ("float32", None), "float32": ("float32", None), "f32": ("float32", None),
              "tf32": ("float32", None),
              "bf16": ("float32", "bfloat16"), "bfloat16": ("float32", "bfloat16"),
              "fp16": ("float32", "float16"), "float16": ("float32", "float16"), "f16": ("float32", "float16")}


def auto_precision(dev):
    """The fastest precision that trains reliably on `dev`: bf16 mixed precision where the GPU has bf16 (NVIDIA
    Ampere and newer, AMD MI200+, Intel XPU), fp16 with loss scaling on older CUDA GPUs and Apple MPS, fp32 on the
    CPU. APSI_PRECISION overrides it for every program."""
    torch = _import()
    env = os.environ.get("APSI_PRECISION", "").strip().lower()
    if env and env != "auto":
        return env
    if dev.type == "cuda":
        try:
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        except Exception:
            return "fp16"
    if dev.type == "xpu":
        return "bf16"
    if dev.type == "mps":
        return "fp16"
    return "fp32"


def _precision(name, dev):
    """(storage dtype, autocast dtype or None). `name` is "auto" (default), a name such as "bf16", or a record /
    dict {store: "float32", compute: "bfloat16"} for any pair torch has."""
    torch = _import()
    if isinstance(name, dict) or hasattr(name, "fields") or hasattr(name, "store"):
        get = (lambda k: name.get(k)) if isinstance(name, dict) else (lambda k: getattr(name, k, None))
        store, amp = get("store") or "float32", get("compute")
        for t in (store, amp):
            if t is not None and not isinstance(getattr(torch, str(t), None), torch.dtype):
                raise RuntimeErr(f"precision: '{t}' is not a torch dtype (e.g. float32, bfloat16, float16)")
        return getattr(torch, str(store)), (getattr(torch, str(amp)) if amp and amp != store else None)
    key = str(name or "auto").lower()
    if key == "auto":
        key = auto_precision(dev)
    from . import lowbit
    if lowbit.parse(key):                   # fp8 / fp6 / fp4 / int8 / int4 weights: float32 master, 16-bit products
        if dev.type == "cuda":
            try:
                half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                half = torch.float16
            return torch.float32, half
        return torch.float32, (torch.bfloat16 if dev.type == "xpu" else None)
    if key not in _PRECISION:
        raise RuntimeErr(f"precision must be auto, fp64, fp32, tf32, bf16, fp16, fp8, fp6, fp6_e2m3, fp4, int8, int4 "
                         f"or {{store, compute}} (got '{name}')")
    store, amp = _PRECISION[key]
    if store == "float64" and dev.type == "mps":
        raise RuntimeErr("precision=\"fp64\" is not supported on Apple MPS; use fp32")
    if key == "tf32" and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return getattr(torch, store), (getattr(torch, amp) if amp else None)


# =============================================================== conversion
def to_tensor(x, dev, dtype):
    """Apsi value -> torch tensor (integer arrays stay integer, e.g. token ids and class labels)."""
    torch = _import()
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    a = np.asarray(ops.raw(x))
    if a.dtype == bool:
        return torch.as_tensor(a, device=dev)
    if np.issubdtype(a.dtype, np.integer):
        return torch.as_tensor(a.astype(np.int64)).to(dev)
    # convert on the CPU first: some devices (Apple MPS) have no float64
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64)).to(dtype).to(dev)


def to_numpy(t):
    torch = _import()
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        a = t.cpu().numpy()
        return a.astype(np.float64) if np.issubdtype(a.dtype, np.floating) else a
    return t


def _inputs(x, dev, dtype):
    if isinstance(x, (list, tuple)):
        return [_inputs(v, dev, dtype) for v in x]
    return to_tensor(x, dev, dtype)


def params_to_torch(params, dev, dtype, requires_grad=True):
    def conv(leaf):
        t = to_tensor(leaf, dev, dtype)
        if t.is_floating_point() and requires_grad:
            t.requires_grad_(True)
        return t
    return tree_map(conv, params)


def params_to_numpy(tparams):
    return tree_map(to_numpy, tparams)


class TorchGraph:
    """A `graph()` with its index arrays on the torch device."""

    def __init__(self, g, dev, dtype):
        torch = _import()
        self.n, self.edges, self.graphs = g.n, g.edges, g.graphs
        self.src = torch.as_tensor(g.src, device=dev)
        self.dst = torch.as_tensor(g.dst, device=dev)
        self.batch = torch.as_tensor(g.batch, device=dev)
        keep = g.src != g.dst
        self.src_nl = torch.as_tensor(g.src[keep], device=dev)
        self.dst_nl = torch.as_tensor(g.dst[keep], device=dev)
        deg = np.maximum(g.in_degree[g.dst] * g.in_degree[g.src], 1.0)
        self.gcn_norm = torch.as_tensor(1.0 / np.sqrt(deg)).to(dtype).to(dev)


def _ctx(ctx, dev, dtype):
    out = dict(ctx)
    g = out.get("graph")
    if g is not None and not isinstance(g, TorchGraph):
        out["graph"] = TorchGraph(g, dev, dtype)
    if out.get("mask") is not None:
        out["mask"] = to_tensor(out["mask"], dev, dtype)
    return out


# =============================================================== graph ops
def segment_sum(data, ids, n):
    torch = _import()
    out = torch.zeros((int(n),) + tuple(data.shape[1:]), dtype=data.dtype, device=data.device)
    return out.index_add(0, ids, data)


def segment_count(ids, n, like):
    torch = _import()
    return torch.bincount(ids, minlength=int(n)).to(like.dtype)


def segment_mean(data, ids, n):
    cnt = segment_count(ids, n, data).clamp(min=1.0)
    s = segment_sum(data, ids, n)
    return s / cnt.reshape((-1,) + (1,) * (s.dim() - 1))


def segment_max(data, ids, n):
    torch = _import()
    out = torch.zeros((int(n),) + tuple(data.shape[1:]), dtype=data.dtype, device=data.device)
    idx = ids.reshape((-1,) + (1,) * (data.dim() - 1)).expand_as(data)
    return out.scatter_reduce(0, idx, data, reduce="amax", include_self=False)


def segment_softmax(scores, ids, n):
    m = segment_max(scores.detach(), ids, n)[ids]
    e = (scores - m).exp()
    return e / (segment_sum(e, ids, n)[ids] + 1e-16)


# =============================================================== activations
def _act(name):
    torch = _import()
    F = torch.nn.functional
    table = {
        "relu": F.relu, "gelu": lambda x: F.gelu(x, approximate="tanh"), "gelu_exact": F.gelu, "tanh": torch.tanh,
        "sigmoid": torch.sigmoid, "softplus": F.softplus, "silu": F.silu, "swish": F.silu,
        "leaky_relu": lambda x: F.leaky_relu(x, 0.01), "elu": F.elu, "mish": F.mish,
        "linear": lambda x: x, "identity": lambda x: x, "softmax": lambda x: torch.softmax(x, dim=-1),
    }
    return table.get(name)


# =============================================================== layers
_IMPL = {}


def impl(*classes):
    def deco(fn):
        for c in classes:
            _IMPL[c] = fn
        return fn
    return deco


def forward(layer, p, x, train=False, **ctx):
    """Run any `use layers` model on torch tensors."""
    fn = _IMPL.get(type(layer))
    if fn is None:
        custom = getattr(layer, "torch_apply", None)
        if custom is None:
            raise RuntimeErr(f"the torch backend has no implementation of {layer.describe()}; "
                             f"use backend=\"numpy\" for this model")
        return custom(p, x, train=train, **ctx)
    return fn(layer, p, x, train, ctx)


def _register():
    from . import layers as L
    torch = _import()
    F = torch.nn.functional

    from .lowbit import matmul as _mm

    @impl(L.Linear)
    def _linear(l, p, x, train, ctx):
        y = _mm(x, p.W)
        return y + p.b if l.bias else y

    @impl(L.Bilinear)
    def _bilinear(l, p, x, train, ctx):
        a, b = x
        return torch.einsum("...i,kij,...j->...k", a, p.W, b) + p.b

    @impl(L.Embedding)
    def _embedding(l, p, x, train, ctx):
        return F.embedding(x.long(), p.E)

    @impl(L.PositionalEncoding)
    def _posenc(l, p, x, train, ctx):
        T = x.shape[-2]
        if l.learned:
            return x + p.P[0:T]
        pos = np.arange(T)[:, None]
        i = np.arange(l.d)[None, :]
        ang = pos / np.power(10000.0, (2 * (i // 2)) / l.d)
        pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
        return x + torch.as_tensor(pe).to(x.dtype).to(x.device)

    @impl(L.Conv2D)
    def _conv(l, p, x, train, ctx):
        return F.conv2d(x, p.W, p.b, stride=l.stride, padding=l.padding)

    @impl(L.MaxPool)
    def _maxpool(l, p, x, train, ctx):
        return F.max_pool2d(x, l.k)

    @impl(L.AvgPool)
    def _avgpool(l, p, x, train, ctx):
        return F.avg_pool2d(x, l.k)

    @impl(L.GlobalAvgPool)
    def _gap(l, p, x, train, ctx):
        return x.mean(dim=(2, 3))

    @impl(L.Flatten)
    def _flatten(l, p, x, train, ctx):
        return x.reshape(x.shape[0], -1)

    @impl(L.Reshape)
    def _reshape(l, p, x, train, ctx):
        return x.reshape(x.shape[0], *l.shape)

    @impl(L.MeanPool)
    def _meanpool(l, p, x, train, ctx):
        return x.mean(dim=l.axis)

    @impl(L.LSTMLayer)
    def _lstm(l, p, x, train, ctx):
        out = lstm(p, x, l.n_in, l.n_hidden)
        return out if l.seq else out[:, -1, :]

    @impl(L.GRULayer)
    def _gru(l, p, x, train, ctx):
        H = l.n_hidden
        h = x.new_zeros((x.shape[0], H))
        outs = []
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            zr = torch.sigmoid(torch.cat([xt, h], dim=-1) @ p.Wzr + p.bzr)
            z, r = zr[:, 0:H], zr[:, H:2 * H]
            cand = torch.tanh(torch.cat([xt, r * h], dim=-1) @ p.Wh + p.bh)
            h = (1 - z) * h + z * cand
            outs.append(h)
        out = torch.stack(outs, dim=1)
        return out if l.seq else out[:, -1, :]

    @impl(L.LastStep)
    def _last(l, p, x, train, ctx):
        return x[:, -1, :]

    @impl(L.MultiHeadAttention)
    def _mha(l, p, x, train, ctx):
        src = x if not isinstance(x, (list, tuple)) else x[0]
        mem = src if not isinstance(x, (list, tuple)) else x[1]
        B, T, D = src.shape
        S = mem.shape[1]
        hd = D // l.h

        def split(t, n):
            return t.reshape(B, n, l.h, hd).permute(0, 2, 1, 3)
        q, k, v = split(_mm(src, p.Wq), T), split(_mm(mem, p.Wk), S), split(_mm(mem, p.Wv), S)
        mask = ctx.get("mask")
        bias = None
        if l.causal:
            bias = torch.triu(torch.full((T, S), -1e9, device=src.device, dtype=q.dtype), diagonal=1)
        if mask is not None:
            bias = mask.to(q.dtype) if bias is None else bias + mask.to(q.dtype)
        sdpa = getattr(F, "scaled_dot_product_attention", None)
        if sdpa is not None:
            o = sdpa(q, k, v, attn_mask=bias)                     # fused / flash attention kernels
        else:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            if bias is not None:
                s = s + bias
            o = torch.softmax(s, dim=-1) @ v
        o = o.permute(0, 2, 1, 3).reshape(B, T, D)
        return _mm(o, p.Wo) + p.bo

    @impl(L.LayerNorm)
    def _ln(l, p, x, train, ctx):
        return F.layer_norm(x, (l.d,), p.gamma, p.beta, l.eps)

    @impl(L.RMSNorm)
    def _rms(l, p, x, train, ctx):
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + l.eps) * p.gamma

    @impl(L.BatchNorm)
    def _bn(l, p, x, train, ctx):
        nd = x.dim()
        axes = (0,) if nd == 2 else tuple(i for i in range(nd) if i != 1)
        shape = (1, l.d) + (1,) * (nd - 2)
        m = x.mean(dim=axes, keepdim=True)
        v = ((x - m) * (x - m)).mean(dim=axes, keepdim=True)
        return (x - m) / torch.sqrt(v + l.eps) * p.gamma.reshape(shape) + p.beta.reshape(shape)

    @impl(L.Dropout)
    def _dropout(l, p, x, train, ctx):
        return F.dropout(x, l.p, training=bool(train)) if l.p > 0 else x

    @impl(L.Activation)
    def _activation(l, p, x, train, ctx):
        fn = _act(l.name) if isinstance(l.name, str) else None
        if fn is None:
            raise RuntimeErr(f"activation({l.name}) is a custom function; the torch backend supports the named "
                             f"activations ({', '.join(L.ACTIVATIONS)})")
        return fn(x)

    @impl(L.LambdaLayer)
    def _lambda(l, p, x, train, ctx):
        tf = getattr(l, "torch_fn", None)
        if tf is None:
            raise RuntimeErr(f"lambda_layer({l.name}) runs NumPy code; give it a torch version with "
                             f"lambda_layer(fn, torch_fn=…) or use backend=\"numpy\"")
        return tf(x)

    @impl(L.Scale)
    def _scale(l, p, x, train, ctx):
        return x * p.s + p.t if l.shift else x * p.s

    @impl(L.Identity)
    def _identity(l, p, x, train, ctx):
        return x

    @impl(L.Select)
    def _select(l, p, x, train, ctx):
        return x[l.i]

    @impl(L.Sequential)
    def _seq(l, params, x, train, ctx):
        for layer, p in zip(l.layers, params):
            x = forward(layer, p, x, train, **ctx)
        return x

    @impl(L.Residual)
    def _res(l, p, x, train, ctx):
        h = forward(l.norm, p.norm, x, train, **ctx) if l.norm else x
        return x + forward(l.inner, p.inner, h, train, **ctx)

    @impl(L.Parallel)
    def _par(l, params, x, train, ctx):
        return _combine([forward(b, p, x, train, **ctx) for b, p in zip(l.branches, params)], l.combine)

    @impl(L.Combine)
    def _comb(l, p, x, train, ctx):
        return _combine(list(x), l.how)

    @impl(L.DAG)
    def _dag(l, params, x, train, ctx):
        ctx = dict(ctx)
        ctx["_dag_params"] = params
        vals = {}
        if len(l.inputs) == 1:
            vals[l.inputs[0]] = x
        else:
            vals.update(zip(l.inputs, x))
        for name, layer, srcs, single in l.nodes:
            inp = vals[srcs[0]] if single else [vals[s] for s in srcs]
            vals[name] = forward(layer, getattr(params, name), inp, train, **ctx)
        return vals[l.output] if isinstance(l.output, str) else [vals[o] for o in l.output]

    @impl(L._Shared)
    def _shared(l, p, x, train, ctx):
        return forward(l.layer, getattr(ctx["_dag_params"], l.owner), x, train, **ctx)

    @impl(L._Reparam)
    def _reparam(l, p, x, train, ctx):
        mu, logvar = x
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if train else mu

    @impl(L.TransformerBlock)
    def _block(l, p, x, train, ctx):
        return forward(l.mlp, p.mlp, forward(l.attn, p.attn, x, train, **ctx), train, **ctx)

    def _g(ctx):
        g = ctx.get("graph")
        if g is None:
            raise RuntimeErr("graph layers need the graph: fit(…, graph=g) / predict(…, graph=g)")
        return g

    @impl(L.GCN)
    def _gcn(l, p, x, train, ctx):
        g = _g(ctx)
        out = segment_sum((x @ p.W)[g.src] * g.gcn_norm[:, None], g.dst, g.n)
        return out + p.b if l.bias else out

    @impl(L.SAGE)
    def _sage(l, p, x, train, ctx):
        g = _g(ctx)
        msg = x[g.src]
        f = {"mean": segment_mean, "sum": segment_sum, "max": segment_max}[l.agg]
        return x @ p.Ws + f(msg, g.dst, g.n) @ p.Wn + p.b

    @impl(L.GAT)
    def _gat(l, p, x, train, ctx):
        g = _g(ctx)
        h = (x @ p.W).reshape(g.n, l.h, l.n_out)
        e = F.leaky_relu((h * p.a_src).sum(-1)[g.src] + (h * p.a_dst).sum(-1)[g.dst], l.slope)
        alpha = segment_softmax(e, g.dst, g.n)
        out = segment_sum(h[g.src] * alpha.reshape(g.edges, l.h, 1), g.dst, g.n)
        out = out.reshape(g.n, l.h * l.n_out) if l.concat else out.mean(dim=1)
        return out + p.b

    @impl(L.GIN)
    def _gin(l, p, x, train, ctx):
        g = _g(ctx)
        agg = segment_sum(x[g.src_nl], g.dst_nl, g.n)
        return forward(l.mlp, p.mlp, (1 + p.eps) * x + agg, train, **ctx)

    @impl(L.GraphPool)
    def _gpool(l, p, x, train, ctx):
        g = _g(ctx)
        f = {"mean": segment_mean, "sum": segment_sum, "max": segment_max}[l.how]
        return f(x, g.batch, g.graphs)


def _combine(outs, how):
    torch = _import()
    if how == "list":
        return list(outs)
    if how == "concat":
        return torch.cat(outs, dim=-1)
    acc = outs[0]
    for o in outs[1:]:
        acc = acc * o if how == "mul" else acc + o
    return acc / len(outs) if how == "mean" else acc


def lstm(p, x, n_in, H):
    """LSTM over [B, T, n_in] with Apsi's fused weights. On GPUs this runs as one cuDNN call."""
    torch = _import()
    W, b = p.W, p.b
    fc = getattr(getattr(torch, "func", None), "functional_call", None)
    if fc is not None and x.device.type in ("cuda", "xpu") and x.dtype in (torch.float32, torch.float16,
                                                                             torch.bfloat16, torch.float64):
        try:                                    # same gate order (i, f, g, o) as torch.nn.LSTM
            mod = _lstm_shell(n_in, H, x.device, W.dtype)
            weights = {"weight_ih_l0": W[:n_in].t().contiguous(), "weight_hh_l0": W[n_in:].t().contiguous(),
                       "bias_ih_l0": b, "bias_hh_l0": torch.zeros_like(b)}
            out, _ = fc(mod, weights, (x.to(W.dtype),))
            return out
        except (RuntimeError, TypeError):
            pass                                # older torch: fall back to the explicit loop
    h = x.new_zeros((x.shape[0], H))
    c = x.new_zeros((x.shape[0], H))
    xw = x @ W[:n_in] + b                       # input projection for all time steps at once
    Wh = W[n_in:]
    outs = []
    for t in range(x.shape[1]):
        z = xw[:, t, :] + h @ Wh
        i, f = torch.sigmoid(z[:, 0:H]), torch.sigmoid(z[:, H:2 * H])
        g, o = torch.tanh(z[:, 2 * H:3 * H]), torch.sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, dim=1)


_SHELLS = {}


def _lstm_shell(n_in, H, dev, dtype):
    key = (n_in, H, str(dev), str(dtype))
    if key not in _SHELLS:
        torch = _import()
        _SHELLS[key] = torch.nn.LSTM(n_in, H, batch_first=True).to(device=dev, dtype=dtype)
    return _SHELLS[key]


_REGISTERED = False


def ensure():
    global _REGISTERED
    _import()
    if not _REGISTERED:
        _register()
        _REGISTERED = True


# =============================================================== nn.Module wrapper
def torch_module(model, params, device="cpu", precision="fp32"):
    """Wrap an Apsi model and its parameters as a torch.nn.Module (forward(x, **context))."""
    ensure()
    torch = _import()
    dev = pick_device(device)
    store, _ = _precision(precision, dev)

    class ApsiModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.apsi_model = model
            self.like = params
            leaves = tree_leaves(params)
            self.flat = torch.nn.ParameterList()
            self.slots = []                                   # index into flat, or a constant buffer
            for i, leaf in enumerate(leaves):
                t = to_tensor(leaf, dev, store)
                if t.is_floating_point():
                    self.slots.append(len(self.flat))
                    self.flat.append(torch.nn.Parameter(t))
                else:
                    self.register_buffer(f"const{i}", t)
                    self.slots.append(f"const{i}")

        def tree(self):
            it = iter([self.flat[s] if isinstance(s, int) else getattr(self, s) for s in self.slots])
            return tree_map(lambda _: next(it), self.like)

        def forward(self, x, **ctx):
            first = next(iter(self.parameters()), None)
            d = first.device if first is not None else dev
            dt = first.dtype if first is not None else store
            return forward(self.apsi_model, self.tree(), _inputs(x, d, dt), self.training, **_ctx(ctx, d, dt))

        def extra_repr(self):
            return model.describe()

    return ApsiModule()


def from_torch(model, module):
    """The parameters of a module made by torch_module(), as an Apsi parameter tree."""
    return params_to_numpy(module.tree())


# =============================================================== training
def _loss(loss):
    torch = _import()
    F = torch.nn.functional
    if callable(loss) and not isinstance(loss, str):
        tf = getattr(loss, "torch_fn", None)
        if tf is not None:
            return tf
        raise RuntimeErr("fit(backend=\"torch\"): a custom loss function must be written for torch tensors; "
                         "use one of the named losses (mse, mae, huber, cross_entropy, bce, vae) or backend=\"numpy\"")

    def ce(logits, y):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.long().reshape(-1))

    def vae(out, x):
        recon, mu, logvar = out
        rec = ((recon - x) ** 2).sum(-1).mean()
        kl = -0.5 * (1 + logvar - mu * mu - logvar.exp()).sum(-1).mean()
        return rec + kl
    table = {"mse": lambda p, y: F.mse_loss(p, y.to(p.dtype)), "mae": lambda p, y: F.l1_loss(p, y.to(p.dtype)),
             "huber": lambda p, y: F.smooth_l1_loss(p, y.to(p.dtype)), "cross_entropy": ce,
             "bce": lambda p, y: F.binary_cross_entropy_with_logits(p, y.to(p.dtype)), "vae": vae}
    if loss not in table:
        raise RuntimeErr(f"fit(): unknown loss '{loss}' (choose from {', '.join(table)})")
    return table[loss]


def _optimizer(opt, tensors):
    torch = _import()
    from .stdlib import SGD, Adam, AdamW
    if opt is None:
        return torch.optim.Adam(tensors, lr=1e-3)
    if isinstance(opt, str):
        name, kw = opt.lower(), {}
    else:
        name = type(opt).__name__.lower()
        kw = {"lr": float(opt.lr)}
        if isinstance(opt, Adam):
            kw.update(betas=(float(opt.b1), float(opt.b2)), eps=float(opt.eps))
        if isinstance(opt, AdamW):
            kw["weight_decay"] = float(opt.wd)
        if isinstance(opt, SGD):
            kw["momentum"] = float(opt.momentum)
    table = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD,
             "rmsprop": torch.optim.RMSprop, "adagrad": torch.optim.Adagrad, "nadam": torch.optim.NAdam,
             "radam": torch.optim.RAdam, "lbfgs": torch.optim.LBFGS}
    if name not in table:
        raise RuntimeErr(f"fit(backend=\"torch\"): unknown optimizer '{name}' (choose from {', '.join(table)})")
    if name == "sgd" and "lr" not in kw:
        kw["lr"] = 0.01
    return table[name](tensors, **kw)


def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, graph=None, mask=None, val=None, device="auto", precision="auto",
        compile=False):
    ensure()
    torch = _import()
    dev = pick_device(device)
    store, amp = _precision(precision, dev)
    from . import lowbit
    kind = lowbit.parse(auto_precision(dev) if str(precision).lower() == "auto" else precision)
    lowbit._HW["on"] = kind == "fp8" and lowbit.fp8_matmul_available(dev)
    if kind and verbose:
        print(f"training with {kind} weights (one scale per 32 values, float32 master copy)"
              + (", products on fp8 tensor cores" if lowbit._HW["on"] else
                 f", products in {str(amp).replace('torch.', '') if amp else 'float32'}"))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    lf = _loss(loss)
    tp = params_to_torch(params, dev, store)
    leaves = [t for t in tree_leaves(tp) if t.is_floating_point()]
    optim = _optimizer(opt, leaves)
    ctx = _ctx({"graph": graph} if graph is not None else {}, dev, store)
    Xt = _inputs(X, dev, store)
    Yt = Xt if y is None else _inputs(y, dev, store)
    sel = torch.as_tensor(np.nonzero(np.asarray(ops.raw(mask)).astype(bool))[0], device=dev) if mask is not None else None

    def run(p, xb, train):
        return forward(model, p, xb, train, **ctx)
    state = {"fn": run, "compiled": False}
    if compile:
        try:
            state["fn"], state["compiled"] = torch.compile(run, dynamic=True), True
        except Exception as e:                                  # compile is an optimisation, never a failure
            if verbose:
                print(f"torch.compile unavailable ({type(e).__name__}: {e}); running eagerly")

    def step_fn(p, xb, train):
        if not state["compiled"]:
            return run(p, xb, train)
        try:
            return state["fn"](p, xb, train)
        except Exception as e:                                  # e.g. no C++ compiler for inductor
            state["compiled"] = False
            if verbose:
                print(f"torch.compile failed ({type(e).__name__}); continuing eagerly")
            return run(p, xb, train)

    use_scaler = amp is torch.float16 and dev.type == "cuda"
    scaler = None
    if use_scaler:
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") \
            else torch.cuda.amp.GradScaler()

    def objective(xb, yb):
        p = lowbit.fake_quant_tree_torch(tp, kind) if kind and not lowbit._HW["on"] else tp
        if amp is not None:
            with torch.autocast(device_type=dev.type, dtype=amp):
                out = step_fn(p, xb, True)
        else:
            out = step_fn(p, xb, True)
        if sel is not None:
            out, yb = out[sel], yb[sel]
        return lf(_float(out), yb)

    def index(t, b):
        return [index(v, b) for v in t] if isinstance(t, list) else t[b]
    n = (Xt[0] if isinstance(Xt, list) else Xt).shape[0]
    full = graph is not None or not batch_size
    rng = np.random.default_rng(int(seed))
    history, val_hist, step = [], [], 0
    for ep in range(int(epochs)):
        if full:
            batches = [None]
        else:
            perm = torch.as_tensor(rng.permutation(n), device=dev)
            batches = [perm[i:i + int(batch_size)] for i in range(0, n, int(batch_size))]
        total = torch.zeros((), device=dev)
        for b in batches:
            xb = Xt if b is None else index(Xt, b)
            yb = Yt if b is None else index(Yt, b)
            if schedule is not None:
                for group in optim.param_groups:
                    group["lr"] = float(ops.raw(schedule(step)))
            optim.zero_grad(set_to_none=True)
            l = objective(xb, yb)
            if scaler is not None:
                scaler.scale(l).backward()
                scaler.unscale_(optim)
            else:
                l.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(leaves, float(clip))
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            total = total + l.detach().float() * (1 if b is None else len(b))
            step += 1
        history.append(float(total) / (1 if full else n))       # one device sync per epoch
        extra = ""
        if val is not None:
            vx, vy = val
            with torch.no_grad():
                pv = lowbit.fake_quant_tree_torch(tp, kind) if kind and not lowbit._HW["on"] else tp
                vl = float(lf(_float(run(pv, _inputs(vx, dev, store), False)), _inputs(vy, dev, store)))
            val_hist.append(vl)
            extra = f"  val {vl:.5f}"
        if verbose and (ep % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1):
            print(f"epoch {ep:4}  loss {history[-1]:.5f}{extra}")
    lowbit._HW["on"] = False
    return Record(params=params_to_numpy(tp), history=history, val_history=val_hist, device=str(dev))


def _float(out):
    torch = _import()
    if isinstance(out, list):
        return [_float(o) for o in out]
    return out.float() if out.dtype in (torch.float16, torch.bfloat16) else out


def predict(model, params, X, device="auto", precision="auto", **ctx):
    ensure()
    torch = _import()
    dev = pick_device(device)
    store, amp = _precision(precision, dev)
    from . import lowbit
    kind = lowbit.parse(auto_precision(dev) if str(precision).lower() == "auto" else precision)
    with torch.no_grad():
        tp = params_to_torch(params, dev, store, requires_grad=False)
        if kind:
            tp = lowbit.fake_quant_tree_torch(tp, kind)
        if amp is not None:
            with torch.autocast(device_type=dev.type, dtype=amp):
                out = forward(model, tp, _inputs(X, dev, store), False, **_ctx(ctx, dev, store))
        else:
            out = forward(model, tp, _inputs(X, dev, store), False, **_ctx(ctx, dev, store))
    return tree_map(to_numpy, out) if isinstance(out, list) else to_numpy(out)


# =============================================================== self check
def check(device="auto", verbose=True):
    """Run every layer type on NumPy and on torch (float64) and compare outputs and gradients (device "auto" = the GPU when there is one)."""
    ensure()
    torch = _import()
    from . import layers as L
    from .stdlib import value_and_grad
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((2, 5, 8))
    img = rng.standard_normal((2, 3, 8, 8))
    ids = rng.integers(0, 20, (2, 5))
    g = L.graph(edges=[[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]], n=4, batch=[0, 0, 1, 1])
    nodes = rng.standard_normal((4, 6))
    cases = [
        ("mlp", L.mlp([8, 16, 4], "gelu", norm=True), rng.standard_normal((3, 8)), {}),
        ("activations", L.Sequential([L.Linear(8, 8)] + [L.Activation(a) for a in
                                                         ("relu", "tanh", "silu", "elu", "leaky_relu", "mish",
                                                          "softplus", "sigmoid")]), rng.standard_normal((3, 8)), {}),
        ("bilinear", L.Bilinear(3, 4, 2), [rng.standard_normal((2, 3)), rng.standard_normal((2, 4))], {}),
        ("cnn", L.cnn(3, 5, image=8, channels=(4,)), img, {}),
        ("conv/avgpool/batchnorm", L.Sequential([L.Conv2D(3, 4, 3), L.BatchNorm(4), L.AvgPool(2), L.GlobalAvgPool()]),
         img, {}),
        ("lstm", L.lstm_classifier(8, 6, 3, layers=2), seq, {}),
        ("gru", L.Sequential([L.GRULayer(8, 6), L.LastStep()]), seq, {}),
        ("transformer_lm", L.transformer_lm(20, d=8, heads=2, layers=1, context=8), ids, {}),
        ("transformer_classifier", L.transformer_classifier(20, 3, d=8, heads=2, layers=1, context=8), ids, {}),
        ("rms/scale/residual", L.Sequential([L.Residual(L.Linear(8, 8), pre_norm=L.RMSNorm(8)), L.Scale(8)]),
         rng.standard_normal((3, 8)), {}),
        ("parallel/dag", L.DAG([["a", L.Linear(8, 4), "x"], ["b", L.Parallel([L.Linear(8, 4), L.Linear(8, 4)], "add"), "x"],
                                ["c", L.Combine("concat"), ["a", "b"]], ["o", L.Linear(8, 2), "c"]]),
         rng.standard_normal((3, 8)), {}),
        ("siamese", L.siamese(L.Linear(8, 4), "concat", L.Linear(8, 1)),
         [rng.standard_normal((3, 8)), rng.standard_normal((3, 8))], {}),
        ("gcn", L.gcn_net(6, 8, 3, dropout=0.0), nodes, {"graph": g}),
        ("sage", L.Sequential([L.SAGE(6, 4, "mean"), L.SAGE(4, 4, "sum")]), nodes, {"graph": g}),
        ("gat", L.gat_net(6, 4, 3, heads=2, dropout=0.0), nodes, {"graph": g}),
        ("gin", L.gin_net(6, 8, 2, layers=2), nodes, {"graph": g}),
        ("vae (eval)", L.vae(8, 2, hidden=(6,)), rng.standard_normal((3, 8)), {}),
    ]
    dev = pick_device(device)
    rows, worst = [], 0.0
    for name, model, x, ctx in cases:
        p = model.init(1)
        ref = model(p, x, **ctx)

        def scalar(out):
            outs = out if isinstance(out, list) else [out]
            return sum(ops.sum_(o * o) for o in outs)
        ref_l, ref_g = value_and_grad(lambda q: scalar(model(q, x, **ctx)))(p)
        tp = params_to_torch(p, dev, torch.float64)
        tout = forward(model, tp, _inputs(x, dev, torch.float64), False, **_ctx(ctx, dev, torch.float64))
        touts = tout if isinstance(tout, list) else [tout]
        tl = sum((o * o).sum() for o in touts)
        tl.backward()
        refs = ref if isinstance(ref, list) else [ref]
        d_out = max(float(np.max(np.abs(np.asarray(ops.raw(r)) - to_numpy(t)))) for r, t in zip(refs, touts))
        d_grad = 0.0
        for a, t in zip(tree_leaves(ref_g), tree_leaves(tp)):
            if getattr(t, "grad", None) is not None:
                d_grad = max(d_grad, float(np.max(np.abs(np.asarray(ops.raw(a), dtype=float) - to_numpy(t.grad)))))
        worst = max(worst, d_out, d_grad)
        rows.append(Record(name=name, output_diff=d_out, grad_diff=d_grad))
        if verbose:
            print(f"{name:<24} output {d_out:9.2e}   gradient {d_grad:9.2e}")
    ok = worst < 1e-6
    if verbose:
        print(f"{'OK' if ok else 'MISMATCH'}: largest difference {worst:.2e} on {dev}")
    return Record(ok=ok, worst=worst, cases=rows, device=str(dev))
