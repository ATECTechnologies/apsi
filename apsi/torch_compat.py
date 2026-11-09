"""`use torch` — PyTorch's layers, losses, optimizers and schedulers, implemented natively in Apsi.

Nothing here calls PyTorch or Python libraries: every layer is built from Apsi's differentiable tensor operations.
The semantics follow PyTorch exactly (defaults, shapes, parameter names, update rules), so programs converted
by `apsi convert` behave like the original and `state_dict`s move between PyTorch and Apsi unchanged.

Parameters use PyTorch's names and layouts (`weight` is [out, in]); a model's parameter tree is its state_dict.
Running statistics of batch norm are buffers held by the layer, as in PyTorch.

Layers     Linear Bilinear Conv1d Conv2d MaxPool1d MaxPool2d AvgPool1d AvgPool2d AdaptiveAvgPool1d AdaptiveAvgPool2d
           AdaptiveMaxPool2d BatchNorm1d BatchNorm2d LayerNorm GroupNorm RMSNorm Embedding Dropout Flatten Unflatten
           Identity LSTM GRU RNN MultiheadAttention TransformerEncoderLayer TransformerEncoder
           TransformerDecoderLayer TransformerDecoder Transformer Sequential ModuleList and every activation
           (ReLU GELU SiLU Tanh Sigmoid Softmax LogSoftmax LeakyReLU ELU SELU CELU Softplus Mish Hardtanh ReLU6
           Hardswish Hardsigmoid PReLU Softsign Tanhshrink GLU)
Functional relu gelu silu … softmax log_softmax dropout linear layer_norm normalize pad one_hot
           scaled_dot_product_attention embedding
Losses     CrossEntropyLoss NLLLoss MSELoss L1Loss SmoothL1Loss HuberLoss BCELoss BCEWithLogitsLoss KLDivLoss
           (and the functional forms) with weight / ignore_index / label_smoothing / pos_weight / reduction
Optimizers SGD (momentum, dampening, nesterov, weight decay) Adam AdamW (amsgrad) RMSprop (centered, momentum)
           Adagrad Adamax NAdam RAdam — PyTorch's update rules
Schedulers StepLR MultiStepLR ExponentialLR CosineAnnealingLR CosineAnnealingWarmRestarts LinearLR ConstantLR
           LambdaLR MultiplicativeLR PolynomialLR OneCycleLR CyclicLR ReduceLROnPlateau SequentialLR ChainedScheduler
State      state_dict(model, params) / load_state_dict(model, params, state) / parameters_count(params)
"""
import copy
import math

import numpy as np

from . import autodiff as ad
from . import ops
from .autodiff import Node
from .errors import RuntimeErr, ShapeError
from .values import Record, tree_map, tree_leaves


# =============================================================================== helpers
def _raw(x):
    return np.asarray(ops.raw(x))


def _shape(x):
    return tuple(np.shape(ops.raw(x)))


def _tuple(v, n):
    if isinstance(v, (list, tuple)):
        return tuple(int(a) for a in v)
    return (int(v),) * n


def _rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))


def _uniform(rng, bound, shape):
    return rng.uniform(-bound, bound, shape)


def _neg_inf_like(shape):
    return np.full(shape, -np.inf)


def _apply_act(name, x, **kw):
    return ACT_FUNCS[name](x, **kw)


def _seed_for(seed, i):
    return None if seed is None else int(seed) * 7919 + i


# =============================================================================== base layer
class TorchLayer:
    """A layer with PyTorch semantics: init(seed) -> parameter tree (PyTorch names), layer(p, *inputs, train=…)."""
    kind = "torch"
    name = "Module"

    def init(self, seed=0):
        return Record()

    def __call__(self, params, *xs, train=None, **kw):
        mode = getattr(self, "training", True) if train is None else bool(train)
        return self.forward(params, *xs, train=mode, **kw)

    def apply(self, params, x, train=False, **ctx):
        return self.forward(params, x, train=bool(train))

    def forward(self, p, x, train=False):
        raise NotImplementedError

    def children(self):                      # (name, layer) pairs for state_dict walking and summaries
        return []

    def describe(self):
        return self.name

    def __repr__(self):
        return self.describe()


# =============================================================================== activations
def _leaky(x, negative_slope=0.01):
    return ops.where(_raw(x) > 0, x, x * negative_slope)


def _elu(x, alpha=1.0):
    return ops.where(_raw(x) > 0, x, alpha * (ops.exp(ops.where(_raw(x) > 0, 0.0 * x, x)) - 1.0))


def _selu(x):
    return 1.0507009873554804934193349852946 * _elu(x, 1.6732632423543772848170429916717)


def _celu(x, alpha=1.0):
    return ops.where(_raw(x) > 0, x, alpha * (ops.exp(ops.where(_raw(x) > 0, 0.0 * x, x) / alpha) - 1.0))


def _softplus(x, beta=1.0, threshold=20.0):
    bx = x * beta
    return ops.where(_raw(bx) > threshold, x, ops.softplus(ops.where(_raw(bx) > threshold, 0.0 * bx, bx)) / beta)


def _hardtanh(x, min_val=-1.0, max_val=1.0):
    r = _raw(x)
    return ops.where(r < min_val, min_val + 0.0 * x, ops.where(r > max_val, max_val + 0.0 * x, x))


def _relu6(x):
    return _hardtanh(ops.relu(x), 0.0, 6.0)


def _hardsigmoid(x):
    return _hardtanh(x / 6.0 + 0.5, 0.0, 1.0)


def _hardswish(x):
    return x * _hardsigmoid(x)


def _mish(x):
    return x * ops.tanh(ops.softplus(x))


def _silu(x):
    return x * ops.sigmoid(x)


def _gelu(x, approximate="none"):
    return ops.gelu(x, "tanh" if approximate == "tanh" else "none")


def _softsign(x):
    return x / (1.0 + ops.abs_(x))


def _tanhshrink(x):
    return x - ops.tanh(x)


def _glu(x, dim=-1):
    n = _shape(x)[dim] // 2
    nd = len(_shape(x))
    d = dim % nd
    a = x[(slice(None),) * d + (slice(0, n),)]
    b = x[(slice(None),) * d + (slice(n, 2 * n),)]
    return a * ops.sigmoid(b)


def _softmax(x, dim=None):
    return ops.softmax(x, axis=-1 if dim is None else int(dim))


def _log_softmax(x, dim=None):
    return ops.log_softmax(x, axis=-1 if dim is None else int(dim))


ACT_FUNCS = {
    "relu": lambda x: ops.relu(x), "gelu": _gelu, "silu": _silu, "tanh": lambda x: ops.tanh(x),
    "sigmoid": lambda x: ops.sigmoid(x), "softmax": _softmax, "log_softmax": _log_softmax, "leaky_relu": _leaky,
    "elu": _elu, "selu": _selu, "celu": _celu, "softplus": _softplus, "mish": _mish, "hardtanh": _hardtanh,
    "relu6": _relu6, "hardswish": _hardswish, "hardsigmoid": _hardsigmoid, "softsign": _softsign,
    "tanhshrink": _tanhshrink, "glu": _glu, "identity": lambda x: x,
}
ACT_CLASSES = {"ReLU": "relu", "GELU": "gelu", "SiLU": "silu", "Tanh": "tanh", "Sigmoid": "sigmoid",
               "Softmax": "softmax", "LogSoftmax": "log_softmax", "LeakyReLU": "leaky_relu", "ELU": "elu",
               "SELU": "selu", "CELU": "celu", "Softplus": "softplus", "Mish": "mish", "Hardtanh": "hardtanh",
               "ReLU6": "relu6", "Hardswish": "hardswish", "Hardsigmoid": "hardsigmoid", "Softsign": "softsign",
               "Tanhshrink": "tanhshrink", "GLU": "glu", "Identity": "identity"}
_ACT_ARGS = {"softmax": ["dim"], "log_softmax": ["dim"], "leaky_relu": ["negative_slope"], "elu": ["alpha"],
             "celu": ["alpha"], "softplus": ["beta", "threshold"], "hardtanh": ["min_val", "max_val"],
             "glu": ["dim"], "gelu": ["approximate"]}


class Activation(TorchLayer):
    def __init__(self, fn, *args, **kw):
        self.fn = fn
        names = _ACT_ARGS.get(fn, [])
        self.kw = {k: v for k, v in zip(names, args)}
        self.kw.update({k: v for k, v in kw.items() if k in names})
        self.name = next((c for c, f in ACT_CLASSES.items() if f == fn), fn)

    def forward(self, p, x, train=False):
        return ACT_FUNCS[self.fn](x, **self.kw)


def _act_class(fn):
    def make(*args, inplace=False, **kw):
        return Activation(fn, *args, **kw)
    make.__name__ = fn
    return make


class PReLU(TorchLayer):
    name = "PReLU"

    def __init__(self, num_parameters=1, init=0.25):
        self.n, self.init_v = int(num_parameters), float(init)

    def init(self, seed=0):
        return Record(weight=np.full(self.n, self.init_v))

    def forward(self, p, x, train=False):
        nd = len(_shape(x))
        w = p.weight if self.n == 1 or nd < 2 else ops.reshape(p.weight, *((1, self.n) + (1,) * (nd - 2)))
        return ops.where(_raw(x) >= 0, x, w * x)


# =============================================================================== dense
class Linear(TorchLayer):
    name = "Linear"

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.i, self.o, self.bias = int(in_features), int(out_features), bool(bias)

    def init(self, seed=0):
        rng = _rng(seed)
        b = 1 / math.sqrt(self.i) if self.i else 0
        w = _uniform(rng, b, (self.o, self.i))
        return Record(weight=w, bias=_uniform(rng, b, (self.o,))) if self.bias else Record(weight=w)

    def forward(self, p, x, train=False):
        y = ops.matmul(x, ops.transpose(p.weight))
        return y + p.bias if self.bias else y

    def describe(self):
        return f"Linear({self.i} → {self.o})"


class Bilinear(TorchLayer):
    name = "Bilinear"

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        self.a, self.b, self.o, self.bias = int(in1_features), int(in2_features), int(out_features), bool(bias)

    def init(self, seed=0):
        rng = _rng(seed)
        bd = 1 / math.sqrt(self.a)
        w = _uniform(rng, bd, (self.o, self.a, self.b))
        return Record(weight=w, bias=_uniform(rng, bd, (self.o,))) if self.bias else Record(weight=w)

    def forward(self, p, x1, x2, train=False):
        outs = [ops.sum_(ops.matmul(x1, p.weight[k]) * x2, axis=-1) for k in range(self.o)]
        y = ops.stack(outs, axis=-1)
        return y + p.bias if self.bias else y


class Identity(TorchLayer):
    name = "Identity"

    def __init__(self, *a, **k):
        pass

    def forward(self, p, x, train=False):
        return x


class Dropout(TorchLayer):
    name = "Dropout"

    def __init__(self, p=0.5, inplace=False):
        self.p = float(p)

    def forward(self, p, x, train=False):
        return dropout(x, self.p, train)


class Flatten(TorchLayer):
    name = "Flatten"

    def __init__(self, start_dim=1, end_dim=-1):
        self.s, self.e = int(start_dim), int(end_dim)

    def forward(self, p, x, train=False):
        return flatten(x, self.s, self.e)


class Unflatten(TorchLayer):
    name = "Unflatten"

    def __init__(self, dim, unflattened_size):
        self.dim, self.sizes = int(dim), tuple(int(s) for s in unflattened_size)

    def forward(self, p, x, train=False):
        sh = list(_shape(x))
        d = self.dim % len(sh)
        return ops.reshape(x, *(sh[:d] + list(self.sizes) + sh[d + 1:]))


class Embedding(TorchLayer):
    name = "Embedding"

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        self.n, self.d = int(num_embeddings), int(embedding_dim)
        self.pad = None if padding_idx is None else int(padding_idx) % self.n
        if max_norm is not None:
            raise RuntimeErr("Embedding(max_norm=…) is not supported")

    def init(self, seed=0):
        w = _rng(seed).standard_normal((self.n, self.d))
        if self.pad is not None:
            w[self.pad] = 0.0
        return Record(weight=w)

    def forward(self, p, idx, train=False):
        return embedding(idx, p.weight, self.pad)


# =============================================================================== convolution and pooling
def _conv_nd(x, w, stride, padding, dilation, groups, nd):
    """N-d convolution (1-D or 2-D) with PyTorch semantics, as one differentiable operation."""
    def prep(xv):
        xv = np.asarray(xv, dtype=float)
        if nd == 1:
            xv = xv[:, :, None, :]
        pads = ((0, 0), (0, 0)) + tuple((p, p) for p in padding)
        return np.pad(xv, pads) if any(padding) else xv

    def windows(xp, kh, kw):
        dh, dw = dilation
        eh, ew = dh * (kh - 1) + 1, dw * (kw - 1) + 1
        win = np.lib.stride_tricks.sliding_window_view(xp, (eh, ew), axis=(2, 3))
        win = win[:, :, ::stride[0], ::stride[1], ::dh, ::dw]        # N, C, OH, OW, kh, kw
        return win

    def fwd(xv, wv):
        wv = np.asarray(wv, dtype=float)
        if nd == 1:
            wv = wv[:, :, None, :]
        xp = prep(xv)
        O, Cg, kh, kw = wv.shape
        win = windows(xp, kh, kw)
        N, C, OH, OW = win.shape[:4]
        outs = []
        og = O // groups
        for g in range(groups):
            wg = win[:, g * Cg:(g + 1) * Cg]
            outs.append(np.einsum("nchwij,ocij->nohw", wg, wv[g * og:(g + 1) * og], optimize=True))
        y = np.concatenate(outs, axis=1) if groups > 1 else outs[0]
        return y[:, :, 0, :] if nd == 1 else y

    def vjp(g, out, need, xv, wv):
        wv4 = np.asarray(wv, dtype=float)
        if nd == 1:
            wv4 = wv4[:, :, None, :]
            g = g[:, :, None, :]
        xv0 = np.asarray(xv, dtype=float)
        xp = prep(xv0)
        O, Cg, kh, kw = wv4.shape
        win = windows(xp, kh, kw)
        OH, OW = win.shape[2:4]
        og = O // groups
        gx = gw = None
        if need[1]:
            gws = [np.einsum("nohw,nchwij->ocij", g[:, k * og:(k + 1) * og], win[:, k * Cg:(k + 1) * Cg], optimize=True)
                   for k in range(groups)]
            gw = np.concatenate(gws, axis=0)
            if nd == 1:
                gw = gw[:, :, 0, :]
        if need[0]:
            dxp = np.zeros_like(xp)
            dh, dw = dilation
            for k in range(groups):
                gk = g[:, k * og:(k + 1) * og]
                wk = wv4[k * og:(k + 1) * og]
                for i in range(kh):
                    for j in range(kw):
                        contrib = np.einsum("nohw,oc->nchw", gk, wk[:, :, i, j], optimize=True)
                        r0, c0 = i * dh, j * dw
                        dxp[:, k * Cg:(k + 1) * Cg, r0:r0 + stride[0] * OH:stride[0],
                            c0:c0 + stride[1] * OW:stride[1]] += contrib
            ph, pw = padding
            H, W = dxp.shape[2] - 2 * ph, dxp.shape[3] - 2 * pw
            gx = dxp[:, :, ph:ph + H, pw:pw + W]
            if nd == 1:
                gx = gx[:, :, 0, :]
        return gx, gw
    return ad._op(fwd, vjp, [x, w], raw_only=True)


class _ConvNd(TorchLayer):
    def __init__(self, nd, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", device=None, dtype=None):
        self.nd = nd
        self.cin, self.cout = int(in_channels), int(out_channels)
        self.k = _tuple(kernel_size, nd)
        self.stride = _tuple(stride, nd)
        self.dil = _tuple(dilation, nd)
        self.groups = int(groups)
        self.bias = bool(bias)
        if padding_mode != "zeros":
            raise RuntimeErr(f"Conv padding_mode='{padding_mode}' is not supported (only 'zeros')")
        if self.cin % self.groups or self.cout % self.groups:
            raise ShapeError("in_channels and out_channels must be divisible by groups")
        if isinstance(padding, str):
            if padding == "valid":
                self.pad = (0,) * nd
            elif padding == "same":
                if any(s != 1 for s in self.stride):
                    raise RuntimeErr("padding='same' needs stride 1")
                tot = [d * (k - 1) for d, k in zip(self.dil, self.k)]
                if any(t % 2 for t in tot):
                    raise RuntimeErr("padding='same' with an even kernel is not supported")
                self.pad = tuple(t // 2 for t in tot)
            else:
                raise RuntimeErr(f"unknown padding {padding!r}")
        else:
            self.pad = _tuple(padding, nd)
        self.name = f"Conv{nd}d"

    def init(self, seed=0):
        rng = _rng(seed)
        fan_in = self.cin // self.groups * int(np.prod(self.k))
        b = 1 / math.sqrt(fan_in)
        w = _uniform(rng, b, (self.cout, self.cin // self.groups) + self.k)
        return Record(weight=w, bias=_uniform(rng, b, (self.cout,))) if self.bias else Record(weight=w)

    def forward(self, p, x, train=False):
        squeeze = len(_shape(x)) == self.nd + 1
        if squeeze:
            x = ops.reshape(x, 1, *_shape(x))
        if self.nd == 1:
            y = _conv_nd(x, p.weight, (1,) + self.stride, (0,) + self.pad, (1,) + self.dil, self.groups, 1)
        else:
            y = _conv_nd(x, p.weight, self.stride, self.pad, self.dil, self.groups, 2)
        if self.bias:
            y = y + ops.reshape(p.bias, *((1, self.cout) + (1,) * self.nd))
        return y[0] if squeeze else y

    def describe(self):
        return f"{self.name}({self.cin} → {self.cout}, k={self.k})"


def Conv1d(*a, **k):
    return _ConvNd(1, *a, **k)


def Conv2d(*a, **k):
    return _ConvNd(2, *a, **k)


def _pool_out(L, k, s, p, d, ceil):
    num = L + 2 * p - d * (k - 1) - 1
    o = (-(-num // s) if ceil else num // s) + 1
    if ceil and (o - 1) * s >= L + p:
        o -= 1
    return o


def _pool(x, kind, k, s, p, nd, ceil=False, count_include_pad=True, divisor_override=None):
    """Max / average pooling with PyTorch's output sizes, padding and ceil_mode, differentiable."""
    xv0 = np.asarray(ops.raw(x), dtype=float)
    squeeze = xv0.ndim == nd + 1
    if squeeze:
        xv0 = xv0[None]
    one = nd == 1
    if one:
        xv0 = xv0[:, :, None, :]
        k, s, p = (1,) + k, (1,) + s, (0,) + p
    N, C, H, W = xv0.shape
    OH = _pool_out(H, k[0], s[0], p[0], 1, ceil)
    OW = _pool_out(W, k[1], s[1], p[1], 1, ceil)
    eh = max(0, (OH - 1) * s[0] + k[0] - H - p[0])
    ew = max(0, (OW - 1) * s[1] + k[1] - W - p[1])
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(xv0, ((0, 0), (0, 0), (p[0], eh), (p[1], ew)), constant_values=fill)
    win = np.lib.stride_tricks.sliding_window_view(xp, (k[0], k[1]), axis=(2, 3))[:, :, ::s[0], ::s[1]][:, :, :OH, :OW]
    if kind == "max":
        flat = win.reshape(N, C, OH, OW, -1)
        arg = flat.argmax(-1)
        out = np.take_along_axis(flat, arg[..., None], -1)[..., 0]
    else:
        out = win.sum((-1, -2))
        if divisor_override:
            div = np.full((OH, OW), float(divisor_override))
        else:
            hs = np.arange(OH) * s[0] - p[0]
            ws = np.arange(OW) * s[1] - p[1]
            if count_include_pad:
                he = np.minimum(hs + k[0], H + p[0])
                we = np.minimum(ws + k[1], W + p[1])
                div = (he - hs)[:, None] * (we - ws)[None, :]
            else:
                he = np.minimum(hs + k[0], H)
                we = np.minimum(ws + k[1], W)
                div = (he - np.maximum(hs, 0))[:, None] * (we - np.maximum(ws, 0))[None, :]
            div = div.astype(float)
        out = out / div

    def vjp(g, o, need, a):
        g = np.asarray(g, dtype=float)
        if squeeze:
            g = g[None]
        if one:
            g = g[:, :, None, :]
        dxp = np.zeros_like(xp)
        if kind == "max":
            ki, kj = np.divmod(arg, k[1])
            n_, c_, oh, ow = np.indices(arg.shape)
            np.add.at(dxp, (n_, c_, oh * s[0] + ki, ow * s[1] + kj), g)
        else:
            gd = g / div
            for i in range(k[0]):
                for j in range(k[1]):
                    dxp[:, :, i:i + s[0] * OH:s[0], j:j + s[1] * OW:s[1]] += gd
        dx = dxp[:, :, p[0]:p[0] + H, p[1]:p[1] + W]
        if one:
            dx = dx[:, :, 0, :]
        if squeeze:
            dx = dx[0]
        return (dx,)
    res = out[:, :, 0, :] if one else out
    res = res[0] if squeeze else res
    return ad._op(lambda a: res, vjp, [x], raw_only=True) if isinstance(x, Node) else res


class _Pool(TorchLayer):
    def __init__(self, kind, nd, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        self.kind, self.nd = kind, nd
        self.k = _tuple(kernel_size, nd)
        self.s = _tuple(stride if stride is not None else kernel_size, nd)
        self.p = _tuple(padding, nd)
        if _tuple(dilation, nd) != (1,) * nd:
            raise RuntimeErr("pooling with dilation is not supported")
        if return_indices:
            raise RuntimeErr("pooling with return_indices=True is not supported")
        self.ceil, self.cip, self.div = bool(ceil_mode), bool(count_include_pad), divisor_override
        self.name = f"{'Max' if kind == 'max' else 'Avg'}Pool{nd}d"

    def forward(self, p, x, train=False):
        return _pool(x, self.kind, self.k, self.s, self.p, self.nd, self.ceil, self.cip, self.div)


def MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    return _Pool("max", 1, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    return _Pool("max", 2, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


def AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return _Pool("avg", 1, kernel_size, stride, padding, 1, False, ceil_mode, count_include_pad)


def AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _Pool("avg", 2, kernel_size, stride, padding, 1, False, ceil_mode, count_include_pad, divisor_override)


def _adaptive(x, out, kind, nd):
    shp = _shape(x)
    sizes = shp[-nd:]
    outs = [sizes[i] if o is None else int(o) for i, o in enumerate(_tuple(out, nd) if not isinstance(out, (list, tuple))
                                                                   else out)]
    if all(s % o == 0 for s, o in zip(sizes, outs)) and kind == "avg":
        k = tuple(s // o for s, o in zip(sizes, outs))
        return _pool(x, "avg", k, k, (0,) * nd, nd)
    if all(s % o == 0 for s, o in zip(sizes, outs)) and kind == "max":
        k = tuple(s // o for s, o in zip(sizes, outs))
        return _pool(x, "max", k, k, (0,) * nd, nd)
    red = ops.mean if kind == "avg" else (lambda t, axis: ops.max_(t, axis=axis))
    lead = len(shp) - nd
    if nd == 1:
        L, O = sizes[0], outs[0]
        cells = [red(x[(slice(None),) * lead + (slice(i * L // O, -(-(i + 1) * L // O)),)], axis=-1) for i in range(O)]
        return ops.stack(cells, axis=-1)
    (H, W), (OH, OW) = sizes, outs
    rows = []
    for i in range(OH):
        h0, h1 = i * H // OH, -(-(i + 1) * H // OH)
        cols = []
        for j in range(OW):
            w0, w1 = j * W // OW, -(-(j + 1) * W // OW)
            cell = x[(slice(None),) * lead + (slice(h0, h1), slice(w0, w1))]
            cols.append(red(red(cell, axis=-1), axis=-1))
        rows.append(ops.stack(cols, axis=-1))
    return ops.stack(rows, axis=-2)


class _Adaptive(TorchLayer):
    def __init__(self, kind, nd, output_size):
        self.kind, self.nd, self.out = kind, nd, output_size
        self.name = f"Adaptive{'Avg' if kind == 'avg' else 'Max'}Pool{nd}d"

    def forward(self, p, x, train=False):
        return _adaptive(x, self.out, self.kind, self.nd)


def AdaptiveAvgPool1d(output_size):
    return _Adaptive("avg", 1, output_size)


def AdaptiveAvgPool2d(output_size):
    return _Adaptive("avg", 2, output_size)


def AdaptiveMaxPool1d(output_size, return_indices=False):
    return _Adaptive("max", 1, output_size)


def AdaptiveMaxPool2d(output_size, return_indices=False):
    return _Adaptive("max", 2, output_size)


# =============================================================================== normalisation
class _BatchNorm(TorchLayer):
    """Batch norm with running statistics (buffers), exactly as PyTorch in train and eval mode."""

    def __init__(self, nd, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        self.c, self.eps, self.mom, self.affine, self.track = int(num_features), float(eps), momentum, bool(affine), \
            bool(track_running_stats)
        self.name = f"BatchNorm{nd}d"
        self.reset_buffers()

    def reset_buffers(self):
        self.running_mean = np.zeros(self.c)
        self.running_var = np.ones(self.c)
        self.num_batches_tracked = 0

    def init(self, seed=0):
        return Record(weight=np.ones(self.c), bias=np.zeros(self.c)) if self.affine else Record()

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var,
                "num_batches_tracked": np.asarray(self.num_batches_tracked, dtype=np.int64)} if self.track else {}

    def set_buffer(self, name, value):
        v = np.asarray(value, dtype=float)
        if name == "running_mean":
            self.running_mean = v.copy()
        elif name == "running_var":
            self.running_var = v.copy()
        elif name == "num_batches_tracked":
            self.num_batches_tracked = int(v)

    def forward(self, p, x, train=False):
        nd = len(_shape(x))
        axes = (0,) if nd == 2 else (0,) + tuple(range(2, nd))
        bshape = (1, self.c) + (1,) * (nd - 2)
        if train or not self.track:
            m = ops.mean(x, axis=axes, keepdims=True)
            d = x - m
            v = ops.mean(d * d, axis=axes, keepdims=True)
            if train and self.track:
                xr = np.asarray(ops.raw(x), dtype=float)
                n = xr.size / self.c
                bm = xr.mean(axis=axes)
                bv = xr.var(axis=axes) * (n / max(n - 1, 1))
                self.num_batches_tracked += 1
                f = (1.0 / self.num_batches_tracked) if self.mom is None else float(self.mom)
                self.running_mean = (1 - f) * self.running_mean + f * bm
                self.running_var = (1 - f) * self.running_var + f * bv
            y = d / ops.sqrt(v + self.eps)
        else:
            y = (x - self.running_mean.reshape(bshape)) / np.sqrt(self.running_var.reshape(bshape) + self.eps)
        if self.affine:
            y = y * ops.reshape(p.weight, *bshape) + ops.reshape(p.bias, *bshape)
        return y


def BatchNorm1d(*a, **k):
    return _BatchNorm(1, *a, **k)


def BatchNorm2d(*a, **k):
    return _BatchNorm(2, *a, **k)


class LayerNorm(TorchLayer):
    name = "LayerNorm"

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        self.shape = _tuple(normalized_shape, 1) if not isinstance(normalized_shape, (list, tuple)) \
            else tuple(int(s) for s in normalized_shape)
        self.eps, self.affine, self.has_bias = float(eps), bool(elementwise_affine), bool(bias)

    def init(self, seed=0):
        if not self.affine:
            return Record()
        return Record(weight=np.ones(self.shape), bias=np.zeros(self.shape)) if self.has_bias \
            else Record(weight=np.ones(self.shape))

    def forward(self, p, x, train=False):
        y = layer_norm(x, self.shape, None, None, self.eps)
        if self.affine:
            y = y * p.weight
            if self.has_bias:
                y = y + p.bias
        return y


class RMSNorm(TorchLayer):
    name = "RMSNorm"

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        self.shape = _tuple(normalized_shape, 1) if not isinstance(normalized_shape, (list, tuple)) \
            else tuple(int(s) for s in normalized_shape)
        self.eps, self.affine = eps, bool(elementwise_affine)

    def init(self, seed=0):
        return Record(weight=np.ones(self.shape)) if self.affine else Record()

    def forward(self, p, x, train=False):
        axes = tuple(range(-len(self.shape), 0))
        eps = float(np.finfo(np.asarray(ops.raw(x)).dtype).eps) if self.eps is None else float(self.eps)
        y = x / ops.sqrt(ops.mean(x * x, axis=axes, keepdims=True) + eps)
        return y * p.weight if self.affine else y


class GroupNorm(TorchLayer):
    name = "GroupNorm"

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        self.g, self.c, self.eps, self.affine = int(num_groups), int(num_channels), float(eps), bool(affine)

    def init(self, seed=0):
        return Record(weight=np.ones(self.c), bias=np.zeros(self.c)) if self.affine else Record()

    def forward(self, p, x, train=False):
        sh = _shape(x)
        N = sh[0]
        xg = ops.reshape(x, N, self.g, -1)
        m = ops.mean(xg, axis=-1, keepdims=True)
        d = xg - m
        v = ops.mean(d * d, axis=-1, keepdims=True)
        y = ops.reshape(d / ops.sqrt(v + self.eps), *sh)
        if self.affine:
            bs = (1, self.c) + (1,) * (len(sh) - 2)
            y = y * ops.reshape(p.weight, *bs) + ops.reshape(p.bias, *bs)
        return y


# =============================================================================== recurrent
class _RNNBase(TorchLayer):
    gates = 1

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0,
                 bidirectional=False, nonlinearity="tanh", proj_size=0, device=None, dtype=None):
        if proj_size:
            raise RuntimeErr("LSTM(proj_size=…) is not supported")
        self.mode = mode
        self.i, self.h, self.L = int(input_size), int(hidden_size), int(num_layers)
        self.bias, self.bf, self.drop, self.bi = bool(bias), bool(batch_first), float(dropout), bool(bidirectional)
        self.nonlin = nonlinearity
        self.gates = {"LSTM": 4, "GRU": 3, "RNN": 1}[mode]
        self.name = mode

    def _names(self):
        for layer in range(self.L):
            for d in range(2 if self.bi else 1):
                yield layer, d, f"_l{layer}{'_reverse' if d else ''}"

    def init(self, seed=0):
        rng = _rng(seed)
        b = 1 / math.sqrt(self.h)
        out = {}
        G = self.gates * self.h
        for layer, d, sfx in self._names():
            nin = self.i if layer == 0 else self.h * (2 if self.bi else 1)
            out["weight_ih" + sfx] = _uniform(rng, b, (G, nin))
            out["weight_hh" + sfx] = _uniform(rng, b, (G, self.h))
            if self.bias:
                out["bias_ih" + sfx] = _uniform(rng, b, (G,))
                out["bias_hh" + sfx] = _uniform(rng, b, (G,))
        return Record(**out)

    def _cell(self, xw, h, c, Whh, bhh):
        H = self.h
        hw = ops.matmul(h, ops.transpose(Whh))
        if bhh is not None:
            hw = hw + bhh
        if self.mode == "LSTM":
            z = xw + hw
            i = ops.sigmoid(z[:, 0:H])
            f = ops.sigmoid(z[:, H:2 * H])
            g = ops.tanh(z[:, 2 * H:3 * H])
            o = ops.sigmoid(z[:, 3 * H:4 * H])
            c = f * c + i * g
            return o * ops.tanh(c), c
        if self.mode == "GRU":
            r = ops.sigmoid(xw[:, 0:H] + hw[:, 0:H])
            zg = ops.sigmoid(xw[:, H:2 * H] + hw[:, H:2 * H])
            n = ops.tanh(xw[:, 2 * H:3 * H] + r * hw[:, 2 * H:3 * H])
            return (1.0 - zg) * n + zg * h, None
        a = xw + hw
        return (ops.tanh(a) if self.nonlin == "tanh" else ops.relu(a)), None

    def forward(self, p, x, hx=None, train=False):
        unbatched = len(_shape(x)) == 2
        if unbatched:
            x = ops.reshape(x, *((1,) + _shape(x))) if self.bf else ops.reshape(x, _shape(x)[0], 1, _shape(x)[1])
        if not self.bf:
            x = ops.swapaxes(x, 0, 1)                      # -> batch first
        N, T = _shape(x)[0], _shape(x)[1]
        D = 2 if self.bi else 1
        h0 = c0 = None
        if hx is not None:
            if self.mode == "LSTM":
                h0, c0 = hx[0], hx[1]
            else:
                h0 = hx
            if unbatched:
                h0 = ops.reshape(h0, _shape(h0)[0], 1, _shape(h0)[-1])
                if c0 is not None:
                    c0 = ops.reshape(c0, _shape(c0)[0], 1, _shape(c0)[-1])
        hn, cn = [], []
        inp = x
        for layer in range(self.L):
            outs_dir = []
            for d in range(D):
                sfx = f"_l{layer}{'_reverse' if d else ''}"
                Wih, Whh = p["weight_ih" + sfx], p["weight_hh" + sfx]
                bih = p["bias_ih" + sfx] if self.bias else None
                bhh = p["bias_hh" + sfx] if self.bias else None
                xw_all = ops.matmul(inp, ops.transpose(Wih))
                if bih is not None:
                    xw_all = xw_all + bih
                k = layer * D + d
                h = h0[k] if h0 is not None else np.zeros((N, self.h))
                c = (c0[k] if c0 is not None else np.zeros((N, self.h))) if self.mode == "LSTM" else None
                steps = range(T - 1, -1, -1) if d else range(T)
                seq = [None] * T
                for t in steps:
                    h, c = self._cell(xw_all[:, t, :], h, c, Whh, bhh)
                    seq[t] = h
                outs_dir.append(ops.stack(seq, axis=1))
                hn.append(h)
                if self.mode == "LSTM":
                    cn.append(c)
            inp = outs_dir[0] if D == 1 else ops.concat(outs_dir, axis=-1)
            if self.drop and train and layer < self.L - 1:
                inp = dropout(inp, self.drop, True)
        out = inp if self.bf else ops.swapaxes(inp, 0, 1)
        h_n = ops.stack(hn, axis=0)
        if unbatched:
            out = out[0] if self.bf else out[:, 0]
            h_n = h_n[:, 0]
        if self.mode == "LSTM":
            c_n = ops.stack(cn, axis=0)
            if unbatched:
                c_n = c_n[:, 0]
            return out, (h_n, c_n)
        return out, h_n

    def describe(self):
        return f"{self.mode}({self.i} → {self.h}, layers={self.L}{', bidirectional' if self.bi else ''})"


def LSTM(*a, **k):
    return _RNNBase("LSTM", *a, **k)


def GRU(*a, **k):
    return _RNNBase("GRU", *a, **k)


def RNN(*a, nonlinearity="tanh", **k):
    return _RNNBase("RNN", *a, nonlinearity=nonlinearity, **k)


# =============================================================================== attention
def _xavier(rng, shape):
    fan_out, fan_in = shape[0], shape[1]
    b = math.sqrt(6.0 / (fan_in + fan_out))
    return _uniform(rng, b, shape)


def _merge_masks(attn_mask, key_padding_mask, N, H, L, S):
    """Additive float mask of shape broadcastable to [N, H, L, S] (−inf where attention is not allowed)."""
    total = None
    if attn_mask is not None:
        m = np.asarray(ops.raw(attn_mask))
        add = np.where(m, -np.inf, 0.0) if m.dtype == bool else m.astype(float)
        add = add.reshape(N, H, L, S) if add.ndim == 3 else add.reshape(1, 1, L, S)
        total = add
    if key_padding_mask is not None:
        k = np.asarray(ops.raw(key_padding_mask))
        add = np.where(k, -np.inf, 0.0) if k.dtype == bool else k.astype(float)
        add = add.reshape(N, 1, 1, S)
        total = add if total is None else total + add
    return total


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 train=True):
    """softmax(Q·Kᵀ/√d + mask)·V over the last two axes (PyTorch's F.scaled_dot_product_attention)."""
    L, S, E = _shape(query)[-2], _shape(key)[-2], _shape(query)[-1]
    sc = (1.0 / math.sqrt(E)) if scale is None else float(scale)
    s = ops.matmul(query, ops.swapaxes(key, -2, -1)) * sc
    if is_causal:
        s = s + np.triu(np.full((L, S), -np.inf), k=1)
    if attn_mask is not None:
        m = np.asarray(ops.raw(attn_mask))
        s = s + (np.where(m, 0.0, -np.inf) if m.dtype == bool else m)     # bool: True means "may attend"
    w = ops.softmax(s, axis=-1)
    if dropout_p and train:
        w = dropout(w, dropout_p, True)
    return ops.matmul(w, value)


class MultiheadAttention(TorchLayer):
    name = "MultiheadAttention"

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        self.E, self.H = int(embed_dim), int(num_heads)
        if self.E % self.H:
            raise ShapeError("embed_dim must be divisible by num_heads")
        if add_bias_kv or add_zero_attn:
            raise RuntimeErr("MultiheadAttention(add_bias_kv / add_zero_attn) is not supported")
        self.drop, self.bias, self.bf = float(dropout), bool(bias), bool(batch_first)
        self.kdim = int(kdim) if kdim is not None else self.E
        self.vdim = int(vdim) if vdim is not None else self.E
        self.same = self.kdim == self.E and self.vdim == self.E

    def init(self, seed=0):
        rng = _rng(seed)
        out = {}
        if self.same:
            out["in_proj_weight"] = _xavier(rng, (3 * self.E, self.E))
        else:
            out["q_proj_weight"] = _xavier(rng, (self.E, self.E))
            out["k_proj_weight"] = _xavier(rng, (self.E, self.kdim))
            out["v_proj_weight"] = _xavier(rng, (self.E, self.vdim))
        if self.bias:
            out["in_proj_bias"] = np.zeros(3 * self.E)
        b = 1 / math.sqrt(self.E)
        op = {"weight": _uniform(rng, b, (self.E, self.E))}
        if self.bias:
            op["bias"] = np.zeros(self.E)
        out["out_proj"] = Record(**op)
        return Record(**out)

    def forward(self, p, query, key=None, value=None, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False, train=False):
        key = query if key is None else key
        value = key if value is None else value
        unbatched = len(_shape(query)) == 2
        if unbatched:
            query, key, value = (ops.reshape(t, 1, *_shape(t)) if self.bf else
                                 ops.reshape(t, _shape(t)[0], 1, _shape(t)[1]) for t in (query, key, value))
            if key_padding_mask is not None:
                key_padding_mask = np.asarray(ops.raw(key_padding_mask))[None]
        if not self.bf:
            query, key, value = (ops.swapaxes(t, 0, 1) for t in (query, key, value))
        N, L, E = _shape(query)
        S = _shape(key)[1]
        H, hd = self.H, self.E // self.H
        if self.same:
            Wq, Wk, Wv = p.in_proj_weight[0:E], p.in_proj_weight[E:2 * E], p.in_proj_weight[2 * E:3 * E]
        else:
            Wq, Wk, Wv = p.q_proj_weight, p.k_proj_weight, p.v_proj_weight
        q = ops.matmul(query, ops.transpose(Wq))
        k = ops.matmul(key, ops.transpose(Wk))
        v = ops.matmul(value, ops.transpose(Wv))
        if self.bias:
            q = q + p.in_proj_bias[0:E]
            k = k + p.in_proj_bias[E:2 * E]
            v = v + p.in_proj_bias[2 * E:3 * E]
        q = ops.permute(ops.reshape(q, N, L, H, hd), 0, 2, 1, 3)
        k = ops.permute(ops.reshape(k, N, S, H, hd), 0, 2, 1, 3)
        v = ops.permute(ops.reshape(v, N, S, H, hd), 0, 2, 1, 3)
        scores = ops.matmul(q, ops.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(hd))
        if is_causal and attn_mask is None:
            attn_mask = np.triu(np.ones((L, S), dtype=bool), k=1)
        mask = _merge_masks(attn_mask, key_padding_mask, N, H, L, S)
        if mask is not None:
            scores = scores + mask
        w = ops.softmax(scores, axis=-1)
        wd = dropout(w, self.drop, True) if (self.drop and train) else w
        o = ops.reshape(ops.permute(ops.matmul(wd, v), 0, 2, 1, 3), N, L, E)
        o = ops.matmul(o, ops.transpose(p.out_proj.weight))
        if self.bias:
            o = o + p.out_proj.bias
        if not self.bf:
            o = ops.swapaxes(o, 0, 1)
        weights = None
        if need_weights:
            weights = ops.mean(w, axis=1) if average_attn_weights else w
        if unbatched:
            o = o[0] if self.bf else o[:, 0]
            if weights is not None:
                weights = weights[0]
        return o, weights

    def describe(self):
        return f"MultiheadAttention({self.E}, heads={self.H})"


class TransformerEncoderLayer(TorchLayer):
    name = "TransformerEncoderLayer"

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        self.d, self.ff, self.drop = int(d_model), int(dim_feedforward), float(dropout)
        self.act = activation
        self.bf, self.norm_first = bool(batch_first), bool(norm_first)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

    def children(self):
        return [("self_attn", self.self_attn), ("linear1", self.linear1), ("linear2", self.linear2),
                ("norm1", self.norm1), ("norm2", self.norm2)]

    def init(self, seed=0):
        return Record(**{n: l.init(_seed_for(seed, i)) for i, (n, l) in enumerate(self.children())})

    def _act(self, x):
        if callable(self.act) and not isinstance(self.act, str):
            return self.act(x)
        return ACT_FUNCS[str(self.act)](x)

    def _sa(self, p, x, mask, kpm, causal, train):
        o, _ = self.self_attn(p.self_attn, x, x, x, attn_mask=mask, key_padding_mask=kpm, need_weights=False,
                              is_causal=causal, train=train)
        return dropout(o, self.drop, train)

    def _ffn(self, p, x, train):
        h = dropout(self._act(self.linear1(p.linear1, x)), self.drop, train)
        return dropout(self.linear2(p.linear2, h), self.drop, train)

    def forward(self, p, src, src_mask=None, src_key_padding_mask=None, is_causal=False, train=False):
        x = src
        if self.norm_first:
            x = x + self._sa(p, self.norm1(p.norm1, x), src_mask, src_key_padding_mask, is_causal, train)
            x = x + self._ffn(p, self.norm2(p.norm2, x), train)
        else:
            x = self.norm1(p.norm1, x + self._sa(p, x, src_mask, src_key_padding_mask, is_causal, train))
            x = self.norm2(p.norm2, x + self._ffn(p, x, train))
        return x


class TransformerDecoderLayer(TorchLayer):
    name = "TransformerDecoderLayer"

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        self.d, self.drop, self.act = int(d_model), float(dropout), activation
        self.norm_first = bool(norm_first)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

    def children(self):
        return [("self_attn", self.self_attn), ("multihead_attn", self.multihead_attn), ("linear1", self.linear1),
                ("linear2", self.linear2), ("norm1", self.norm1), ("norm2", self.norm2), ("norm3", self.norm3)]

    def init(self, seed=0):
        return Record(**{n: l.init(_seed_for(seed, i)) for i, (n, l) in enumerate(self.children())})

    def _act(self, x):
        if callable(self.act) and not isinstance(self.act, str):
            return self.act(x)
        return ACT_FUNCS[str(self.act)](x)

    def forward(self, p, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False, train=False):
        def sa(x):
            o, _ = self.self_attn(p.self_attn, x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
                                  need_weights=False, is_causal=tgt_is_causal, train=train)
            return dropout(o, self.drop, train)

        def ca(x):
            o, _ = self.multihead_attn(p.multihead_attn, x, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask, need_weights=False,
                                       is_causal=memory_is_causal, train=train)
            return dropout(o, self.drop, train)

        def ff(x):
            h = dropout(self._act(self.linear1(p.linear1, x)), self.drop, train)
            return dropout(self.linear2(p.linear2, h), self.drop, train)
        x = tgt
        if self.norm_first:
            x = x + sa(self.norm1(p.norm1, x))
            x = x + ca(self.norm2(p.norm2, x))
            x = x + ff(self.norm3(p.norm3, x))
        else:
            x = self.norm1(p.norm1, x + sa(x))
            x = self.norm2(p.norm2, x + ca(x))
            x = self.norm3(p.norm3, x + ff(x))
        return x


class _Stack(TorchLayer):
    def __init__(self, layer, num_layers, norm=None, kind="encoder"):
        self.layers = [copy.deepcopy(layer) for _ in range(int(num_layers))]
        self.norm = norm
        self.kind = kind
        self.name = "TransformerEncoder" if kind == "encoder" else "TransformerDecoder"

    def children(self):
        out = [(f"layers.{i}", l) for i, l in enumerate(self.layers)]
        if self.norm is not None:
            out.append(("norm", self.norm))
        return out

    def init(self, seed=0):
        r = {"layers": [l.init(_seed_for(seed, i)) for i, l in enumerate(self.layers)]}
        if self.norm is not None:
            r["norm"] = self.norm.init(_seed_for(seed, 999))
        return Record(**r)

    def forward(self, p, x, *rest, train=False, **kw):
        kw = {k: v for k, v in kw.items() if k not in ("enable_nested_tensor",)}
        if "mask" in kw:
            kw["src_mask"] = kw.pop("mask")
        if "src_key_padding_mask" in kw or "is_causal" in kw or "src_mask" in kw:
            pass
        for l, lp in zip(self.layers, p.layers):
            x = l(lp, x, *rest, train=train, **kw)
        if self.norm is not None:
            x = self.norm(p.norm, x)
        return x


def TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
    return _Stack(encoder_layer, num_layers, norm, "encoder")


def TransformerDecoder(decoder_layer, num_layers, norm=None):
    return _Stack(decoder_layer, num_layers, norm, "decoder")


class Transformer(TorchLayer):
    name = "Transformer"

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        enc = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                      batch_first, norm_first, bias)
        dec = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                      batch_first, norm_first, bias)
        self.encoder = custom_encoder or TransformerEncoder(enc, num_encoder_layers,
                                                            LayerNorm(d_model, eps=layer_norm_eps, bias=bias))
        self.decoder = custom_decoder or TransformerDecoder(dec, num_decoder_layers,
                                                            LayerNorm(d_model, eps=layer_norm_eps, bias=bias))

    def children(self):
        return [("encoder", self.encoder), ("decoder", self.decoder)]

    def init(self, seed=0):
        return Record(encoder=self.encoder.init(_seed_for(seed, 1)), decoder=self.decoder.init(_seed_for(seed, 2)))

    def forward(self, p, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, src_is_causal=None, tgt_is_causal=None,
                memory_is_causal=False, train=False):
        mem = self.encoder(p.encoder, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                           train=train)
        return self.decoder(p.decoder, tgt, mem, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask, train=train)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return np.triu(np.full((int(sz), int(sz)), -np.inf), k=1)


def generate_square_subsequent_mask(sz):
    """Causal mask: −inf above the diagonal (as nn.Transformer.generate_square_subsequent_mask)."""
    return np.triu(np.full((int(sz), int(sz)), -np.inf), k=1)


# =============================================================================== containers
class Sequential(TorchLayer):
    name = "Sequential"

    def __init__(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        self.layers = list(layers)

    def children(self):
        return [(str(i), l) for i, l in enumerate(self.layers)]

    def init(self, seed=0):
        return [l.init(_seed_for(seed, i)) if hasattr(l, "init") else Record() for i, l in enumerate(self.layers)]

    def forward(self, p, x, train=False):
        for l, lp in zip(self.layers, p):
            x = l(lp, x, train=train)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


# =============================================================================== functional
def dropout(x, p=0.5, train=True, training=None, inplace=False):
    """Inverted dropout (as F.dropout): zero with probability p and scale the rest by 1/(1-p) while training."""
    if training is not None:
        train = training
    p = float(p)
    if not train or p <= 0:
        return x
    if p >= 1:
        return x * 0.0
    keep = (ops._rng(None).random(_shape(x)) >= p) / (1.0 - p)
    return x * keep


def linear(x, weight, bias=None):
    y = ops.matmul(x, ops.transpose(weight))
    return y + bias if bias is not None else y


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    shp = _tuple(normalized_shape, 1) if not isinstance(normalized_shape, (list, tuple)) else tuple(normalized_shape)
    axes = tuple(range(-len(shp), 0))
    m = ops.mean(x, axis=axes, keepdims=True)
    d = x - m
    v = ops.mean(d * d, axis=axes, keepdims=True)
    y = d / ops.sqrt(v + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def normalize(x, p=2.0, dim=1, eps=1e-12):
    n = ops.sum_(ops.abs_(x) ** p, axis=dim, keepdims=True) ** (1.0 / p)
    nr = np.asarray(ops.raw(n))
    return x / ops.where(nr > eps, n, eps + 0.0 * n)


def embedding(idx, weight, padding_idx=None):
    ix = np.asarray(ops.raw(idx)).astype(np.int64)
    out = weight[ix]
    if padding_idx is not None and isinstance(out, Node):
        keep = (ix != padding_idx)[..., None].astype(float)
        out = out * keep + ad.const(out) * (1.0 - keep)        # no gradient into the padding row
    return out


def one_hot(t, num_classes=-1):
    ix = np.asarray(ops.raw(t)).astype(np.int64)
    k = int(ix.max()) + 1 if num_classes in (-1, None) else int(num_classes)
    return np.eye(k, dtype=np.int64)[ix]


def pad(x, pad, mode="constant", value=0.0):
    """F.pad for constant padding: pad = (last_lo, last_hi, prev_lo, prev_hi, …)."""
    if mode != "constant":
        raise RuntimeErr(f"pad(mode='{mode}') is not supported (only 'constant')")
    shp = list(_shape(x))
    out = x
    for i in range(len(pad) // 2):
        ax = len(shp) - 1 - i
        lo, hi = int(pad[2 * i]), int(pad[2 * i + 1])
        cur = list(_shape(out))
        parts = []
        if lo:
            cur_lo = cur.copy()
            cur_lo[ax] = lo
            parts.append(np.full(cur_lo, float(value)))
        parts.append(out)
        if hi:
            cur_hi = cur.copy()
            cur_hi[ax] = hi
            parts.append(np.full(cur_hi, float(value)))
        out = ops.concat(parts, axis=ax) if len(parts) > 1 else out
    return out


def flatten(x, start_dim=0, end_dim=-1):
    sh = list(_shape(x))
    n = len(sh)
    s, e = start_dim % n, end_dim % n
    new = sh[:s] + [int(np.prod(sh[s:e + 1]))] + sh[e + 1:]
    return ops.reshape(x, *new)


# =============================================================================== losses
def _reduce(v, reduction, denom=None):
    if reduction == "none":
        return v
    if reduction == "sum":
        return ops.sum_(v)
    if reduction == "mean":
        return ops.sum_(v) / denom if denom is not None else ops.mean(v)
    raise RuntimeErr(f"reduction must be 'mean', 'sum' or 'none', not {reduction!r}")


def _class_last(x):
    """[N, C, d1, …] -> [N·d1·…, C] (and the matching flattening for targets)."""
    sh = _shape(x)
    if len(sh) <= 2:
        return x, None
    perm = (0,) + tuple(range(2, len(sh))) + (1,)
    return ops.reshape(ops.permute(x, *perm), -1, sh[1]), sh


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean",
                  label_smoothing=0.0):
    """F.cross_entropy with class-index or class-probability targets, weights, ignore_index and smoothing."""
    x, sh = _class_last(input)
    unbatched = len(_shape(input)) == 1
    if unbatched:
        x = ops.reshape(x, 1, -1)
    C = _shape(x)[1]
    lp = ops.log_softmax(x, axis=-1)
    tr = np.asarray(ops.raw(target))
    w = None if weight is None else np.asarray(ops.raw(weight), dtype=float)
    eps = float(label_smoothing)
    if np.issubdtype(tr.dtype, np.floating) and tr.shape == _shape(input):     # probabilities
        t = tr if sh is None else np.moveaxis(tr, 1, -1).reshape(-1, C)
        if unbatched:
            t = t.reshape(1, -1)
        if eps:
            t = t * (1 - eps) + eps / C
        per = -ops.sum_(lp * (t * (w if w is not None else 1.0)), axis=-1)
        if reduction == "mean":
            return ops.sum_(per) / per_count(per)
        return _reduce(per if not unbatched else per[0], reduction)
    ti = tr.astype(np.int64).reshape(-1)
    valid = ti != ignore_index
    safe = np.where(valid, ti, 0)
    n = safe.shape[0]
    picked = lp[(np.arange(n), safe)]
    wy = w[safe] * valid if w is not None else valid.astype(float)
    nll = -picked * wy
    if eps:
        wc = w if w is not None else np.ones(C)
        smooth = -ops.sum_(lp * wc, axis=-1) * valid
        if reduction == "mean":
            denom = wy.sum()
            return (1 - eps) * ops.sum_(nll) / denom + eps / C * ops.sum_(smooth) / denom
        per = (1 - eps) * nll + eps / C * smooth
    else:
        if reduction == "mean":
            return ops.sum_(nll) / wy.sum()
        per = nll
    if reduction == "none":
        out = per if sh is None else ops.reshape(per, *((sh[0],) + sh[2:]))
        return out[0] if unbatched else out
    return _reduce(per, reduction)


def per_count(per):
    return float(np.prod(_shape(per))) if _shape(per) else 1.0


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
    x, sh = _class_last(input)
    ti = np.asarray(ops.raw(target)).astype(np.int64).reshape(-1)
    valid = ti != ignore_index
    safe = np.where(valid, ti, 0)
    w = None if weight is None else np.asarray(ops.raw(weight), dtype=float)
    wy = w[safe] * valid if w is not None else valid.astype(float)
    per = -x[(np.arange(safe.shape[0]), safe)] * wy
    if reduction == "mean":
        return ops.sum_(per) / wy.sum()
    if reduction == "none" and sh is not None:
        return ops.reshape(per, *((sh[0],) + sh[2:]))
    return _reduce(per, reduction)


def mse_loss(input, target, size_average=None, reduce=None, reduction="mean", weight=None):
    d = input - target
    return _reduce(d * d, reduction)


def l1_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    return _reduce(ops.abs_(input - target), reduction)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction="mean", beta=1.0):
    d = input - target
    a = np.abs(np.asarray(ops.raw(d)))
    if beta == 0:
        return _reduce(ops.abs_(d), reduction)
    v = ops.where(a < beta, 0.5 * d * d / beta, ops.abs_(d) - 0.5 * beta)
    return _reduce(v, reduction)


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    d = input - target
    a = np.abs(np.asarray(ops.raw(d)))
    v = ops.where(a <= delta, 0.5 * d * d, delta * (ops.abs_(d) - 0.5 * delta))
    return _reduce(v, reduction)


def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction="mean"):
    with np.errstate(divide="ignore"):
        lp = ops.log(input)
        lq = ops.log(1.0 - input)
    lp = ops.where(np.asarray(ops.raw(lp)) > -100.0, lp, -100.0)
    lq = ops.where(np.asarray(ops.raw(lq)) > -100.0, lq, -100.0)
    v = -(target * lp + (1.0 - target) * lq)
    if weight is not None:
        v = v * weight
    return _reduce(v, reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction="mean",
                                     pos_weight=None):
    """PyTorch's stable form: (1−y)·x + (1 + (pw−1)·y)·(log(1+e^(−|x|)) + max(−x, 0))."""
    x = input
    xr = np.asarray(ops.raw(x))
    neg = ops.where(xr < 0, -x, 0.0 * x)                          # max(−x, 0)
    lse = ops.log(1.0 + ops.exp(-ops.abs_(x))) + neg
    if pos_weight is not None:
        lw = 1.0 + (pos_weight - 1.0) * target
        v = (1.0 - target) * x + lw * lse
    else:
        v = (1.0 - target) * x + lse
    if weight is not None:
        v = v * weight
    return _reduce(v, reduction)


def kl_div(input, target, size_average=None, reduce=None, reduction="mean", log_target=False):
    if log_target:
        v = ops.exp(target) * (target - input)
    else:
        tr = np.asarray(ops.raw(target))
        safe = np.where(tr > 0, tr, 1.0)
        v = ops.where(tr > 0, target * (np.log(safe) - input), 0.0 * input)
    if reduction == "batchmean":
        return ops.sum_(v) / _shape(input)[0]
    return _reduce(v, reduction)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    num = ops.sum_(x1 * x2, axis=dim)
    n1 = ops.sqrt(ops.sum_(x1 * x1, axis=dim))
    n2 = ops.sqrt(ops.sum_(x2 * x2, axis=dim))
    d = n1 * n2
    return num / ops.where(np.asarray(ops.raw(d)) > eps, d, eps + 0.0 * d)


class _Loss:
    """A loss module: nn.CrossEntropyLoss(...)(input, target)."""

    def __init__(self, fn, **kw):
        self.fn, self.kw = fn, kw

    def __call__(self, input, target):
        return self.fn(input, target, **self.kw)


def _loss_class(fn, names):
    def make(*args, **kw):
        k = dict(zip(names, args))
        k.update({a: b for a, b in kw.items() if a not in ("size_average", "reduce")})
        return _Loss(fn, **k)
    make.__name__ = fn.__name__
    return make


CrossEntropyLoss = _loss_class(cross_entropy, ["weight", "size_average", "ignore_index", "reduce", "reduction",
                                               "label_smoothing"])
NLLLoss = _loss_class(nll_loss, ["weight", "size_average", "ignore_index", "reduce", "reduction"])
MSELoss = _loss_class(mse_loss, ["size_average", "reduce", "reduction"])
L1Loss = _loss_class(l1_loss, ["size_average", "reduce", "reduction"])
SmoothL1Loss = _loss_class(smooth_l1_loss, ["size_average", "reduce", "reduction", "beta"])
HuberLoss = _loss_class(huber_loss, ["reduction", "delta"])
BCELoss = _loss_class(binary_cross_entropy, ["weight", "size_average", "reduce", "reduction"])
BCEWithLogitsLoss = _loss_class(binary_cross_entropy_with_logits, ["weight", "size_average", "reduce", "reduction",
                                                                   "pos_weight"])
KLDivLoss = _loss_class(kl_div, ["size_average", "reduce", "reduction", "log_target"])


# =============================================================================== optimizers (PyTorch's rules)
class _Optim:
    def __init__(self, lr, defaults):
        self.lr = float(lr)
        self.defaults = defaults
        self.state = None
        self.t = 0
        self.initial_lr = self.lr

    @property
    def param_groups(self):
        return [{"lr": self.lr}]

    def _raw_tree(self, tree):
        return tree_map(lambda g: np.asarray(ops.raw(g), dtype=float), tree)

    def step(self, params, grads):
        g = self._raw_tree(grads)
        if self.defaults.get("maximize"):
            g = tree_map(lambda a: -a, g)
        p = self._raw_tree(params)
        self.t += 1
        if self.state is None:
            self.state = self.new_state(p)
        out = self.update(p, g)
        return out

    def zero_grad(self, set_to_none=True):
        return None

    def get_lr(self):
        return self.lr


def _zeros_like(tree):
    return tree_map(np.zeros_like, tree)


def _tmap(f, *trees):
    return tree_map(f, *trees)


class SGD(_Optim):
    def __init__(self, params=None, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
                 maximize=False, foreach=None, differentiable=False, fused=None):
        super().__init__(lr, dict(momentum=float(momentum), dampening=float(dampening), wd=float(weight_decay),
                                  nesterov=bool(nesterov), maximize=bool(maximize)))

    def new_state(self, p):
        return {"buf": None}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        if d["momentum"]:
            if self.state["buf"] is None:
                self.state["buf"] = tree_map(np.copy, g)
            else:
                self.state["buf"] = _tmap(lambda b, gi: d["momentum"] * b + (1 - d["dampening"]) * gi,
                                          self.state["buf"], g)
            if d["nesterov"]:
                g = _tmap(lambda gi, b: gi + d["momentum"] * b, g, self.state["buf"])
            else:
                g = self.state["buf"]
        return _tmap(lambda pi, gi: pi - self.lr * gi, p, g)


class Adam(_Optim):
    decoupled = False

    def __init__(self, params=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                 maximize=False, foreach=None, capturable=False, differentiable=False, fused=None,
                 decoupled_weight_decay=False):
        super().__init__(lr, dict(b1=float(betas[0]), b2=float(betas[1]), eps=float(eps), wd=float(weight_decay),
                                  amsgrad=bool(amsgrad), maximize=bool(maximize)))
        if decoupled_weight_decay:
            self.decoupled = True

    @property
    def b1(self):
        return self.defaults["b1"]

    @b1.setter
    def b1(self, v):
        self.defaults["b1"] = float(v)

    def new_state(self, p):
        s = {"m": _zeros_like(p), "v": _zeros_like(p)}
        if self.defaults["amsgrad"]:
            s["vmax"] = _zeros_like(p)
        return s

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            if self.decoupled:
                p = _tmap(lambda pi: pi * (1 - self.lr * d["wd"]), p)
            else:
                g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        b1, b2, t = d["b1"], d["b2"], self.t
        self.state["m"] = _tmap(lambda m, gi: b1 * m + (1 - b1) * gi, self.state["m"], g)
        self.state["v"] = _tmap(lambda v, gi: b2 * v + (1 - b2) * gi * gi, self.state["v"], g)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        v = self.state["v"]
        if d["amsgrad"]:
            self.state["vmax"] = _tmap(np.maximum, self.state["vmax"], v)
            v = self.state["vmax"]
        step = self.lr / bc1
        sq = math.sqrt(bc2)
        return _tmap(lambda pi, m, vi: pi - step * m / (np.sqrt(vi) / sq + d["eps"]), p, self.state["m"], v)


class AdamW(Adam):
    decoupled = True

    def __init__(self, params=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 maximize=False, foreach=None, capturable=False, differentiable=False, fused=None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize)


class Adamax(_Optim):
    def __init__(self, params=None, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **_):
        super().__init__(lr, dict(b1=float(betas[0]), b2=float(betas[1]), eps=float(eps), wd=float(weight_decay)))

    def new_state(self, p):
        return {"m": _zeros_like(p), "u": _zeros_like(p)}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        self.state["m"] = _tmap(lambda m, gi: d["b1"] * m + (1 - d["b1"]) * gi, self.state["m"], g)
        self.state["u"] = _tmap(lambda u, gi: np.maximum(d["b2"] * u, np.abs(gi) + d["eps"]), self.state["u"], g)
        clr = self.lr / (1 - d["b1"] ** self.t)
        return _tmap(lambda pi, m, u: pi - clr * m / u, p, self.state["m"], self.state["u"])


class RMSprop(_Optim):
    def __init__(self, params=None, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False,
                 capturable=False, foreach=None, maximize=False, differentiable=False):
        super().__init__(lr, dict(alpha=float(alpha), eps=float(eps), wd=float(weight_decay), mom=float(momentum),
                                  centered=bool(centered), maximize=bool(maximize)))

    def new_state(self, p):
        return {"sq": _zeros_like(p), "ga": _zeros_like(p), "buf": _zeros_like(p)}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        a = d["alpha"]
        self.state["sq"] = _tmap(lambda s, gi: a * s + (1 - a) * gi * gi, self.state["sq"], g)
        if d["centered"]:
            self.state["ga"] = _tmap(lambda s, gi: a * s + (1 - a) * gi, self.state["ga"], g)
            avg = _tmap(lambda s, ga: np.sqrt(s - ga * ga) + d["eps"], self.state["sq"], self.state["ga"])
        else:
            avg = _tmap(lambda s: np.sqrt(s) + d["eps"], self.state["sq"])
        if d["mom"]:
            self.state["buf"] = _tmap(lambda b, gi, av: d["mom"] * b + gi / av, self.state["buf"], g, avg)
            return _tmap(lambda pi, b: pi - self.lr * b, p, self.state["buf"])
        return _tmap(lambda pi, gi, av: pi - self.lr * gi / av, p, g, avg)


class Adagrad(_Optim):
    def __init__(self, params=None, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                 eps=1e-10, foreach=None, maximize=False, differentiable=False, fused=None):
        super().__init__(lr, dict(lr_decay=float(lr_decay), wd=float(weight_decay),
                                  init=float(initial_accumulator_value), eps=float(eps), maximize=bool(maximize)))

    def new_state(self, p):
        return {"sum": tree_map(lambda a: np.full_like(a, self.defaults["init"]), p)}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        clr = self.lr / (1 + (self.t - 1) * d["lr_decay"])
        self.state["sum"] = _tmap(lambda s, gi: s + gi * gi, self.state["sum"], g)
        return _tmap(lambda pi, gi, s: pi - clr * gi / (np.sqrt(s) + d["eps"]), p, g, self.state["sum"])


class NAdam(_Optim):
    def __init__(self, params=None, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3,
                 decoupled_weight_decay=False, **_):
        super().__init__(lr, dict(b1=float(betas[0]), b2=float(betas[1]), eps=float(eps), wd=float(weight_decay),
                                  md=float(momentum_decay), dec=bool(decoupled_weight_decay)))
        self.mu_product = 1.0

    def new_state(self, p):
        return {"m": _zeros_like(p), "v": _zeros_like(p)}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            if d["dec"]:
                p = _tmap(lambda pi: pi * (1 - self.lr * d["wd"]), p)
            else:
                g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        t, b1, b2 = self.t, d["b1"], d["b2"]
        mu = b1 * (1 - 0.5 * 0.96 ** (t * d["md"]))
        mu_next = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * d["md"]))
        self.mu_product *= mu
        self.state["m"] = _tmap(lambda m, gi: b1 * m + (1 - b1) * gi, self.state["m"], g)
        self.state["v"] = _tmap(lambda v, gi: b2 * v + (1 - b2) * gi * gi, self.state["v"], g)
        bc2 = 1 - b2 ** t
        mp, mpn = self.mu_product, self.mu_product * mu_next
        return _tmap(lambda pi, gi, m, v: pi - self.lr * ((1 - mu) / (1 - mp)) * gi / (np.sqrt(v / bc2) + d["eps"])
                     - self.lr * (mu_next / (1 - mpn)) * m / (np.sqrt(v / bc2) + d["eps"]), p, g, self.state["m"],
                     self.state["v"])


class RAdam(_Optim):
    def __init__(self, params=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled_weight_decay=False, **_):
        super().__init__(lr, dict(b1=float(betas[0]), b2=float(betas[1]), eps=float(eps), wd=float(weight_decay),
                                  dec=bool(decoupled_weight_decay)))

    def new_state(self, p):
        return {"m": _zeros_like(p), "v": _zeros_like(p)}

    def update(self, p, g):
        d = self.defaults
        if d["wd"]:
            if d["dec"]:
                p = _tmap(lambda pi: pi * (1 - self.lr * d["wd"]), p)
            else:
                g = _tmap(lambda gi, pi: gi + d["wd"] * pi, g, p)
        t, b1, b2 = self.t, d["b1"], d["b2"]
        self.state["m"] = _tmap(lambda m, gi: b1 * m + (1 - b1) * gi, self.state["m"], g)
        self.state["v"] = _tmap(lambda v, gi: b2 * v + (1 - b2) * gi * gi, self.state["v"], g)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        rho_inf = 2 / (1 - b2) - 1
        rho_t = rho_inf - 2 * t * b2 ** t / bc2
        if rho_t > 5:
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            return _tmap(lambda pi, m, v: pi - self.lr * (m / bc1) * rect * math.sqrt(bc2) / (np.sqrt(v) + d["eps"]),
                         p, self.state["m"], self.state["v"])
        return _tmap(lambda pi, m: pi - self.lr * m / bc1, p, self.state["m"])


# =============================================================================== learning-rate schedulers
class _Sched:
    def __init__(self, optimizer, last_epoch=-1):
        self.opt = optimizer
        self.base_lr = getattr(optimizer, "initial_lr", optimizer.lr)
        self.last_epoch = last_epoch
        self._last_lr = [optimizer.lr]
        self.step()

    def get_last_lr(self):
        return list(self._last_lr)

    def step(self, epoch=None):
        self.last_epoch += 1
        self.opt.lr = float(self.compute())
        self._last_lr = [self.opt.lr]

    def state_dict(self):
        return Record(last_epoch=self.last_epoch, lr=self.opt.lr)


class StepLR(_Sched):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = int(step_size), float(gamma)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e = self.last_epoch
        if e == 0 or e % self.step_size:
            return self.opt.lr
        return self.opt.lr * self.gamma


class MultiStepLR(_Sched):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.ms = [int(m) for m in milestones]
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        c = self.ms.count(self.last_epoch)
        return self.opt.lr * self.gamma ** c if c else self.opt.lr


class ExponentialLR(_Sched):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        return self.opt.lr if self.last_epoch == 0 else self.opt.lr * self.gamma


class CosineAnnealingLR(_Sched):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T, self.eta = int(T_max), float(eta_min)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e, lr, T, eta = self.last_epoch, self.opt.lr, self.T, self.eta
        if e == 0:
            return self.base_lr
        if (e - 1 - T) % (2 * T) == 0:
            return lr + (self.base_lr - eta) * (1 - math.cos(math.pi / T)) / 2
        return (1 + math.cos(math.pi * e / T)) / (1 + math.cos(math.pi * (e - 1) / T)) * (lr - eta) + eta


class CosineAnnealingWarmRestarts(_Sched):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        self.T0, self.Ti, self.mult, self.eta = int(T_0), int(T_0), int(T_mult), float(eta_min)
        self.Tcur = last_epoch
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            self.Tcur += 1
            if self.Tcur >= self.Ti:
                self.Tcur -= self.Ti
                self.Ti *= self.mult
        else:
            epoch = float(epoch)
            if epoch >= self.T0:
                if self.mult == 1:
                    self.Tcur = epoch % self.T0
                else:
                    n = int(math.log(epoch / self.T0 * (self.mult - 1) + 1, self.mult))
                    self.Tcur = epoch - self.T0 * (self.mult ** n - 1) / (self.mult - 1)
                    self.Ti = self.T0 * self.mult ** n
            else:
                self.Ti, self.Tcur = self.T0, epoch
            self.last_epoch = math.floor(epoch)
        self.opt.lr = self.eta + (self.base_lr - self.eta) * (1 + math.cos(math.pi * self.Tcur / self.Ti)) / 2
        self._last_lr = [self.opt.lr]

    def compute(self):
        return self.opt.lr


class LinearLR(_Sched):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
        self.sf, self.ef, self.total = float(start_factor), float(end_factor), int(total_iters)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e, lr = self.last_epoch, self.opt.lr
        if e == 0:
            return lr * self.sf
        if e > self.total:
            return lr
        return lr * (1.0 + (self.ef - self.sf) / (self.total * self.sf + (e - 1) * (self.ef - self.sf)))


class ConstantLR(_Sched):
    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        self.f, self.total = float(factor), int(total_iters)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e, lr = self.last_epoch, self.opt.lr
        if e == 0:
            return lr * self.f
        if e != self.total:
            return lr
        return lr * (1.0 / self.f)


class LambdaLR(_Sched):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.fn = lr_lambda
        super().__init__(optimizer, last_epoch)

    def compute(self):
        return self.base_lr * float(self.fn(self.last_epoch))


class MultiplicativeLR(_Sched):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.fn = lr_lambda
        super().__init__(optimizer, last_epoch)

    def compute(self):
        return self.opt.lr * float(self.fn(self.last_epoch)) if self.last_epoch > 0 else self.opt.lr


class PolynomialLR(_Sched):
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total, self.power = int(total_iters), float(power)
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e, lr = self.last_epoch, self.opt.lr
        if e == 0 or e > self.total:
            return lr
        return lr * ((1.0 - e / self.total) / (1.0 - (e - 1) / self.total)) ** self.power


class OneCycleLR(_Sched):
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3,
                 anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                 final_div_factor=1e4, three_phase=False, last_epoch=-1):
        self.total = int(total_steps) if total_steps is not None else int(epochs) * int(steps_per_epoch)
        self.max_lr = float(max_lr)
        self.init_lr = self.max_lr / float(div_factor)
        self.min_lr = self.init_lr / float(final_div_factor)
        self.cos = anneal_strategy == "cos"
        self.cycle_m = bool(cycle_momentum) and hasattr(optimizer, "defaults") and (
            "b1" in optimizer.defaults or "momentum" in optimizer.defaults)
        self.bm, self.mm = float(base_momentum), float(max_momentum)
        p = float(pct_start)
        if three_phase:
            self.phases = [(p * self.total - 1, "init", "max", "max_m", "base_m"),
                           (2 * p * self.total - 2, "max", "init", "base_m", "max_m"),
                           (self.total - 1, "init", "min", "max_m", "max_m")]
        else:
            self.phases = [(p * self.total - 1, "init", "max", "max_m", "base_m"),
                           (self.total - 1, "max", "min", "base_m", "max_m")]
        optimizer.lr = self.init_lr
        optimizer.initial_lr = self.init_lr
        self.opt = optimizer
        if self.cycle_m:
            self._set_m(self.mm)
        super().__init__(optimizer, last_epoch)

    def _set_m(self, m):
        d = self.opt.defaults
        if "b1" in d:
            d["b1"] = m
        else:
            d["momentum"] = m

    def _anneal(self, start, end, pct):
        if self.cos:
            return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)
        return (end - start) * pct + start

    def compute(self):
        vals = {"init": self.init_lr, "max": self.max_lr, "min": self.min_lr, "max_m": self.mm, "base_m": self.bm}
        step = self.last_epoch
        if step > self.total:
            raise RuntimeErr(f"OneCycleLR stepped {step} times; total_steps is {self.total}")
        start = 0.0
        for i, (end, a, b, ma, mb) in enumerate(self.phases):
            if step <= end or i == len(self.phases) - 1:
                pct = (step - start) / (end - start)
                lr = self._anneal(vals[a], vals[b], pct)
                if self.cycle_m:
                    self._set_m(self._anneal(vals[ma], vals[mb], pct))
                return lr
            start = end
        return self.opt.lr


class CyclicLR(_Sched):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode="triangular",
                 gamma=1.0, scale_fn=None, scale_mode="cycle", cycle_momentum=True, base_momentum=0.8,
                 max_momentum=0.9, last_epoch=-1):
        self.lo, self.hi = float(base_lr), float(max_lr)
        up = float(step_size_up)
        down = float(step_size_down) if step_size_down is not None else up
        self.total, self.ratio = up + down, up / (up + down)
        self.mode, self.gamma = mode, float(gamma)
        optimizer.lr = self.lo
        optimizer.initial_lr = self.lo
        super().__init__(optimizer, last_epoch)

    def compute(self):
        e = self.last_epoch
        cycle = math.floor(1 + e / self.total)
        x = 1.0 + e / self.total - cycle
        scale = x / self.ratio if x <= self.ratio else (x - 1) / (self.ratio - 1)
        if self.mode == "triangular":
            f = 1.0
        elif self.mode == "triangular2":
            f = 1 / (2.0 ** (cycle - 1))
        else:
            f = self.gamma ** e
        return self.lo + (self.hi - self.lo) * scale * f


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel",
                 cooldown=0, min_lr=0.0, eps=1e-8):
        self.opt, self.mode, self.factor, self.patience = optimizer, mode, float(factor), int(patience)
        self.th, self.thm, self.cooldown, self.min_lr, self.eps = float(threshold), threshold_mode, int(cooldown), \
            float(min_lr), float(eps)
        self.best = math.inf if mode == "min" else -math.inf
        self.bad = 0
        self.cool = 0
        self.last_epoch = 0
        self._last_lr = [optimizer.lr]

    def _better(self, a):
        if self.mode == "min":
            return a < (self.best * (1 - self.th) if self.thm == "rel" else self.best - self.th)
        return a > (self.best * (1 + self.th) if self.thm == "rel" else self.best + self.th)

    def step(self, metrics, epoch=None):
        cur = float(ops.raw(metrics))
        self.last_epoch += 1
        if self._better(cur):
            self.best, self.bad = cur, 0
        else:
            self.bad += 1
        if self.cool > 0:
            self.cool -= 1
            self.bad = 0
        if self.bad > self.patience:
            new = max(self.opt.lr * self.factor, self.min_lr)
            if self.opt.lr - new > self.eps:
                self.opt.lr = new
            self.cool = self.cooldown
            self.bad = 0
        self._last_lr = [self.opt.lr]

    def get_last_lr(self):
        return list(self._last_lr)


class SequentialLR:
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.opt, self.scheds, self.ms = optimizer, list(schedulers), [int(m) for m in milestones]
        self.last_epoch = last_epoch + 1
        for s in self.scheds[1:]:
            pass
        self.opt.lr = self.scheds[0]._last_lr[0] if self.scheds else self.opt.lr
        self._last_lr = [self.opt.lr]

    def step(self):
        self.last_epoch += 1
        import bisect
        i = bisect.bisect_right(self.ms, self.last_epoch)
        s = self.scheds[i]
        if i > 0 and self.ms[i - 1] == self.last_epoch:
            s.last_epoch = -1
            s.base_lr = getattr(self.opt, "initial_lr", s.base_lr)
            s.step()
        else:
            s.step()
        self._last_lr = [self.opt.lr]

    def get_last_lr(self):
        return list(self._last_lr)


class ChainedScheduler:
    def __init__(self, schedulers, optimizer=None):
        self.scheds = list(schedulers)

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return self.scheds[-1].get_last_lr()



# =============================================================================== training helpers
def value_and_grad_aux(fn):
    """value_and_grad for a function returning [loss, aux]: returns (loss, aux, grads). aux values (e.g. the
    logits used to compute the loss) are returned without their gradient graph, so nothing is computed twice."""
    from .stdlib import _wrap_leaf, _unwrap_leaf, _TRACE_DEPTH
    from .units import DIMLESS

    def run(params):
        wrapped = tree_map(_wrap_leaf, params)
        _TRACE_DEPTH[0] += 1
        try:
            out = fn(wrapped)
        finally:
            _TRACE_DEPTH[0] -= 1
        if isinstance(out, (list, tuple)) and len(out) == 2:
            loss, aux = out
        else:
            loss, aux = out, None
        if isinstance(loss, Node):
            grads = ad.gradients(loss)
            value = float(np.asarray(loss.value))
        else:
            grads, value = {}, float(np.asarray(ops.raw(loss)))
        g = tree_map(_unwrap_leaf(grads, DIMLESS), wrapped)
        aux = tree_map(lambda v: np.asarray(v.value) if isinstance(v, Node) else v, aux) if aux is not None else None
        return value, aux, g
    return run


def checkpoint_layer(layer, params, *xs, train=None, **kw):
    """Apply layer(params, *xs) with gradient checkpointing (torch.utils.checkpoint): activations inside the
    layer are not stored but recomputed in the backward pass, which trades compute for memory."""
    leaves = tree_leaves(params)
    k = len(leaves)
    mode = getattr(layer, "training", True) if train is None else bool(train)

    def rebuild(vals):
        it = iter(vals)
        return tree_map(lambda _: next(it), params)

    def f(*args):
        return layer(rebuild(args[:k]), *args[k:], train=mode, **kw)
    return ad.checkpoint(f, *leaves, *xs)


def checkpoint(function, *args, use_reentrant=None, **kw):
    """torch.utils.checkpoint.checkpoint(fn, *args): every differentiable value fn uses must be an argument."""
    return ad.checkpoint(lambda *a: function(*a, **kw), *args)


def clip_grad_norm(grads, max_norm, norm_type=2.0, error_if_nonfinite=False):
    """PyTorch's clip_grad_norm_: scale every gradient by max_norm / (total_norm + 1e-6) when that is below 1."""
    leaves = [np.asarray(ops.raw(g), dtype=float) for g in tree_leaves(grads)]
    nt = float(norm_type)
    if nt == math.inf:
        total = max(float(np.max(np.abs(l))) for l in leaves) if leaves else 0.0
    else:
        total = float(sum(float(np.sum(np.abs(l) ** nt)) for l in leaves)) ** (1.0 / nt) if leaves else 0.0
    coef = min(float(max_norm) / (total + 1e-6), 1.0)
    return tree_map(lambda g: np.asarray(ops.raw(g), dtype=float) * coef, grads)


def clip_grad_value(grads, clip_value):
    c = float(clip_value)
    return tree_map(lambda g: np.clip(np.asarray(ops.raw(g), dtype=float), -c, c), grads)


class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = [t for t in tensors]
        n = {np.shape(ops.raw(t))[0] for t in self.tensors}
        if len(n) != 1:
            raise ShapeError("TensorDataset: all tensors need the same first dimension")
        self.n = n.pop()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


def _ds_len(ds):
    if hasattr(ds, "__len__"):
        return len(ds)
    m = _method(ds, "__len__") or _method(ds, "len")
    return int(m())


def _method(obj, name):
    try:
        from .values import BoundMethod
        t = getattr(obj, "type", None)
        if t is not None and name in getattr(t, "methods", {}):
            return BoundMethod(t.methods[name], obj)
    except Exception:
        pass
    return None


def _collate(items):
    first = items[0]
    if isinstance(first, (tuple, list)):
        return tuple(_collate([it[k] for it in items]) for k in range(len(first)))
    if isinstance(first, Record):
        return Record(**{k: _collate([it[k] for it in items]) for k in first.fields})
    return np.stack([np.asarray(ops.raw(it)) for it in items])


class DataLoader:
    """Mini-batches from a dataset (TensorDataset, or any object with __len__ / __getitem__), reshuffled every
    epoch when shuffle=true; batches are stacked like PyTorch's default collate."""

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 generator=None, prefetch_factor=None, persistent_workers=False, pin_memory_device="", seed=None):
        self.ds, self.bs, self.shuffle, self.drop = dataset, int(batch_size), bool(shuffle), bool(drop_last)
        self.collate = collate_fn
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = _ds_len(self.ds)
        return n // self.bs if self.drop else -(-n // self.bs)

    def __iter__(self):
        n = _ds_len(self.ds)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        if isinstance(self.ds, TensorDataset) and self.collate is None:
            for i in range(0, n, self.bs):
                idx = order[i:i + self.bs]
                if self.drop and len(idx) < self.bs:
                    break
                yield tuple(t[idx] for t in self.ds.tensors)
            return
        get = self.ds.__getitem__ if hasattr(self.ds, "__getitem__") and not hasattr(self.ds, "type") \
            else _method(self.ds, "__getitem__") or _method(self.ds, "get")
        for i in range(0, n, self.bs):
            idx = order[i:i + self.bs]
            if self.drop and len(idx) < self.bs:
                break
            items = [get(int(j)) for j in idx]
            yield self.collate(items) if self.collate is not None else _collate(items)


def _set_mode(layer, mode):
    layer.training = bool(mode)
    for _, sub in (layer.children() if isinstance(layer, TorchLayer) else []):
        _set_mode(sub, mode)
    return layer


TorchLayer.training = True
TorchLayer.train = lambda self, mode=True: _set_mode(self, mode)
TorchLayer.eval = lambda self: _set_mode(self, False)


# =============================================================================== state dicts
def _walk(layer, prefix=""):
    """(prefix, layer) for every layer that owns buffers, in PyTorch's naming."""
    from .layers import CustomLayer
    out = [(prefix, layer)]
    if isinstance(layer, CustomLayer):
        parts = layer.parts.fields if isinstance(layer.parts, Record) else {}
        for name, sub in parts.items():
            if isinstance(sub, list):
                for i, s in enumerate(sub):
                    out.extend(_walk(s, f"{prefix}{name}.{i}."))
            else:
                out.extend(_walk(sub, f"{prefix}{name}."))
    elif isinstance(layer, TorchLayer):
        for name, sub in layer.children():
            out.extend(_walk(sub, f"{prefix}{name}."))
    elif hasattr(layer, "layers") and isinstance(getattr(layer, "layers"), list):
        for i, s in enumerate(layer.layers):
            out.extend(_walk(s, f"{prefix}{i}."))
    return out


def _flat_params(tree, prefix=""):
    out = {}
    if isinstance(tree, Record):
        for k, v in tree.fields.items():
            out.update(_flat_params(v, f"{prefix}{k}."))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(_flat_params(v, f"{prefix}{i}."))
    elif tree is not None:
        out[prefix[:-1]] = tree
    return out


def state_dict(model, params):
    """PyTorch-style state_dict of a model: parameters and buffers as one flat record of tensors
    ("encoder.layers.0.self_attn.in_proj_weight", "bn.running_mean", …). Saved with models.save_model it is a
    safetensors file PyTorch can load."""
    flat = {k: np.asarray(ops.raw(v)) for k, v in _flat_params(params).items()}
    for prefix, layer in _walk(model):
        if hasattr(layer, "buffers"):
            for name, val in layer.buffers().items():
                flat[prefix + name] = np.asarray(val)
    r = Record()
    r.fields = flat
    return r


def load_state_dict(model, params, state, strict=True):
    """Copy a PyTorch state_dict (or one saved by Apsi) into the parameter tree and the buffers of `model`.
    Missing or unexpected keys are errors when strict=true; shapes must match."""
    st = dict(state.fields) if isinstance(state, Record) else dict(state)
    if any(isinstance(v, (Record, list, dict)) for v in st.values()):        # nested (e.g. load_safetensors)
        nested = Record()
        nested.fields = {k: (Record(**v) if isinstance(v, dict) else v) for k, v in st.items()}
        st = _flat_params(nested)

    def arr(v):
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        return np.asarray(ops.raw(v), dtype=float)
    want = _flat_params(params)
    used = set()

    def rebuild(tree, prefix=""):
        if isinstance(tree, Record):
            return Record(**{k: rebuild(v, f"{prefix}{k}.") for k, v in tree.fields.items()})
        if isinstance(tree, list):
            return [rebuild(v, f"{prefix}{i}.") for i, v in enumerate(tree)]
        name = prefix[:-1]
        if name not in st:
            if strict:
                raise RuntimeErr(f"load_state_dict: '{name}' is missing from the state_dict")
            return tree
        a = arr(st[name])
        if a.shape != np.shape(ops.raw(tree)):
            raise ShapeError(f"load_state_dict: '{name}' has shape {a.shape}, the model expects "
                             f"{np.shape(ops.raw(tree))}")
        used.add(name)
        return a
    out = rebuild(params)
    for prefix, layer in _walk(model):
        if hasattr(layer, "buffers"):
            for name in list(layer.buffers()):
                key = prefix + name
                if key in st:
                    layer.set_buffer(name, arr(st[key]))
                    used.add(key)
    extra = [k for k in st if k not in used and k not in want]
    if strict and extra:
        raise RuntimeErr(f"load_state_dict: unexpected keys {', '.join(sorted(extra)[:6])}"
                         f"{' …' if len(extra) > 6 else ''}")
    return out


def parameters_count(params):
    return int(sum(int(np.size(ops.raw(l))) for l in tree_leaves(params)))


def module():
    names = {}
    g = globals()
    for k in ("Linear", "Bilinear", "Identity", "Dropout", "Flatten", "Unflatten", "Embedding", "Conv1d", "Conv2d",
              "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
              "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "BatchNorm1d", "BatchNorm2d", "LayerNorm", "RMSNorm",
              "GroupNorm", "LSTM", "GRU", "RNN", "MultiheadAttention", "TransformerEncoderLayer",
              "TransformerEncoder", "TransformerDecoderLayer", "TransformerDecoder", "Transformer", "Sequential",
              "PReLU", "generate_square_subsequent_mask",
              "dropout", "linear", "layer_norm", "normalize", "embedding", "one_hot", "pad", "flatten",
              "scaled_dot_product_attention", "cosine_similarity",
              "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
              "binary_cross_entropy", "binary_cross_entropy_with_logits", "kl_div",
              "CrossEntropyLoss", "NLLLoss", "MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss", "BCELoss",
              "BCEWithLogitsLoss", "KLDivLoss",
              "SGD", "Adam", "AdamW", "Adamax", "RMSprop", "Adagrad", "NAdam", "RAdam",
              "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts",
              "LinearLR", "ConstantLR", "LambdaLR", "MultiplicativeLR", "PolynomialLR", "OneCycleLR", "CyclicLR",
              "ReduceLROnPlateau", "SequentialLR", "ChainedScheduler",
              "state_dict", "load_state_dict", "parameters_count", "value_and_grad_aux", "clip_grad_norm",
              "clip_grad_value", "TensorDataset", "DataLoader", "checkpoint", "checkpoint_layer"):
        names[k] = g[k]
    for cls, fn in ACT_CLASSES.items():
        names[cls] = _act_class(fn)
    for fn, f in ACT_FUNCS.items():
        names[fn] = f
    names["ModuleList"] = lambda mods=(): list(mods)

    def Module(init, forward, name="model", parts=None):
        """A model from its init(seed) and forward(p, x…, train) functions (models converted from PyTorch)."""
        from .layers import custom_layer
        return custom_layer(init, forward, name, parts)
    names["Module"] = Module
    return names
