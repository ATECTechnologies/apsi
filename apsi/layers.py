"""`use layers` — build any neural-network architecture as a graph of layers.

Every layer is a pair of pure functions, so models stay ordinary values that `grad`, optimizers and
`save_params` understand:

    let model = sequential([linear(2, 64), activation(relu), linear(64, 3)])
    var params = model.init(seed=0)            # a parameter tree (lists and records of tensors)
    let logits = model(params, X)              # apply; model(params, X, train=true) enables dropout
    let fit = fit(model, params, X, y, loss="cross_entropy", epochs=50)

Building blocks
  dense:        linear, bilinear, embedding, positional_encoding
  conv/pool:    conv2d_layer, max_pool, avg_pool, global_avg_pool, flatten_layer
  recurrent:    lstm_layer, gru_layer, last_step
  attention:    multi_head_attention, transformer_block, transformer_encoder
  graph (GNN):  gcn, sage, gat, gin, graph_pool   (+ graph(), segment_sum/mean/max/softmax ops)
  norm/reg:     layer_norm_layer, batch_norm_layer, rms_norm_layer, dropout_layer
  shape:        reshape_layer, select, identity
  composition:  sequential, residual, parallel, dag (any directed acyclic graph of layers),
                lambda_layer (any function), activation, scale
Ready-made models
  mlp, cnn, lstm_classifier, transformer_lm, transformer_classifier, autoencoder, vae, gcn_net, gat_net,
  gin_net, resnet_mlp, siamese
Training and inspection: fit, predict, summary, count_params, vae_loss
Backends: every model runs on NumPy (default) or PyTorch — fit(…, backend="torch", device="cuda",
  precision="bf16", compile=true), torch_module / params_from_torch (a real torch.nn.Module), devices(),
  check_torch(). The torch backend runs PyTorch, so it needs the 'python' permission (--allow-python).
"""
import math

import numpy as np

from . import ops
from . import autodiff as ad
from .autodiff import Node
from .errors import RuntimeErr, ShapeError
from .values import Record, tree_leaves


# =============================================================== graph ops (differentiable)
def _ids(x):
    return np.asarray(ops.raw(x)).astype(np.int64).ravel()


def segment_sum(data, ids, n):
    """out[k] = sum of data[i] where ids[i] == k  (k < n). Differentiable in `data`."""
    ids = _ids(ids)
    n = int(n)
    shape = (n,) + tuple(np.shape(ops.raw(data))[1:])
    if isinstance(data, Node):
        return ad.scatter(data, ids, shape)
    out = np.zeros(shape)
    np.add.at(out, ids, np.asarray(ops.raw(data), dtype=float))
    return out


def segment_count(ids, n):
    return np.bincount(_ids(ids), minlength=int(n)).astype(float)


def segment_mean(data, ids, n):
    cnt = np.maximum(segment_count(ids, n), 1.0)
    s = segment_sum(data, ids, n)
    return s / cnt.reshape((-1,) + (1,) * (len(np.shape(ops.raw(s))) - 1))


def segment_max(data, ids, n):
    """Per-segment maximum (no gradient through the choice of maximum)."""
    ids = _ids(ids)
    raw = np.asarray(ops.raw(data), dtype=float)
    out = np.full((int(n),) + raw.shape[1:], -np.inf)
    np.maximum.at(out, ids, raw)
    out[np.isinf(out)] = 0.0
    return out


def segment_softmax(scores, ids, n):
    """Softmax of `scores` within each segment (e.g. over the incoming edges of every node)."""
    ids = _ids(ids)
    m = segment_max(ad.const(scores) if isinstance(scores, Node) else scores, ids, n)[ids]
    e = ops.exp(scores - m)
    den = segment_sum(e, ids, n)[ids]
    return e / (den + 1e-16)


def gather(x, ids):
    """Rows of x at ids (differentiable)."""
    return x[_ids(ids)]


class Graph:
    """A graph for graph neural networks: n nodes and directed edges src -> dst."""

    def __init__(self, src, dst, n, self_loops=True, undirected=False):
        src, dst = _ids(src), _ids(dst)
        if undirected:
            src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
        if self_loops:
            loops = np.arange(int(n))
            keep = src != dst
            src, dst = np.concatenate([src[keep], loops]), np.concatenate([dst[keep], loops])
        if len(src) and (src.max() >= n or dst.max() >= n or src.min() < 0 or dst.min() < 0):
            raise ShapeError(f"graph(): edge refers to a node outside 0..{int(n) - 1}")
        self.src, self.dst, self.n = src, dst, int(n)
        self.in_degree = np.bincount(dst, minlength=self.n).astype(float)
        self.out_degree = np.bincount(src, minlength=self.n).astype(float)
        self.edges = len(src)

    def __repr__(self):
        return f"graph({self.n} nodes, {self.edges} edges)"

    def adjacency(self):
        a = np.zeros((self.n, self.n))
        np.add.at(a, (self.dst, self.src), 1.0)
        return a


def graph(edges=None, n=None, src=None, dst=None, self_loops=True, undirected=False, batch=None):
    """graph(edges=[[0, 1], [1, 2]], n=3) or graph(src=tensor, dst=tensor, n=…). `batch` assigns nodes to
    separate graphs for graph-level tasks (used by graph_pool)."""
    if edges is not None:
        e = np.asarray(ops.raw(edges)).astype(np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), int)
        src, dst = e[:, 0], e[:, 1]
    if src is None or dst is None:
        raise RuntimeErr("graph() needs edges=[[from, to], …] or src= and dst=")
    if n is None:
        n = int(max(np.max(_ids(src), initial=-1), np.max(_ids(dst), initial=-1))) + 1
    g = Graph(src, dst, n, self_loops=self_loops, undirected=undirected)
    g.batch = _ids(batch) if batch is not None else np.zeros(g.n, dtype=np.int64)
    g.graphs = int(g.batch.max()) + 1 if g.n else 0
    return g


# =============================================================== layer base
def _rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))


def _glorot(rng, n_in, n_out):
    s = math.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-s, s, (int(n_in), int(n_out)))


class Layer:
    """A layer: init(seed) -> params, and layer(params, x, train=false, **context) -> output."""
    kind = "layer"

    def init(self, seed=0):
        return Record()                      # no parameters (an empty record keeps optimizers happy)

    def __call__(self, params, x, train=False, **ctx):
        return self.apply(params, x, train=train, **ctx)

    def apply(self, params, x, train=False, **ctx):
        raise NotImplementedError

    def describe(self):
        return self.kind

    def children(self):
        return []

    def __repr__(self):
        return self.describe()


def _seed_of(seed, i):
    return None if seed is None else int(seed) * 1009 + i


class Linear(Layer):
    kind = "linear"

    def __init__(self, n_in, n_out, bias=True, init="he"):
        self.n_in, self.n_out, self.bias, self.how = int(n_in), int(n_out), bool(bias), init

    def init(self, seed=0):
        rng = _rng(seed)
        if self.how == "glorot":
            W = _glorot(rng, self.n_in, self.n_out)
        else:
            W = rng.standard_normal((self.n_in, self.n_out)) * math.sqrt(2.0 / self.n_in)
        return Record(W=W, b=np.zeros(self.n_out)) if self.bias else Record(W=W)

    def apply(self, p, x, train=False, **ctx):
        y = x @ p.W
        return y + p.b if self.bias else y

    def describe(self):
        return f"linear({self.n_in} → {self.n_out})"


class Bilinear(Layer):
    kind = "bilinear"

    def __init__(self, n1, n2, n_out):
        self.n1, self.n2, self.n_out = int(n1), int(n2), int(n_out)

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(W=rng.standard_normal((self.n_out, self.n1, self.n2)) / math.sqrt(self.n1 * self.n2),
                      b=np.zeros(self.n_out))

    def apply(self, p, x, train=False, **ctx):
        a, b = x
        return ops.stack([ops.sum_((a @ p.W[k]) * b, axis=-1) for k in range(self.n_out)], axis=-1) + p.b

    def describe(self):
        return f"bilinear({self.n1} × {self.n2} → {self.n_out})"


class Embedding(Layer):
    kind = "embedding"

    def __init__(self, vocab, d):
        self.vocab, self.d = int(vocab), int(d)

    def init(self, seed=0):
        return Record(E=_rng(seed).standard_normal((self.vocab, self.d)) * 0.02)

    def apply(self, p, x, train=False, **ctx):
        return p.E[np.asarray(ops.raw(x)).astype(np.int64)]

    def describe(self):
        return f"embedding({self.vocab} × {self.d})"


class PositionalEncoding(Layer):
    kind = "positional_encoding"

    def __init__(self, max_len, d, learned=False):
        self.max_len, self.d, self.learned = int(max_len), int(d), bool(learned)

    def init(self, seed=0):
        return Record(P=_rng(seed).standard_normal((self.max_len, self.d)) * 0.02) if self.learned else Record()

    def apply(self, p, x, train=False, **ctx):
        T = np.shape(ops.raw(x))[-2]
        if self.learned:
            return x + p.P[0:T]
        pos = np.arange(T)[:, None]
        i = np.arange(self.d)[None, :]
        ang = pos / np.power(10000.0, (2 * (i // 2)) / self.d)
        pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
        return x + pe

    def describe(self):
        return f"positional_encoding({'learned' if self.learned else 'sinusoidal'}, {self.d})"


class Conv2D(Layer):
    kind = "conv2d"

    def __init__(self, c_in, c_out, k=3, stride=1, padding="same"):
        self.c_in, self.c_out, self.k, self.stride = int(c_in), int(c_out), int(k), int(stride)
        self.padding = self.k // 2 if padding == "same" else int(padding)

    def init(self, seed=0):
        fan = self.c_in * self.k * self.k
        return Record(W=_rng(seed).standard_normal((self.c_out, self.c_in, self.k, self.k)) * math.sqrt(2.0 / fan),
                      b=np.zeros(self.c_out))

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import conv2d
        return conv2d(x, p.W, p.b, stride=self.stride, padding=self.padding)

    def describe(self):
        return f"conv2d({self.c_in} → {self.c_out}, {self.k}×{self.k}, stride {self.stride})"


class MaxPool(Layer):
    kind = "max_pool"

    def __init__(self, k=2):
        self.k = int(k)

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import max_pool2d
        return max_pool2d(x, self.k)

    def describe(self):
        return f"max_pool({self.k})"


class AvgPool(Layer):
    kind = "avg_pool"

    def __init__(self, k=2):
        self.k = int(k)

    def apply(self, p, x, train=False, **ctx):
        N, C, H, W = np.shape(ops.raw(x))
        k = self.k
        h, w = H // k, W // k
        x = x[:, :, 0:h * k, 0:w * k]
        return ops.mean(ops.reshape(x, N, C, h, k, w, k), axis=(3, 5))

    def describe(self):
        return f"avg_pool({self.k})"


class GlobalAvgPool(Layer):
    kind = "global_avg_pool"

    def apply(self, p, x, train=False, **ctx):
        return ops.mean(x, axis=(2, 3))


class Flatten(Layer):
    kind = "flatten"

    def apply(self, p, x, train=False, **ctx):
        n = np.shape(ops.raw(x))[0]
        return ops.reshape(x, n, -1)


class Reshape(Layer):
    kind = "reshape"

    def __init__(self, *shape):
        self.shape = tuple(int(s) for s in shape)

    def apply(self, p, x, train=False, **ctx):
        n = np.shape(ops.raw(x))[0]
        return ops.reshape(x, n, *self.shape)

    def describe(self):
        return f"reshape(N, {', '.join(map(str, self.shape))})"


class LSTMLayer(Layer):
    kind = "lstm"

    def __init__(self, n_in, n_hidden, return_sequences=True):
        self.n_in, self.n_hidden, self.seq = int(n_in), int(n_hidden), bool(return_sequences)

    def init(self, seed=0):
        from .stdlib import lstm_init
        return lstm_init(self.n_in, self.n_hidden, seed=seed)

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import lstm
        out = lstm(p, x)
        return out if self.seq else out[:, -1, :]

    def describe(self):
        return f"lstm({self.n_in} → {self.n_hidden}{'' if self.seq else ', last step'})"


class GRULayer(Layer):
    kind = "gru"

    def __init__(self, n_in, n_hidden, return_sequences=True):
        self.n_in, self.n_hidden, self.seq = int(n_in), int(n_hidden), bool(return_sequences)

    def init(self, seed=0):
        from .stdlib import gru_init
        return gru_init(self.n_in, self.n_hidden, seed=seed)

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import gru
        out = gru(p, x)
        return out if self.seq else out[:, -1, :]

    def describe(self):
        return f"gru({self.n_in} → {self.n_hidden}{'' if self.seq else ', last step'})"


class LastStep(Layer):
    kind = "last_step"

    def apply(self, p, x, train=False, **ctx):
        return x[:, -1, :]


class MultiHeadAttention(Layer):
    kind = "attention"

    def __init__(self, d, heads=4, causal=False):
        self.d, self.h, self.causal = int(d), int(heads), bool(causal)
        if self.d % self.h:
            raise ShapeError(f"multi_head_attention: d={self.d} must be divisible by heads={self.h}")

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(Wq=_glorot(rng, self.d, self.d), Wk=_glorot(rng, self.d, self.d),
                      Wv=_glorot(rng, self.d, self.d), Wo=_glorot(rng, self.d, self.d), bo=np.zeros(self.d))

    def apply(self, p, x, train=False, mask=None, **ctx):
        src = x if not isinstance(x, (list, tuple)) else x[0]
        mem = src if not isinstance(x, (list, tuple)) else x[1]          # cross-attention: [query, memory]
        B, T, D = np.shape(ops.raw(src))
        S = np.shape(ops.raw(mem))[1]
        hd = D // self.h

        def split(t, L):
            return ops.transpose(ops.reshape(t, B, L, self.h, hd), (0, 2, 1, 3))
        q, k, v = split(src @ p.Wq, T), split(mem @ p.Wk, S), split(mem @ p.Wv, S)
        s = (q @ ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(hd)
        if self.causal:
            s = s + np.triu(np.full((T, S), -1e9), k=1)
        if mask is not None:
            s = s + mask
        o = ops.softmax(s, axis=-1) @ v
        o = ops.reshape(ops.transpose(o, (0, 2, 1, 3)), B, T, D)
        return o @ p.Wo + p.bo

    def describe(self):
        return f"multi_head_attention(d={self.d}, heads={self.h}{', causal' if self.causal else ''})"


class LayerNorm(Layer):
    kind = "layer_norm"

    def __init__(self, d, eps=1e-5):
        self.d, self.eps = int(d), float(eps)

    def init(self, seed=0):
        return Record(gamma=np.ones(self.d), beta=np.zeros(self.d))

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import layer_norm
        return layer_norm(x, p.gamma, p.beta, self.eps)

    def describe(self):
        return f"layer_norm({self.d})"


class RMSNorm(Layer):
    kind = "rms_norm"

    def __init__(self, d, eps=1e-6):
        self.d, self.eps = int(d), float(eps)

    def init(self, seed=0):
        return Record(gamma=np.ones(self.d))

    def apply(self, p, x, train=False, **ctx):
        return x / ops.sqrt(ops.mean(x * x, axis=-1, keepdims=True) + self.eps) * p.gamma

    def describe(self):
        return f"rms_norm({self.d})"


class BatchNorm(Layer):
    """Normalises over the batch (and spatial) axes with the current batch's statistics."""
    kind = "batch_norm"

    def __init__(self, d, eps=1e-5):
        self.d, self.eps = int(d), float(eps)

    def init(self, seed=0):
        return Record(gamma=np.ones(self.d), beta=np.zeros(self.d))

    def apply(self, p, x, train=False, **ctx):
        nd = len(np.shape(ops.raw(x)))
        axes = (0,) if nd == 2 else tuple(i for i in range(nd) if i != 1)
        shape = (1, self.d) + (1,) * (nd - 2)
        m = ops.mean(x, axis=axes, keepdims=True)
        v = ops.mean((x - m) * (x - m), axis=axes, keepdims=True)
        y = (x - m) / ops.sqrt(v + self.eps)
        return y * ops.reshape(p.gamma, *shape) + ops.reshape(p.beta, *shape)

    def describe(self):
        return f"batch_norm({self.d})"


class Dropout(Layer):
    kind = "dropout"

    def __init__(self, p=0.1):
        self.p = float(p)

    def apply(self, params, x, train=False, **ctx):
        if not train or self.p <= 0:
            return x
        mask = (np.random.random(np.shape(ops.raw(x))) >= self.p) / (1.0 - self.p)
        return x * mask

    def describe(self):
        return f"dropout({self.p})"


class Activation(Layer):
    kind = "activation"

    def __init__(self, fn, name=None):
        self.fn = _activation(fn)
        self.name = name or (fn if isinstance(fn, str) else getattr(fn, "__name__", "fn"))

    def apply(self, p, x, train=False, **ctx):
        return self.fn(x)

    def describe(self):
        return f"activation({self.name})"


def _leaky(x, a=0.01):
    return ops.where(np.asarray(ops.raw(x)) > 0, x, a * x)


def _elu(x):
    return ops.where(np.asarray(ops.raw(x)) > 0, x, ops.exp(ops.where(np.asarray(ops.raw(x)) > 0, 0 * x, x)) - 1)


ACTIVATIONS = {
    "relu": ops.relu, "gelu": ops.gelu, "tanh": ops.tanh, "sigmoid": ops.sigmoid,
    "softplus": lambda x: ops.log(1 + ops.exp(x)), "silu": lambda x: x * ops.sigmoid(x),
    "swish": lambda x: x * ops.sigmoid(x), "leaky_relu": _leaky, "elu": _elu,
    "mish": lambda x: x * ops.tanh(ops.log(1 + ops.exp(x))), "linear": lambda x: x, "identity": lambda x: x,
    "softmax": lambda x: ops.softmax(x, axis=-1),
}


def _activation(fn):
    if isinstance(fn, str):
        if fn not in ACTIVATIONS:
            raise RuntimeErr(f"unknown activation '{fn}' (choose from {', '.join(ACTIVATIONS)})")
        return ACTIVATIONS[fn]
    if not callable(fn):
        raise RuntimeErr("activation() needs a function or a name such as \"relu\"")
    return fn


class LambdaLayer(Layer):
    kind = "lambda"

    def __init__(self, fn, name="lambda", torch_fn=None):
        self.fn, self.name, self.torch_fn = fn, name, torch_fn

    def apply(self, p, x, train=False, **ctx):
        return self.fn(x)

    def describe(self):
        return f"lambda_layer({self.name})"


class MeanPool(Layer):
    """Mean over one axis, e.g. over time: [B, T, D] -> [B, D]."""
    kind = "mean_pool"

    def __init__(self, axis=1):
        self.axis = int(axis)

    def apply(self, p, x, train=False, **ctx):
        return ops.mean(x, axis=self.axis)

    def describe(self):
        return f"mean_pool(axis={self.axis})"


class Scale(Layer):
    """A learnable per-feature scale (and shift)."""
    kind = "scale"

    def __init__(self, d, shift=True):
        self.d, self.shift = int(d), bool(shift)

    def init(self, seed=0):
        return Record(s=np.ones(self.d), t=np.zeros(self.d)) if self.shift else Record(s=np.ones(self.d))

    def apply(self, p, x, train=False, **ctx):
        return x * p.s + p.t if self.shift else x * p.s


class Identity(Layer):
    kind = "identity"

    def apply(self, p, x, train=False, **ctx):
        return x


class Select(Layer):
    """Pick one element of a list/tuple input (e.g. the output of parallel)."""
    kind = "select"

    def __init__(self, i):
        self.i = int(i)

    def apply(self, p, x, train=False, **ctx):
        return x[self.i]

    def describe(self):
        return f"select({self.i})"


# ---------------------------------------------------------------- composition
class Sequential(Layer):
    kind = "sequential"

    def __init__(self, layers):
        self.layers = [_as_layer(l) for l in layers]

    def init(self, seed=0):
        return [l.init(_seed_of(seed, i)) for i, l in enumerate(self.layers)]

    def apply(self, params, x, train=False, **ctx):
        if params is None:
            params = [None] * len(self.layers)
        if len(params) != len(self.layers):
            raise ShapeError(f"sequential: {len(self.layers)} layers but {len(params)} parameter entries "
                             f"(create them with model.init())")
        for layer, p in zip(self.layers, params):
            x = layer(p, x, train=train, **ctx)
        return x

    def children(self):
        return self.layers

    def describe(self):
        return f"sequential({len(self.layers)} layers)"


class Residual(Layer):
    """x + f(x), optionally followed by a normalisation layer (post-norm) or preceded by one (pre_norm)."""
    kind = "residual"

    def __init__(self, inner, pre_norm=None):
        self.inner = _as_layer(inner)
        self.norm = _as_layer(pre_norm) if pre_norm is not None else None

    def init(self, seed=0):
        return Record(inner=self.inner.init(_seed_of(seed, 0)),
                      norm=self.norm.init(_seed_of(seed, 1)) if self.norm else Record())

    def apply(self, p, x, train=False, **ctx):
        h = self.norm(p.norm, x, train=train, **ctx) if self.norm else x
        return x + self.inner(p.inner, h, train=train, **ctx)

    def children(self):
        return [self.inner] + ([self.norm] if self.norm else [])


class Parallel(Layer):
    """Run several branches on the same input; combine with "concat" (last axis), "add", "mul", "mean" or "list"."""
    kind = "parallel"

    def __init__(self, branches, combine="concat"):
        self.branches = [_as_layer(b) for b in branches]
        if combine not in ("concat", "add", "mul", "mean", "list"):
            raise RuntimeErr("parallel(): combine must be concat, add, mul, mean or list")
        self.combine = combine

    def init(self, seed=0):
        return [b.init(_seed_of(seed, i)) for i, b in enumerate(self.branches)]

    def apply(self, params, x, train=False, **ctx):
        outs = [b(p, x, train=train, **ctx) for b, p in zip(self.branches, params)]
        return _combine(outs, self.combine)

    def children(self):
        return self.branches

    def describe(self):
        return f"parallel({len(self.branches)} branches, {self.combine})"


def _combine(outs, how):
    if how == "list":
        return list(outs)
    if how == "concat":
        return ops.concat(outs, axis=-1)
    acc = outs[0]
    for o in outs[1:]:
        acc = acc * o if how == "mul" else acc + o
    return acc / len(outs) if how == "mean" else acc


class DAG(Layer):
    """Any directed acyclic graph of layers.

    dag(nodes=[["h1", linear(4, 16), "x"], ["h2", linear(4, 16), "x"], ["sum", add(), ["h1", "h2"]],
               ["out", linear(16, 1), "sum"]], inputs=["x"], output="out")
    Each node is [name, layer, input name(s)]; a node with several inputs receives them as a list.
    `output` may be a name or a list of names (the model then returns a list).
    """
    kind = "dag"

    def __init__(self, nodes, inputs=("x",), output=None):
        self.inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        self.nodes = []
        known = set(self.inputs)
        for item in nodes:
            if len(item) != 3:
                raise RuntimeErr("dag(): every node is [name, layer, input or [inputs]]")
            name, layer, src = item
            srcs = [src] if isinstance(src, str) else list(src)
            for s in srcs:
                if s not in known:
                    raise RuntimeErr(f"dag(): node '{name}' uses '{s}', which is not defined before it")
            if name in known:
                raise RuntimeErr(f"dag(): the name '{name}' is used twice")
            known.add(name)
            self.nodes.append((name, _as_layer(layer), srcs, isinstance(src, str)))
        self.output = output if output is not None else self.nodes[-1][0]
        outs = [self.output] if isinstance(self.output, str) else list(self.output)
        for o in outs:
            if o not in known:
                raise RuntimeErr(f"dag(): output '{o}' is not a node")

    def init(self, seed=0):
        return Record(**{name: layer.init(_seed_of(seed, i)) for i, (name, layer, _, _) in enumerate(self.nodes)})

    def apply(self, params, x, train=False, **ctx):
        vals = {}
        if len(self.inputs) == 1:
            vals[self.inputs[0]] = x
        else:
            if not isinstance(x, (list, tuple)) or len(x) != len(self.inputs):
                raise ShapeError(f"dag(): expected {len(self.inputs)} inputs {self.inputs} as a list")
            vals.update(zip(self.inputs, x))
        for name, layer, srcs, single in self.nodes:
            inp = vals[srcs[0]] if single else [vals[s] for s in srcs]
            vals[name] = layer(getattr(params, name), inp, train=train, **ctx)
        return vals[self.output] if isinstance(self.output, str) else [vals[o] for o in self.output]

    def children(self):
        return [l for _, l, _, _ in self.nodes]

    def describe(self):
        return f"dag({len(self.nodes)} nodes, inputs {', '.join(self.inputs)})"


class Combine(Layer):
    def __init__(self, how):
        self.how, self.kind = how, how

    def apply(self, p, x, train=False, **ctx):
        return _combine(list(x), self.how)


def _as_layer(x):
    if isinstance(x, Layer):
        return x
    if isinstance(x, str):
        return Activation(x)
    if callable(x):
        return LambdaLayer(x, getattr(x, "__name__", getattr(x, "name", "fn")))
    raise RuntimeErr(f"not a layer: {x!r}")


# ---------------------------------------------------------------- graph neural network layers
def _graph(ctx):
    g = ctx.get("graph")
    if g is None:
        raise RuntimeErr("graph layers need the graph: model(params, X, graph=g)  (create g with graph(edges, n))")
    return g


class GCN(Layer):
    """Graph convolution (Kipf & Welling): h_i = Σ_j x_j W / sqrt(deg_i deg_j)."""
    kind = "gcn"

    def __init__(self, n_in, n_out, bias=True):
        self.n_in, self.n_out, self.bias = int(n_in), int(n_out), bool(bias)

    def init(self, seed=0):
        return Record(W=_glorot(_rng(seed), self.n_in, self.n_out), b=np.zeros(self.n_out))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        h = x @ p.W
        norm = 1.0 / np.sqrt(np.maximum(g.in_degree[g.dst] * g.in_degree[g.src], 1.0))
        out = segment_sum(h[g.src] * norm[:, None], g.dst, g.n)
        return out + p.b if self.bias else out

    def describe(self):
        return f"gcn({self.n_in} → {self.n_out})"


class SAGE(Layer):
    """GraphSAGE: h_i = x_i W_self + mean_{j→i} x_j W_neigh."""
    kind = "sage"

    def __init__(self, n_in, n_out, aggregate="mean"):
        self.n_in, self.n_out, self.agg = int(n_in), int(n_out), aggregate
        if aggregate not in ("mean", "sum", "max"):
            raise RuntimeErr("sage(): aggregate must be mean, sum or max")

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(Ws=_glorot(rng, self.n_in, self.n_out), Wn=_glorot(rng, self.n_in, self.n_out),
                      b=np.zeros(self.n_out))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        msg = x[g.src]
        if self.agg == "mean":
            neigh = segment_mean(msg, g.dst, g.n)
        elif self.agg == "sum":
            neigh = segment_sum(msg, g.dst, g.n)
        else:
            neigh = segment_max(msg, g.dst, g.n)
        return x @ p.Ws + neigh @ p.Wn + p.b

    def describe(self):
        return f"sage({self.n_in} → {self.n_out}, {self.agg})"


class GAT(Layer):
    """Graph attention (Veličković et al.) with several heads; heads are concatenated (or averaged)."""
    kind = "gat"

    def __init__(self, n_in, n_out, heads=4, concat=True, slope=0.2):
        self.n_in, self.n_out, self.h, self.concat, self.slope = int(n_in), int(n_out), int(heads), bool(concat), slope

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(W=_glorot(rng, self.n_in, self.n_out * self.h),
                      a_src=rng.standard_normal((self.h, self.n_out)) * 0.1,
                      a_dst=rng.standard_normal((self.h, self.n_out)) * 0.1,
                      b=np.zeros(self.n_out * self.h if self.concat else self.n_out))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        h = ops.reshape(x @ p.W, g.n, self.h, self.n_out)                  # [N, H, F]
        e_src = ops.sum_(h * p.a_src, axis=-1)                               # [N, H]
        e_dst = ops.sum_(h * p.a_dst, axis=-1)
        e = _leaky(e_src[g.src] + e_dst[g.dst], self.slope)                 # [E, H]
        alpha = segment_softmax(e, g.dst, g.n)                              # softmax over incoming edges
        out = segment_sum(h[g.src] * ops.reshape(alpha, g.edges, self.h, 1), g.dst, g.n)   # [N, H, F]
        out = ops.reshape(out, g.n, self.h * self.n_out) if self.concat else ops.mean(out, axis=1)
        return out + p.b

    def describe(self):
        return f"gat({self.n_in} → {self.n_out} × {self.h} heads{'' if self.concat else ', averaged'})"


class GIN(Layer):
    """Graph isomorphism network layer: MLP((1 + eps) x_i + Σ_j x_j)."""
    kind = "gin"

    def __init__(self, n_in, n_out, hidden=None):
        hidden = int(hidden or n_out)
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.mlp = Sequential([Linear(n_in, hidden), Activation("relu"), Linear(hidden, n_out)])

    def init(self, seed=0):
        return Record(eps=np.zeros(1), mlp=self.mlp.init(seed))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        keep = g.src != g.dst                                               # neighbours only
        agg = segment_sum(x[g.src[keep]], g.dst[keep], g.n)
        return self.mlp(p.mlp, (1 + p.eps) * x + agg, train=train, **ctx)

    def describe(self):
        return f"gin({self.n_in} → {self.n_out})"


class GraphPool(Layer):
    """Node features -> one vector per graph (mean, sum or max over each graph's nodes)."""
    kind = "graph_pool"

    def __init__(self, how="mean"):
        self.how = how

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        f = {"mean": segment_mean, "sum": segment_sum, "max": segment_max}[self.how]
        return f(x, g.batch, g.graphs)

    def describe(self):
        return f"graph_pool({self.how})"


# ---------------------------------------------------------------- transformer
class TransformerBlock(Layer):
    """Pre-norm transformer block: x + attention(norm(x)), then x + MLP(norm(x))."""
    kind = "transformer_block"

    def __init__(self, d, heads=4, ff=None, causal=False, dropout=0.0, activation="gelu"):
        ff = int(ff or 4 * d)
        self.d = int(d)
        self.attn = Residual(MultiHeadAttention(d, heads, causal), pre_norm=LayerNorm(d))
        self.mlp = Residual(Sequential([Linear(d, ff), Activation(activation), Dropout(dropout), Linear(ff, d)]),
                            pre_norm=LayerNorm(d))
        self.heads, self.ff, self.causal = int(heads), ff, bool(causal)

    def init(self, seed=0):
        return Record(attn=self.attn.init(_seed_of(seed, 0)), mlp=self.mlp.init(_seed_of(seed, 1)))

    def apply(self, p, x, train=False, **ctx):
        return self.mlp(p.mlp, self.attn(p.attn, x, train=train, **ctx), train=train, **ctx)

    def children(self):
        return [self.attn, self.mlp]

    def describe(self):
        return f"transformer_block(d={self.d}, heads={self.heads}, ff={self.ff}{', causal' if self.causal else ''})"


# =============================================================== constructors exposed to Apsi
def linear(n_in, n_out, bias=True, init="he"):
    return Linear(n_in, n_out, bias, init)


def dense(n_in, n_out, act=None):
    return Linear(n_in, n_out) if act is None else Sequential([Linear(n_in, n_out), Activation(act)])


def activation(fn):
    return Activation(fn)


def sequential(layers):
    return Sequential(layers)


def residual(inner, pre_norm=None):
    return Residual(inner, pre_norm)


def parallel(branches, combine="concat"):
    return Parallel(branches, combine)


def dag(nodes, inputs=("x",), output=None):
    return DAG(nodes, inputs, output)


def add():
    return Combine("add")


def multiply():
    return Combine("mul")


def concatenate():
    return Combine("concat")


def transformer_encoder(d, heads=4, layers=2, ff=None, causal=False, dropout=0.0):
    return Sequential([TransformerBlock(d, heads, ff, causal, dropout) for _ in range(int(layers))] + [LayerNorm(d)])


# ---------------------------------------------------------------- model zoo
def mlp(sizes, act="relu", dropout=0.0, norm=False, out_act=None):
    """Multilayer perceptron, e.g. mlp([2, 64, 64, 3])."""
    sizes = [int(s) for s in sizes]
    L = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        L.append(Linear(a, b))
        if i < len(sizes) - 2:
            if norm:
                L.append(LayerNorm(b))
            L.append(Activation(act))
            if dropout:
                L.append(Dropout(dropout))
    if out_act:
        L.append(Activation(out_act))
    return Sequential(L)


def resnet_mlp(d_in, d_out, width=64, blocks=3, act="gelu"):
    """Deep MLP with residual blocks and layer norm — trains stably at any depth."""
    body = [Residual(Sequential([Linear(width, width), Activation(act), Linear(width, width)]), pre_norm=LayerNorm(width))
            for _ in range(int(blocks))]
    return Sequential([Linear(d_in, width)] + body + [LayerNorm(width), Linear(width, d_out)])


def cnn(in_channels, classes, image=28, channels=(16, 32), dense_width=64):
    """Small convolutional classifier for images shaped [N, C, H, W]."""
    L, c, size = [], int(in_channels), int(image)
    for ch in channels:
        L += [Conv2D(c, ch, 3), Activation("relu"), MaxPool(2)]
        c, size = int(ch), size // 2
    L += [Flatten(), Linear(c * size * size, dense_width), Activation("relu"), Linear(dense_width, classes)]
    return Sequential(L)


def lstm_classifier(n_in, hidden, classes, layers=1):
    L, d = [], int(n_in)
    for i in range(int(layers)):
        L.append(LSTMLayer(d, hidden, return_sequences=i < int(layers) - 1))
        d = int(hidden)
    return Sequential(L + [Linear(d, classes)])


def transformer_lm(vocab, d=64, heads=4, layers=2, context=128, dropout=0.0):
    """Decoder-only (GPT-style) language model: token ids [B, T] -> logits [B, T, vocab]."""
    return Sequential([Embedding(vocab, d), PositionalEncoding(context, d, learned=True),
                       transformer_encoder(d, heads, layers, causal=True, dropout=dropout), Linear(d, vocab)])


def transformer_classifier(vocab, classes, d=64, heads=4, layers=2, context=128):
    """Encoder for sequences of token ids [B, T] -> class logits [B, classes] (mean pooling)."""
    return Sequential([Embedding(vocab, d), PositionalEncoding(context, d),
                       transformer_encoder(d, heads, layers), MeanPool(1),
                       Linear(d, classes)])


def autoencoder(d_in, latent, hidden=(64,), act="relu"):
    """Returns a DAG with outputs [reconstruction, code]."""
    enc = mlp([d_in, *hidden, latent], act)
    dec = mlp([latent, *reversed(hidden), d_in], act)
    return DAG([["code", enc, "x"], ["recon", dec, "code"]], inputs=["x"], output=["recon", "code"])


class _Reparam(Layer):
    kind = "reparameterize"

    def apply(self, p, x, train=False, **ctx):
        mu, logvar = x
        if not train:
            return mu
        eps = np.random.standard_normal(np.shape(ops.raw(mu)))
        return mu + ops.exp(0.5 * logvar) * eps


def vae(d_in, latent, hidden=(64,), act="relu"):
    """Variational autoencoder. Returns [reconstruction, mu, logvar]; train with vae_loss."""
    body = mlp([d_in, *hidden], act)
    h = int(hidden[-1]) if hidden else int(d_in)
    nodes = [["h", Sequential([body, Activation(act)]), "x"], ["mu", Linear(h, latent), "h"],
             ["logvar", Linear(h, latent), "h"], ["z", _Reparam(), ["mu", "logvar"]],
             ["recon", mlp([latent, *reversed(hidden), d_in], act), "z"]]
    return DAG(nodes, inputs=["x"], output=["recon", "mu", "logvar"])


def vae_loss(outputs, x, beta=1.0):
    recon, mu, logvar = outputs
    rec = ops.mean(ops.sum_((recon - x) * (recon - x), axis=-1))
    kl = -0.5 * ops.mean(ops.sum_(1 + logvar - mu * mu - ops.exp(logvar), axis=-1))
    return rec + beta * kl


def gcn_net(n_in, hidden, classes, layers=2, dropout=0.5):
    L, d = [], int(n_in)
    for i in range(int(layers) - 1):
        L += [Dropout(dropout), GCN(d, hidden), Activation("relu")]
        d = int(hidden)
    return Sequential(L + [Dropout(dropout), GCN(d, classes)])


def gat_net(n_in, hidden, classes, heads=4, dropout=0.3):
    return Sequential([Dropout(dropout), GAT(n_in, hidden, heads), Activation("elu"), Dropout(dropout),
                       GAT(int(hidden) * int(heads), classes, heads=1, concat=False)])


def gin_net(n_in, hidden, classes, layers=3, pool="sum"):
    """Graph-level classifier: GIN layers, pooling per graph, linear head."""
    L, d = [], int(n_in)
    for _ in range(int(layers)):
        L += [GIN(d, hidden), Activation("relu")]
        d = int(hidden)
    return Sequential(L + [GraphPool(pool), Linear(d, classes)])


def siamese(encoder, combine="concat", head=None):
    """Shared encoder applied to [a, b]; the outputs are combined (concat/add/mul/list) and optionally fed to head."""
    enc = _as_layer(encoder)
    nodes = [["ea", enc, "a"], ["eb", _Shared(enc, "ea"), "b"], ["both", Combine(combine), ["ea", "eb"]]]
    if head is not None:
        nodes.append(["out", _as_layer(head), "both"])
    return DAG(nodes, inputs=["a", "b"])


class _Shared(Layer):
    """Re-uses the parameters of another DAG node."""

    def __init__(self, layer, owner):
        self.layer, self.owner, self.kind = layer, owner, f"shared({owner})"

    def apply(self, p, x, train=False, **ctx):
        return self.layer(getattr(ctx["_dag_params"], self.owner), x, train=train, **ctx)


_dag_apply = DAG.apply


def _dag_apply_shared(self, params, x, train=False, **ctx):
    ctx = dict(ctx)
    ctx["_dag_params"] = params
    return _dag_apply(self, params, x, train=train, **ctx)


DAG.apply = _dag_apply_shared


# =============================================================== training & inspection
def _loss_fn(loss):
    from .stdlib import mse, cross_entropy
    if callable(loss):
        return loss
    table = {"mse": mse, "cross_entropy": cross_entropy, "vae": vae_loss,
             "mae": lambda p, y: ops.mean(ops.abs_(p - y)),
             "bce": lambda p, y: -ops.mean(y * ops.log(ops.sigmoid(p) + 1e-12) + (1 - y) * ops.log(1 - ops.sigmoid(p) + 1e-12))}
    if loss not in table:
        raise RuntimeErr(f"fit(): unknown loss '{loss}' (choose from {', '.join(table)} or pass a function)")
    return table[loss]


def _rows(x):
    """Number of examples in an input (a list of inputs for models with several inputs)."""
    if isinstance(x, (list, tuple)):
        sizes = {_rows(v) for v in x}
        if len(sizes) != 1:
            raise ShapeError(f"fit(): the inputs have different numbers of rows {sorted(sizes)}")
        return sizes.pop()
    return np.shape(ops.raw(x))[0]


def _take(x, rows):
    return [_take(v, rows) for v in x] if isinstance(x, (list, tuple)) else x[rows]


def _backend(backend):
    b = str(backend or "numpy").lower()
    if b == "auto":
        from . import torch_backend
        return "torch" if torch_backend.available() else "numpy"
    if b not in ("numpy", "torch"):
        raise RuntimeErr(f"backend must be \"numpy\", \"torch\" or \"auto\" (got '{backend}')")
    return b


def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, graph=None, mask=None, val=None, backend="numpy", device="auto",
        precision="fp32", compile=False):
    """Train `model`. Returns record(params, history, val_history, device).

    loss: "mse", "cross_entropy", "mae", "bce", "vae" ("huber" on torch) or a function loss(prediction, target).
    graph / mask: node-level graph tasks — the whole graph is used every step and only nodes where
    mask is true count in the loss. val=(X_val, y_val) records a validation loss each epoch.
    backend="torch" trains with PyTorch on device ("auto", "cpu", "cuda", "cuda:1", "mps", "xpu") with
    precision "fp64", "fp32", "tf32", "bf16" or "fp16" (mixed precision) and optional torch.compile.
    """
    if _backend(backend) == "torch":
        from . import torch_backend
        return torch_backend.fit(model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip, schedule,
                                 graph, mask, val, device=device, precision=precision, compile=compile)
    from .stdlib import Adam, value_and_grad, clip_grad_norm
    lf = _loss_fn(loss)
    opt = opt or Adam(1e-3)
    ctx = {"graph": graph} if graph is not None else {}
    target = X if y is None else y
    sel = np.asarray(ops.raw(mask)).astype(bool) if mask is not None else None

    def objective(p, xb, yb):
        out = model(p, xb, train=True, **ctx)
        if sel is not None:
            out, yb = out[np.nonzero(sel)[0]], yb[np.nonzero(sel)[0]]
        return lf(out, yb)
    vg = value_and_grad(objective)
    Xr, Yr = X, target
    n = _rows(Xr)
    full = graph is not None or not batch_size
    rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed))
    history, val_hist, step = [], [], 0
    for ep in range(int(epochs)):
        if full:
            batches = [None]
        else:
            idx = rng.permutation(n)
            batches = [idx[i:i + int(batch_size)] for i in range(0, n, int(batch_size))]
        total = 0.0
        for b in batches:
            xb = Xr if b is None else _take(Xr, b)
            yb = Yr if b is None else _take(Yr, b)
            if schedule is not None:
                opt.lr = schedule(step)
            l, g = vg(params, xb, yb)
            if clip:
                g = clip_grad_norm(g, clip)
            params = opt.step(params, g)
            total += float(ops.raw(l)) * (1 if b is None else len(b))
            step += 1
        history.append(total / (1 if full else n))
        extra = ""
        if val is not None:
            vx, vy = val
            vl = float(ops.raw(lf(model(params, vx, **ctx), vy)))
            val_hist.append(vl)
            extra = f"  val {vl:.5f}"
        if verbose and (ep % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1):
            print(f"epoch {ep:4}  loss {history[-1]:.5f}{extra}")
    return Record(params=params, history=history, val_history=val_hist, device="numpy")


def predict(model, params, X, backend="numpy", device="auto", precision="fp32", **ctx):
    if _backend(backend) == "torch":
        from . import torch_backend
        return torch_backend.predict(model, params, X, device=device, precision=precision, **ctx)
    return model(params, X, train=False, **ctx)


def torch_module(model, params, device="cpu", precision="fp32"):
    """The model with its parameters as a torch.nn.Module: module(x) / module(x, graph=g)."""
    from . import torch_backend
    return torch_backend.torch_module(model, params, device, precision)


def params_from_torch(model, module):
    """Parameters of a module made by torch_module (after training it in Python) as an Apsi parameter tree."""
    from . import torch_backend
    return torch_backend.from_torch(model, module)


def devices(torch=True):
    """The compute devices available: NumPy, and PyTorch CPU / CUDA / ROCm / MPS / XPU when installed."""
    from . import torch_backend
    if torch:
        return torch_backend.devices()
    import importlib.util
    out = [Record(name="numpy", kind="cpu", detail=f"NumPy {np.__version__}, float64")]
    if importlib.util.find_spec("torch") is not None:
        out.append(Record(name="torch", kind="locked", detail="PyTorch is installed; run with --allow-python to use it"))
    return out


def check_torch(device="auto", verbose=True):
    """Compare every layer type on NumPy and PyTorch (outputs and gradients, float64)."""
    from . import torch_backend
    return torch_backend.check(device, verbose)


def count_params(params):
    return int(sum(int(np.size(ops.raw(l))) for l in tree_leaves(params)
                   if isinstance(ops.raw(l), np.ndarray) or isinstance(ops.raw(l), float)))


def summary(model, params=None):
    """A table of the model's layers and parameter counts (printed, and returned as a string)."""
    rows = []

    def walk(layer, p, depth):
        n = count_params(p) if p is not None else 0
        rows.append(("  " * depth + layer.describe(), n))
        kids = layer.children()
        if not kids:
            return
        if isinstance(layer, (Sequential, Parallel)):
            subs = p if isinstance(p, list) else [None] * len(kids)
        elif isinstance(layer, DAG):
            subs = [getattr(p, name, None) if p is not None else None for name, _, _, _ in layer.nodes]
        elif isinstance(layer, Residual):
            subs = [p.inner if p is not None else None] + ([p.norm if p is not None else None] if layer.norm else [])
        elif isinstance(layer, TransformerBlock):
            subs = [p.attn, p.mlp] if p is not None else [None, None]
        else:
            subs = [None] * len(kids)
        for k, sp in zip(kids, subs):
            walk(k, sp, depth + 1)
    walk(model, params if params is not None else model.init(0), 0)
    w = max(len(r[0]) for r in rows)
    lines = [f"{'layer'.ljust(w)}  parameters", "-" * (w + 12)]
    lines += [f"{name.ljust(w)}  {n:>10,}" for name, n in rows]
    total = count_params(params if params is not None else model.init(0))
    lines += ["-" * (w + 12), f"{'total'.ljust(w)}  {total:>10,}"]
    text = "\n".join(lines)
    print(text)
    return text


def module(interp=None):
    from .security import need, _policy

    def permitted():
        return interp is None or bool(_policy(interp).python)

    def resolve(backend, what):
        b = str(backend or "numpy").lower()
        if b == "auto":
            from . import torch_backend
            return "torch" if permitted() and torch_backend.available() else "numpy"
        if b == "torch" and interp is not None:
            need(interp, "python", what)
        return b

    def fit_(model, params, X, y=None, backend="numpy", **kw):
        return fit(model, params, X, y, backend=resolve(backend, 'fit(backend="torch")'), **kw)

    def predict_(model, params, X, backend="numpy", **kw):
        return predict(model, params, X, backend=resolve(backend, 'predict(backend="torch")'), **kw)

    def guarded(fn, what):
        def g(*a, **k):
            if interp is not None:
                need(interp, "python", what)
            return fn(*a, **k)
        g.__name__, g.__doc__ = fn.__name__, fn.__doc__
        return g

    def devices_():
        return devices(torch=permitted())
    devices_.__doc__ = devices.__doc__
    fit_.__doc__, predict_.__doc__ = fit.__doc__, predict.__doc__
    names = dict(
        # ops for graphs
        graph=graph, segment_sum=segment_sum, segment_mean=segment_mean, segment_max=segment_max,
        segment_softmax=segment_softmax, segment_count=segment_count, gather=gather,
        # layers
        linear=linear, dense=dense, bilinear=Bilinear, embedding=Embedding, positional_encoding=PositionalEncoding,
        conv2d_layer=Conv2D, max_pool=MaxPool, avg_pool=AvgPool, global_avg_pool=GlobalAvgPool,
        flatten_layer=Flatten, reshape_layer=Reshape, lstm_layer=LSTMLayer, gru_layer=GRULayer, last_step=LastStep,
        multi_head_attention=MultiHeadAttention, transformer_block=TransformerBlock,
        transformer_encoder=transformer_encoder, layer_norm_layer=LayerNorm, rms_norm_layer=RMSNorm,
        batch_norm_layer=BatchNorm, dropout_layer=Dropout, activation=activation, lambda_layer=LambdaLayer,
        scale=Scale, identity=Identity, select=Select, mean_pool=MeanPool,
        gcn=GCN, sage=SAGE, gat=GAT, gin=GIN, graph_pool=GraphPool,
        # composition
        sequential=sequential, residual=residual, parallel=parallel, dag=dag, add=add, multiply=multiply,
        concatenate=concatenate,
        # model zoo
        mlp=mlp, resnet_mlp=resnet_mlp, cnn=cnn, lstm_classifier=lstm_classifier, transformer_lm=transformer_lm,
        transformer_classifier=transformer_classifier, autoencoder=autoencoder, vae=vae, vae_loss=vae_loss,
        gcn_net=gcn_net, gat_net=gat_net, gin_net=gin_net, siamese=siamese,
        # training
        fit=fit_, predict=predict_, summary=summary, count_params=count_params,
        # backends
        torch_module=guarded(torch_module, "torch_module()"), params_from_torch=guarded(params_from_torch, "params_from_torch()"),
        devices=devices_, check_torch=guarded(check_torch, "check_torch()"),
    )
    return names
