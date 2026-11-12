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
  long range:   graph_transformer, global_attention, mix_hop, k_hop, hop_distances, eccentricity (graphs.py)
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
from .values import Record, tree_leaves, tree_map


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
    """A graph for graph neural networks: n nodes and directed edges src -> dst. Every edge also has a relation type
    (`rel`, 0 when the graph has one kind of edge) and a hop count (`hop`: 1 for an ordinary edge, k for a shortcut
    made by k_hop(), 0 for a self loop)."""

    def __init__(self, src, dst, n, self_loops=True, undirected=False, rel=None, hop=None):
        src, dst = _ids(src), _ids(dst)
        rel = np.zeros(len(src), dtype=np.int64) if rel is None else _ids(rel)
        hop = np.ones(len(src), dtype=np.int64) if hop is None else _ids(hop)
        if len(rel) != len(src) or len(hop) != len(src):
            raise ShapeError("graph(): types= needs one entry per edge")
        if undirected:
            src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
            rel, hop = np.concatenate([rel, rel]), np.concatenate([hop, hop])
        if self_loops:
            loops = np.arange(int(n))
            keep = src != dst
            src, dst = np.concatenate([src[keep], loops]), np.concatenate([dst[keep], loops])
            rel = np.concatenate([rel[keep], np.zeros(int(n), dtype=np.int64)])
            hop = np.concatenate([hop[keep], np.zeros(int(n), dtype=np.int64)])
        if len(src) and (src.max() >= n or dst.max() >= n or src.min() < 0 or dst.min() < 0):
            raise ShapeError(f"graph(): edge refers to a node outside 0..{int(n) - 1}")
        self.src, self.dst, self.n = src, dst, int(n)
        self.rel, self.hop = rel, hop
        self.relations = int(rel.max()) + 1 if len(rel) else 1
        self.in_degree = np.bincount(dst, minlength=self.n).astype(float)
        self.out_degree = np.bincount(src, minlength=self.n).astype(float)
        self.edges = len(src)
        self.batch = np.zeros(self.n, dtype=np.int64)
        self.graphs = 1 if self.n else 0

    def __repr__(self):
        extra = f", {self.relations} relations" if self.relations > 1 else ""
        return f"graph({self.n} nodes, {self.edges} edges{extra})"

    def adjacency(self):
        a = np.zeros((self.n, self.n))
        np.add.at(a, (self.dst, self.src), 1.0)
        return a

    def _derived(self, src, dst, rel, hop):
        g = Graph(src, dst, self.n, self_loops=False, rel=rel, hop=hop)
        g.batch, g.graphs = self.batch, self.graphs
        return g


def graph(edges=None, n=None, src=None, dst=None, self_loops=True, undirected=False, batch=None, types=None,
          reverse=False):
    """graph(edges=[[0, 1], [1, 2]], n=3) or graph(src=tensor, dst=tensor, n=…). `batch` assigns nodes to
    separate graphs for graph-level tasks (used by graph_pool). `types` gives every edge a relation (0, 1, …) for
    graph_transformer layers; reverse=true adds each edge backwards as its own relation, so information flows both
    ways with separate weights (use -> definition as well as definition -> use)."""
    if edges is not None:
        e = np.asarray(ops.raw(edges)).astype(np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), int)
        src, dst = e[:, 0], e[:, 1]
    if src is None or dst is None:
        raise RuntimeErr("graph() needs edges=[[from, to], …] or src= and dst=")
    src, dst = _ids(src), _ids(dst)
    if n is None:
        n = int(max(np.max(src, initial=-1), np.max(dst, initial=-1))) + 1
    rel = _ids(types) if types is not None else np.zeros(len(src), dtype=np.int64)
    if reverse:
        R = int(rel.max()) + 1 if len(rel) else 1
        src, dst, rel = np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([rel, rel + R])
    g = Graph(src, dst, n, self_loops=self_loops, undirected=undirected, rel=rel)
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


def rope_tables(T, hd, base=10000.0, offset=0):
    """cos and sin of the rotary position angles, [T, hd] (the half-split layout: pairs (i, i + hd / 2))."""
    half = hd // 2
    inv = 1.0 / (float(base) ** (np.arange(half) * 2.0 / hd))
    ang = (np.arange(int(offset), int(offset) + int(T))[:, None] * inv[None, :])
    return np.concatenate([np.cos(ang)] * 2, 1), np.concatenate([np.sin(ang)] * 2, 1)


def _rotate_half_matrix(hd):
    """R with x @ R = rotate_half(x) = [-x2, x1]."""
    half = hd // 2
    R = np.zeros((hd, hd))
    R[half:, :half] = -np.eye(half)
    R[:half, half:] = np.eye(half)
    return R


def apply_rope(x, base=10000.0, offset=0):
    """Rotary position embedding on [..., T, hd] (positions offset.., the half-split layout of LLaMA / Qwen)."""
    sh = np.shape(ops.raw(x))
    T, hd = sh[-2], sh[-1]
    if hd % 2:
        raise ShapeError(f"rope: the head size {hd} must be even")
    cos, sin = rope_tables(T, hd, base, offset)
    return x * cos + (x @ _rotate_half_matrix(hd)) * sin


class MultiHeadAttention(Layer):
    """Self-attention (or cross-attention when called on [query, memory]).

    kv_heads: grouped-query attention (fewer key / value heads, each shared by heads / kv_heads query heads);
    rope: rotary positions on the queries and keys (rope_base); qk_norm: an RMS norm of each head's query and key
    (learned scale per head dimension) before the positions, as in Qwen3 / Gemma 3."""
    kind = "attention"

    def __init__(self, d, heads=4, causal=False, kv_heads=None, rope=False, rope_base=10000.0, qk_norm=False,
                 eps=1e-6):
        self.d, self.h, self.causal = int(d), int(heads), bool(causal)
        if self.d % self.h:
            raise ShapeError(f"multi_head_attention: d={self.d} must be divisible by heads={self.h}")
        self.kvh = int(kv_heads) if kv_heads else self.h
        if self.kvh < 1 or self.h % self.kvh:
            raise ShapeError(f"multi_head_attention: heads={self.h} must be a multiple of kv_heads={self.kvh}")
        self.hd = self.d // self.h
        self.rope, self.rope_base, self.qk_norm, self.eps = bool(rope), float(rope_base), bool(qk_norm), float(eps)
        if self.rope and self.hd % 2:
            raise ShapeError(f"multi_head_attention: rope needs an even head size (d / heads = {self.hd})")

    def init(self, seed=0):
        rng = _rng(seed)
        dkv = self.kvh * self.hd
        f = dict(Wq=_glorot(rng, self.d, self.d), Wk=_glorot(rng, self.d, dkv), Wv=_glorot(rng, self.d, dkv),
                 Wo=_glorot(rng, self.d, self.d), bo=np.zeros(self.d))
        if self.qk_norm:
            f.update(q_norm=np.ones(self.hd), k_norm=np.ones(self.hd))
        return Record(**f)

    def apply(self, p, x, train=False, mask=None, **ctx):
        src = x if not isinstance(x, (list, tuple)) else x[0]
        mem = src if not isinstance(x, (list, tuple)) else x[1]          # cross-attention: [query, memory]
        B, T, D = np.shape(ops.raw(src))
        S = np.shape(ops.raw(mem))[1]
        hd, h, kvh = self.hd, self.h, self.kvh

        def split(t, L, n):
            return ops.transpose(ops.reshape(t, B, L, n, hd), (0, 2, 1, 3))
        q, k, v = split(src @ p.Wq, T, h), split(mem @ p.Wk, S, kvh), split(mem @ p.Wv, S, kvh)
        if self.qk_norm:
            q = q / ops.sqrt(ops.mean(q * q, axis=-1, keepdims=True) + self.eps) * p.q_norm
            k = k / ops.sqrt(ops.mean(k * k, axis=-1, keepdims=True) + self.eps) * p.k_norm
        if self.rope:
            q, k = apply_rope(q, self.rope_base), apply_rope(k, self.rope_base)
        if kvh != h:                                                       # each key / value head serves h / kvh
            g = h // kvh
            one = np.ones((1, 1, g, 1, 1))
            k = ops.reshape(ops.reshape(k, B, kvh, 1, S, hd) * one, B, h, S, hd)
            v = ops.reshape(ops.reshape(v, B, kvh, 1, S, hd) * one, B, h, S, hd)
        s = (q @ ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(hd)
        if self.causal:
            s = s + np.triu(np.full((T, S), -1e9), k=1)
        if mask is not None:
            s = s + mask
        o = ops.softmax(s, axis=-1) @ v
        o = ops.reshape(ops.transpose(o, (0, 2, 1, 3)), B, T, D)
        return o @ p.Wo + p.bo

    def describe(self):
        extra = (f", kv_heads={self.kvh}" if self.kvh != self.h else "") + (", rope" if self.rope else "") + \
            (", qk_norm" if self.qk_norm else "")
        return f"multi_head_attention(d={self.d}, heads={self.h}{', causal' if self.causal else ''}{extra})"


class SwiGLU(Layer):
    """Gated feed-forward block: down(act(x gate) * (x up)), act "silu" (SwiGLU, LLaMA / Qwen) or "gelu" (GeGLU)."""
    kind = "swiglu"

    def __init__(self, d, ff=None, act="silu", d_out=None):
        self.d, self.ff = int(d), int(ff or (8 * int(d)) // 3)
        self.d_out = int(d_out or d)
        self.act = str(act).lower()
        if self.act not in ("silu", "swish", "gelu", "relu"):
            raise ShapeError(f"swiglu: act '{act}' (silu, gelu, relu)")

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(Wg=_glorot(rng, self.d, self.ff), Wu=_glorot(rng, self.d, self.ff),
                      Wd=_glorot(rng, self.ff, self.d_out))

    def apply(self, p, x, train=False, **ctx):
        g = x @ p.Wg
        a = {"silu": lambda t: t * ops.sigmoid(t), "swish": lambda t: t * ops.sigmoid(t), "gelu": ops.gelu,
             "relu": ops.relu}[self.act](g)
        return (a * (x @ p.Wu)) @ p.Wd

    def describe(self):
        return f"swiglu({self.d} -> {self.ff} -> {self.d_out}, {self.act})"


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
    "relu": ops.relu, "gelu": ops.gelu, "gelu_exact": ops.gelu_exact, "tanh": ops.tanh, "sigmoid": ops.sigmoid,
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
    """Pre-norm transformer block: x + attention(norm(x)), then x + MLP(norm(x)).

    norm="layer" | "rms"; ffn="mlp" (Linear, activation, Linear) | "swiglu"; kv_heads / rope / qk_norm: see
    multi_head_attention. transformer_block(d, heads, kv_heads=2, rope=true, qk_norm=true, norm="rms",
    ffn="swiglu") is a LLaMA / Qwen3 block."""
    kind = "transformer_block"

    def __init__(self, d, heads=4, ff=None, causal=False, dropout=0.0, activation="gelu", kv_heads=None, rope=False,
                 rope_base=10000.0, qk_norm=False, norm="layer", ffn="mlp"):
        self.d = int(d)
        self.norm_kind, self.ffn_kind = str(norm).lower(), str(ffn).lower()
        if self.norm_kind not in ("layer", "rms"):
            raise ShapeError(f"transformer_block: norm '{norm}' (layer, rms)")
        if self.ffn_kind not in ("mlp", "swiglu"):
            raise ShapeError(f"transformer_block: ffn '{ffn}' (mlp, swiglu)")
        mk = (lambda: LayerNorm(d)) if self.norm_kind == "layer" else (lambda: RMSNorm(d))
        self.attn = Residual(MultiHeadAttention(d, heads, causal, kv_heads, rope, rope_base, qk_norm), pre_norm=mk())
        if self.ffn_kind == "swiglu":
            ff = int(ff or (8 * int(d)) // 3)
            body = SwiGLU(d, ff, "silu" if str(activation).lower() == "gelu" else activation)   # (gelu: the default)
            if dropout:
                body = Sequential([body, Dropout(dropout)])
        else:
            ff = int(ff or 4 * d)
            body = Sequential([Linear(d, ff), Activation(activation), Dropout(dropout), Linear(ff, d)])
        self.mlp = Residual(body, pre_norm=mk())
        self.heads, self.ff, self.causal = int(heads), ff, bool(causal)

    def init(self, seed=0):
        return Record(attn=self.attn.init(_seed_of(seed, 0)), mlp=self.mlp.init(_seed_of(seed, 1)))

    def apply(self, p, x, train=False, **ctx):
        return self.mlp(p.mlp, self.attn(p.attn, x, train=train, **ctx), train=train, **ctx)

    def children(self):
        return [self.attn, self.mlp]

    def describe(self):
        a = self.attn.inner
        extra = (f", kv_heads={a.kvh}" if a.kvh != a.h else "") + (", rope" if a.rope else "") + \
            (", qk_norm" if a.qk_norm else "") + (", rms" if self.norm_kind == "rms" else "") + \
            (", swiglu" if self.ffn_kind == "swiglu" else "")
        return f"transformer_block(d={self.d}, heads={self.heads}, ff={self.ff}{', causal' if self.causal else ''}{extra})"


# =============================================================== constructors exposed to Apsi
def linear(n_in, n_out, bias=True, init="he"):
    return Linear(n_in, n_out, bias, init)


def dense(n_in, n_out, act=None):
    return Linear(n_in, n_out) if act is None else Sequential([Linear(n_in, n_out), Activation(act)])


def activation(fn):
    return Activation(fn)


class CustomLayer(Layer):
    """A layer defined by two Apsi functions: init(seed) -> parameters and forward(p, x..., train) -> output.
    `parts` names the sub-layers (a record of layers or lists of layers) so summaries and weight import can see
    inside. Models converted from PyTorch (`apsi convert`) are custom layers."""
    kind = "custom"

    def __init__(self, init_fn, forward_fn, name="model", parts=None):
        self.init_fn, self.forward_fn, self.name = init_fn, forward_fn, str(name)
        self.parts = parts
        self.training = True                  # like a PyTorch module: train() / eval() switch the mode

    def init(self, seed=0):
        return self.init_fn(seed=0 if seed is None else int(seed))

    def __call__(self, params, *xs, train=None, **kw):
        return self.forward_fn(params, *xs, train=self.training if train is None else bool(train), **kw)

    def train(self, mode=True):
        self.training = bool(mode)
        for c in self.children():
            if hasattr(c, "train") and callable(c.train):
                c.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def apply(self, params, x, train=False, **ctx):
        return self.forward_fn(params, x, train=train)

    def children(self):
        if self.parts is None:
            return []
        out = []
        for v in (self.parts.fields.values() if isinstance(self.parts, Record) else self.parts):
            out.extend(v if isinstance(v, list) else [v])
        return [c for c in out if hasattr(c, "init") and callable(getattr(c, "init"))]

    def describe(self):
        return self.name


class Checkpoint(Layer):
    """Wraps a layer so that its activations are recomputed in the backward pass instead of stored."""
    kind = "checkpoint"

    def __init__(self, inner):
        self.inner = _as_layer(inner)

    def init(self, seed=0):
        return self.inner.init(seed)

    def apply(self, params, x, train=False, **ctx):
        from .torch_compat import checkpoint_layer
        return checkpoint_layer(self.inner, params, x, train=train, **ctx)

    def children(self):
        return [self.inner]

    def describe(self):
        return f"checkpoint({self.inner.describe()})"


def checkpoint(layer):
    """Gradient checkpointing: checkpoint(layer) gives the same results with much less activation memory."""
    return Checkpoint(layer)


def custom_layer(init, forward, name="model", parts=None):
    """A layer from your own functions: init(seed) returns the parameter tree, forward(p, x, train) the output."""
    return CustomLayer(init, forward, name, parts)


def _np_of(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(ops.raw(t), dtype=np.float64)


def load_torch_weights(model, params, state, prefix=""):
    """Copy a PyTorch state_dict (or a flat record/dict of arrays with the same names, e.g. from
    load_safetensors(path, nested=false)) into an Apsi parameter tree of the same architecture.
    Linear weights are transposed, LSTM/GRU gate weights fused, embeddings and norms renamed. Returns the new
    parameter tree; raises a clear error naming any missing or mis-shaped tensor."""
    st = dict(state.fields) if isinstance(state, Record) else dict(state)
    st = {k.replace("__", "."): v for k, v in st.items()} if not any("." in k for k in st) else st
    used = set()

    def get(name, shape=None):
        if name not in st:
            raise RuntimeErr(f"load_torch_weights: '{name}' is not in the state_dict "
                             f"(it has {', '.join(sorted(st)[:8])}{' …' if len(st) > 8 else ''})")
        used.add(name)
        a = _np_of(st[name])
        if shape is not None and tuple(a.shape) != tuple(shape):
            raise ShapeError(f"load_torch_weights: '{name}' has shape {tuple(a.shape)}, the Apsi layer expects "
                             f"{tuple(shape)}")
        return a

    def leaf(layer, p, pre):
        k = layer.kind
        if isinstance(layer, CustomLayer):
            out = {}
            for name, sub in (layer.parts.fields.items() if isinstance(layer.parts, Record) else []):
                if isinstance(sub, list):
                    out[name] = [leaf(l, pp, f"{pre}{name}.{i}.") if isinstance(l, Layer) else pp
                                 for i, (l, pp) in enumerate(zip(sub, p[name]))]
                elif isinstance(sub, Layer):
                    out[name] = leaf(sub, p[name], f"{pre}{name}.")
            for name in p.fields:                              # nn.Parameter / buffers of this module
                if name not in out:
                    full = f"{pre}{name}"
                    out[name] = get(full, np.shape(ops.raw(p[name]))) if full in st else p[name]
            return Record(**out)
        if isinstance(layer, Sequential):
            return [leaf(l, pp, f"{pre}{i}.") for i, (l, pp) in enumerate(zip(layer.layers, p))]
        if k == "linear":
            W = get(pre + "weight", (layer.n_out, layer.n_in)).T
            return Record(W=W, b=get(pre + "bias", (layer.n_out,))) if layer.bias else Record(W=W)
        if k == "conv2d":
            W = get(pre + "weight", (layer.c_out, layer.c_in, layer.k, layer.k))
            return Record(W=W, b=get(pre + "bias", (layer.c_out,))) if "b" in p.fields else Record(W=W)
        if k == "embedding":
            return Record(E=get(pre + "weight", (layer.vocab, layer.d)))
        if k in ("layer_norm", "batch_norm"):
            return Record(gamma=get(pre + "weight", (layer.d,)), beta=get(pre + "bias", (layer.d,)))
        if k == "rms_norm":
            return Record(gamma=get(pre + "weight", (layer.d,)))
        if k in ("lstm", "gru"):
            H = layer.n_hidden
            g = 4 if k == "lstm" else 3
            wi = get(pre + "weight_ih_l0", (g * H, layer.n_in))
            wh = get(pre + "weight_hh_l0", (g * H, H))
            bi = get(pre + "bias_ih_l0", (g * H,))
            bh = get(pre + "bias_hh_l0", (g * H,))
            if k == "gru":
                raise RuntimeErr("load_torch_weights: GRU weights use a different gate layout; train the GRU in "
                                 "Apsi or keep it in PyTorch")
            return Record(W=np.concatenate([wi.T, wh.T], axis=0), b=bi + bh)
        if count_params(p) == 0:
            return p
        raise RuntimeErr(f"load_torch_weights: layer kind '{k}' at '{pre or 'model'}' has no PyTorch mapping yet")

    out = leaf(model, params, str(prefix))
    return out


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


def transformer_encoder(d, heads=4, layers=2, ff=None, causal=False, dropout=0.0, **block):
    """layers transformer blocks and a final norm; block: kv_heads, rope, rope_base, qk_norm, norm, ffn, activation."""
    blocks = [TransformerBlock(d, heads, ff, causal, dropout, **block) for _ in range(int(layers))]
    return Sequential(blocks + [RMSNorm(d) if str(block.get("norm", "layer")).lower() == "rms" else LayerNorm(d)])


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


def transformer_lm(vocab, d=64, heads=4, layers=2, context=128, dropout=0.0, ff=None, **block):
    """Decoder-only (GPT-style) language model: token ids [B, T] -> logits [B, T, vocab].

    block options (kv_heads, rope, rope_base, qk_norm, norm, ffn, activation) shape every block; with rope=true the
    model has no learned position table. transformer_lm(32000, d=768, heads=12, layers=12, kv_heads=4, rope=true,
    qk_norm=true, norm="rms", ffn="swiglu") is a modern (LLaMA / Qwen3 style) model."""
    pos = [] if block.get("rope") else [PositionalEncoding(context, d, learned=True)]
    return Sequential([Embedding(vocab, d)] + pos +
                      [transformer_encoder(d, heads, layers, ff=ff, causal=True, dropout=dropout, **block),
                       Linear(d, vocab)])


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


def _backend(backend, model=None, loss=None, plain=True):
    """numpy, torch or gpu. "auto": Apsi's own GPU trainer when there is an NVIDIA card and it takes the model,
    else PyTorch when installed, else NumPy."""
    b = str(backend or "numpy").lower()
    b = {"apsi-gpu": "gpu", "native-gpu": "gpu", "cuda": "gpu", "apsi": "numpy", "cpu": "numpy"}.get(b, b)
    if b in ("auto", "auto-notorch"):
        if model is not None and plain and loss in ("cross_entropy", "mse") and _gpu_ready() and \
                _gpu_supports(model) is None:
            return "gpu"
        if b == "auto-notorch":
            return "numpy"
        from . import torch_backend
        return "torch" if torch_backend.available() else "numpy"
    if b not in ("numpy", "torch", "gpu"):
        raise RuntimeErr(f"backend must be \"numpy\", \"gpu\" (Apsi's own GPU trainer), \"torch\" or \"auto\" "
                         f"(got '{backend}')")
    return b


def _gpu_ready():
    try:
        from .native import cuda
        drv = cuda.driver()
        return drv is not None and bool(drv.devices)
    except Exception:
        return False


def _gpu_supports(model):
    from . import gpu_train
    return gpu_train.supports(model)


def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, graph=None, mask=None, val=None, backend="numpy", device="auto",
        precision="auto", compile=False, workers=1, ema=None, ewc=None, parallel="auto", stages=None,
        micro_batches=None):
    """Train `model`. Returns record(params, history, val_history, device).

    loss: "mse", "cross_entropy", "mae", "bce", "vae" ("huber" on torch) or a function loss(prediction, target).
    graph / mask: node-level graph tasks — the whole graph is used every step and only nodes where
    mask is true count in the loss. val=(X_val, y_val) records a validation loss each epoch.
    backend="torch" trains with PyTorch on device ("auto", "cpu", "cuda", "cuda:1", "mps", "xpu") with
    precision "auto" (default: bf16 mixed precision on GPUs that have it, fp16 with loss scaling on older GPUs, fp32
    on the CPU; APSI_PRECISION overrides it), "fp64", "fp32", "tf32", "bf16", "fp16" or {store, compute} for any
    pair of torch dtypes, and optional torch.compile. The numpy backend trains in fp32 for "auto".
    workers=N (numpy backend, mini-batches): data-parallel training on N processes — every batch is split into N
    shards, the gradients of the shards are averaged (weighted by shard size, so the result is the gradient of the
    whole batch) and one optimizer step is taken. "auto" uses every core.
    backend="gpu" on several cards (device="cuda:all" or "cuda:0,1,2"): parallel="data" (a copy of the model on every
    card), "pipeline" (the model split across the cards; stages=N pieces or a list of the units in each;
    micro_batches=M) or "auto" (copies when the model fits one card, else the fewest pieces that fit).
    """
    which = _backend(backend, model, loss if isinstance(loss, str) else None, graph is None and mask is None)
    if which == "gpu":
        from . import gpu_train
        if graph is not None or mask is not None:
            raise RuntimeErr("backend=\"gpu\": graph tasks train with backend=\"torch\" or \"numpy\"")
        dev = "cuda" if str(device).lower() in ("auto", "gpu", "cuda") else device
        try:
            return gpu_train.fit(model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip, schedule, val,
                                 device=dev, precision=precision, ema=ema, ewc=ewc, parallel=parallel, stages=stages,
                                 micro_batches=micro_batches)
        except gpu_train.TrainError as e:
            raise RuntimeErr(f"backend=\"gpu\": {e}")
    if ema is not None or ewc is not None:
        raise RuntimeErr("ema= and ewc= train on the GPU trainer: fit(..., backend=\"gpu\")")
    if which == "torch":
        from . import torch_backend
        return torch_backend.fit(model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip, schedule,
                                 graph, mask, val, device=device, precision=precision, compile=compile)
    from .stdlib import Adam, value_and_grad
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
    from . import lowbit
    kind = lowbit.parse(precision)
    vg = value_and_grad(objective)
    if kind:                                  # weights in the low-bit format forward, straight-through gradient
        _vg_full = vg

        def vg(p, *a):
            return _vg_full(lowbit.round_tree(p, kind), *a)
    vparams = (lambda p: lowbit.round_tree(p, kind)) if kind else (lambda p: p)
    Xr, Yr = X, target
    low = kind is not None or str(precision).lower() in ("auto", "fp32", "float32", "f32", "tf32", "bf16", "fp16")
    if low:                                   # half the memory of float64 for weights, data and activations
        def _f32(t):
            return tree_map(lambda a: a.astype(np.float32) if isinstance(a, np.ndarray) and a.dtype.kind == "f"
                            else a, t)
        params, Xr = _f32(params), _f32(Xr)
        Yr = _f32(Yr) if np.asarray(ops.raw(Yr)).dtype.kind == "f" else Yr
    n = _rows(Xr)
    full = graph is not None or not batch_size
    rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed))
    history, val_hist, step = [], [], 0
    team = _Team(vg, Xr, Yr, workers) if not full and _want_workers(workers) > 1 else None
    if team is not None:
        vg = team.vg
    try:
        return _fit_loop(locals())
    finally:
        if team is not None:
            team.close()


def _fit_loop(env):
    env.setdefault("vparams", (lambda p: p))
    (model, epochs, full, rng, n, batch_size, Xr, Yr, schedule, opt, vg, params, clip, low, history, val, lf, ctx,
     val_hist, verbose) = (env[k] for k in ("model", "epochs", "full", "rng", "n", "batch_size", "Xr", "Yr",
                                             "schedule", "opt", "vg", "params", "clip", "low", "history", "val",
                                             "lf", "ctx", "val_hist", "verbose"))
    from .stdlib import clip_grad_norm
    _f32 = env.get("_f32")
    step = 0
    from . import thermal
    cool = thermal.cpu_governor()                      # keeps the processor near APSI_CPU_TEMP (77 C)
    for ep in range(int(epochs)):
        if full:
            batches = [None]
        else:
            idx = rng.permutation(n)
            batches = [idx[i:i + int(batch_size)] for i in range(0, n, int(batch_size))]
        total = 0.0
        for b in batches:
            if schedule is not None:
                opt.lr = schedule(step)
            if hasattr(vg, "shards"):
                l, g = vg(params, b)
            else:
                l, g = vg(params, Xr if b is None else _take(Xr, b), Yr if b is None else _take(Yr, b))
            if clip:
                g = clip_grad_norm(g, clip)
            params = opt.step(params, g)
            if low:
                params = _f32(params)
            total += float(ops.raw(l)) * (1 if b is None else len(b))
            step += 1
            cool.tick()
        history.append(total / (1 if full else n))
        extra = ""
        if val is not None:
            vx, vy = val
            vl = float(ops.raw(lf(model(env["vparams"](params), vx, **ctx), vy)))
            val_hist.append(vl)
            extra = f"  val {vl:.5f}"
        if verbose and (ep % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1):
            print(f"epoch {ep:4}  loss {history[-1]:.5f}{extra}")
    return Record(params=params, history=history, val_history=val_hist, device="numpy")


def _want_workers(w):
    import os as _os
    if w in (None, 0, 1, False):
        return 1
    n = (_os.cpu_count() or 1) if str(w).lower() == "auto" else int(w)
    return max(1, min(int(n), 256))


class _Team:
    """Data-parallel helpers: forked processes that compute the loss and gradients of one shard of each batch."""

    def __init__(self, vg, X, Y, workers):
        import multiprocessing as mp
        import sys as _sys
        self.n = _want_workers(workers)
        self.procs = []
        if _sys.platform.startswith("win") or "fork" not in mp.get_all_start_methods():
            print("apsi: note: fit(workers=…) needs fork (Linux, macOS); training on one process", file=_sys.stderr)
            self.vg = vg
            return
        global _TEAM_STATE
        _TEAM_STATE = (vg, X, Y)
        ctx = mp.get_context("fork")
        self.pipes = []
        for _ in range(self.n - 1):                       # the parent computes one shard itself
            a, b = ctx.Pipe()
            p = ctx.Process(target=_team_worker, args=(b,), daemon=True)
            p.start()
            self.procs.append(p)
            self.pipes.append(a)
        self.local = vg
        self.X, self.Y = X, Y

        def parallel(params, b):
            shards = [s for s in np.array_split(np.asarray(b), self.n) if len(s)]
            for pipe, sh in zip(self.pipes, shards[1:]):
                pipe.send((params, sh))
            l0, g0 = self.local(params, _take(self.X, shards[0]), _take(self.Y, shards[0]))
            total = len(b)
            loss = float(ops.raw(l0)) * len(shards[0]) / total
            grad = tree_map(lambda a: np.asarray(ops.raw(a)) * (len(shards[0]) / total), g0)
            for pipe, sh in zip(self.pipes, shards[1:]):
                ok, res = pipe.recv()
                if not ok:
                    raise RuntimeErr(f"fit(workers=…): a worker failed: {res}")
                l, g = res
                w = len(sh) / total
                loss += l * w
                grad = tree_map2(lambda a, c: a + np.asarray(c) * w, grad, g)
            return loss, grad
        parallel.shards = True
        self.vg = parallel

    def close(self):
        for pipe in getattr(self, "pipes", []):
            try:
                pipe.send(None)
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()


_TEAM_STATE = None


def tree_map2(f, a, b):
    """f over the leaves of two trees of the same shape (records, dicts, lists of arrays)."""
    it = iter([f(x, y) for x, y in zip(tree_leaves(a), tree_leaves(b))])
    return tree_map(lambda _: next(it), a)


def _team_worker(pipe):
    vg, X, Y = _TEAM_STATE
    while True:
        msg = pipe.recv()
        if msg is None:
            return
        params, sh = msg
        try:
            l, g = vg(params, _take(X, sh), _take(Y, sh))
            pipe.send((True, (float(ops.raw(l)), tree_map(lambda a: np.asarray(ops.raw(a)), g))))
        except Exception as e:                            # noqa: BLE001 - reported to the parent
            pipe.send((False, f"{type(e).__name__}: {e}"))


def predict(model, params, X, backend="numpy", device="auto", precision="auto", **ctx):
    which = _backend(backend)
    if which == "gpu":
        from . import gpu_train
        dev = "cuda" if str(device).lower() in ("auto", "gpu", "cuda") else device
        return gpu_train.predict(model, params, X, device=dev, precision=precision)
    if which == "torch":
        from . import torch_backend
        return torch_backend.predict(model, params, X, device=device, precision=precision, **ctx)
    from . import lowbit
    kind = lowbit.parse(precision)
    if kind:
        params = lowbit.round_tree(params, kind)
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
        if b == "auto":                     # Apsi's GPU trainer first, then PyTorch when allowed, then NumPy
            return "auto" if permitted() else "auto-notorch"
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
        multi_head_attention=MultiHeadAttention, transformer_block=TransformerBlock, swiglu=SwiGLU,
        apply_rope=apply_rope,
        transformer_encoder=transformer_encoder, layer_norm_layer=LayerNorm, rms_norm_layer=RMSNorm,
        batch_norm_layer=BatchNorm, dropout_layer=Dropout, activation=activation, lambda_layer=LambdaLayer,
        scale=Scale, identity=Identity, select=Select, mean_pool=MeanPool,
        gcn=GCN, sage=SAGE, gat=GAT, gin=GIN, graph_pool=GraphPool,
        # composition
        sequential=sequential, residual=residual, parallel=parallel, dag=dag, add=add, multiply=multiply,
        concatenate=concatenate, custom_layer=custom_layer, checkpoint=checkpoint, diagnose=diagnose, load_torch_weights=load_torch_weights,
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
    from . import graphs                     # long-range graph layers (graph_transformer, global_attention, …)
    names.update(graphs.names())
    return names


# =============================================================== diagnose: what is wrong with a model built from scratch
def diagnose(model, params, X, y=None, loss="mse", lr=1e-3, steps=60, batch=16, seed=0, verbose=True):
    """Checks a new model before a long training run, the way an experienced person would, and says what to fix:

    1. data: NaN / inf in inputs or targets, labels outside the model's classes, one class only, unscaled inputs
    2. forward: the output shape against the targets, NaN / inf in the output
    3. the first loss against what an untrained model should give (log(classes) for cross_entropy)
    4. gradients: parameters that get no gradient (unused or dead), NaN / inf, very large or vanishing ones
    5. it can learn: a small batch is trained for `steps` steps and the loss must fall (otherwise: a bug or the
       learning rate), and the learning rates 10x around `lr` are tried to suggest a better one.

    Returns record(ok, problems, warnings, notes, suggested_lr, losses) and prints a report."""
    from .stdlib import Adam, value_and_grad
    lf = _loss_fn(loss)
    problems, warnings, notes = [], [], []
    target = X if y is None else y
    Xr = np.asarray(ops.raw(X)) if not isinstance(X, (list, tuple)) else None
    Yr = np.asarray(ops.raw(target)) if not isinstance(target, (list, tuple)) else None

    # 1. data
    for name, a in (("inputs", Xr), ("targets", Yr)):
        if a is not None and a.dtype.kind == "f":
            bad = int((~np.isfinite(a)).sum())
            if bad:
                problems.append(f"{bad} NaN/inf value(s) in the {name}: clean them (e.g. where(isnan(x), 0, x))")
    if Xr is not None and Xr.dtype.kind == "f" and np.isfinite(Xr).all() and Xr.size:
        spread = float(np.abs(Xr).max())
        if spread > 1e3:
            warnings.append(f"inputs reach {spread:.3g}: scale them (standardize: (x - mean) / std) or training will "
                            "be slow or unstable")
    n = _rows(X)
    k = min(int(batch), n)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=k, replace=False))
    xb, yb = _take(X, idx), _take(target, idx)

    # 2. forward
    try:
        out = model(params, xb, train=False)
    except Exception as e:
        problems.append(f"the model cannot run on the inputs: {type(e).__name__}: {e}")
        return _report(False, problems, warnings, notes, None, [], verbose)
    o = np.asarray(ops.raw(out)) if not isinstance(out, (list, tuple)) else None
    if o is not None:
        if o.dtype.kind == "f" and not np.isfinite(o).all():
            problems.append("the output has NaN/inf before any training: check the initialisation and operations "
                            "such as log / sqrt / division of values that can be 0 or negative")
        yb_a = np.asarray(ops.raw(yb)) if not isinstance(yb, (list, tuple)) else None
        if loss == "cross_entropy" and yb_a is not None:
            classes = o.shape[-1]
            labels = np.asarray(ops.raw(target)).astype(int) if Yr is not None else yb_a.astype(int)
            valid = labels[labels >= 0]
            if valid.size and valid.max() >= classes:
                problems.append(f"labels go up to {int(valid.max())} but the model has {classes} outputs: "
                                f"the last layer needs {int(valid.max()) + 1} outputs")
            if valid.size and len(np.unique(valid)) == 1:
                warnings.append("every label is the same class: the model can only learn to predict it")
            if o.shape[:-1] != yb_a.shape:
                problems.append(f"output {tuple(o.shape)} does not fit labels {tuple(yb_a.shape)} "
                                "(cross_entropy wants outputs [..., classes] and labels [...])")
        elif yb_a is not None and loss in ("mse", "mae", "bce") and o.shape != yb_a.shape:
            if o.size == yb_a.size:
                warnings.append(f"output {tuple(o.shape)} and targets {tuple(yb_a.shape)} have the same values but "
                                "different shapes: reshape the targets, or the loss may broadcast wrongly")
            else:
                problems.append(f"output {tuple(o.shape)} does not match targets {tuple(yb_a.shape)}")
    if problems:
        return _report(False, problems, warnings, notes, None, [], verbose)

    # 3. first loss
    try:
        l0 = float(ops.raw(lf(model(params, xb, train=True), yb)))
    except Exception as e:
        problems.append(f"the loss cannot be computed: {type(e).__name__}: {e}")
        return _report(False, problems, warnings, notes, None, [], verbose)
    if not np.isfinite(l0):
        problems.append("the first loss is NaN/inf")
        return _report(False, problems, warnings, notes, None, [], verbose)
    if loss == "cross_entropy" and o is not None:
        expect = float(np.log(o.shape[-1]))
        notes.append(f"first loss {l0:.4g} (an untrained model with {o.shape[-1]} classes gives about {expect:.4g})")
        if l0 > 3 * expect + 1:
            warnings.append("the first loss is far above log(classes): the last layer starts with too large weights "
                            "(the model is confidently wrong); initialise it smaller")
    else:
        notes.append(f"first loss {l0:.4g}")

    # 4. gradients
    def obj(p):
        return lf(model(p, xb, train=True), yb)
    _, g = value_and_grad(obj)(params)
    from .models import _flatten
    flat_g = _flatten(g)
    flat_p = _flatten(params)
    dead, huge = [], []
    for name, gv in flat_g.items():
        a = np.asarray(ops.raw(gv), np.float64)
        if not np.isfinite(a).all():
            problems.append(f"gradient of '{name}' has NaN/inf")
            continue
        g_rms = float(np.sqrt((a * a).mean())) if a.size else 0.0
        pv = np.asarray(ops.raw(flat_p.get(name, 0)), np.float64)
        p_rms = float(np.sqrt((pv * pv).mean())) if pv.size else 0.0
        if g_rms == 0.0:
            dead.append(name)
        elif g_rms / max(p_rms, 0.05) > 100:             # one step at lr 1e-3 would move it by over 10 %
            huge.append(name)
    if dead:
        warnings.append(f"{len(dead)} parameter(s) get no gradient (not used by the output, or dead ReLUs): "
                        + ", ".join(dead[:6]) + (" …" if len(dead) > 6 else ""))
    if huge:
        warnings.append("very large gradients for " + ", ".join(huge[:4]) + ": add normalisation or clip=1.0")
    if problems:
        return _report(False, problems, warnings, notes, None, [], verbose)

    # 5. can it learn? a small batch, a few learning rates
    def run(rate):
        p, o_ = params, Adam(rate)
        vg = value_and_grad(obj)
        hist = []
        for _ in range(int(steps)):
            lv, gg = vg(p)
            lv = float(ops.raw(lv))
            hist.append(lv)
            if not np.isfinite(lv):
                break
            p = o_.step(p, gg)
        return hist
    tried = {}
    for rate in (lr / 10, lr, lr * 10):
        h = run(rate)
        tried[rate] = h
    def score(h):
        return h[-1] if h and np.isfinite(h[-1]) else np.inf
    best = min(tried, key=lambda r: score(tried[r]))
    h = tried[lr]
    drop = (h[0] - min(h)) / (abs(h[0]) + 1e-12) if h and np.isfinite(min(h)) else 0.0
    if not np.isfinite(score(h)):
        problems.append(f"training diverges at lr={lr:g} (the loss became NaN/inf): lower the learning rate")
    elif drop < 0.1 and score(tried[best]) > 0.9 * h[0]:
        problems.append(f"the loss does not fall on {k} examples in {steps} steps at any learning rate "
                        f"({h[0]:.4g} → {h[-1]:.4g}): the model cannot fit even a tiny batch — usually a bug (wrong "
                        "targets, a detached part of the network, an activation at the output where none belongs)")
    else:
        notes.append(f"learns: loss {h[0]:.4g} → {h[-1]:.4g} on {k} examples in {steps} steps at lr={lr:g}")
    if best != lr and score(tried[best]) < 0.7 * score(h):
        notes.append(f"lr={best:g} learned faster ({score(tried[best]):.4g} vs {score(h):.4g}): try it")
    return _report(not problems, problems, warnings, notes, best, h, verbose)


def _report(ok, problems, warnings, notes, best, losses, verbose):
    if verbose:
        print("diagnose: " + ("no problems found" if ok and not warnings else
                              f"{len(problems)} problem(s), {len(warnings)} warning(s)"))
        for p in problems:
            print("  ✗ " + p)
        for w in warnings:
            print("  ! " + w)
        for n_ in notes:
            print("  · " + n_)
    return Record(ok=bool(ok), problems=problems, warnings=warnings, notes=notes, suggested_lr=best, losses=losses)
