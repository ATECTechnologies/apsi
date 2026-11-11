"""Long-range graph learning: information that travels many hops, or any distance in one step.

Ordinary message passing (gcn, sage, gat) moves information one edge per layer and squeezes everything a node has
heard into one vector, so in practice values travel only a few hops. This module adds what large graph models use
to reach further:

  graph_transformer   pre-norm residual block, attention over neighbours only, one set of key/value weights per
                      relation and a learned bias per hop distance. The residual stream is carried unnormalised, so
                      a value survives many layers (a 5-hop chain is learned with 5 layers, also with 18).
  global_attention    every node attends to every node of its own graph by content: one hop to anywhere. Graphs of
                      a batch never see each other. Above `max_keys` nodes a graph's keys are an evenly strided
                      subset (queries are always all nodes), so the cost stays O(nodes x max_keys).
  mix_hop             several hop distances in parallel in one layer: [x, Ax, A²x, …] each with its own weights.
  graph_memory        the k most salient node states of each graph ([graphs, k, d]) for a decoder to attend into.
  k_hop(g, k)         a graph with shortcut edges to everything within k hops (edge.hop = distance), so one layer
                      reads k hops at once.
  hop_distances(g, s) breadth-first hop counts from node s; eccentricity(g) the longest shortest path.
  graph_transformer_net  a complete model: input projection, local blocks with a global block every
                      `global_every` layers, final norm, node- or graph-level output.

All layers are differentiable with Apsi's autodiff and train with fit() like any other layer.
"""
import collections
import math

import numpy as np

from . import ops
from .errors import RuntimeErr, ShapeError
from .layers import (Layer, Linear, Activation, Sequential, LayerNorm, GraphPool, Graph, _glorot, _rng, _seed_of,
                     _graph, segment_sum, segment_softmax, segment_mean)
from .values import Record


# ============================================================== graph structure
def _adjacency_lists(g, undirected=False):
    out = [[] for _ in range(g.n)]
    for s, d in zip(g.src.tolist(), g.dst.tolist()):
        if s != d:
            out[s].append(d)
            if undirected:
                out[d].append(s)
    return out


def hop_distances(g, source, undirected=True):
    """Number of hops from `source` to every node (-1 when unreachable), by breadth-first search."""
    adj = _adjacency_lists(g, undirected)
    dist = np.full(g.n, -1, dtype=np.int64)
    s = int(source)
    if not 0 <= s < g.n:
        raise ShapeError(f"hop_distances(): node {s} is outside 0..{g.n - 1}")
    dist[s] = 0
    q = collections.deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def eccentricity(g, undirected=True, samples=64, seed=0):
    """The longest shortest path (in hops) seen from up to `samples` start nodes: how far information must travel."""
    if g.n == 0:
        return 0
    rng = np.random.default_rng(seed)
    starts = range(g.n) if g.n <= samples else rng.choice(g.n, samples, replace=False)
    return int(max(hop_distances(g, s, undirected).max() for s in starts))


def k_hop(g, k, undirected=False):
    """A graph with an edge u -> v for every v within 1..k hops of u (edge.hop = the distance), plus self loops.
    One layer over this graph reads k hops at once; graph_transformer learns a separate bias per distance."""
    k = int(k)
    if k < 1:
        raise RuntimeErr("k_hop(): k must be at least 1")
    adj = _adjacency_lists(g, undirected)
    src, dst, hop = [], [], []
    for s in range(g.n):
        seen = {s: 0}
        frontier = [s]
        for depth in range(1, k + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = depth
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        for v, dpt in seen.items():
            if v != s:
                src.append(s)
                dst.append(v)
                hop.append(dpt)
    src += list(range(g.n))
    dst += list(range(g.n))
    hop += [0] * g.n
    out = Graph(np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), g.n, self_loops=False,
                hop=np.array(hop, dtype=np.int64))
    out.batch, out.graphs = g.batch, g.graphs
    return out


# ============================================================== layers
class GraphTransformer(Layer):
    """Pre-norm residual graph block: x + attention over neighbours (by relation and hop), then x + MLP."""
    kind = "graph_transformer"

    def __init__(self, d, heads=4, relations=1, ff=None, max_hop=16, activation="gelu"):
        self.d, self.h, self.R = int(d), int(heads), max(1, int(relations))
        if self.d % self.h:
            raise ShapeError(f"graph_transformer: d={self.d} must be divisible by heads={self.h}")
        self.ff, self.max_hop = int(ff or 4 * self.d), int(max_hop)
        self.norm1, self.norm2 = LayerNorm(self.d), LayerNorm(self.d)
        self.mlp = Sequential([Linear(self.d, self.ff), Activation(activation), Linear(self.ff, self.d)])

    def init(self, seed=0):
        rng = _rng(seed)
        d, R = self.d, self.R
        return Record(n1=self.norm1.init(), n2=self.norm2.init(), Wq=_glorot(rng, d, d),
                      Wk=np.stack([_glorot(rng, d, d) for _ in range(R)]),
                      Wv=np.stack([_glorot(rng, d, d) for _ in range(R)]),
                      Wo=_glorot(rng, d, d) * 0.5, bo=np.zeros(d),
                      rel_bias=np.zeros((R, self.h)), hop_bias=np.zeros((self.max_hop + 1, self.h)),
                      mlp=self.mlp.init(_seed_of(seed, 1)))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        if g.relations > self.R:
            raise RuntimeErr(f"graph_transformer(relations={self.R}) got a graph with {g.relations} relations")
        N, H, hd = g.n, self.h, self.d // self.h
        keep = g.src != g.dst                       # neighbours only: the residual already carries the node itself
        src, dst, rel = g.src[keep], g.dst[keep], g.rel[keep]
        hop = np.minimum(g.hop[keep], self.max_hop)
        E = len(src)
        h = self.norm1(p.n1, x)
        q = ops.reshape(h @ p.Wq, N, H, hd)
        if E:
            hs = h[src]                                                        # [E, d]
            k_parts, v_parts, order = [], [], []
            for r in range(self.R):                                            # one projection per relation
                idx = np.nonzero(rel == r)[0]
                if len(idx):
                    k_parts.append(hs[idx] @ p.Wk[r])
                    v_parts.append(hs[idx] @ p.Wv[r])
                    order.append(idx)
            order = np.concatenate(order)
            inv = np.empty(E, dtype=np.int64)
            inv[order] = np.arange(E)
            ke = ops.reshape(ops.concat(k_parts, 0)[inv], E, H, hd)
            ve = ops.reshape(ops.concat(v_parts, 0)[inv], E, H, hd)
            score = ops.sum_(q[dst] * ke, axis=-1) / math.sqrt(hd) + p.rel_bias[rel] + p.hop_bias[hop]   # [E, H]
            alpha = segment_softmax(score, dst, N)
            msg = segment_sum(ve * ops.reshape(alpha, E, H, 1), dst, N)        # [N, H, hd]
            x = x + ops.reshape(msg, N, self.d) @ p.Wo + p.bo
        return x + self.mlp(p.mlp, self.norm2(p.n2, x), train=train, **ctx)

    def children(self):
        return [self.mlp]

    def describe(self):
        rel = f", {self.R} relations" if self.R > 1 else ""
        return f"graph_transformer(d={self.d}, heads={self.h}{rel})"


class GlobalAttention(Layer):
    """Pre-norm residual self-attention over all nodes of the same graph: any node reaches any other in one step."""
    kind = "global_attention"

    def __init__(self, d, heads=4, max_keys=4096):
        self.d, self.h, self.max_keys = int(d), int(heads), int(max_keys)
        if self.d % self.h:
            raise ShapeError(f"global_attention: d={self.d} must be divisible by heads={self.h}")
        self.norm = LayerNorm(self.d)

    def init(self, seed=0):
        rng = _rng(seed)
        d = self.d
        return Record(n=self.norm.init(), Wq=_glorot(rng, d, d), Wk=_glorot(rng, d, d), Wv=_glorot(rng, d, d),
                      Wo=_glorot(rng, d, d) * 0.5, bo=np.zeros(d))

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        N = int(np.shape(ops.raw(x))[0])
        batch = g.batch if g is not None else np.zeros(N, dtype=np.int64)
        H, hd = self.h, self.d // self.h
        h = self.norm(p.n, x)
        outs, rows = [], []
        for b in np.unique(batch):
            idx = np.nonzero(batch == b)[0]
            keys = idx if len(idx) <= self.max_keys else idx[np.linspace(0, len(idx) - 1, self.max_keys).astype(int)]
            hb, hk = h[idx], h[keys]
            n, m = len(idx), len(keys)
            q = ops.transpose(ops.reshape(hb @ p.Wq, n, H, hd), (1, 0, 2))           # [H, n, hd]
            k = ops.transpose(ops.reshape(hk @ p.Wk, m, H, hd), (1, 2, 0))           # [H, hd, m]
            v = ops.transpose(ops.reshape(hk @ p.Wv, m, H, hd), (1, 0, 2))           # [H, m, hd]
            a = ops.softmax((q @ k) / math.sqrt(hd), axis=-1)
            o = ops.reshape(ops.transpose(a @ v, (1, 0, 2)), n, self.d)
            outs.append(o)
            rows.append(idx)
        out = segment_sum(ops.concat(outs, 0), np.concatenate(rows), N) if len(outs) > 1 else \
            segment_sum(outs[0], rows[0], N)
        return x + out @ p.Wo + p.bo

    def describe(self):
        return f"global_attention(d={self.d}, heads={self.h}, max_keys={self.max_keys})"


class MixHop(Layer):
    """Several hop distances in parallel: concat_k (Â^k x) W_k with Â the mean over incoming neighbours."""
    kind = "mix_hop"

    def __init__(self, n_in, n_out, hops=(0, 1, 2)):
        self.n_in, self.n_out, self.hops = int(n_in), int(n_out), tuple(int(k) for k in hops)
        if not self.hops or min(self.hops) < 0:
            raise RuntimeErr("mix_hop(): hops must be a list of distances >= 0")

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(W=np.stack([_glorot(rng, self.n_in, self.n_out) for _ in self.hops]),
                      b=np.zeros(self.n_out * len(self.hops)))

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        keep = g.src != g.dst
        src, dst = g.src[keep], g.dst[keep]
        powers, cur = {0: x}, x
        for k in range(1, max(self.hops) + 1):
            cur = segment_mean(cur[src], dst, g.n) if len(src) else cur * 0.0
            powers[k] = cur
        return ops.concat([powers[k] @ p.W[i] for i, k in enumerate(self.hops)], 1) + p.b

    def describe(self):
        return f"mix_hop({self.n_in} → {self.n_out} × hops {list(self.hops)})"


class GraphMemory(Layer):
    """The k most salient node states of every graph, weighted by a learned score: [graphs, k, d]. A decoder or any
    other model reads a whole graph through these k vectors with cross-attention (multi_head_attention([q, mem]))
    instead of one pooled vector, so it can still tell WHICH node said what. The choice of the top k is not
    differentiable; the weights (sigmoid of the score) are, so the scorer learns what to keep."""
    kind = "graph_memory"

    def __init__(self, d, k=16):
        self.d, self.k = int(d), int(k)

    def init(self, seed=0):
        return Record(w=_glorot(_rng(seed), self.d, 1), b=np.zeros(1))

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        N = int(np.shape(ops.raw(x))[0])
        batch = g.batch if g is not None else np.zeros(N, dtype=np.int64)
        graphs = int(batch.max()) + 1 if N else 0
        score = ops.reshape(x @ p.w + p.b, N)
        raw = np.asarray(ops.raw(score))
        rows, valid = [], np.zeros((graphs, self.k))
        for b in range(graphs):
            idx = np.nonzero(batch == b)[0]
            top = idx[np.argsort(-raw[idx], kind="stable")[:self.k]]
            valid[b, :len(top)] = 1.0
            rows.append(np.concatenate([top, np.full(self.k - len(top), top[0] if len(top) else 0)]))
        sel = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        weight = ops.reshape(1.0 / (1.0 + ops.exp(-score[sel])), graphs, self.k, 1) * valid[:, :, None]
        return ops.reshape(x[sel], graphs, self.k, self.d) * weight

    def describe(self):
        return f"graph_memory(d={self.d}, k={self.k})"


def graph_transformer_net(n_in, d, out, layers=4, heads=4, relations=1, global_every=2, max_keys=4096,
                          task="node", pool="mean"):
    """Input projection, `layers` graph_transformer blocks with a global_attention block after every
    `global_every`-th one (0: none), a final layer norm, then a linear output per node (task="node") or per graph
    (task="graph", pooled with `pool`)."""
    L = [Linear(n_in, d)]
    for i in range(int(layers)):
        L.append(GraphTransformer(d, heads, relations))
        if global_every and (i + 1) % int(global_every) == 0:
            L.append(GlobalAttention(d, heads, max_keys))
    L.append(LayerNorm(d))
    if task == "graph":
        L.append(GraphPool(pool))
    elif task != "node":
        raise RuntimeErr('graph_transformer_net(): task must be "node" or "graph"')
    L.append(Linear(d, out))
    return Sequential(L)


def names():
    return dict(graph_transformer=GraphTransformer, global_attention=GlobalAttention, mix_hop=MixHop,
                graph_memory=GraphMemory,
                graph_transformer_net=graph_transformer_net, k_hop=k_hop, hop_distances=hop_distances,
                eccentricity=eccentricity)
