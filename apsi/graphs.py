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
from .values import Record, tree_map


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
    """Pre-norm residual self-attention over all nodes of the same graph: any node reaches any other in one step.
    Graphs of a batch never see each other. Above `max_keys` nodes a graph's keys are a subset: evenly strided, or with
    learned_keys the max_keys nodes of highest learned score (landmark keys: every node's value is also scaled by
    2 sigmoid(score), 1 at the start, so the scorer learns which nodes are worth finding). ratio: the attention runs
    with round(heads * ratio) heads of the same size (narrower than the model); bias: biases on q / k / v; zero_init:
    the output projection starts at zero (the layer starts as a no-op); types: only the nodes of these types attend and
    are attended (the others pass through)."""
    kind = "global_attention"

    def __init__(self, d, heads=4, max_keys=4096, ratio=1.0, learned_keys=False, bias=False, zero_init=False,
                 types=None, dropout=0.0, region=None):
        self.d, self.h, self.max_keys = int(d), int(heads), int(max_keys)
        self.p_drop = float(dropout)
        self.region = None if region is None else int(region)    # only the nodes of this region (graph.region)
        if self.d % self.h:
            raise ShapeError(f"global_attention: d={self.d} must be divisible by heads={self.h}")
        self.hd = self.d // self.h
        self.ratio = float(ratio)
        self.H = max(1, int(round(self.h * self.ratio)))
        self.inner = self.H * self.hd
        self.learned, self.bias, self.zero = bool(learned_keys), bool(bias), bool(zero_init)
        self.T, self.type_ids = _owned(types) if types is not None else (None, None)
        self.norm = LayerNorm(self.d)

    def init(self, seed=0):
        rng = _rng(seed)
        d, i = self.d, self.inner
        f = dict(n=self.norm.init(), Wq=_glorot(rng, d, i), Wk=_glorot(rng, d, i), Wv=_glorot(rng, d, i),
                 Wo=np.zeros((i, d)) if self.zero else _glorot(rng, i, d) * 0.5, bo=np.zeros(d))
        if self.bias:
            f.update(bq=np.zeros(i), bk=np.zeros(i), bv=np.zeros(i))
        if self.learned:
            f.update(ks_w=rng.normal(size=(d, 1)) * 0.02, ks_b=np.zeros(1))
        return Record(**f)

    def own(self, g, N):
        m = np.ones((N, 1), bool) if self.T is None else _owned_mask(g, N, self.T, self.type_ids)
        return m & region_mask(g, N, self.region) if self.region is not None else m

    def masked(self):
        return self.T is not None or self.region is not None

    def keys_of(self, idx, score):
        """The key rows of one graph (its query rows idx): all of them, or max_keys of them."""
        if len(idx) <= self.max_keys or self.max_keys <= 0:
            return idx
        if self.learned:
            return idx[np.sort(np.argsort(-score[idx], kind="stable")[:self.max_keys])]
        return idx[np.linspace(0, len(idx) - 1, self.max_keys).astype(int)]

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        N = int(np.shape(ops.raw(x))[0])
        batch = np.asarray(g.batch, np.int64) if g is not None else np.zeros(N, dtype=np.int64)
        own = self.own(g, N)
        H, hd, inner = self.H, self.hd, self.inner
        h = self.norm(p.n, x)
        q, k, v = h @ p.Wq, h @ p.Wk, h @ p.Wv
        if self.bias:
            q, k, v = q + p.bq, k + p.bk, v + p.bv
        score = None
        if self.learned:
            s = h @ p.ks_w + p.ks_b                                              # [N, 1]
            v = v * (2.0 * ops.sigmoid(s))
            score = np.asarray(ops.raw(s)).reshape(-1)
        outs, rows = [], []
        for b in np.unique(batch[own.reshape(-1)]):
            idx = np.nonzero((batch == b) & own.reshape(-1))[0]
            keys = self.keys_of(idx, score)
            n, m = len(idx), len(keys)
            qb = ops.transpose(ops.reshape(q[idx], n, H, hd), (1, 0, 2))          # [H, n, hd]
            kb = ops.transpose(ops.reshape(k[keys], m, H, hd), (1, 2, 0))         # [H, hd, m]
            vb = ops.transpose(ops.reshape(v[keys], m, H, hd), (1, 0, 2))         # [H, m, hd]
            a = ops.softmax((qb @ kb) / math.sqrt(hd), axis=-1)
            outs.append(ops.reshape(ops.transpose(a @ vb, (1, 0, 2)), n, inner))
            rows.append(idx)
        if not outs:
            return x
        out = segment_sum(ops.concat(outs, 0) if len(outs) > 1 else outs[0], np.concatenate(rows), N)
        delta = _drop(out @ p.Wo + p.bo, self.p_drop, train)
        return x + (ops.where(own, delta, 0.0) if self.masked() else delta)

    def describe(self):
        extra = (f", ratio={self.ratio}" if self.H != self.h else "") + (", learned keys" if self.learned else "")
        return f"global_attention(d={self.d}, heads={self.h}, max_keys={self.max_keys}{extra})"


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
        lead = getattr(g, "leading", None) if g is not None else None
        graphs = min(graphs, int(lead)) if lead is not None else graphs      # graph(leading=n): the first n only
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


class NodeTextEncoder(Layer):
    """Node features from each node's text (a token-efficient text encoder): the masked means of a token-id and a
    hash-id embedding of its tokens, projected; plus a learned attention pooling of the tokens with their positions
    (gated, starts small); plus embeddings of the language, the language family, the node type and five structural
    codes, each with a learned mix. Input [N, 2 L + 9] integers per node: L token ids, L hash ids, the number of tokens
    (1..L), language, family, node type, then the 5 structural codes (depth, sibling, in-degree, out-degree, line)."""
    kind = "node_text_encoder"
    takes_ids = True                                     # (its input is integers: on the GPU trainer they travel as bits)

    def __init__(self, d, vocab=8192, hashes=4096, types=15, langs=1094, families=9, max_pos=128,
                 struct=(32, 32, 16, 16, 32)):
        self.d, self.V, self.Hh, self.T, self.Lg, self.F = int(d), int(vocab), int(hashes), int(types), int(langs), \
            int(families)
        self.P, self.struct = int(max_pos), tuple(int(v) for v in struct)

    def init(self, seed=0):
        rng = _rng(seed)
        d = self.d
        xav = lambda n: rng.uniform(-1, 1, (n, d)) * math.sqrt(6.0 / (n + d))      # noqa: E731
        return Record(token_emb=xav(self.V), hash_emb=xav(self.Hh), type_emb=rng.normal(size=(self.T, d)),
                      lang_emb=xav(self.Lg), family_emb=xav(self.F),
                      proj=Record(W=rng.uniform(-1, 1, (2 * d, d)) / math.sqrt(2 * d), b=np.zeros(d)),
                      pos_emb=rng.normal(size=(self.P, d)) * 0.02, pool_query=rng.normal(size=d) * 0.02,
                      pool_out=Record(W=rng.uniform(-1, 1, (d, d)) * math.sqrt(6.0 / (2 * d)), b=np.zeros(d)),
                      focus_gate=np.array(0.01), lang_mix=np.array(0.12), family_mix=np.array(0.08),
                      type_mix=np.array(0.1), struct_emb=[rng.normal(size=(n, d)) * 0.02 for n in self.struct],
                      struct_mix=np.array(0.5))

    def columns(self, x):
        a = np.asarray(ops.raw(x)).astype(np.int64)
        L = (a.shape[1] - 9) // 2
        return a, L

    def apply(self, p, x, train=False, **ctx):
        a, L = self.columns(x)
        N, d = a.shape[0], self.d
        tid, hid, ln = a[:, :L], a[:, L:2 * L], np.clip(a[:, 2 * L], 1, L)
        lang, fam, typ, st = a[:, 2 * L + 1], a[:, 2 * L + 2], a[:, 2 * L + 3], a[:, 2 * L + 4:2 * L + 9]
        mask = np.arange(L)[None, :] < ln[:, None]                        # [N, L]
        rows, cols = np.nonzero(mask)
        inv = (1.0 / ln)[rows][:, None]
        te = segment_sum(p.token_emb[tid[rows, cols]] * inv, rows, N)
        he = segment_sum(p.hash_emb[hid[rows, cols]] * inv, rows, N)
        base = ops.concat([te, he], 1) @ p.proj.W + p.proj.b
        rep = p.token_emb[tid[rows, cols]] + p.hash_emb[hid[rows, cols]] + p.pos_emb[np.minimum(cols, self.P - 1)]
        sc = ops.sum_(rep * p.pool_query, axis=-1) / math.sqrt(d)          # [E]
        w = segment_softmax(ops.reshape(sc, -1, 1), rows, N)                # [E, 1]
        focus = segment_sum(rep * w, rows, N) @ p.pool_out.W + p.pool_out.b
        node = (typ >= 0)[:, None]                         # a type < 0: a plain text (no type, no structure)
        out = (base + p.lang_emb[lang % self.Lg] * p.lang_mix + p.family_emb[fam % self.F] * p.family_mix
               + focus * p.focus_gate + ops.where(node, p.type_emb[np.maximum(typ, 0)] * p.type_mix, 0.0))
        se = p.struct_emb[0][st[:, 0]]
        for k in range(1, len(self.struct)):
            se = se + p.struct_emb[k][st[:, k]]
        return out + ops.where(node, se * p.struct_mix, 0.0)

    def describe(self):
        return f"node_text_encoder(d={self.d})"


def _ln(x, w, b, eps=1e-5):
    m = ops.mean(x, axis=-1, keepdims=True)
    v = ops.mean((x - m) * (x - m), axis=-1, keepdims=True)
    return (x - m) / ops.sqrt(v + eps) * w + b


class BrainRouting(Layer):
    """The routing of three graph encoders' summaries through ONE mixture of experts and ONE fusion bridge, in the
    order the original runs them: C (task 2), then A (task 0, fused with C's routed vector), then B (task 1, fused
    with A's). Router: layer norm, gate + task bias (+ noise while training), the top_k experts (chosen with a
    balancing bias that follows the experts' load), each a pre-norm two-layer GELU block blended with a running mean of
    its outputs; the chosen experts' outputs mixed by a one-layer attention between them (gated, starts nearly closed),
    then weighted by their router probabilities; an auxiliary loss (load balance, z-loss, importance spread). Bridge:
    fused = g a + (1 - g) out([a, b]) through a layer norm, g = sigmoid(gate([a, b])), a = proj_a(other brain, no
    gradient), b = proj_b(routed); plus the anti-forgetting penalty of the other brain's centre.
    Input [sem_c, sem_a, sem_b] ([B, d] each). Output [B, 3 d + 3]: routed C, fused A, fused B, the penalties for A and
    B, the auxiliary loss (the mean of the three calls). Running statistics are state of the layer (state_get /
    state_set; saved with the model)."""
    kind = "brain_routing"
    stateful = True

    def __init__(self, d, experts=8, top_k=2, tasks=3, noise=0.06, comm_heads=8):
        self.d, self.E, self.k, self.tasks, self.noise = int(d), int(experts), int(top_k), int(tasks), float(noise)
        h = max(1, int(comm_heads))
        while h > 1 and self.d % h:
            h -= 1
        self.ch = h
        self.hidden = max(2 * self.d, self.d + self.d // 4)
        self.reset_state()

    def reset_state(self):
        E, d = self.E, self.d
        self.st = dict(load_ema=np.full(E, 1.0 / E), importance_ema=np.full(E, 1.0 / E), route_bias=np.zeros(E),
                       ema_out=np.zeros((E, d)), fisher_diag=np.zeros(d), anchor_a=np.zeros(d), reward_ema=0.5)

    def state_get(self):
        return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in self.st.items()}

    def state_set(self, st):
        self.st = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in st.items()}

    def init(self, seed=0):
        rng = _rng(seed)
        d, E, hd = self.d, self.E, self.hidden
        u = lambda i, o: rng.uniform(-1, 1, (i, o)) / math.sqrt(i)        # noqa: E731
        ln = lambda n: Record(w=np.ones(n), b=np.zeros(n))                # noqa: E731
        experts = [Record(input_norm=ln(d), l1=Record(W=u(d, hd), b=np.zeros(hd)), n2=ln(hd),
                          l2=Record(W=u(hd, d), b=np.zeros(d))) for _ in range(E)]
        router = Record(task_bias=np.zeros((self.tasks, E)), pre_norm=ln(d), gate=Record(W=u(d, E), b=np.zeros(E)),
                        experts=experts,
                        comm=Record(gate=np.array(-3.0), norm1=ln(d), in_proj=Record(W=u(d, 3 * d), b=np.zeros(3 * d)),
                                    out_proj=Record(W=u(d, d), b=np.zeros(d)), norm2=ln(d),
                                    f1=Record(W=u(d, 2 * d), b=np.zeros(2 * d)), f2=Record(W=u(2 * d, d), b=np.zeros(d))))
        bridge = Record(proj_a=Record(W=u(d, d), b=np.zeros(d)), proj_b=Record(W=u(d, d), b=np.zeros(d)),
                        gate=Record(W=u(2 * d, d), b=np.zeros(d)), out=Record(W=u(2 * d, d), b=np.zeros(d)),
                        out_norm=ln(d))
        return Record(router=router, bridge=bridge)

    def _expert(self, e, x, train):
        q = _ln(x, e.input_norm.w, e.input_norm.b)
        h = ops.gelu_exact(q @ e.l1.W + e.l1.b)
        return _ln(h, e.n2.w, e.n2.b) @ e.l2.W + e.l2.b

    def _comm(self, c, seq):
        """seq [B, k, d]: one attention layer and a feed-forward block across the chosen experts (gated)."""
        B, k, d = np.shape(ops.raw(seq))
        if k < 2:
            return seq
        g = ops.sigmoid(c.gate)
        h = _ln(seq, c.norm1.w, c.norm1.b)
        qkv = h @ c.in_proj.W + c.in_proj.b                                   # [B, k, 3d]
        H, hd = self.ch, d // self.ch
        sp = lambda t: ops.transpose(ops.reshape(t, B, k, H, hd), (0, 2, 1, 3))    # noqa: E731
        q, kk, v = sp(qkv[:, :, :d]), sp(qkv[:, :, d:2 * d]), sp(qkv[:, :, 2 * d:])
        a = ops.softmax((q @ ops.transpose(kk, (0, 1, 3, 2))) / math.sqrt(hd), axis=-1)
        att = ops.reshape(ops.transpose(a @ v, (0, 2, 1, 3)), B, k, d) @ c.out_proj.W + c.out_proj.b
        seq = seq + g * att
        h2 = _ln(seq, c.norm2.w, c.norm2.b)
        return seq + g * (ops.gelu_exact(h2 @ c.f1.W + c.f1.b) @ c.f2.W + c.f2.b)

    def _route(self, r, x, task, train, sample=None):
        B, E, k = int(np.shape(ops.raw(x))[0]), self.E, self.k
        st = self.st
        logits = _ln(x, r.pre_norm.w, r.pre_norm.b) @ r.gate.W + r.gate.b + r.task_bias[task]
        if train and self.noise > 0:
            logits = logits + ops._RNG[0].normal(size=(B, E)) * self.noise
        probs = ops.softmax(logits, axis=-1)
        pd = np.asarray(ops.raw(probs), np.float64)
        pick = pd + st["route_bias"][None, :] if train else pd
        sel = np.argsort(-pick, axis=1, kind="stable")[:, :k]                # [B, k]
        if sample is not None and not train:            # exploring: Gumbel top-k (Plackett-Luce), the choices kept
            s0, recs = sample
            if s0 < B:
                u = np.clip(ops._RNG[0].random(size=(B - s0, E)), 1e-9, 1.0 - 1e-9)
                lg = np.asarray(ops.raw(logits), np.float64)[s0:]
                sel = sel.copy()
                sel[s0:] = np.argsort(-(lg - np.log(-np.log(u))), axis=1, kind="stable")[:, :k]
                xr = np.asarray(ops.raw(x), np.float64)
                for i in range(s0, B):
                    if sum(1 for r_ in recs if r_[3] == i) < 16:
                        recs.append([xr[i:i + 1].copy(), int(task), [int(v) for v in sel[i]], i])
        rows = np.arange(B)[:, None]
        sv = probs[rows, sel]
        w = sv / ops.where(np.asarray(ops.raw(ops.sum_(sv, axis=1, keepdims=True))) > 1e-9,
                           ops.sum_(sv, axis=1, keepdims=True), 1e-9)
        hard = np.zeros((B, E))
        hard[rows, sel] = 1.0 / k
        hard = hard.mean(0)
        aux = None
        if train:
            dB = 0.99 ** B
            imp_prev = st["importance_ema"].copy()
            st["load_ema"] = st["load_ema"] * dB + hard * (1 - dB)
            st["importance_ema"] = st["importance_ema"] * dB + pd.mean(0) * (1 - dB)
            st["route_bias"] = np.clip(st["route_bias"] + 1e-3 * min(B, 4) * np.sign(1.0 / E - st["load_ema"]),
                                       -0.05, 0.05)
            f = st["load_ema"].copy()
            lb = ops.mean(E * ops.sum_(probs * f, axis=-1))
            mx = np.max(np.asarray(ops.raw(logits)), axis=-1, keepdims=True)
            lse = ops.log(ops.sum_(ops.exp(logits - mx), axis=-1)) + mx[:, 0]
            z = ops.mean(lse * lse)
            imp = 0.01 * probs + 0.99 * imp_prev
            mu = ops.mean(imp, axis=-1, keepdims=True)
            var = ops.mean((imp - mu) * (imp - mu), axis=-1)
            cv2 = ops.mean(var / (ops.reshape(mu, B) * ops.reshape(mu, B) + 1e-9))
            aux = 0.01 * lb + 1e-3 * z + 0.01 * cv2
        outs = [None] * (B * k)
        flat = sel.reshape(-1)
        parts = []
        for e in range(E):
            slots = np.nonzero(flat == e)[0]
            if not len(slots):
                continue
            o = self._expert(r.experts[e], x[slots // k], train)
            if train:
                st["ema_out"][e] = 0.95 * st["ema_out"][e] + 0.05 * np.asarray(ops.raw(o), np.float64).mean(0)
            ema = st["ema_out"][e]
            if np.abs(ema).sum() > 0:
                o = 0.88 * o + 0.12 * ema
            parts.append((slots, o))
        order = np.concatenate([sl for sl, _ in parts])
        allo = ops.concat([o for _, o in parts], 0) if len(parts) > 1 else parts[0][1]
        inv = np.empty(B * k, np.int64)
        inv[order] = np.arange(B * k)
        eo = ops.reshape(allo[inv], B, k, self.d)
        eo = self._comm(r.comm, eo)
        out = ops.sum_(ops.reshape(w, B, k, 1) * eo, axis=1)
        return out, aux

    def _bridge(self, b, emb_other, routed):
        a = np.asarray(ops.raw(emb_other), np.float64) @ b.proj_a.W + b.proj_a.b      # (no gradient to the other brain)
        bb = routed @ b.proj_b.W + b.proj_b.b
        cat = ops.concat([a, bb], 1)
        g = ops.sigmoid(cat @ b.gate.W + b.gate.b)
        fused = g * a + (1 - g) * (cat @ b.out.W + b.out.b)
        return _ln(fused, b.out_norm.w, b.out_norm.b)

    def _ewc(self, emb):
        st = self.st
        centre = ops.mean(emb, axis=0)
        diff = centre - st["anchor_a"]
        return 0.12 * ops.sum_(st["fisher_diag"] * diff * diff)

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        mode = getattr(g, "route_mode", "joint") if g is not None else "joint"
        if mode in ("lang", "lang_eval"):              # lang_eval: the same, as evaluation (no statistics)
            return self._apply_lang(p, x, train and mode == "lang", g)
        if mode == "none":
            train = False                                # (a batch that must not touch the running statistics)
        sc, sa, sb = x
        B = int(np.shape(ops.raw(sa))[0])
        smp = getattr(g, "route_sample", None) if g is not None else None   # a list: explore, the choices kept
        if smp is not None:                             # rows from graph.route_sample_from on (the first: as usual)
            smp = (int(getattr(g, "route_sample_from", 0) or 0), smp)
        rc, ac = self._route(p.router, sc, 2, train, smp)
        ra, aa = self._route(p.router, sa, 0, train, smp)
        fa = self._bridge(p.bridge, rc, ra)
        ea = self._ewc(rc)
        rb, ab = self._route(p.router, sb, 1, train, smp)
        fb = self._bridge(p.bridge, fa, rb)
        eb = self._ewc(fa)
        if train:                                       # the bridge's reward: a running mean of cos(a, fused)
            for o_, f_ in ((rc, fa), (fa, fb)):
                A_ = np.asarray(ops.raw(o_), np.float64)
                F_ = np.asarray(ops.raw(f_), np.float64)
                cs = (A_ * F_).sum(1) / np.maximum(np.linalg.norm(A_, axis=1) * np.linalg.norm(F_, axis=1), 1e-8)
                for c_ in cs:
                    self.st["reward_ema"] = 0.9 * self.st["reward_ema"] + 0.1 * float((c_ + 1.0) / 2.0)
        aux = (ac + aa + ab) / 3.0 if train else 0.0
        one = np.ones((B, 1))
        return ops.concat([rc, fa, fb, one * ea, one * eb, one * aux], 1)

    def _apply_lang(self, p, x, train, g):
        """A text micro-batch: the third encoder's summaries through the router (task 2), one call per group of rows
        (graph.route_split: e.g. the texts, then the responses), the auxiliary loss their mean; the first encoder's view
        of the texts (its first rows) through the router (task 0) and the bridge with the texts' routed summaries, as
        evaluation (no gradient, no running statistics, no auxiliary loss). Rows beyond the first encoder's are zero in
        its columns."""
        sc, sa, _sb = x
        R = int(np.shape(ops.raw(sc))[0])
        split = [int(v) for v in (getattr(g, "route_split", None) or [R])]
        outs, auxs, a0 = [], [], 0
        for n_ in split:
            if n_ <= 0:
                continue
            r_, a_ = self._route(p.router, sc[a0:a0 + n_], 2, train)
            outs.append(r_)
            if a_ is not None:
                auxs.append(a_)
            a0 += n_
        rc = ops.concat(outs, 0) if len(outs) > 1 else outs[0]
        Ba = int(np.shape(ops.raw(sa))[0])
        cols = []
        if Ba > 0:
            sa_c = np.asarray(ops.raw(sa), np.float64)
            pr = Record(**{k: (tree_map(lambda v: np.asarray(ops.raw(v), np.float64), p.fields[k]))
                           for k in p.fields})
            ra, _ = self._route(pr.router, sa_c, 0, False)
            rc0 = np.asarray(ops.raw(rc), np.float64)[:Ba]
            fa = np.asarray(ops.raw(self._bridge(pr.bridge, rc0, ra)), np.float64)
            fa = np.concatenate([fa, np.zeros((R - Ba, self.d))], 0) if R > Ba else fa[:R]
        else:
            fa = np.zeros((R, self.d))
        aux = sum(auxs[1:], auxs[0]) / len(auxs) if (train and auxs) else 0.0
        one = np.ones((R, 1))
        return ops.concat([rc, fa, np.zeros((R, self.d)), np.zeros((R, 1)), np.zeros((R, 1)), one * aux], 1)

    def rl_policy_grad(self, p, items, entropy=0.01):
        """REINFORCE on the router's choice (Plackett-Luce): items = [[records of one sample, advantage], ...], a
        record [input [1, d], task, chosen experts, row] as route_sample keeps them. -> (loss, gradient) for the router's
        pre_norm, gate and task_bias (a tree like p.router with zeros elsewhere)."""
        from .stdlib import value_and_grad
        r0 = p.router

        def loss(pre_w, pre_b, gw, gb, tb):
            terms = []
            for recs, adv in items:
                adv = float(adv)
                if not recs or abs(adv) < 1e-8:
                    continue
                lp_sum, ent_sum = 0.0, 0.0
                for rec in recs:
                    xr, tid, sel = rec[0], rec[1], rec[2]
                    lg = ops.reshape(_ln(np.asarray(xr, np.float64), pre_w, pre_b) @ gw + gb + tb[int(tid)], -1)
                    rem = np.ones(self.E, bool)
                    lp = 0.0
                    for j in [int(v) for v in sel]:
                        m = np.where(rem, 0.0, -1e30)
                        z = lg + m
                        mx = float(np.max(np.asarray(ops.raw(z))))
                        lse = ops.log(ops.sum_(ops.exp(z - mx))) + mx
                        lp = lp + lg[j] - lse
                        rem = rem.copy()
                        rem[j] = False
                    pr = ops.softmax(lg, axis=-1)
                    ent = -ops.sum_(pr * ops.log_softmax(lg, axis=-1))
                    lp_sum = lp_sum + lp
                    ent_sum = ent_sum + ent
                n = float(len(recs))
                terms.append(-(adv * lp_sum / n) - entropy * ent_sum / n)
            if not terms:
                return 0.0 * ops.sum_(gb)
            return sum(terms[1:], terms[0]) / len(terms)
        f = lambda *a: loss(*a)                                              # noqa: E731
        lv, g = value_and_grad(f, wrt=(0, 1, 2, 3, 4))(r0.pre_norm.w, r0.pre_norm.b, r0.gate.W, r0.gate.b,
                                                         r0.task_bias)
        return float(ops.raw(lv)), Record(pre_norm=Record(w=g[0], b=g[1]), gate=Record(W=g[2], b=g[3]), task_bias=g[4])

    def describe(self):
        return f"brain_routing(d={self.d}, {self.E} experts, top {self.k})"


class MaskRows(Layer):
    """Its input with the rows graph.picks[name] replaced by one learned vector (masked-feature training: the
    model has to tell what was there from the context)."""
    kind = "mask_rows"

    def __init__(self, d, name="mask"):
        self.d, self.name = int(d), str(name)

    def init(self, seed=0):
        return Record(mask_emb=_rng(seed).normal(size=self.d) * 0.02)

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        picks = getattr(g, "picks", None) or {}
        idx = np.asarray(picks.get(self.name, []), np.int64).reshape(-1)
        if not len(idx):
            return x
        N = int(np.shape(ops.raw(x))[0])
        m = np.zeros((N, 1), bool)
        m[idx] = True
        return ops.where(m, p.mask_emb * np.ones((N, 1)), x)

    def describe(self):
        return f"mask_rows({self.name})"


class GatedDelta(Layer):
    """x + out(gelu(inp(f))) * f[:, 0]: a correction from side features that is exactly zero for the rows whose first
    feature is 0 (a math / physics adapter: the features of the numbers and units in a text; out starts at 0).
    Input [x, f]."""
    kind = "gated_delta"

    def __init__(self, d, f, hidden=None):
        self.d, self.f = int(d), int(f)
        self.h = int(hidden or max(64, self.d // 4))

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(inp=Record(W=rng.uniform(-1, 1, (self.f, self.h)) / math.sqrt(self.f), b=np.zeros(self.h)),
                      out=Record(W=np.zeros((self.h, self.d)), b=np.zeros(self.d)))

    def apply(self, p, x, train=False, **ctx):
        a, f = x
        fa = np.asarray(ops.raw(f), np.float64)
        dlt = ops.gelu_exact(fa @ p.inp.W + p.inp.b) @ p.out.W + p.out.b
        return a + dlt * fa[:, :1]

    def describe(self):
        return f"gated_delta(d={self.d}, {self.f} features)"


class MemorySlots(Layer):
    """A decoder's memory from two sets of node states and two summaries (a writer's graph memory): of each set, the k best
    nodes of every row (by a learned score; their projection scaled by sigmoid(score)) plus the embedding of their node
    type and of the set's slot; each summary projected plus its slot; a learned null slot; an RMS norm over all.
    Input [A, C, sum_a, sum_c]: A and C node states [N, d_in] of the batch's graph, sum_a / sum_c [B, d_in]. The nodes
    each set reads and the row (0..B-1) each belongs to: graph.picks[pick_a] / [pick_a + "_row"] (and pick_c). Output
    [B, ka + kc + 3, d + 1]: the last column says which slots are filled (multi_head_attention(memory_mask=true))."""
    kind = "memory_slots"

    def __init__(self, d_in, d, ka, kc, node_types, slots=8, pick_a="mem_a", pick_c="mem_c", eps=1e-6):
        self.d_in, self.d, self.ka, self.kc = int(d_in), int(d), int(ka), int(kc)
        self.nt, self.slots, self.eps = int(node_types), int(slots), float(eps)
        self.pick_a, self.pick_c = str(pick_a), str(pick_c)

    def init(self, seed=0):
        rng = _rng(seed)
        di, d = self.d_in, self.d
        u = lambda i, o: rng.uniform(-1, 1, (i, o)) / math.sqrt(i)        # noqa: E731
        return Record(score_a=Record(W=u(di, 1), b=np.zeros(1)), score_c=Record(W=u(di, 1), b=np.zeros(1)),
                      node_proj_a=u(di, d), node_proj_c=u(di, d), sum_proj_a=u(di, d), sum_proj_c=u(di, d),
                      type_emb=rng.normal(size=(self.nt, d)) * 0.02, slot_emb=rng.normal(size=(self.slots, d)) * 0.02,
                      null=rng.normal(size=d) * 0.02, norm=np.ones(d))

    def _rows(self, g, name):
        picks = getattr(g, "picks", None) or {}
        if name not in picks or name + "_row" not in picks:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return (np.asarray(picks[name], np.int64).reshape(-1), np.asarray(picks[name + "_row"], np.int64).reshape(-1))

    def _nodes(self, x, sw, sb, proj, p, g, name, B, K, slot):
        """[B, K, d] and its mask [B, K]: the K best-scoring picked nodes of every row (best first)."""
        rows, row_of = self._rows(g, name)
        keep = row_of < B
        rows, row_of = rows[keep], row_of[keep]
        valid = np.zeros((B, K))
        if not len(rows) or K == 0:
            return None, valid
        xs = x[rows]
        s = ops.reshape(xs @ sw + sb, len(rows))
        raw = np.asarray(ops.raw(s), np.float64)
        order = np.argsort(-raw, kind="stable")
        order = order[np.argsort(row_of[order], kind="stable")]
        rr = row_of[order]
        start = np.concatenate([[0], np.cumsum(np.bincount(rr, minlength=B))])[:-1]
        rank = np.arange(len(order)) - start[rr]
        ok = rank < K
        idx, rr, rank = order[ok], rr[ok], rank[ok]
        picks = getattr(g, "picks", None) or {}
        if name + "_type" in picks:                        # the node types to embed, when not graph.ntype
            types = np.asarray(picks[name + "_type"], np.int64).reshape(-1)[keep][idx]
        else:
            types = np.asarray(g.ntype, np.int64)[rows[idx]] if hasattr(g, "ntype") else np.zeros(len(idx), np.int64)
        vals = (xs[idx] @ proj) * ops.reshape(ops.sigmoid(s[idx]), -1, 1) + p.type_emb[types] + p.slot_emb[slot]
        pos = rr * K + rank
        valid[rr, rank] = 1.0
        sel = np.zeros(B * K, np.int64)                    # rows of vals per slot (empty slots: any row, times 0)
        sel[pos] = np.arange(len(pos))
        out = ops.reshape(vals[sel], B, K, self.d) * valid[:, :, None]
        return out, valid

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        A, C, sa, sc = x
        given = getattr(g, "mem_given", None) if g is not None else None
        if given is not None:                           # a memory made earlier (e.g. while sampling): as given,
            return np.asarray(ops.raw(given), np.float64)   # no gradient back
        if getattr(g, "mem_a_const", False):            # the A nodes and summary as given, no gradient back
            A = np.asarray(ops.raw(A), np.float64)
            sa = np.asarray(ops.raw(sa), np.float64)
        B = int(np.shape(ops.raw(sa))[0])
        ma, va = self._nodes(A, p.score_a.W, p.score_a.b, p.node_proj_a, p, g, self.pick_a, B, self.ka, 0)
        mc, vc = self._nodes(C, p.score_c.W, p.score_c.b, p.node_proj_c, p, g, self.pick_c, B, self.kc, 1)
        z = lambda K: np.zeros((B, K, self.d))                                # noqa: E731
        sum_a = ops.reshape(sa @ p.sum_proj_a + p.slot_emb[2], B, 1, self.d)
        sum_c = ops.reshape(sc @ p.sum_proj_c + p.slot_emb[3], B, 1, self.d)
        null = ops.reshape(p.null + p.slot_emb[4], 1, 1, self.d) * np.ones((B, 1, 1))
        mem = ops.concat([ma if ma is not None else z(self.ka), mc if mc is not None else z(self.kc), sum_a, sum_c,
                          null], 1)
        mem = mem / ops.sqrt(ops.mean(mem * mem, axis=-1, keepdims=True) + self.eps) * p.norm
        mask = np.concatenate([va, vc, np.ones((B, 3))], 1)[:, :, None]
        return ops.concat([mem, mask], -1)

    def describe(self):
        return f"memory_slots({self.ka} + {self.kc} nodes, 2 summaries, null; d={self.d})"


class PickRows(Layer):
    """The rows of its input named by the graph of the batch: graph(...).picks[name] (a list of row ids, repeats
    allowed). For heads that read some nodes only - the masked words of a masked-word loss, the node pairs of a
    structure loss - without sending every node state to the loss."""
    kind = "pick_rows"

    def __init__(self, name):
        self.name = str(name)

    def init(self, seed=0):
        return Record()

    def rows(self, g):
        picks = getattr(g, "picks", None) or {}
        if self.name not in picks:
            raise ShapeError(f"pick_rows('{self.name}'): the batch's graph has no picks['{self.name}']")
        return np.asarray(picks[self.name], dtype=np.int64).reshape(-1)

    def apply(self, p, x, train=False, **ctx):
        idx = self.rows(ctx.get("graph"))
        return x[idx] if len(idx) else ops.reshape(x[np.zeros(1, np.int64)], 1, -1) * 0.0

    def describe(self):
        return f"pick_rows('{self.name}')"


# ============================================================== heterogeneous graphs (a type per node)
def _owned(types):
    """(count, ids): types=T owns the node types 0..T-1 (ids None); a list owns exactly those type ids, in order (the
    i-th id gets the i-th set of weights). A layer leaves the nodes of the types it does not own unchanged."""
    if hasattr(types, "__len__") or isinstance(types, np.ndarray):
        ids = tuple(int(t) for t in np.asarray(ops.raw(types)).reshape(-1))
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ShapeError(f"node types {list(ids)}: give distinct type ids >= 0")
        return len(ids), ids
    T = int(types)
    if T < 1:
        raise ShapeError("types must be >= 1 (or a list of node type ids)")
    return T, None


def local_types(g, n, T, ids=None):
    """Each node's index among the types a layer owns (0..T-1), T for a node of a type it does not own."""
    t = np.asarray(g.ntype if g is not None and hasattr(g, "ntype") else np.zeros(n, np.int64), np.int64)
    if ids is None:
        return np.where(t < T, t, T)
    lut = np.full(max(int(t.max(initial=0)), max(ids)) + 1, T, np.int64)
    lut[list(ids)] = np.arange(T)
    return lut[t]


def _type_masks(g, n, types, ids=None):
    loc = local_types(g, n, types, ids)
    return [(loc == k)[:, None] for k in range(types)]


def owned_edges(g, R, rel_ids=None):
    """(src, dst, rel) of the edges a layer reads: no self loops; with rel_ids only the edges of those relations, their
    relation renumbered 0..len(rel_ids)-1."""
    keep = g.src != g.dst
    src, dst, rel = g.src[keep], g.dst[keep], g.rel[keep]
    if rel_ids is not None:
        lut = np.full(max(int(rel.max(initial=0)), max(rel_ids)) + 1, -1, np.int64)
        lut[list(rel_ids)] = np.arange(len(rel_ids))
        r = lut[rel]
        k = r >= 0
        src, dst, rel = src[k], dst[k], r[k]
    elif len(rel) and rel.max() >= R:
        raise ShapeError(f"a layer with {R} relations got a graph with relation {int(rel.max())}")
    return src, dst, rel


def _owned_mask(g, n, T, ids):
    return (local_types(g, n, T, ids) < T)[:, None]


class TypedLinear(Layer):
    """A linear layer with its own weights for every node type: row i goes through W[type(i)] + b[type(i)]. Rows of
    a type the layer does not own give zeros. types: a count (types 0..T-1) or a list of node type ids; the types come
    from graph(..., node_types=)."""
    kind = "typed_linear"

    def __init__(self, n_in, n_out, types, bias=True, std=None):
        self.n_in, self.n_out, self.bias = int(n_in), int(n_out), bool(bias)
        self.T, self.type_ids = _owned(types)
        self.std = std

    def init(self, seed=0):
        rng = _rng(seed)
        s = self.std if self.std else 1.0 / math.sqrt(self.n_in)
        W = rng.normal(size=(self.T, self.n_in, self.n_out)) * s
        return Record(W=W, b=np.zeros((self.T, self.n_out))) if self.bias else Record(W=W)

    def apply(self, p, x, train=False, **ctx):
        masks = _type_masks(ctx.get("graph"), int(np.shape(ops.raw(x))[0]), self.T, self.type_ids)
        acc = None
        for t, m in enumerate(masks):
            y = x @ p.W[t] + (p.b[t] if self.bias else 0.0)
            y = ops.where(m, y, 0.0)
            acc = y if acc is None else acc + y
        return acc

    def describe(self):
        return f"typed_linear({self.n_in} -> {self.n_out}, {self.T} types)"


class TypedNorm(Layer):
    """Layer norm with its own scale and shift for every node type; rows of a type the layer does not own give
    zeros."""
    kind = "typed_norm"

    def __init__(self, d, types, eps=1e-5):
        self.d, self.eps = int(d), float(eps)
        self.T, self.type_ids = _owned(types)

    def init(self, seed=0):
        return Record(gamma=np.ones((self.T, self.d)), beta=np.zeros((self.T, self.d)))

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import layer_norm
        masks = _type_masks(ctx.get("graph"), int(np.shape(ops.raw(x))[0]), self.T, self.type_ids)
        acc = None
        for t, m in enumerate(masks):
            y = ops.where(m, layer_norm(x, p.gamma[t], p.beta[t], self.eps), 0.0)
            acc = y if acc is None else acc + y
        return acc

    def describe(self):
        return f"typed_norm({self.d}, {self.T} types)"


class HGT(Layer):
    """Heterogeneous graph transformer layer: a layer norm, query, key, value and output weights
    per node type; the key of every edge seen through its relation (a d_head x d_head matrix per relation and head);
    ONE softmax over all incoming edges of a node (self loops left out); a residual that leaves nodes without incoming
    edges unchanged; then a shared pre-norm feed-forward block (gelu) for the nodes of the layer's types. types and
    relations: counts (0..n-1) or lists of the node type and relation ids the layer owns; edges of other relations are
    not read and nodes of other types pass through unchanged (several encoders over one graph, each with its own
    types)."""
    kind = "hgt"

    def __init__(self, d, heads=4, types=1, relations=1, ffn_mult=4, dropout=0.0, out_bias="node", act="gelu"):
        self.d, self.h, self.fm = int(d), int(heads), int(ffn_mult)
        self.p_drop = float(dropout)
        if out_bias not in ("node", "type"):
            raise ShapeError('hgt: out_bias is "node" (nodes with incoming edges) or "type" (every node of a type '
                             'that has incoming edges in the batch)')
        self.out_bias = out_bias
        self.T, self.type_ids = _owned(types)
        self.R, self.rel_ids = _owned(relations)
        if self.d % self.h:
            raise ShapeError(f"hgt: d={self.d} must be divisible by heads={self.h}")
        self.hd = self.d // self.h
        self.norm = TypedNorm(d, types)
        self.q, self.k, self.v, self.o = (TypedLinear(d, d, types) for _ in range(4))
        self.act = str(act)                              # "gelu" (tanh form) or "gelu_exact" (erf, as PyTorch's GELU)
        self.ffn = Sequential([LayerNorm(d), Linear(d, d * self.fm), Activation(self.act), Linear(d * self.fm, d)]) \
            if self.fm > 0 else None

    def init(self, seed=0):
        rng = _rng(seed)
        f = dict(norm=self.norm.init(_seed_of(seed, 0)), q=self.q.init(_seed_of(seed, 1)),
                 k=self.k.init(_seed_of(seed, 2)), v=self.v.init(_seed_of(seed, 3)), o=self.o.init(_seed_of(seed, 4)),
                 rel=rng.normal(size=(self.R, self.h, self.hd, self.hd)) * 0.02)
        if self.ffn is not None:
            f["ffn"] = self.ffn.init(_seed_of(seed, 5))
        return Record(**f)

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        N, H, hd = int(np.shape(ops.raw(x))[0]), self.h, self.hd
        src, dst, rel = owned_edges(g, self.R, self.rel_ids)
        xn = self.norm(p.norm, x, **ctx)
        q = ops.reshape(self.q(p.q, xn, **ctx), N, H, hd)
        k = ops.reshape(self.k(p.k, xn, **ctx), N, H, hd)
        v = ops.reshape(self.v(p.v, xn, **ctx), N, H, hd)
        any_type = _owned_mask(g, N, self.T, self.type_ids)
        if len(src):
            E = len(src)                                                  # the key through the edge's relation
            kr = ops.reshape(ops.reshape(k[src], E, H, 1, hd) @ p.rel[rel], E, H, hd)
            score = ops.sum_(q[dst] * kr, axis=-1) / math.sqrt(hd)        # [E, H]
            a = segment_softmax(score, dst, N)
            msg = ops.reshape(ops.reshape(a, -1, H, 1) * v[src], -1, self.d)
            agg = segment_sum(msg, dst, N)
            x = x + ops.where(self.out_mask(g, N, dst), _drop(self.o(p.o, agg, **ctx), self.p_drop, train), 0.0)
        if self.ffn is None:
            return x
        f = _drop(self.ffn(p.ffn, x, train=train), self.p_drop, train)
        return x + ops.where(any_type, f, 0.0)

    def out_mask(self, g, N, dst):
        """The nodes the output projection is added to: those with incoming edges ("node"), or every owned node of a
        type that receives edges somewhere in the batch ("type": its output bias reaches nodes without edges too)."""
        if self.out_bias == "node":
            return (np.bincount(dst, minlength=N) > 0)[:, None]
        t = np.asarray(g.ntype if g is not None and hasattr(g, "ntype") else np.zeros(N, np.int64), np.int64)
        recv = np.zeros(int(t.max(initial=0)) + 1, bool)
        recv[np.unique(t[dst])] = True
        return (recv[t] & _owned_mask(g, N, self.T, self.type_ids).reshape(-1))[:, None]

    def children(self):
        return [self.ffn] if self.ffn is not None else []

    def describe(self):
        return f"hgt(d={self.d}, heads={self.h}, {self.T} node types, {self.R} relations)"


def region_mask(g, N, region):
    """[N, 1] bool: the nodes of `region` (graph.region, one int per node; set it on the batch's graph), or all."""
    if region is None:
        return np.ones((N, 1), bool)
    r = getattr(g, "region", None) if g is not None else None
    if r is None:
        raise ShapeError(f"region={region}: the batch's graph has no .region (one region number per node)")
    r = np.asarray(r, np.int64).reshape(-1)
    if len(r) != N:
        raise ShapeError(f"graph.region has {len(r)} entries for {N} nodes")
    return (r == int(region))[:, None]


def _drop(x, p, train):
    """Dropout of a branch while training (the rest of the time x)."""
    if not train or p <= 0:
        return x
    return x * ((ops._RNG[0].random(np.shape(ops.raw(x))) >= p) / (1.0 - p))


def _graph_ids(g, N):
    batch = np.asarray(g.batch if g is not None else np.zeros(N, np.int64), np.int64)
    G = int(g.graphs) if g is not None else (1 if N else 0)
    return batch, max(G, int(batch.max(initial=-1)) + 1)


def context_delta(x, own, batch, G, W, b, gate):
    """A graph's summary: the mean of a graph's (owned) nodes, projected, WITHOUT its component along the mean
    itself (adding the plain mean every layer over-smooths), scaled by tanh(gate): [G, d]."""
    o = own.reshape(-1).astype(bool)
    ids = batch[o]
    cnt = np.maximum(np.bincount(ids, minlength=G), 1).astype(float)[:, None]
    xs = x[np.nonzero(o)[0]] if not o.all() else x
    mu = segment_sum(xs, ids, G) / cnt
    gproj = mu @ W + b
    ss = ops.sum_(mu * mu, axis=-1, keepdims=True)
    # (a graph without owned nodes has mu = 0: the root is taken of 1e-12 there, so its gradient stays finite)
    nrm = ops.sqrt(ops.where(np.asarray(ops.raw(ss)) > 1e-12, ss, 1e-12))
    nraw = np.asarray(ops.raw(nrm))
    u = mu / ops.where(nraw > 1e-6, nrm, 1e-6)
    gc = gproj - ops.sum_(gproj * u, axis=-1, keepdims=True) * u
    return ops.tanh(gate) * gc


class GraphContext(Layer):
    """x + tanh(gate) * (the graph's summary, projected, without its common direction) for every owned node: every
    node hears its whole graph after any layer (whole-graph context; the gate starts closed)."""
    kind = "graph_context"

    def __init__(self, d, types=None):
        self.d = int(d)
        self.T, self.type_ids = _owned(types) if types is not None else (None, None)

    def init(self, seed=0):
        return Record(W=_rng(seed).normal(size=(self.d, self.d)) * 0.02, b=np.zeros(self.d), gate=np.zeros(1))

    def own(self, g, N):
        return np.ones((N, 1), bool) if self.T is None else _owned_mask(g, N, self.T, self.type_ids)

    def apply(self, p, x, train=False, **ctx):
        g = ctx.get("graph")
        N = int(np.shape(ops.raw(x))[0])
        own = self.own(g, N)
        batch, G = _graph_ids(g, N)
        d = context_delta(x, own, batch, G, p.W, p.b, p.gate[0])
        return x + ops.where(own, d[batch], 0.0)

    def describe(self):
        return f"graph_context({self.d})"


class TypedPool(Layer):
    """One vector per graph: the mean of every node type, mixed with a learned positive weight per type
    (exp(log_weight), init `init`). presence="batch": a type counts in a graph's mix when
    the batch has nodes of it (a graph without any adds zeros); "graph": only the types the graph itself has."""
    kind = "typed_pool"

    def __init__(self, types, init=None, presence="batch", others=None):
        self.T, self.type_ids = _owned(types)
        self.w0 = [1.0] * self.T if init is None else [float(v) for v in init]
        if len(self.w0) != self.T or min(self.w0) <= 0:
            raise ShapeError(f"typed_pool: init needs {self.T} positive weights")
        if presence not in ("batch", "graph"):
            raise ShapeError('typed_pool: presence is "batch" or "graph"')
        self.presence = presence
        # others: more node type ids in the mix with a fixed weight of 1 (no parameter of their own)
        self.others = tuple(int(t) for t in np.asarray(ops.raw(others)).reshape(-1)) if others is not None else ()
        if self.others and self.type_ids is None:
            raise ShapeError("typed_pool: others= needs types= as a list of type ids")
        self.TT = self.T + len(self.others)

    def init(self, seed=0):
        return Record(log_weight=np.log(np.array(self.w0, float)))

    def weights(self, log_weight):
        w = ops.exp(log_weight)
        return w if not self.others else ops.concat([w, np.ones(len(self.others))], 0)

    def plan(self, g, N):
        """(local type of each node, graph of each node, graphs, counts [G, T], which (g, t) count in the mix)."""
        loc = local_types(g, N, self.TT, (tuple(self.type_ids) + self.others) if self.others else self.type_ids)
        batch, G = _graph_ids(g, N)
        cnt = np.zeros((G, self.TT))
        o = loc < self.TT
        np.add.at(cnt, (batch[o], loc[o]), 1.0)
        if self.presence == "batch":
            on = np.repeat((cnt.sum(0) > 0)[None, :], G, 0).astype(float)
        else:
            on = (cnt > 0).astype(float)
        return loc, batch, G, cnt, on

    def apply(self, p, x, train=False, **ctx):
        N = int(np.shape(ops.raw(x))[0])
        loc, batch, G, cnt, on = self.plan(ctx.get("graph"), N)
        w = self.weights(p.log_weight)                                     # [T (+ others)]
        num, den = None, None
        for t in range(self.TT):
            if not on[:, t].any():
                continue
            idx = np.nonzero(loc == t)[0]
            m = segment_sum(x[idx], batch[idx], G) / np.maximum(cnt[:, t], 1e-8)[:, None]
            term = w[t] * m
            num = term if num is None else num + term
            wt = w[t] * on[:, t:t + 1]
            den = wt if den is None else den + wt
        if num is None:
            return x[np.zeros(G, np.int64)] * 0.0
        return num / ops.where(np.asarray(ops.raw(den)) > 1e-8, den, 1e-8)

    def describe(self):
        return f"typed_pool({self.T} types)"


class TypedGraphEncoder(Layer):
    """An encoder over the node types it owns: an input projection per type (gelu), `layers` hgt layers, a
    global_attention after every `global_every`-th layer (learned keys, `ratio` of the heads, starting as a no-op), the
    graph context after every layer (one projection shared by the layers, a gate per layer that starts closed), a final
    layer norm. Nodes of other types pass through unchanged. Pool with typed_pool, add heads as you need."""
    kind = "typed_graph_encoder"

    def __init__(self, d, heads, layers, types, relations, global_every=1, max_keys=4096, ratio=0.5, ffn_mult=4,
                 context=True, scope="owned", dropout=0.0, out_bias="node", act="gelu", region=None):
        self.d, self.h, self.L = int(d), int(heads), int(layers)
        self.T, self.type_ids = _owned(types)
        self.types = list(self.type_ids) if self.type_ids is not None else self.T
        self.every, self.context = int(global_every), bool(context)
        if scope not in ("owned", "all"):
            raise ShapeError('typed_graph_encoder: scope is "owned" (global attention, graph context and the final '
                             'norm over the owned nodes) or "all" (over every node of the graph)')
        self.scope, self.p_drop = scope, float(dropout)
        self.region = None if region is None else int(region)    # scope="all": the nodes of this region only
        self.inp = TypedLinear(d, d, self.types)
        self.act = str(act)
        self.layers = [HGT(d, heads, self.types, relations, ffn_mult, dropout=dropout, out_bias=out_bias, act=act)
                       for _ in range(self.L)]
        self.ga = {li: GlobalAttention(d, heads, max_keys, ratio=ratio, learned_keys=True, bias=True, zero_init=True,
                                       types=self.types if scope == "owned" else None, dropout=dropout,
                                       region=self.region if scope == "all" else None)
                   for li in range(self.L) if self.every > 0 and (li + 1) % self.every == 0}
        self.final = LayerNorm(d)

    def init(self, seed=0):
        rng = _rng(_seed_of(seed, 99))
        f = dict(input=self.inp.init(_seed_of(seed, 0)),
                 layers=[l_.init(_seed_of(seed, 1 + i)) for i, l_ in enumerate(self.layers)],
                 attend=[self.ga[li].init(_seed_of(seed, 1000 + li)) for li in sorted(self.ga)],
                 final=self.final.init())
        if self.context:
            f.update(ctx_W=rng.normal(size=(self.d, self.d)) * 0.02, ctx_b=np.zeros(self.d),
                     ctx_gate=np.zeros(self.L))
        return Record(**f)

    def own(self, g, N):
        return _owned_mask(g, N, self.T, self.type_ids)

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        N = int(np.shape(ops.raw(x))[0])
        own = self.own(g, N)
        batch, G = _graph_ids(g, N)
        from .layers import ACTIVATIONS
        h = ops.where(own, ACTIVATIONS[self.act](self.inp(p.input, x, **ctx)), x)
        gi = 0
        order = sorted(self.ga)
        reach = own if self.scope == "owned" else region_mask(g, N, self.region)   # where the global parts act
        for li, layer in enumerate(self.layers):
            h = layer(p.layers[li], h, train=train, **ctx)
            if li in self.ga:
                h = self.ga[li](p.attend[order.index(li)], h, train=train, **ctx)
                gi += 1
            if self.context:
                dlt = context_delta(h, reach, batch, G, p.ctx_W, p.ctx_b, p.ctx_gate[li])
                h = h + ops.where(reach, dlt[batch], 0.0)
        return ops.where(reach, self.final(p.final, h), h)

    def children(self):
        return [self.inp] + self.layers + [self.ga[k] for k in sorted(self.ga)] + [self.final]

    def describe(self):
        return f"typed_graph_encoder(d={self.d}, {self.L} layers, {self.T} node types, global every {self.every})"


# ============================================================== a typed encoder in pieces (to cut it across cards)
class TypedEncoderInput(Layer):
    """The first piece of a split typed_graph_encoder: the typed input projection (gelu) of the owned nodes."""
    kind = "typed_encoder_input"

    def __init__(self, d, types):
        self.d = int(d)
        self.T, self.type_ids = _owned(types)
        self.inp = TypedLinear(d, d, list(self.type_ids) if self.type_ids else self.T)

    def init(self, seed=0):
        return Record(input=self.inp.init(_seed_of(seed, 0)))

    def apply(self, p, x, train=False, **ctx):
        own = _owned_mask(ctx.get("graph"), int(np.shape(ops.raw(x))[0]), self.T, self.type_ids)
        return ops.where(own, ops.gelu(self.inp(p.input, x, **ctx)), x)


class TypedEncoderShared(Layer):
    """The weights the layers of a split typed encoder share (the graph context's projection, a gate per layer): passes its
    input through; the typed_encoder_layer pieces read them by this node's name."""
    kind = "typed_encoder_shared"

    def __init__(self, d, layers):
        self.d, self.L = int(d), int(layers)

    def init(self, seed=0):
        rng = _rng(_seed_of(seed, 99))
        return Record(ctx_W=rng.normal(size=(self.d, self.d)) * 0.02, ctx_b=np.zeros(self.d), ctx_gate=np.zeros(self.L))

    def apply(self, p, x, train=False, **ctx):
        return x


class TypedEncoderLayer(Layer):
    """One layer of a split typed encoder: hgt, the global attention when it has one, the graph context with the weights of
    the typed_encoder_shared node `owner` (gate number `li`)."""
    kind = "typed_encoder_layer"

    def __init__(self, hgt, ga, li, owner, types, context=True):
        self.hgt, self.ga, self.li, self.owner, self.context = hgt, ga, int(li), str(owner), bool(context)
        self.T, self.type_ids = _owned(types)

    def init(self, seed=0):
        f = dict(hgt=self.hgt.init(_seed_of(seed, 0)))
        if self.ga is not None:
            f["attend"] = self.ga.init(_seed_of(seed, 1))
        return Record(**f)

    def apply(self, p, x, train=False, **ctx):
        g = _graph(ctx)
        N = int(np.shape(ops.raw(x))[0])
        h = self.hgt(p.hgt, x, train=train, **ctx)
        if self.ga is not None:
            h = self.ga(p.attend, h, train=train, **ctx)
        if not self.context:
            return h
        sp = getattr(ctx["_dag_params"], self.owner)
        own = _owned_mask(g, N, self.T, self.type_ids)
        batch, G = _graph_ids(g, N)
        dlt = context_delta(h, own, batch, G, sp.ctx_W, sp.ctx_b, sp.ctx_gate[self.li])
        return h + ops.where(own, dlt[batch], 0.0)

    def children(self):
        return [self.hgt] + ([self.ga] if self.ga is not None else [])


class TypedEncoderOutput(Layer):
    """The last piece of a split typed encoder: the final layer norm of the owned nodes."""
    kind = "typed_encoder_output"

    def __init__(self, d, types):
        self.d = int(d)
        self.T, self.type_ids = _owned(types)
        self.final = LayerNorm(d)

    def init(self, seed=0):
        return Record(final=self.final.init())

    def apply(self, p, x, train=False, **ctx):
        own = _owned_mask(ctx.get("graph"), int(np.shape(ops.raw(x))[0]), self.T, self.type_ids)
        return ops.where(own, self.final(p.final, x), x)


def split_typed_encoder(name, encoder, src):
    """An typed_graph_encoder as dag nodes [name_in, name_shared, name_l0 … name_out] (the same computation and the same
    weights; the pieces can go to different cards). Returns (nodes, the name of the encoder's output)."""
    b = encoder
    types = list(b.type_ids) if b.type_ids is not None else b.T
    nodes = [[f"{name}_in", TypedEncoderInput(b.d, types), src], [f"{name}_shared", TypedEncoderShared(b.d, b.L), f"{name}_in"]]
    prev = f"{name}_shared"
    for li, l_ in enumerate(b.layers):
        nodes.append([f"{name}_l{li}", TypedEncoderLayer(l_, b.ga.get(li), li, f"{name}_shared", types, b.context), prev])
        prev = f"{name}_l{li}"
    nodes.append([f"{name}_out", TypedEncoderOutput(b.d, types), prev])
    return nodes, f"{name}_out"


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
                graph_memory=GraphMemory, pick_rows=PickRows, hgt=HGT, typed_linear=TypedLinear, typed_norm=TypedNorm,
                typed_graph_encoder=TypedGraphEncoder, typed_pool=TypedPool, memory_slots=MemorySlots, mask_rows=MaskRows, gated_delta=GatedDelta, brain_routing=BrainRouting, node_text_encoder=NodeTextEncoder, graph_context=GraphContext, split_typed_encoder=split_typed_encoder,
                graph_transformer_net=graph_transformer_net, k_hop=k_hop, hop_distances=hop_distances,
                eccentricity=eccentricity)
