"""GLA 3.0 in Apsi: the code-and-language model of GLA.py / gla_brain_dual.py built from `use layers`, so it trains
with fit_batches on NumPy, on Apsi's GPU trainer and split across many cards.

  gla(size)            the whole model (sizes "21b", "3b", "small", "tiny", or a record of your own): node encoder,
                       Brain A (code), Brain B (the editing brain), Brain C (text), the A -> B bridge, the FlyPrompt
                       router, the graph memory and the writer (a decoder with gated cross-attention into the memory)
  gla_config(size)     the sizes of a model; gla_count(cfg) its parameters part by part with GLA's own formula
                       (estimate_model_params), which the built model matches exactly
  gla_batch(samples)   samples (a code graph, a text graph, the answer to write, labels) -> (inputs, targets, graph)
  gla_loss(outs, y)    the losses of GLA 3.0: next-token loss of the writer, validity and issue heads, A <-> C
                       alignment (InfoNCE), router balance (load, z-loss, importance), value head
  gla_task(cfg, ...)   a built-in task (program graphs whose called functions the writer must write, in order)

Layers of GLA's own, usable in any model:
  node_encoder         a node's ids (hashed text, sub-tokens, node type, language slot, family, flags, five
                       structural codes) -> a vector
  brain_memory         what the writer reads: the k best-scored node states of two brains, both brains' summaries and
                       a null slot (RMS-normed), with a last column that marks the slots a graph does not fill
  expert_router        FlyPrompt: the top-k of `experts` experts per sample, the chosen experts attend to each other,
                       mixed by their renormalised probabilities; outputs [routed, router logits]
  gated_fusion         the AntiForgetBridge: two vectors fused through a learned gate
  param_bank           parameters held for parts trained elsewhere (GLA's PPO policy, math core): identity

The model runs on one graph per batch: the code graph and the text graph of every sample are one graph (node_types
say which node is which, batch says which sample), and each brain owns its node types and relations, so the others
pass through it unchanged - the same computation as GLA's per-type dictionaries.
"""
import math

import numpy as np

from . import ops
from .errors import RuntimeErr, ShapeError
from .layers import (Layer, Linear, LayerNorm, RMSNorm, Embedding, Activation, Sequential, Residual, SwiGLU,
                     MultiHeadAttention, Combine, DAG, Slice, TiedOutput, _rng, _seed_of, _glorot)
from .values import Record

# ======================================================================================================== GLA's tables
NODE_TYPES = ["ast", "cfg", "dfg", "pdg", "call", "sdg", "token", "func", "block", "expr", "stmt", "lang", "api",
              "config", "word"]
CODE_TYPES = ["ast", "cfg", "dfg", "pdg", "call", "sdg", "func", "block", "stmt", "lang", "config"]
LANG_TYPES = ["word"]
EDGE_TYPES = [
    ("ast", "child", "ast"), ("ast", "next", "ast"), ("cfg", "flow", "cfg"), ("cfg", "branch", "cfg"),
    ("dfg", "dataflow", "dfg"), ("pdg", "control_dep", "pdg"), ("pdg", "data_dep", "pdg"), ("call", "invokes", "func"),
    ("call", "returns", "call"), ("sdg", "inter_proc", "sdg"), ("ast", "maps_to", "cfg"), ("cfg", "maps_to", "pdg"),
    ("dfg", "maps_to", "pdg"), ("ast", "maps_to", "dfg"), ("ast", "same_line", "dfg"), ("ast", "same_line", "pdg"),
    ("ast", "same_line", "ast"), ("ast", "same_line", "call"), ("ast", "same_line", "cfg"),
    ("stmt", "same_line", "dfg"), ("stmt", "same_line", "pdg"), ("cfg", "same_line", "dfg"),
    ("cfg", "same_line", "pdg"), ("token", "belongs", "ast"), ("func", "contains", "ast"),
    ("func", "cross_lang", "func"), ("lang", "hosts", "func"), ("api", "binds", "call"), ("config", "wires", "stmt"),
    ("lang", "same_family", "lang"), ("lang", "coexists", "lang"), ("ast", "child", "call"), ("ast", "child", "stmt"),
    ("call", "child", "sdg"), ("cfg", "branch", "block"), ("lang", "hosts", "ast"), ("lang", "wires", "config"),
    ("sdg", "invokes", "func"), ("stmt", "child", "call"), ("word", "next", "word"), ("word", "near", "word")]
ISSUE_CATEGORIES = ["syntax", "security", "safety", "undefined_var", "unused_import", "logic", "_other"]
LANGUAGE_FAMILIES = ["c_like", "js_like", "script", "functional", "data", "markup", "config", "shell", "other"]
LANG_SLOTS = 70 + 1024
STRUCT_FIELDS = [("depth", 32), ("sibling", 32), ("in_degree", 16), ("out_degree", 16), ("line", 32)]
TOKEN_BUCKETS, SUBTOKEN_BUCKETS, EXTRA_SLOTS = 8192, 4096, 128
MP_FEAT_DIM, MP_BASE_UNITS = 16, 7
POOL_WEIGHTS_INIT = {"func": 2.5, "call": 2.0, "dfg": 1.8, "sdg": 1.8, "pdg": 1.5, "cfg": 1.3, "stmt": 1.0,
                     "ast": 0.8, "token": 0.5, "lang": 1.2, "api": 1.5, "config": 0.7, "block": 0.8, "expr": 0.9}
# the columns of a node's ids (node_encoder's input)
NODE_COLUMNS = ["tok", "sub", "ntype", "slot", "fam", "extra"] + [f for f, _ in STRUCT_FIELDS]


def brain_relations(types):
    """GLA's edge types whose both ends the brain carries, then every one backwards (rev_)."""
    rels = [e for e in EDGE_TYPES if e[0] in types and e[2] in types]
    for e in list(rels):
        back = (e[2], "rev_" + e[1], e[0])
        if back not in rels:
            rels.append(back)
    return rels


def relation_table():
    """Every relation of the model: Brain A's (= Brain B's), then Brain C's. An edge's relation id is its index."""
    return brain_relations(CODE_TYPES) + brain_relations(LANG_TYPES)


# ======================================================================================================== sizes
_SIZES = {
    "21b": dict(d=2048, heads=32, a_layers=24, b_layers=18, c_layers=34, a_every=1, b_every=2, c_every=2, ffn_mult=4,
                ga_ratio=0.5, max_keys=4096, experts=16, top_k=4, tasks=3, w_vocab=65536, w_dim=4096, w_layers=40,
                w_heads=32, w_kv=8, w_ffn=10976, w_cross_every=2, mem_a=256, mem_c=128, rope_theta=5000000.0),
    "3b": dict(d=1024, heads=16, a_layers=12, b_layers=9, c_layers=16, a_every=1, b_every=2, c_every=2, ffn_mult=4,
               ga_ratio=0.5, max_keys=4096, experts=16, top_k=4, tasks=3, w_vocab=65536, w_dim=2048, w_layers=20,
               w_heads=16, w_kv=4, w_ffn=5504, w_cross_every=2, mem_a=128, mem_c=64, rope_theta=1000000.0),
    "small": dict(d=64, heads=4, a_layers=3, b_layers=2, c_layers=3, a_every=1, b_every=2, c_every=2, ffn_mult=4,
                  ga_ratio=0.5, max_keys=512, experts=6, top_k=2, tasks=3, w_vocab=512, w_dim=96, w_layers=4,
                  w_heads=4, w_kv=2, w_ffn=256, w_cross_every=2, mem_a=16, mem_c=8, rope_theta=10000.0),
    "tiny": dict(d=32, heads=4, a_layers=2, b_layers=2, c_layers=2, a_every=1, b_every=2, c_every=2, ffn_mult=2,
                 ga_ratio=0.5, max_keys=64, experts=4, top_k=2, tasks=3, w_vocab=96, w_dim=32, w_layers=2, w_heads=4,
                 w_kv=2, w_ffn=64, w_cross_every=2, mem_a=16, mem_c=8, rope_theta=10000.0),
}


def gla_config(size="tiny", **override):
    """The sizes of a GLA model: "21b" (the GLA 3.0 release), "3b", "small", "tiny", or any field overridden."""
    if isinstance(size, Record):
        base = dict(size.fields)
    else:
        key = str(size).lower()
        if key not in _SIZES:
            raise RuntimeErr(f"gla: unknown size '{size}' ({', '.join(_SIZES)})")
        base = dict(_SIZES[key], name=key)
    base.setdefault("name", "custom")
    base.setdefault("math_core", True)
    base.setdefault("tie", True)
    base.setdefault("split", True)
    base.update(override)
    return Record(**base)


def _keys(c):
    return list(c.fields) if isinstance(c, Record) else list(vars(c))


def ga_heads(c):
    return max(1, int(round(c.heads * c.ga_ratio)))


def gla_count(c):
    """The parameters of every part with GLA's own formula (GLA.py estimate_model_params)."""
    c = c if isinstance(c, Record) else gla_config(c)

    def lin(i, o, bias=True):
        return i * o + (o if bias else 0)

    def ln(n):
        return 2 * n
    d, hd, fm = c.d, c.d // c.heads, c.ffn_mult

    def layer_n(types):
        n, e = len(types), len(brain_relations(types))
        ffn = (ln(d) + lin(d, d * fm) + lin(d * fm, d)) if fm > 0 else 0
        return 4 * n * lin(d, d) + e * c.heads * hd * hd + n * ln(d) + ffn

    def extra_n(types):
        n = len(types)
        return n * lin(d, d) + lin(d, d) + ln(d) + lin(d, d) + lin(d, 1) + lin(d, len(ISSUE_CATEGORIES)) + n + ln(d)
    gi = ga_heads(c) * hd
    ga_block = ln(d) + lin(d, 3 * gi) + lin(gi, d) + lin(d, 1)

    def brain_n(layers, types, every):
        n_ga = sum(1 for li in range(layers) if every > 0 and (li + 1) % every == 0)
        return layers * layer_n(types) + extra_n(types) + (lin(d, d) + layers) + n_ga * ga_block
    E = c.experts
    expert = ln(d) + lin(d, 2 * d) + ln(2 * d) + lin(2 * d, d)
    comm = 2 * ln(d) + 4 * d * d + 4 * d + lin(d, 2 * d) + lin(2 * d, d) + 1
    struct_rows = sum(f[1] for f in STRUCT_FIELDS)
    parts = {
        "brain_a": brain_n(c.a_layers, CODE_TYPES, c.a_every),
        "brain_b": brain_n(c.b_layers, CODE_TYPES, c.b_every),
        "brain_c": brain_n(c.c_layers, LANG_TYPES, c.c_every),
        "router": ln(d) + lin(d, E) + E * expert + comm + c.tasks * E,
        "bridge": 2 * lin(d, d) + 2 * lin(2 * d, d) + ln(d),
        "writer": _writer_n(c),
        "encoder": (TOKEN_BUCKETS + SUBTOKEN_BUCKETS + len(NODE_TYPES) + LANG_SLOTS + len(LANGUAGE_FAMILIES)
                    + EXTRA_SLOTS + struct_rows) * d + 1 + lin(2 * d, d) + lin(d, d) + d + 4,
        "policy": lin(d, d) + lin(d, 1024) + lin(d, d // 2) + lin(d // 2, 1) + lin(d, d // 4) + lin(d // 4, 1),
        "fusion": 2 * lin(2 * d, d),
        "small_heads": 3 * lin(d, d)}
    if c.math_core:
        h = max(64, d // 4)
        parts["math_core"] = lin(MP_FEAT_DIM, h) + lin(h, d) + ln(d) + lin(d, MP_BASE_UNITS) + 2 * lin(d, 1)
    return parts


def _writer_n(c):
    d, hd = c.w_dim, c.w_dim // c.w_heads
    attn = d * c.w_heads * hd * 2 + d * c.w_kv * hd * 2 + 2 * hd
    per = attn + 3 * d * c.w_ffn + 2 * d
    n_cross = sum(1 for i in range(c.w_layers) if c.w_cross_every > 0 and i % c.w_cross_every == c.w_cross_every - 1)
    bd = c.d
    memory = 2 * (bd + 1) + 4 * bd * d + len(NODE_TYPES) * d + 8 * d + d + d
    tie = getattr(c, "tie", True)
    return c.w_vocab * d * (1 if tie else 2) + c.w_layers * per + n_cross * (attn + d + 1) + d + memory


# ======================================================================================================== layers
class NodeEncoder(Layer):
    """A node's ids -> a vector, as GLA's TokenEfficientEncoder: [N, 11] ids (columns NODE_COLUMNS: hashed text,
    sub-token, node type, language slot, language family, flag, then depth, sibling, in / out degree, line bucket).
    what the node says + what it is + where it sits, mixed (gelu), plus a gated copy of what it says."""
    kind = "node_encoder"
    takes_ids = True

    def __init__(self, d, tables=None):
        self.d = int(d)
        self.tables = list(tables or [TOKEN_BUCKETS, SUBTOKEN_BUCKETS, len(NODE_TYPES), LANG_SLOTS,
                                      len(LANGUAGE_FAMILIES), EXTRA_SLOTS] + [n for _, n in STRUCT_FIELDS])
        if len(self.tables) != 11:
            raise ShapeError("node_encoder: 11 tables (6 for what a node is and says, 5 for where it sits)")

    def init(self, seed=0):
        rng = _rng(seed)
        d = self.d
        emb = [rng.normal(size=(n, d)) * 0.02 for n in self.tables]
        return Record(tok=emb[0], sub=emb[1], ntype=emb[2], slot=emb[3], fam=emb[4], extra=emb[5], struct=emb[6:],
                      struct_scale=np.array([0.5]), mix=Linear(2 * d, d).init(_seed_of(seed, 1)),
                      out=Linear(d, d).init(_seed_of(seed, 2)), out_gain=np.ones(d), lang_gate=np.ones(4))

    def apply(self, p, x, train=False, **ctx):
        ids = np.asarray(ops.raw(x)).astype(np.int64)
        if ids.ndim != 2 or ids.shape[1] != 11:
            raise ShapeError(f"node_encoder: ids [nodes, 11], got {list(ids.shape)}")
        for i, n in enumerate(self.tables):
            ids[:, i] = np.clip(ids[:, i], 0, n - 1)
        words = p.tok[ids[:, 0]] + p.sub[ids[:, 1]]
        what = p.ntype[ids[:, 2]] + p.lang_gate[0] * p.slot[ids[:, 3]] + p.lang_gate[1] * p.fam[ids[:, 4]] + \
            p.lang_gate[2] * p.extra[ids[:, 5]]
        where = None
        for i in range(5):
            e = p.struct[i][ids[:, 6 + i]]
            where = e if where is None else where + e
        what = what + p.struct_scale[0] * where
        h = ops.gelu(ops.concat([words, what], 1) @ p.mix.W + p.mix.b)
        return (h @ p.out.W + p.out.b) * p.out_gain + p.lang_gate[3] * words

    def describe(self):
        return f"node_encoder(11 tables -> {self.d})"


class BrainMemory(Layer):
    """What a writer reads while it writes (GLA's GraphMemory): per graph, the k_a best-scored node states of one brain
    and the k_c best of another (a learned score picks them and scales them by sigmoid(score), so the picking learns),
    projected to the writer's width with a type embedding and a slot embedding; both brains' summaries; a learned null
    slot; RMS-normed. Input [nodes_a, nodes_c, summary_a, summary_c]; output [graphs, k_a + k_c + 3, d + 1], the last
    column 1 for a slot that may be read and 0 for one a small graph leaves empty (multi_head_attention(...,
    memory_mask=true) reads it)."""
    kind = "brain_memory"

    def __init__(self, d_brain, d, k_a, k_c, types_a, types_c, node_types=None):
        from .graphs import _owned
        self.bd, self.d, self.ka, self.kc = int(d_brain), int(d), int(k_a), int(k_c)
        self.Ta, self.ids_a = _owned(types_a)
        self.Tc, self.ids_c = _owned(types_c)
        self.NT = int(node_types or len(NODE_TYPES))

    def init(self, seed=0):
        rng = _rng(seed)
        bd, d = self.bd, self.d

        def w(i, o):
            return rng.uniform(-1, 1, size=(i, o)) / math.sqrt(i)
        return Record(score_a=Record(W=w(bd, 1), b=np.zeros(1)), score_c=Record(W=w(bd, 1), b=np.zeros(1)),
                      node_a=w(bd, d), node_c=w(bd, d), sum_a=w(bd, d), sum_c=w(bd, d),
                      type_emb=rng.normal(size=(self.NT, d)) * 0.02, slot_emb=rng.normal(size=(8, d)) * 0.02,
                      null=rng.normal(size=d) * 0.02, norm=np.ones(d))

    def _nodes(self, h, ps, W, owned, T, batch, G, K, ntype, p, slot):
        from .graphs import _owned_mask
        N = int(np.shape(ops.raw(h))[0])
        own = _owned_mask(owned[0], N, T, owned[1]).reshape(-1)
        s = h @ ps.W + ps.b                                                # [N, 1]
        raw = np.asarray(ops.raw(s)).reshape(-1)
        rows, pos = [], []
        for b in range(G):
            idx = np.nonzero((batch == b) & own)[0]
            top = idx[np.argsort(-raw[idx], kind="stable")[:K]]
            rows.append(top)
            pos.append(b * K + np.arange(len(top)))
        rows = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
        pos = np.concatenate(pos).astype(np.int64) if pos else np.zeros(0, np.int64)
        valid = np.zeros(G * K)
        valid[pos] = 1.0
        if not len(rows):
            return None, valid
        vals = (h[rows] @ W) * ops.sigmoid(s[rows]) + p.type_emb[np.minimum(ntype[rows], self.NT - 1)] + \
            p.slot_emb[slot]
        from .layers import segment_sum
        return segment_sum(vals, pos, G * K), valid                        # empty slots: zeros

    def apply(self, p, x, train=False, **ctx):
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise ShapeError("brain_memory: [nodes_a, nodes_c, summary_a, summary_c]")
        ha, hc, sa, sc = x
        g = ctx.get("graph")
        N = int(np.shape(ops.raw(ha))[0])
        batch = np.asarray(g.batch, np.int64) if g is not None else np.zeros(N, np.int64)
        G = int(np.shape(ops.raw(sa))[0])
        ntype = np.asarray(g.ntype if g is not None else np.zeros(N, np.int64), np.int64)
        d = self.d
        ma, va = self._nodes(ha, p.score_a, p.node_a, (g, self.ids_a), self.Ta, batch, G, self.ka, ntype, p, 0)
        mc, vc = self._nodes(hc, p.score_c, p.node_c, (g, self.ids_c), self.Tc, batch, G, self.kc, ntype, p, 1)
        parts = []
        if self.ka:
            parts.append(ops.reshape(ma, G, self.ka, d) if ma is not None else np.zeros((G, self.ka, d)))
        if self.kc:
            parts.append(ops.reshape(mc, G, self.kc, d) if mc is not None else np.zeros((G, self.kc, d)))
        parts.append(ops.reshape(sa @ p.sum_a + p.slot_emb[2], G, 1, d))
        parts.append(ops.reshape(sc @ p.sum_c + p.slot_emb[3], G, 1, d))
        parts.append(ops.reshape(p.null + p.slot_emb[4], 1, 1, d) * np.ones((G, 1, 1)))
        mem = ops.concat(parts, 1)
        mem = mem / ops.sqrt(ops.mean(mem * mem, axis=-1, keepdims=True) + 1e-6) * p.norm
        keep = np.concatenate([va.reshape(G, self.ka), vc.reshape(G, self.kc), np.ones((G, 3))], 1)
        return ops.concat([mem, keep[:, :, None]], 2)

    def describe(self):
        return f"brain_memory({self.ka} + {self.kc} nodes + 3 slots, {self.bd} -> {self.d})"


class ExpertRouter(Layer):
    """FlyPrompt (GLA's router) on one vector per sample: a gate (plus a learned bias per task) picks the top_k of
    `experts` expert blocks, the chosen experts attend to each other (one gated attention layer whose tokens are the
    k expert outputs), and their outputs are mixed by their renormalised probabilities. Output [B, d + experts]:
    the routed vector, then the router's logits (gla_loss turns them into the balance loss)."""
    kind = "expert_router"

    def __init__(self, d, experts=16, top_k=4, tasks=3, task=0):
        self.d, self.E, self.k, self.tasks, self.task = int(d), int(experts), int(top_k), int(tasks), int(task)
        if not 1 <= self.k <= self.E:
            raise ShapeError(f"expert_router: top_k={self.k} must be between 1 and experts={self.E}")

    def init(self, seed=0):
        d, E = self.d, self.E
        L_ = lambda i, o, k: Linear(i, o).init(_seed_of(seed, k))          # noqa: E731
        ln = lambda n: Record(gamma=np.ones(n), beta=np.zeros(n))          # noqa: E731
        rng = _rng(_seed_of(seed, 999))
        ex = [Record(norm=ln(d), l1=L_(d, 2 * d, 10 + 3 * e), mid=ln(2 * d), l2=L_(2 * d, d, 11 + 3 * e))
              for e in range(E)]
        comm = Record(norm1=ln(d), wq=rng.normal(size=(d, d)) / math.sqrt(d), wk=rng.normal(size=(d, d)) / math.sqrt(d),
                      wv=rng.normal(size=(d, d)) / math.sqrt(d), wo=rng.normal(size=(d, d)) / math.sqrt(d),
                      bqkvo=np.zeros((4, d)), norm2=ln(d), f1=L_(d, 2 * d, 1), f2=L_(2 * d, d, 2),
                      gate=np.array([-3.0]))
        return Record(pre_norm=ln(d), gate=L_(d, E, 3), experts=ex, comm=comm, task_bias=np.zeros((self.tasks, E)))

    @staticmethod
    def _ln(p, x):
        from .stdlib import layer_norm
        return layer_norm(x, p.gamma, p.beta, 1e-5)

    def apply(self, p, x, train=False, **ctx):
        B, d, E, K = int(np.shape(ops.raw(x))[0]), self.d, self.E, self.k
        logits = self._ln(p.pre_norm, x) @ p.gate.W + p.gate.b + p.task_bias[self.task]
        probs = ops.softmax(logits, axis=-1)
        order = np.argsort(-np.asarray(ops.raw(probs)), axis=1, kind="stable")[:, :K]       # [B, K]
        rows = np.repeat(np.arange(B), K)
        flat = order.reshape(-1)
        # every expert on the rows that chose it
        outs, where = [], []
        for e in range(E):
            pos = np.nonzero(flat == e)[0]
            if not len(pos):
                continue
            pe = p.experts[e]
            xe = x[rows[pos]]
            h = self._ln(pe.mid, ops.gelu(self._ln(pe.norm, xe) @ pe.l1.W + pe.l1.b))
            outs.append(h @ pe.l2.W + pe.l2.b)
            where.append(pos)
        from .layers import segment_sum
        seq = ops.reshape(segment_sum(ops.concat(outs, 0), np.concatenate(where), B * K), B, K, d)
        w = ops.reshape(ops.reshape(probs, B * E)[rows * E + flat], B, K)
        w = w / ops.sum_(w, axis=-1, keepdims=True)
        talked = self._talk(p.comm, seq, B, K)
        routed = ops.sum_(talked * ops.reshape(w, B, K, 1), axis=1)
        return ops.concat([routed, logits], 1)

    def _talk(self, c, seq, B, K):
        d = self.d
        heads = 8 if d % 8 == 0 else 1
        hd = d // heads
        g = ops.sigmoid(c.gate[0])
        h = self._ln(c.norm1, seq)

        def split(t):
            return ops.transpose(ops.reshape(t, B, K, heads, hd), (0, 2, 1, 3))
        q, k, v = split(h @ c.wq + c.bqkvo[0]), split(h @ c.wk + c.bqkvo[1]), split(h @ c.wv + c.bqkvo[2])
        att = ops.softmax(q @ ops.transpose(k, (0, 1, 3, 2)) / math.sqrt(hd), axis=-1) @ v
        a = ops.reshape(ops.transpose(att, (0, 2, 1, 3)), B, K, d) @ c.wo + c.bqkvo[3]
        s1 = seq + g * a
        f = ops.gelu(self._ln(c.norm2, s1) @ c.f1.W + c.f1.b) @ c.f2.W + c.f2.b
        return s1 + g * f

    def describe(self):
        return f"expert_router(d={self.d}, {self.E} experts, top {self.k})"


def router_balance(logits, E):
    """GLA's router terms: every expert used (load x probability), small logits (z-loss), equal importance (CV^2)."""
    probs = ops.softmax(logits, axis=-1)
    load = ops.mean(probs, axis=0)
    lb = E * ops.sum_(load * load)
    z = ops.mean(ops.log(ops.sum_(ops.exp(logits), axis=-1)) ** 2)
    cv2 = ops.mean((load - ops.mean(load)) ** 2) / (ops.mean(load) ** 2 + 1e-9)
    return 0.01 * lb + 0.001 * z + 0.01 * cv2


class GatedFusion(Layer):
    """GLA's AntiForgetBridge: [a, b] -> layer_norm(g * a' + (1 - g) * out([a', b'])), a' and b' projected, g a learned
    gate on [a', b']."""
    kind = "gated_fusion"

    def __init__(self, d):
        self.d = int(d)

    def init(self, seed=0):
        d = self.d
        return Record(proj_a=Linear(d, d).init(_seed_of(seed, 0)), proj_b=Linear(d, d).init(_seed_of(seed, 1)),
                      gate=Linear(2 * d, d).init(_seed_of(seed, 2)), out=Linear(2 * d, d).init(_seed_of(seed, 3)),
                      out_norm=Record(gamma=np.ones(d), beta=np.zeros(d)))

    def apply(self, p, x, train=False, **ctx):
        from .stdlib import layer_norm
        a = x[0] @ p.proj_a.W + p.proj_a.b
        b = x[1] @ p.proj_b.W + p.proj_b.b
        cat = ops.concat([a, b], -1)
        g = ops.sigmoid(cat @ p.gate.W + p.gate.b)
        return layer_norm(g * a + (1.0 - g) * (cat @ p.out.W + p.out.b), p.out_norm.gamma, p.out_norm.beta, 1e-5)

    def describe(self):
        return f"gated_fusion({self.d})"


class ParamBank(Layer):
    """Parameters held for parts of a model trained elsewhere (GLA's PPO policy heads, math core), so a checkpoint
    holds every weight of the original: passes its input through unchanged. shapes: {name: [rows, cols] or [n]}."""
    kind = "param_bank"

    def __init__(self, shapes):
        self.shapes = {str(k): tuple(int(v) for v in s) for k, s in dict(shapes).items()}

    def init(self, seed=0):
        rng = _rng(seed)
        return Record(**{k: (_glorot(rng, *s) if len(s) == 2 else np.zeros(s)) for k, s in self.shapes.items()})

    def apply(self, p, x, train=False, **ctx):
        return x

    def describe(self):
        return f"param_bank({sum(int(np.prod(s)) for s in self.shapes.values()):,} parameters)"


# ======================================================================================================== the model
def _type_ids(names):
    return [NODE_TYPES.index(t) for t in names]


def gla(size="tiny", **override):
    """GLA 3.0 as a dag of layers. Inputs ["nodes", "tokens"]: node ids [N, 11] (NODE_COLUMNS) of the code and text
    graphs of every sample, the writer's input tokens [B, T]; the graph (gla_batch makes it) brings node types,
    relations and which sample every node belongs to. Outputs [logits, validity, issues, code vector, text summary,
    router (routed + logits), value]: gla_loss trains them all."""
    from .graphs import HGTBrain, TypedPool
    c = gla_config(size, **override)
    d, dw = c.d, c.w_dim
    rels = relation_table()
    ra = list(range(len(brain_relations(CODE_TYPES))))
    rc = list(range(len(ra), len(rels)))
    ta, tc = _type_ids(CODE_TYPES), _type_ids(LANG_TYPES)
    from .graphs import split_brain
    split = bool(getattr(c, "split", True))
    nodes = [["enc", NodeEncoder(d), "nodes"]]
    outs = {}
    for name, layers, every, types, rr, tnames in (("a", c.a_layers, c.a_every, ta, ra, CODE_TYPES),
                                                  ("b", c.b_layers, c.b_every, ta, ra, CODE_TYPES),
                                                  ("c", c.c_layers, c.c_every, tc, rc, LANG_TYPES)):
        brain = HGTBrain(d, c.heads, layers, types, rr, every, c.max_keys, c.ga_ratio, c.ffn_mult)
        if split:                                  # a node per layer: the brain can be cut across cards
            more, outs[name] = split_brain(f"brain_{name}", brain, "enc")
            nodes += more
        else:
            nodes.append([f"brain_{name}", brain, "enc"])
            outs[name] = f"brain_{name}"
        nodes += [[f"pool_{name}", TypedPool(types, [POOL_WEIGHTS_INIT.get(t, 1.0) for t in tnames]), outs[name]],
                  [f"sem_{name}", Sequential([Linear(d, d), Activation("gelu"), LayerNorm(d), Linear(d, d)]),
                   f"pool_{name}"],
                  [f"valid_{name}", Linear(d, 1), f"sem_{name}"],
                  [f"issues_{name}", Linear(d, len(ISSUE_CATEGORIES)), f"sem_{name}"]]
    h = max(64, d // 4)
    bank = {"policy_trunk_W": (d, d), "policy_trunk_b": (d,), "policy_action_W": (d, 1024), "policy_action_b": (1024,),
            "policy_r1_W": (d, d // 4), "policy_r1_b": (d // 4,), "policy_r2_W": (d // 4, 1), "policy_r2_b": (1,)}
    for i in range(3):
        bank[f"small{i}_W"], bank[f"small{i}_b"] = (d, d), (d,)
    if c.math_core:
        bank.update(math_f1_W=(MP_FEAT_DIM, h), math_f1_b=(h,), math_f2_W=(h, d), math_f2_b=(d,), math_norm_g=(d,),
                    math_norm_b=(d,), math_units_W=(d, MP_BASE_UNITS), math_units_b=(MP_BASE_UNITS,),
                    math_ok_W=(d, 1), math_ok_b=(1,), math_scale_W=(d, 1), math_scale_b=(1,))
    nodes += [
        ["bridge", GatedFusion(d), ["sem_a", "sem_b"]],
        ["route", ExpertRouter(d, c.experts, c.top_k, c.tasks), "bridge"],
        ["routed", Slice(0, d), "route"],
        ["both", Combine("concat"), ["bridge", "routed"]],
        ["code_vec", Linear(2 * d, d), "both"],
        ["fusion_1", Linear(2 * d, d), "both"],
        ["value", Sequential([Linear(d, d // 2), Activation("gelu"), Linear(d // 2, 1)]), "code_vec"],
        ["held", ParamBank(bank), "code_vec"],
        ["memory", BrainMemory(d, dw, c.mem_a, c.mem_c, ta, tc), [outs["a"], outs["c"], "sem_a", "sem_c"]],
        ["w_emb", Embedding(c.w_vocab, dw), "tokens"]]
    prev = "w_emb"
    for i in range(c.w_layers):
        nodes.append([f"w{i}_self", Residual(MultiHeadAttention(dw, c.w_heads, causal=True, kv_heads=c.w_kv, rope=True,
                                                                rope_base=c.rope_theta, qk_norm=True, bias=False),
                                             pre_norm=RMSNorm(dw)), prev])
        prev = f"w{i}_self"
        if c.w_cross_every > 0 and i % c.w_cross_every == c.w_cross_every - 1:
            nodes += [[f"w{i}_cn", RMSNorm(dw), prev],
                      [f"w{i}_cross", MultiHeadAttention(dw, c.w_heads, kv_heads=c.w_kv, qk_norm=True, bias=False,
                                                         gate=0.25, memory_mask=True), [f"w{i}_cn", "memory"]],
                      [f"w{i}_x", Combine("add"), [prev, f"w{i}_cross"]]]
            prev = f"w{i}_x"
        nodes.append([f"w{i}_ffn", Residual(SwiGLU(dw, c.w_ffn), pre_norm=RMSNorm(dw)), prev])
        prev = f"w{i}_ffn"
    nodes.append(["w_norm", RMSNorm(dw), prev])
    nodes.append(["logits", TiedOutput("w_emb") if c.tie else Linear(dw, c.w_vocab, bias=False), "w_norm"])
    m = DAG(nodes, inputs=["nodes", "tokens"],
            output=["logits", "valid_a", "issues_a", "code_vec", "sem_c", "route", "value"])
    m.config = c
    return m


def gla_parts(model, params):
    """The built model's parameters grouped as gla_count groups them."""
    from .layers import count_params
    c = model.config
    out = {k: 0 for k in gla_count(c)}
    for name, _, _, _ in model.nodes:
        n = count_params(getattr(params, name))
        if name.startswith(("brain_a", "pool_a", "sem_a", "valid_a", "issues_a")):
            k = "brain_a"
        elif name.startswith(("brain_b", "pool_b", "sem_b", "valid_b", "issues_b")):
            k = "brain_b"
        elif name.startswith(("brain_c", "pool_c", "sem_c", "valid_c", "issues_c")):
            k = "brain_c"
        elif name == "enc":
            k = "encoder"
        elif name == "bridge":
            k = "bridge"
        elif name == "route":
            k = "router"
        elif name in ("code_vec", "fusion_1"):
            k = "fusion"
        elif name == "value":
            k = "policy"
        elif name == "held":
            p = getattr(params, name)
            for f in _keys(p):
                a = getattr(p, f)
                key = "policy" if f.startswith("policy") else "small_heads" if f.startswith("small") else "math_core"
                out[key] += int(np.size(a))
            continue
        else:
            k = "writer"
        out[k] += n
    return out


# ======================================================================================================== data
BOS, EOS, PAD = 1, 2, 0


class Sample:
    """One training sample: a code graph and a text graph (each: node type names, ids per column, edges as
    (from, relation name, to)), the answer the writer must write (token ids), validity and issue labels."""

    def __init__(self, code_nodes, code_edges, text_nodes, text_edges, answer, valid=1.0, issues=None):
        self.code_nodes, self.code_edges = code_nodes, code_edges
        self.text_nodes, self.text_edges = text_nodes, text_edges
        self.answer, self.valid = list(answer), float(valid)
        self.issues = list(issues) if issues is not None else [0.0] * len(ISSUE_CATEGORIES)


def node(ntype, tok, depth=0, sibling=0, line=0, slot=0, fam=8, extra=0, sub=None, in_deg=0, out_deg=0):
    """A node of a sample: its type name and ids."""
    return dict(ntype=ntype, tok=int(tok), sub=int((tok * 7 + 3) % SUBTOKEN_BUCKETS if sub is None else sub),
                slot=int(slot), fam=int(fam), extra=int(extra), depth=int(depth), sibling=int(sibling),
                in_degree=int(in_deg), out_degree=int(out_deg), line=int(line))


def gla_batch(samples):
    """A list of Samples -> (inputs [node ids, writer input], targets, graph) for fit_batches with gla()."""
    from .layers import graph
    rels = relation_table()
    rid = {r: i for i, r in enumerate(rels)}
    ids, ntypes, batch, src, dst, rel = [], [], [], [], [], []
    for b, s in enumerate(samples):
        for nodes_, edges in ((s.code_nodes, s.code_edges), (s.text_nodes, s.text_edges)):
            base = len(ids)
            for nd in nodes_:
                t = NODE_TYPES.index(nd["ntype"])
                ids.append([nd["tok"] % TOKEN_BUCKETS, nd["sub"], t, nd["slot"], nd["fam"], nd["extra"],
                            min(nd["depth"], 31), min(nd["sibling"], 31), min(nd["in_degree"], 15),
                            min(nd["out_degree"], 15), min(nd["line"], 31)])
                ntypes.append(t)
                batch.append(b)
            for a_, r_, z_ in edges:
                fwd = (nodes_[a_]["ntype"], r_, nodes_[z_]["ntype"])
                if fwd not in rid:
                    continue                                    # a relation no brain carries
                back = (fwd[2], "rev_" + fwd[1], fwd[0])
                src += [base + a_, base + z_]
                dst += [base + z_, base + a_]
                rel += [rid[fwd], rid[back]]
    g = graph(src=np.array(src, np.int64), dst=np.array(dst, np.int64), n=len(ids), self_loops=False,
              batch=np.array(batch, np.int64), types=np.array(rel, np.int64), node_types=np.array(ntypes, np.int64))
    Lmax = max(2, max(len(s.answer) + 1 for s in samples))
    w_in, w_out, w_mask = [], [], []
    for s in samples:
        a_in, a_out = [BOS] + s.answer, s.answer + [EOS]
        w_in.append(a_in + [PAD] * (Lmax - len(a_in)))
        w_out.append(a_out + [PAD] * (Lmax - len(a_out)))
        w_mask.append([1.0] * len(a_out) + [0.0] * (Lmax - len(a_out)))
    y = Record(w_out=np.array(w_out, np.int64), w_mask=np.array(w_mask, float),
               valid=np.array([s.valid for s in samples], float), issues=np.array([s.issues for s in samples], float))
    return [np.array(ids, np.int64), np.array(w_in, np.int64)], y, g


def gla_loss(outs, y, d=None, experts=None, weights=None):
    """GLA 3.0's training loss on the outputs of gla(): writer + 0.3 validity + 0.3 issues + 0.2 alignment + router
    balance + 0.1 value. On the card the writer's cross-entropy (the large output) stays on the card; the small heads
    go to the processor (gla_loss.on_card)."""
    logits = outs[0]
    V = int(np.shape(ops.raw(logits))[-1])
    tgt = np.asarray(ops.raw(y.w_out)).reshape(-1).astype(np.int64)
    m = np.asarray(ops.raw(y.w_mask)).reshape(-1)
    lp = ops.log_softmax(ops.reshape(logits, -1, V), axis=-1)
    picked = lp[np.arange(len(tgt)), tgt]
    writer = -ops.sum_(picked * m) / (m.sum() + 1e-9)
    return writer + _gla_rest(outs, y, d, experts, weights)


def _gla_rest(outs, y, d=None, experts=None, weights=None):
    """Every term of gla_loss but the writer's cross-entropy (outs[0] is not read)."""
    _, valid, issues, code_vec, sem_c, route, value = outs
    w = dict(validity=0.3, issues=0.3, align=0.2, value=0.1)
    w.update(weights or {})
    D = int(np.shape(ops.raw(code_vec))[-1]) if d is None else int(d)
    E = int(np.shape(ops.raw(route))[-1]) - D if experts is None else int(experts)
    vy = np.asarray(ops.raw(y.valid)).reshape(-1)
    vl = ops.reshape(valid, -1)
    sp = lambda z: ops.log(1.0 + ops.exp(-ops.abs_(z))) + ops.relu(z)            # noqa: E731  softplus
    validity = ops.mean(sp(vl) - vy * vl)
    iy = np.asarray(ops.raw(y.issues))
    iss = ops.mean(sp(issues) - iy * issues)
    align = _info_nce(code_vec, sem_c, 0.07)
    router = router_balance(route[:, D:D + E], E)
    value_l = ops.mean((ops.reshape(value, -1) - vy) ** 2)
    return w["validity"] * validity + w["issues"] * iss + w["align"] * align + router + w["value"] * value_l


def _writer_labels(y):
    """The writer's targets for the card's cross-entropy (-1 where the mask is 0), or None for a mask that is not
    0 / 1 (then the whole loss runs on the processor)."""
    m = np.asarray(ops.raw(y.w_mask)).reshape(-1)
    if not np.all((m == 0) | (m == 1)):
        return None
    tgt = np.asarray(ops.raw(y.w_out)).reshape(-1).astype(np.int64)
    return np.where(m > 0, tgt, -1)


gla_loss.on_card = (0, _writer_labels, _gla_rest)        # (output, labels of the targets, the rest of the loss)


def gla_accuracy(model, params, batch):
    """The share of answers the writer writes exactly right (every token, teacher-forced, greedy)."""
    X, y, g = batch
    logits = np.asarray(ops.raw(model(params, X, graph=g)[0]))
    pred = logits.argmax(-1)
    want, m = np.asarray(ops.raw(y.w_out)), np.asarray(ops.raw(y.w_mask)) > 0
    return float(np.mean([bool(np.all(pred[b][m[b]] == want[b][m[b]])) for b in range(len(want))]))


def _info_nce(u, v, temp):
    un = u / (ops.sqrt(ops.sum_(u * u, axis=-1, keepdims=True)) + 1e-6)
    vn = v / (ops.sqrt(ops.sum_(v * v, axis=-1, keepdims=True)) + 1e-6)
    sim = un @ ops.transpose(vn, (1, 0)) / temp
    n = int(np.shape(ops.raw(sim))[0])
    lab = np.arange(n)
    a = -ops.mean(ops.log_softmax(sim, axis=-1)[lab, lab])
    b = -ops.mean(ops.log_softmax(ops.transpose(sim, (1, 0)), axis=-1)[lab, lab])
    return 0.5 * (a + b)


# ======================================================================================================== a built-in task
def gla_task(c, B, seed=0):
    """B samples of GLA21B's task: a small program graph (functions contain statements, statements call functions) and
    a text graph that describes it; the writer must write the names of the functions that are CALLED, in call order
    (it can only know them from the graph memory); a call to an unknown name makes the program invalid."""
    c = c if isinstance(c, Record) else gla_config(c)
    rng = np.random.default_rng(seed)
    names = max(2, min(20, c.w_vocab - 12))
    out = []
    for _ in range(B):
        n_funcs, n_calls = 2 + int(rng.integers(0, 2)), 2 + int(rng.integers(0, 3))
        cn, ce, fnames = [], [], []
        for f in range(n_funcs):
            nm = 10 + int(rng.integers(0, names))
            fnames.append(nm)
            cn.append(node("func", nm, 0, f, fam=2))
        root = len(cn)
        cn.append(node("ast", 3, 1, 0, fam=2))
        ce.append((0, "contains", root))
        called, prev, bad = [], root, 0.0
        for k in range(n_calls):
            st = len(cn)
            cn.append(node("stmt", 4, 2, k, line=k, fam=2))
            ce.append((root, "child", st))
            cl = len(cn)
            unknown = int(rng.integers(0, 10)) == 0
            target = int(rng.integers(0, n_funcs))
            cn.append(node("call", 5 if unknown else fnames[target], 3, 0, line=k, fam=2))
            ce.append((st, "child", cl))
            if unknown:
                bad = 1.0
            else:
                ce.append((cl, "invokes", target))
                called.append(fnames[target])
            a2 = len(cn)
            cn.append(node("ast", 6, 3, 1, line=k, fam=2))
            ce.append((st, "child", a2))
            ce.append((prev, "next", a2))
            prev = a2
        words = [7] + called + [8]
        tn = [node("word", w_, 0, i) for i, w_ in enumerate(words)]
        te = [(i - 1, "next", i) for i in range(1, len(words))] + [(i - 2, "near", i) for i in range(2, len(words))]
        out.append(Sample(cn, ce, tn, te, called, 1.0 - bad, [0, 0, 0, bad, 0, 0, 0]))
    return out


def names():
    return dict(gla=gla, gla_config=gla_config, gla_count=gla_count, gla_parts=gla_parts, gla_batch=gla_batch,
                gla_loss=gla_loss, gla_accuracy=gla_accuracy, gla_task=gla_task, gla_sample=Sample, gla_node=node, node_encoder=NodeEncoder,
                brain_memory=BrainMemory, expert_router=ExpertRouter, router_balance=router_balance,
                gated_fusion=GatedFusion, param_bank=ParamBank, gla_node_types=list(NODE_TYPES),
                gla_relations=lambda: [list(r) for r in relation_table()])
