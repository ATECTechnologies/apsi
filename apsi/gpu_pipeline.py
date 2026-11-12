"""One model split across several cards (model parallelism) for Apsi's own GPU trainer.

The model's layers are cut into stages, one stage per card, at the points that balance the work (and, on real
cards, keep every stage inside its card's memory). A batch is cut into micro-batches that flow through the stages
in a one-forward-one-backward schedule: while card 2 runs micro-batch 1, card 1 already runs micro-batch 2, so the
cards work at the same time instead of waiting for each other. Only the activations at the cuts travel between
the cards (card to card where the driver allows it), never the weights. Each card keeps only its own stage's
weights, gradients and optimizer state, so the model can be as large as the sum of the cards.

With more cards than stages, the pipeline is copied: stages x replicas cards, each replica takes its share of the
batch, and the copies of a stage sum their gradients and share the optimizer's work (data parallel on top).

The result equals training on one card with the whole batch (the gradient of a batch is the sum over its
micro-batches), up to the order of floating-point additions."""
import math
import queue
import threading
import time

import numpy as np

from .values import Record, tree_leaves, tree_map


# ============================================================================== the model in pieces
def units(model, params):
    """The model as a flat list of (layer, parameters): the places where it can be cut. A sequence is opened, a
    transformer block gives its attention half and its MLP half."""
    from . import layers as L
    out = []

    def walk(layer, p):
        if isinstance(layer, L.Sequential):
            for l_, pp in zip(layer.layers, p):
                walk(l_, pp)
        elif isinstance(layer, L.TransformerBlock):
            walk(layer.attn, p.attn)
            walk(layer.mlp, p.mlp)
        else:
            out.append((layer, p))
    walk(model, params)
    return out


def _count(p):
    mats = vecs = 0
    for a in tree_leaves(p):
        if isinstance(a, np.ndarray):
            if a.ndim >= 2:
                mats += a.size
            else:
                vecs += a.size
    return mats, vecs


def unit_costs(us, sample_shape):
    """(work, parameters) of each unit for one sample: the work in multiply-adds of the forward and backward
    passes (matrices 3 x 2 per weight and token, attention 3 x 4 T d per token, the rest a few per value)."""
    from . import layers as L
    T = int(sample_shape[0]) if len(sample_shape) >= 1 else 1
    work, size = [], []
    for layer, p in us:
        mats, vecs = _count(p)
        inner = layer
        while isinstance(inner, L.Residual):
            inner = inner.inner
        if isinstance(layer, L.Embedding):
            w = 3.0 * layer.d * T                       # a lookup and a scatter-add per token
        else:
            w = 6.0 * mats * T + 10.0 * vecs * T
        if isinstance(inner, L.MultiHeadAttention):
            w += 12.0 * T * T * inner.d
        work.append(max(w, 1.0))
        size.append(mats + vecs)
    return work, size


def partition(work, size, stages, bytes_per_param=16.0, budget=None):
    """Cut the units into `stages` consecutive groups (each at least one unit) so that the largest group's work is
    the smallest possible, each group's parameters (weights, gradients, optimizer state) inside `budget` bytes when
    a budget is given. Returns the number of units in each group, or None when no cut fits."""
    n = len(work)
    if stages < 1 or stages > n:
        return None
    pre_w = np.concatenate([[0.0], np.cumsum(work)])
    pre_s = np.concatenate([[0.0], np.cumsum(size)])
    INF = float("inf")
    # best[k][i]: the smallest largest-work over the first i units in k groups
    best = [[INF] * (n + 1) for _ in range(stages + 1)]
    cut = [[0] * (n + 1) for _ in range(stages + 1)]
    best[0][0] = 0.0
    for k in range(1, stages + 1):
        for i in range(k, n - (stages - k) + 1):
            for j in range(k - 1, i):
                if best[k - 1][j] == INF:
                    continue
                if budget is not None and (pre_s[i] - pre_s[j]) * bytes_per_param > budget:
                    continue
                v = max(best[k - 1][j], pre_w[i] - pre_w[j])
                if v < best[k][i]:
                    best[k][i], cut[k][i] = v, j
    if best[stages][n] == INF:
        return None
    sizes, i = [], n
    for k in range(stages, 0, -1):
        j = cut[k][i]
        sizes.append(i - j)
        i = j
    return sizes[::-1]


def plan(model, params, sample_shape, n_cards, stages=None, budget=None, adam=True):
    """(stage sizes in units, replicas): how the cards are used. stages=None: the fewest stages that fit the cards
    (one stage when the model fits one card), the cards then shared between stages and replicas; an int: that many
    stages; a list: the units of each stage, as given."""
    from .gpu_train import TrainError
    us = units(model, params)
    work, size = unit_costs(us, sample_shape)
    bpp = 16.0 if adam else 12.0
    if isinstance(stages, (list, tuple)):
        sizes = [int(s) for s in stages]
        if any(s < 1 for s in sizes) or sum(sizes) != len(us):
            raise TrainError(f"stages={list(stages)}: give the units of each stage, each at least 1, "
                             f"{len(us)} in all ({describe_units(us)})")
        S = len(sizes)
        if S > n_cards:
            raise TrainError(f"stages={list(stages)}: {S} stages on {n_cards} cards")
        return sizes, max(1, n_cards // S)
    if stages is not None:
        S = int(stages)
        if not 1 <= S <= min(n_cards, len(us)):
            raise TrainError(f"stages={S}: between 1 and {min(n_cards, len(us))} (the cards, and the model's "
                             f"{len(us)} cut points)")
        sizes = partition(work, size, S, bpp, budget) or partition(work, size, S, bpp, None)
        return sizes, max(1, n_cards // S)
    for S in range(1, min(n_cards, len(us)) + 1):            # the fewest stages that fit
        if partition(work, size, S, bpp, budget) is not None:
            break
    else:
        need = sum(size) * bpp / 2 ** 30
        raise TrainError(f"the model needs about {need:.1f} GB for its weights, gradients and optimizer state; "
                         f"{n_cards} cards of {budget / 2 ** 30:.1f} GB usable each cannot hold it even split "
                         f"into {min(n_cards, len(us))} stages (more cards; precision=\"fp8\"; SGD)")
    R = max(1, n_cards // S)
    S = n_cards // R                                     # the spare cards deepen the pipeline (less per card)
    S = min(S, len(us))
    sizes = partition(work, size, S, bpp, budget) or partition(work, size, S, bpp, None)
    return sizes, R


def describe_units(us):
    return ", ".join(getattr(l, "kind", type(l).__name__) for l, _ in us[:12]) + (" ..." if len(us) > 12 else "")


# ============================================================================== one stage on one card
class _Abort(Exception):
    pass


class _Link:
    """Messages from one stage to the next (or back), in order; a failure anywhere stops every waiting stage."""

    def __init__(self, stop):
        self.q, self.stop = queue.Queue(), stop

    def put(self, msg):
        self.q.put(msg)

    def get(self):
        while True:
            try:
                return self.q.get(timeout=0.25)
            except queue.Empty:
                if self.stop.is_set():
                    raise _Abort() from None


def _all_nodes(nd):
    """Every object of a compiled model that keeps per-batch state (the nodes and their helpers), found by walking
    their attributes: the state of each micro-batch in flight is saved after its forward pass and put back for its
    backward pass."""
    from . import gpu_train as GT
    kinds = (GT._Node, GT._TypedLin, GT._TypedNormOp)
    out, seen, todo = [], set(), [nd]
    while todo:
        x = todo.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, kinds):
            out.append(x)
            vals = list(vars(x).values())
        elif isinstance(x, (list, tuple)):
            vals = list(x)
        elif isinstance(x, dict):
            vals = list(x.values())
        else:
            continue
        for v in vals:
            if isinstance(v, kinds + (list, tuple, dict)) and not isinstance(v, GT.Net):
                todo.append(v)
    return out


class Stage:
    """A run of the model's units on one card: its own Net (weights, gradients), activation slots for the
    micro-batches in flight, and the optimizer's state for the slice of the weights it updates."""

    def __init__(self, ops, layers, plist, precision, index, count, slots):
        from . import layers as L
        from .gpu_train import Net, _Arena
        self.ops, self.index, self.count = ops, index, count
        self.first, self.last = index == 0, index == count - 1
        self.net = Net(ops, L.Sequential(layers), list(plist), precision)
        self.garena = self.net.arena                     # the backward pass's buffers: one micro-batch at a time
        self.net.garena = self.garena
        self.slots = [_Arena(ops) for _ in range(max(1, slots))]
        self.nodes = _all_nodes(self.net.root)
        self.saved, self.incount = {}, {}
        self.extra = []                                  # buffers this stage owns besides the net's

    def alloc(self, n):
        b = self.ops.alloc(max(1, n))
        self.extra.append(b)
        return b

    def use_slot(self, m):
        self.net.arena = self.slots[m % len(self.slots)]
        return self.net.arena

    def forward(self, m, x, shape):
        self.use_slot(m)
        y, ys = self.net.root.fwd(x, shape)
        self.incount[m] = int(np.prod(shape))
        self.saved[m] = [dict(vars(nd)) for nd in self.nodes]   # what the backward pass of m needs
        return y, ys

    def backward(self, m, dy):
        self.use_slot(m)
        for nd, st in zip(self.nodes, self.saved.pop(m)):
            nd.__dict__.update(st)
        return self.net.root.bwd(dy, not self.first), self.incount.pop(m)

    def close(self):
        for a in self.slots:
            a.close()
        for b in self.extra:
            self.ops.free(b)
        self.extra = []
        self.net.arena = self.garena
        self.net.close()


def schedule(stage, count, micro):
    """The one-forward-one-backward order of a stage: a few forwards to fill the pipeline, then a forward and a
    backward in turn (at most count - stage micro-batches in flight), then the backwards that are left."""
    warm = min(count - stage - 1, micro)
    out = [("F", m) for m in range(warm)]
    f, b = warm, 0
    while b < micro:
        if f < micro:
            out.append(("F", f))
            f += 1
        out.append(("B", b))
        b += 1
    return out


# ============================================================================== training
def fit_pipeline(ops_list, model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip, schedule_fn, val,
                 precision, ema, ewc, stages=None, micro_batches=None, comp=False):
    from . import layers as L, ops as aops, thermal
    from .stdlib import Adam, AdamW, SGD
    from .gpu_train import TrainError, _inputs, _rows, _eval_loss_host
    loss = str(loss)
    from .gpu_train import LOSSES
    if loss not in LOSSES:
        raise TrainError(f"loss '{loss}' is not on the GPU trainer ({', '.join(LOSSES)})")
    opt = opt or Adam(1e-3)
    if not isinstance(opt, (Adam, SGD)):
        raise TrainError(f"optimizer {type(opt).__name__} is not on the GPU trainer (adam, adamw, sgd)")
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    ids = isinstance(first, L.Embedding)
    Xa = _inputs(aops.raw(X), ids)
    Ya = np.asarray(aops.raw(X if y is None else y))
    ce = loss == "cross_entropy"
    Ya = np.ascontiguousarray(Ya.astype(np.int32) if ce else Ya.astype(np.float32))
    n, E = Xa.shape[0], int(epochs)
    bs = int(batch_size) if batch_size else n
    adam = isinstance(opt, Adam)
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    mems = [getattr(getattr(o, "dev", None), "memory", None) for o in ops_list]
    budget = 0.8 * min(mems) if all(mems) else None
    if stages == "all":                                  # parallel="pipeline" with no count: every card a stage
        stages = min(len(ops_list), len(units(model, params)))
    sizes, R = plan(model, params, Xa.shape[1:], len(ops_list), stages, budget, adam)
    S = len(sizes)
    us = units(model, params)
    cuts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    rng = np.random.default_rng(int(seed))
    orders = [rng.permutation(n) if batch_size else np.arange(n) for _ in range(E)]
    per_rep = -(-min(bs, n) // R)
    M = int(micro_batches) if micro_batches else (min(per_rep, 4 * S) if S > 1 else 1)
    M = max(1, min(M, per_rep))
    xw, yw = int(np.prod(Xa.shape[1:])), int(np.prod(Ya.shape[1:]))
    grid = []                                             # grid[r][s]: replica r, stage s
    try:
        for r in range(R):
            row = []
            for s in range(S):
                o = ops_list[r * S + s]
                lay = [l_ for l_, _ in us[cuts[s]:cuts[s + 1]]]
                pl = [p_ for _, p_ in us[cuts[s]:cuts[s + 1]]]
                st = Stage(o, lay, pl, precision, s, S, min(S - s, M))
                st.net.seed = int(seed) * 7919 + 1 + 104729 * (r * S + s)
                st.gov = thermal.governor(getattr(o, "dev", None))
                st.loss_name = loss
                row.append(st)
                if st.first:
                    st.XA, st.IDX = st.alloc(Xa.size), st.alloc(max(1, n * E))
                    o.put(st.XA, Xa.reshape(-1))
                    if E:
                        o.put(st.IDX, np.concatenate(orders).astype(np.int32))
                if st.last:
                    st.YA, st.IDXy = st.alloc(Ya.size), (st.IDX if st.first else st.alloc(max(1, n * E)))
                    o.put(st.YA, Ya.reshape(-1))
                    if E and not st.first:
                        o.put(st.IDXy, np.concatenate(orders).astype(np.int32))
                    st.LOSSE = st.alloc(max(1, E))
                    o.fill((st.LOSSE, 0), max(1, E), 0.0)
            grid.append(row)
        # the optimizer: the copies of a stage share its weights out in slices, each updates its own
        for s in range(S):
            group = [grid[r][s] for r in range(R)]
            nP = group[0].net.n
            for k, st in enumerate(group):
                o = st.ops
                st.lo, st.hi = nP * k // R, nP * (k + 1) // R
                ln = max(1, st.hi - st.lo)
                st.st1, st.st2 = st.alloc(ln), (st.alloc(ln) if adam else None)
                o.fill((st.st1, 0), ln, 0.0)
                if st.st2 is not None:
                    o.fill((st.st2, 0), ln, 0.0)
                st.SQ, st.TMP = st.alloc(1), st.alloc(ln)
                st.COMP = st.alloc(ln) if comp else None
                if st.COMP is not None:
                    o.fill((st.COMP, 0), ln, 0.0)
        EMA = None
        if ema is not None:
            ema = float(ema)
            if not 0.0 < ema < 1.0:
                raise TrainError(f"ema={ema}: give the decay, between 0 and 1 (0.999 is usual)")
            EMA = []
            for st in grid[0]:
                b = st.alloc(st.net.n)
                st.ops.fill((b, 0), st.net.n, 0.0)
                st.ops.axpy((b, 0), (st.net.P, 0), st.net.n, 1.0)
                EMA.append(b)
        lam = None
        if ewc is not None:
            try:
                anchor, fisher, lam = ewc
            except (TypeError, ValueError):
                raise TrainError("ewc=(anchor_params, fisher, strength)") from None
            lam = float(lam)
            mine = tree_leaves(params)
            amap = {id(a): t for a, t in zip(mine, tree_leaves(anchor))}
            fmap = {id(a): t for a, t in zip(mine, tree_leaves(fisher))}
            if len(amap) != len(tree_leaves(anchor)) or len(tree_leaves(fisher)) != len(mine):
                raise TrainError("ewc: the anchor and fisher must have the shapes of the parameters")
            for row in grid:
                for st in row:
                    o = st.ops
                    st.EA, st.EF = st.alloc(st.net.n), st.alloc(st.net.n)
                    o.put(st.EA, _flat(st.net, amap))
                    o.put(st.EF, _flat(st.net, fmap))
                    st.ET = st.alloc(max(1, st.hi - st.lo))
        t_step, step, history, val_hist, ep_seconds = int(getattr(opt, "t", 0)), 0, [], [], []
        for ep in range(E):
            t0 = time.perf_counter()
            for s0 in range(0, n, bs):
                b = orders[ep][s0:s0 + bs]
                yb = Ya[b]
                lr = float(aops.raw(schedule_fn(step))) if schedule_fn is not None else float(opt.lr)
                valid = max(int((yb >= 0).sum()), 1) if ce else None
                parts = np.array_split(np.arange(len(b)), R)
                _run_step(grid, parts, b, ep, n, s0, xw, yw, Xa.shape[1:], M, ce, valid, ep)
                t_step += 1
                _update(grid, opt, adam, wd, lr, t_step, clip, lam)
                if EMA is not None:
                    for st, eb in zip(grid[0], EMA):
                        st.ops.axpy((eb, 0), (eb, 0), st.net.n, ema - 1.0)
                        st.ops.axpy((eb, 0), (st.net.P, 0), st.net.n, 1.0 - ema)
                step += 1
                for row in grid:
                    for st in row:
                        st.gov.tick(st.ops.sync)
            for row in grid:
                for st in row:
                    st.ops.sync()
            ep_seconds.append(round(time.perf_counter() - t0, 3))
            if val is not None:
                val_hist.append(_eval_loss_host(lambda xb: pipe_forward(grid[0], xb), val, ids, loss, bs))
            if verbose and (ep % max(1, E // 10) == 0 or ep == E - 1):
                lv = sum(float(row[-1].ops.get(row[-1].LOSSE, 1, ep)[0]) for row in grid)
                print(f"epoch {ep:4}  loss {lv:.5f}  ({S} stages x {R} replicas)")
        history = [float(sum(v)) for v in zip(*[row[-1].ops.get(row[-1].LOSSE, max(1, E))[:E] for row in grid])]
        opt.t = t_step
        out_params = _gather_params(params, grid[0])
        ema_params = _gather_params(params, grid[0], EMA) if EMA is not None else None
        peak = max(st.ops.used for row in grid for st in row)
        temps = [st.gov.status() for row in grid for st in row]
    finally:
        for row in grid:
            for st in row:
                st.close()
    return Record(params=out_params, history=history, val_history=val_hist, gpu_memory_mb=round(peak / 2 ** 20),
                  gpu_temperature=temps[0], gpu_temperatures=temps, epoch_seconds=ep_seconds,
                  epoch_temperature_c=[t.get("temperature_c") if t else None for t in temps], ema_params=ema_params,
                  stages=list(sizes), replicas=R, micro_batches=M,
                  device=f"{S * R} GPUs ({', '.join(getattr(st.ops, 'name', '?') for row in grid for st in row)}), "
                         f"Apsi kernels, model split into {S} stages" + (f" x {R} replicas" if R > 1 else "")
                         + f", {M} micro-batches")


def _flat(net, idmap):
    flat = np.zeros(net.n, np.float32)
    for a in net.leaves:
        o = net.offsets[id(a)]
        flat[o:o + a.size] = np.asarray(idmap[id(a)], np.float32).reshape(-1)
    return flat


def _gather_params(params, stages, bufs=None):
    got = {}
    for i, st in enumerate(stages):
        flat = st.ops.get(bufs[i] if bufs is not None else st.net.P, st.net.n)
        for a in st.net.leaves:
            o = st.net.offsets[id(a)]
            got[id(a)] = flat[o:o + a.size].reshape(a.shape).astype(a.dtype if a.dtype.kind == "f" else np.float32)
    return tree_map(lambda a: got.get(id(a), a) if isinstance(a, np.ndarray) else a, params)


def _run_step(grid, parts, b, ep, n, s0, xw, yw, xshape, M, ce, valid, lossi):
    """Every stage of every replica on its own thread, each following its one-forward-one-backward order; they
    wait only for the activation or gradient they need."""
    stop = threading.Event()
    errors = []
    threads = []
    for row, part in zip(grid, parts):
        S = len(row)
        k = len(part)
        if not k:
            for st in row:
                st.ops.fill((st.net.G, 0), st.net.n, 0.0)
            continue
        mb = [p for p in np.array_split(np.arange(k), min(M, k)) if len(p)]
        fwd = [_Link(stop) for _ in range(S + 1)]         # fwd[s]: into stage s
        bwd = [_Link(stop) for _ in range(S + 1)]         # bwd[s]: gradients into stage s (from s + 1)
        sends = [threading.Semaphore(2) for _ in range(S)]
        for st in row:
            args = (st, row, mb, part, b, ep, n, s0, xw, yw, xshape, ce, valid, fwd, bwd, sends, lossi)
            t = threading.Thread(target=_stage_main, args=(args, stop, errors), daemon=True)
            threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def _stage_main(args, stop, errors):
    try:
        _stage_work(*args)
    except _Abort:
        pass
    except BaseException as e:                           # noqa: BLE001
        errors.append(e)
        stop.set()


def _stage_work(st, row, mb, part, b, ep, n, s0, xw, yw, xshape, ce, valid, fwd, bwd, sends, lossi):
    from .gpu_train import _rows, _regress
    o, net, s, S = st.ops, st.net, st.index, st.count
    loss_name = st.loss_name
    o.fill((net.G, 0), net.n, 0.0)
    net.round_weights()
    net.train = True
    ring = 0
    for kind, m in schedule(s, S, len(mb)):
        rows = mb[m]
        k = len(rows)
        start = ep * n + s0 + int(part[0]) + int(rows[0])
        if kind == "F":
            a = st.use_slot(m)
            if st.first:
                x = a.get("X", k * xw)
                o.gather((st.XA, 0), (st.IDX, 0), x, k, xw, start)
                shape = (k,) + tuple(xshape)
            else:
                got, view, cnt, shape, src = fwd[s].get()
                assert got == m
                x = a.get("recv", cnt)
                o.pull(x, src, view, cnt)
            out, shape = st.forward(m, x, shape)
            if not st.last:
                o.sync()
                fwd[s + 1].put((m, out, int(np.prod(shape)), shape, o))
                continue
            # the last stage: the loss of this micro-batch, then its backward at once
            r_ = _rows(shape)
            yb = a.get("Y", k * yw)
            o.gather((st.YA, 0), (st.IDXy, 0), yb, k, yw, start)
            dout = st.garena.get("dout", r_ * shape[-1])
            lossv = (st.LOSSE, lossi)
            if ce:
                w = 1.0 / valid
                o.ce(out, yb, dout, lossv, r_, shape[-1], w, w * len(b) / n)
            else:
                m_all = len(b) * int(np.prod(shape[1:]))
                _regress(o, loss_name, out, yb, dout, lossv, r_ * shape[-1], 1.0 / m_all, len(b) / (n * m_all))
            dy = dout
            release = None
        else:
            if st.last:
                continue                                 # (done with its forward)
            got, view, cnt, src, sem = bwd[s].get()
            assert got == m
            dy = st.garena.get("din", cnt)
            o.pull(dy, src, view, cnt)
            release = sem
        if kind == "B" or st.last:
            dx, cnt = st.backward(m, dy)
            if st.first:
                o.sync()
            else:
                sends[s].acquire()                       # a free send buffer (the stage before has read it)
                buf = st.garena.get(("send", ring), cnt)
                ring ^= 1
                o.pull(buf, o, dx, cnt)
                o.sync()
                bwd[s - 1].put((m, buf, cnt, o, sends[s]))
            if release is not None:
                release.release()
    o.sync()



def _update(grid, opt, adam, wd, lr, t_step, clip, lam):
    """The copies of each stage sum their gradients (each the slice it updates), then every slice is updated and
    the copies take the updated slices from each other. clip: the norm of the whole model's gradient."""
    R = len(grid)
    S = len(grid[0])
    for s in range(S):
        group = [grid[r][s] for r in range(R)]
        if R > 1:
            for st in group:
                st.ops.sync()
            for st in group:
                o, lo, k = st.ops, st.lo, st.hi - st.lo
                if k <= 0:
                    continue
                for q in group:
                    if q is not st:
                        o.pull((st.TMP, 0), q.ops, (q.net.G, lo), k)
                        o.axpy((st.net.G, lo), (st.TMP, 0), k, 1.0)
        for st in group:
            o, net, lo, k = st.ops, st.net, st.lo, st.hi - st.lo
            if lam is not None and k > 0:
                o.fill((st.ET, 0), k, 0.0)
                o.axpy((st.ET, 0), (net.P, lo), k, 1.0)
                o.axpy((st.ET, 0), (st.EA, lo), k, -1.0)
                o.mul3((st.ET, 0), (st.ET, 0), (st.EF, lo), k)
                o.axpy((net.G, lo), (st.ET, 0), k, lam)
            o.fill((st.SQ, 0), 1, 0.0)
            if clip and k > 0:
                o.sumsq((net.G, lo), k, (st.SQ, 0))
    everyone = [st for row in grid for st in row]
    if clip:
        total = sum(float(st.ops.get(st.SQ, 1)[0]) for st in everyone)
        for st in everyone:
            st.ops.fill((st.SQ, 0), 1, total)
    for st in everyone:
        o, net, lo, k = st.ops, st.net, st.lo, st.hi - st.lo
        if k <= 0:
            continue
        if adam:
            c1, c2 = 1 - opt.b1 ** t_step, 1 - opt.b2 ** t_step
            nd = min(max(net.n_decay - lo, 0), k)
            o.adam((net.P, lo), (net.G, lo), (st.st1, 0), (st.st2, 0), (st.SQ, 0), k, nd, lr, opt.b1, opt.b2,
                   opt.eps, c1, c2, wd, float(clip or 0.0), (st.COMP, 0) if st.COMP else None)
        else:
            o.sgd((net.P, lo), (net.G, lo), (st.st1, 0), (st.SQ, 0), k, lr, float(opt.momentum), float(clip or 0.0),
                  (st.COMP, 0) if st.COMP else None)
    if R > 1:
        for st in everyone:
            st.ops.sync()
        for s in range(S):
            group = [grid[r][s] for r in range(R)]
            for st in group:
                for q in group:
                    if q is not st and q.hi > q.lo:
                        st.ops.pull((st.net.P, q.lo), q.ops, (q.net.P, q.lo), q.hi - q.lo)
        for st in everyone:
            st.ops.sync()


def pipe_forward(stages, xb):
    """The model's output for a batch, stage after stage (evaluation)."""
    x_host, shape = np.ascontiguousarray(xb), xb.shape
    out = None
    for st in stages:
        o, net = st.ops, st.net
        net.train = False
        a = st.use_slot(0)
        flat = x_host.reshape(-1)
        xd = a.get("Xp", flat.size)
        o.put(xd[0], flat)
        net.round_weights()
        y, shape = net.root.fwd(xd, tuple(shape))
        out = o.get(y[0], int(np.prod(shape)), y[1]).reshape(shape)
        x_host = out
        net.train = True
    return out


# ============================================================================== batches of their own shape, any dag
def dag_view(model, params):
    """(nodes [(name, layer, srcs, single, params)], inputs, output): a dag as it is, a sequence as a chain."""
    from . import layers as L
    from .gpu_train import TrainError
    if isinstance(model, L.DAG):
        if not isinstance(model.output, str):
            raise TrainError("a split dag has one output (output=\"name\")")
        return ([(n, l_, list(srcs), single, getattr(params, n)) for n, l_, srcs, single in model.nodes],
                list(model.inputs), model.output)
    nodes, prev = [], "x"
    for i, (l_, p) in enumerate(units(model, params)):
        nodes.append((f"u{i}", l_, [prev], True, p))
        prev = f"u{i}"
    return nodes, ["x"], prev


def live_sets(nodes, inputs, output, a, b):
    """The values a stage over nodes[a:b] receives and the values it hands on."""
    last = {}
    for i, (n, _, srcs, _, _) in enumerate(nodes):
        for s_ in srcs:
            last[s_] = max(last.get(s_, -1), i)
    last[output] = len(nodes)

    def before(k):
        return list(inputs) + [nodes[i][0] for i in range(k)]
    return ([x for x in before(a) if last.get(x, -1) >= a], [x for x in before(b) if last.get(x, -1) >= b])


class SegStage(Stage):
    """A stretch of a dag's nodes on one card: it receives several named values and hands several on."""

    def __init__(self, ops, seg, live_in, live_out, precision, index, count, slots):
        from . import layers as L
        from .gpu_train import Net, _Arena
        self.ops, self.index, self.count = ops, index, count
        self.first, self.last = index == 0, index == count - 1
        self.live_in, self.live_out = live_in, live_out
        model = L.DAG([[n, l_, srcs[0] if single else srcs] for n, l_, srcs, single, _ in seg], inputs=live_in,
                      output=seg[-1][0])
        self.net = Net(ops, model, Record(**{n: p for n, _, _, _, p in seg}), precision)
        self.garena = self.net.arena
        self.net.garena = self.garena
        self.slots = [_Arena(ops) for _ in range(max(1, slots))]
        self.nodes = _all_nodes(self.net.root)
        self.saved, self.incount = {}, {}
        self.extra = []

    def forward_seg(self, m, ins):
        self.use_slot(m)
        vals = self.net.root.run_all(dict(ins))
        self.saved[m] = [dict(vars(nd)) for nd in self.nodes]
        return vals

    def backward_seg(self, m, seeds):
        self.use_slot(m)
        for nd, st in zip(self.nodes, self.saved.pop(m)):
            nd.__dict__.update(st)
        return self.net.root.back_from(seeds)


def fit_batches_pipeline(ops_list, model, params, batches, loss, opt, epochs, verbose, clip, schedule_fn, precision,
                         stages=None, micro=None, compensated=False, every=None):
    """fit_batches on several cards: the model (a dag or a sequence) cut into stages at the points that balance the
    work, `micro` batches of a step flowing through them in a one-forward-one-backward order, the pipeline copied
    over the cards left (replicas). Every batch may bring its own shapes and graph. One step averages the gradients
    of its batches."""
    from . import ops as aops, thermal
    from .stdlib import Adam, AdamW, SGD
    from .gpu_train import TrainError, LOSSES, _batch_parts
    custom = callable(loss)
    loss_fn, loss = (loss, "custom") if custom else (None, str(loss))
    if not custom and loss not in LOSSES:
        raise TrainError(f"loss '{loss}' is not on the GPU trainer ({', '.join(LOSSES)})")
    opt = opt or Adam(1e-3)
    if not isinstance(opt, (Adam, SGD)):
        raise TrainError(f"optimizer {type(opt).__name__} is not on the GPU trainer (adam, adamw, sgd)")
    adam = isinstance(opt, Adam)
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    nodes, inputs, output = dag_view(model, params)
    first_items = list(batches(0) if callable(batches) else batches)
    if not first_items:
        raise TrainError("no batches")
    X0, _, _ = _batch_parts(first_items[0])
    x0 = np.asarray(aops.raw(X0[0] if isinstance(X0, (list, tuple)) else X0))
    work, size = unit_costs([(l_, p) for _, l_, _, _, p in nodes], x0.shape[1:])
    n_cards = len(ops_list)
    if isinstance(stages, (list, tuple)):
        sizes = [int(v) for v in stages]
        if sum(sizes) != len(nodes) or any(v < 1 for v in sizes) or len(sizes) > n_cards:
            raise TrainError(f"stages={list(stages)}: the nodes of each stage, {len(nodes)} in all, at most "
                             f"{n_cards} stages")
    else:
        S = min(int(stages) if stages else n_cards, len(nodes), n_cards)
        mems = [getattr(getattr(o, "dev", None), "memory", None) for o in ops_list]
        budget = 0.8 * min(mems) if all(mems) else None
        sizes = partition(work, size, S, 16.0 if adam else 12.0, budget) or partition(work, size, S)
    S = len(sizes)
    R = max(1, n_cards // S)
    M = max(1, int(micro) if micro else 2 * S)
    cuts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    grid = []
    try:
        for r in range(R):
            row = []
            for s in range(S):
                li, lo_ = live_sets(nodes, inputs, output, cuts[s], cuts[s + 1])
                st = SegStage(ops_list[r * S + s], nodes[cuts[s]:cuts[s + 1]], li, lo_, precision, s, S,
                              min(S - s, M))
                st.gov = thermal.governor(getattr(st.ops, "dev", None))
                st.LOSSB = st.alloc(max(1, M))
                row.append(st)
            grid.append(row)
        for s in range(S):
            group = [grid[r][s] for r in range(R)]
            nP = group[0].net.n
            for k, st in enumerate(group):
                o = st.ops
                st.lo, st.hi = nP * k // R, nP * (k + 1) // R
                ln = max(1, st.hi - st.lo)
                st.st1, st.st2 = st.alloc(ln), (st.alloc(ln) if adam else None)
                st.SQ, st.TMP = st.alloc(1), st.alloc(ln)
                st.COMP = st.alloc(ln) if compensated else None
                for b_ in (st.st1, st.st2, st.COMP):
                    if b_ is not None:
                        o.fill((b_, 0), ln, 0.0)
        t, step, history = int(getattr(opt, "t", 0)), 0, []
        E = int(epochs)
        every = int(every or max(1, E // 10))
        for ep in range(E):
            items = first_items if ep == 0 else list(batches(ep) if callable(batches) else batches)
            losses = []
            per = R * M
            for s0 in range(0, len(items), per):
                chunk = items[s0:s0 + per]
                parts = [chunk[r::R] for r in range(R)]
                lr = float(aops.raw(schedule_fn(step))) if schedule_fn is not None else float(opt.lr)
                losses += _seg_step(grid, parts, inputs, output, loss, loss_fn, len(chunk))
                t += 1
                _update(grid, opt, adam, wd, lr, t, clip, None)
                step += 1
                for row in grid:
                    for st in row:
                        st.gov.tick(st.ops.sync)
            history.append(float(np.mean(losses)))
            if verbose and (ep % every == 0 or ep == E - 1):
                print(f"epoch {ep:4}  loss {history[-1]:.5f}  ({S} stages x {R} replicas, {len(items)} batches)")
        opt.t = t
        out = _gather_params(params, grid[0])
        temps = [st.gov.status() for row in grid for st in row]
    finally:
        for row in grid:
            for st in row:
                st.net.clear_graphs()
                st.close()
    return Record(params=out, history=history, stages=list(sizes), replicas=R, micro_batches=M, gpu_temperatures=temps,
                  device=f"{S * R} GPUs, a {len(nodes)}-node model split into {S} stages"
                         + (f" x {R} replicas" if R > 1 else ""))


def _seg_step(grid, parts, inputs, output, loss, loss_fn, total):
    stop, errors, threads = threading.Event(), [], []
    host = {}
    for r, (row, items) in enumerate(zip(grid, parts)):
        S = len(row)
        if not items:
            for st in row:
                st.ops.fill((st.net.G, 0), st.net.n, 0.0)
            continue
        fwd = [_Link(stop) for _ in range(S + 1)]
        bwd = [_Link(stop) for _ in range(S + 1)]
        sends = [threading.Semaphore(2) for _ in range(S)]
        for st in row:
            args = (st, items, inputs, output, loss, loss_fn, total, fwd, bwd, sends, host, r)
            threads.append(threading.Thread(target=_seg_main, args=(args, stop, errors), daemon=True))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for row in grid:
        for st in row:
            st.net.clear_graphs()
    if errors:
        raise errors[0]
    out = []
    for r, (row, items) in enumerate(zip(grid, parts)):
        if items:
            last = row[-1]
            vals = last.ops.get(last.LOSSB, len(items))
            out += [host[(r, m)] if (r, m) in host else float(vals[m]) for m in range(len(items))]
    return out


def _seg_main(args, stop, errors):
    try:
        _seg_work(*args)
    except _Abort:
        pass
    except BaseException as e:                           # noqa: BLE001
        errors.append(e)
        stop.set()


def _seg_work(st, items, inputs, output, loss, loss_fn, total, fwd, bwd, sends, host, r):
    from . import ops as aops
    from .gpu_train import _rows, _regress, _batch_parts, _as_input, TrainError
    from .stdlib import value_and_grad
    o, net, s = st.ops, st.net, st.index
    o.fill((net.G, 0), net.n, 0.0)
    o.fill((st.LOSSB, 0), max(1, len(items)), 0.0)
    net.round_weights()
    net.train = True
    ring = 0
    parts = [_batch_parts(it) for it in items]
    for kind, m in schedule(s, st.count, len(items)):
        X, Y, g = parts[m]
        net.set_graph(g, keep=True)
        release = None
        if kind == "F":
            a = st.use_slot(m)
            ins = {}
            if st.first:
                xs = list(X) if isinstance(X, (list, tuple)) else [X]
                if len(xs) != len(inputs):
                    raise TrainError(f"the model takes {len(inputs)} inputs {inputs}, a batch gave {len(xs)}")
                for name, arr in zip(inputs, xs):
                    if name in st.live_in:
                        arr = _as_input(arr)
                        v = a.get(("in", name), arr.size)
                        o.put(v[0], arr.reshape(-1), v[1])
                        ins[name] = (v, arr.shape)
            else:
                got, msg = fwd[s].get()
                assert got == m
                for name, view, cnt, shape, src in msg:
                    v = a.get(("recv", name), cnt)
                    o.pull(v, src, view, cnt)
                    ins[name] = (v, shape)
            vals = st.forward_seg(m, ins)
            if not st.last:
                o.sync()
                fwd[s + 1].put((m, [(n, vals[n][0], int(np.prod(vals[n][1])), vals[n][1], o) for n in st.live_out]))
                continue
            out, shape = vals[output]
            rows, C = _rows(shape), shape[-1]
            mm = rows * C
            dout = st.garena.get("dout", mm)
            Yr = np.asarray(aops.raw(Y))
            if loss == "custom":
                oh = o.get(out[0], mm, out[1]).reshape(shape).astype(np.float64)
                lv, gh = value_and_grad(lambda o_: loss_fn(o_, Yr))(oh)
                o.put(dout[0], (np.asarray(aops.raw(gh), np.float32) / total).reshape(-1), dout[1])
                host[(r, m)] = float(aops.raw(lv))
            else:
                ya = np.ascontiguousarray(Yr.astype(np.int32) if loss == "cross_entropy" else
                                          Yr.astype(np.float32)).reshape(-1)
                yd = st.use_slot(m).get("Y", ya.size)
                o.put(yd[0], ya, yd[1])
                if loss == "cross_entropy":
                    if ya.size != rows:
                        raise TrainError(f"cross_entropy: {ya.size} labels for {rows} rows of output")
                    w = 1.0 / max(int((ya >= 0).sum()), 1)
                    o.ce(out, yd, dout, (st.LOSSB, m), rows, C, w / total, w)
                else:
                    if ya.size != mm:
                        raise TrainError(f"{loss}: targets of {ya.size} values for an output of {mm}")
                    _regress(o, loss, out, yd, dout, (st.LOSSB, m), mm, 1.0 / (mm * total), 1.0 / mm)
            seeds = {output: dout}
        else:
            if st.last:
                continue
            got, msg, sem = bwd[s].get()
            assert got == m
            seeds = {}
            for name, view, cnt, src in msg:
                v = st.garena.get(("din", name), cnt)
                o.pull(v, src, view, cnt)
                seeds[name] = v
            release = sem
        if kind == "B" or st.last:
            grads = st.backward_seg(m, seeds)
            if st.first:
                o.sync()
            else:
                sends[s].acquire()
                msg = []
                for name in st.live_in:
                    gv = grads.get(name)
                    if gv is None or name not in st.net.root.shapes:
                        continue
                    cnt = int(np.prod(st.net.root.shapes[name]))
                    buf = st.garena.get(("send", ring, name), cnt)
                    o.pull(buf, o, gv, cnt)
                    msg.append((name, buf, cnt, o))
                ring ^= 1
                o.sync()
                bwd[s - 1].put((m, msg, sends[s]))
            if release is not None:
                release.release()
    o.sync()
    net.set_graph(None, keep=True)
