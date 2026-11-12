"""Physics-informed networks: train a network so that its output satisfies equations written with its derivatives.

    derivatives(model, params, x)  ->  Record(u, du, d2u)
        u   [N, m]     the network at the points x [N, d]
        du  [N, d, m]  du[:, i, k]  = d u_k / d x_i
        d2u [N, d, m]  d2u[:, i, k] = d2 u_k / d x_i^2

    fit(model, params, points, loss, epochs=2000, opt=adam(1e-3), backend="auto", ...)
        loss(x, u, du, d2u) -> a number: the equation's residual at the points, the boundary and initial conditions
        (select their points from x), measured data; any formula of these. Returns Record(params, history, backend).

On the GPU ("gpu") a network of linear layers and smooth activations (tanh, sin, cos, sigmoid, silu, softplus) runs in
Taylor mode: every layer carries the values, their first and their second derivatives along each input in one stack
(one matrix product a layer for all of them), and the gradient flows back through the same stack (kernels with the
activations' first three derivatives). Only u, du, d2u and their gradients cross to the processor, where the loss is
evaluated with Apsi's autodiff. On the processor ("numpy") the derivatives come from Apsi's autodiff of any order, for
any model. The two give the same training (tested to float32 precision)."""
import math
import time

import numpy as np

from . import ops
from .values import Record, tree_leaves, tree_map

KINDS = {"tanh": 0, "sin": 1, "sigmoid": 2, "silu": 3, "swish": 3, "softplus": 4, "linear": 5, "identity": 5,
         "cos": 6}


class PinnError(Exception):
    pass


# ============================================================================== derivatives with autodiff (any model)
def _jets(model, p, x, ctx):
    """u, du, d2u of the model at x, as differentiable values (the processor's autodiff, any order)."""
    from .stdlib import grad
    X = np.asarray(ops.raw(x), np.float64)
    N, d = X.shape
    u = model(p, X, **ctx)
    m = int(np.shape(ops.raw(u))[1])
    eye_m, eye_d = np.eye(m), np.eye(d)
    dus, d2us = [], []
    for k in range(m):
        ek = eye_m[:, k]

        def uk(Z, ek=ek):
            return ops.sum_(model(p, Z, **ctx) @ ek)
        g = grad(uk)                                       # du_k / dx  [N, d]
        dk = g(X)
        dus.append(ops.reshape(dk, N, d, 1))
        cols = []
        for i in range(d):
            ei = eye_d[:, i]

            def gi(Z, ei=ei, g=g):
                return ops.sum_(g(Z) @ ei)
            cols.append(ops.reshape(grad(gi)(X) @ ei, N, 1, 1))
        d2us.append(ops.concat(cols, 1))
    return u, ops.concat(dus, 2), ops.concat(d2us, 2)


def derivatives(model, params, x, graph=None):
    """Record(u, du, d2u): the network and its first and second derivatives along each input at the points x."""
    ctx = {"graph": graph} if graph is not None else {}
    u, du, d2u = _jets(model, params, x, ctx)
    return Record(u=np.asarray(ops.raw(u)), du=np.asarray(ops.raw(du)), d2u=np.asarray(ops.raw(d2u)))


# ============================================================================== the GPU path: Taylor mode
def taylor_plan(model):
    """The model as [("linear", layer) | ("act", kind)], or the reason it is not a Taylor-mode network."""
    from . import layers as L
    out = []

    def walk(l_):
        if isinstance(l_, L.Sequential):
            for x in l_.layers:
                walk(x)
        elif isinstance(l_, L.Linear):
            out.append(("linear", l_))
        elif isinstance(l_, L.Activation):
            name = str(l_.name).lower()
            if name not in KINDS:
                raise PinnError(f"activation '{l_.name}' has no Taylor kernel ({', '.join(KINDS)}): "
                                "backend=\"numpy\" trains it")
            out.append(("act", KINDS[name]))
        elif isinstance(l_, L.Dropout) and not l_.rate:
            pass
        else:
            raise PinnError(f"layer '{getattr(l_, 'kind', type(l_).__name__)}' is not a Taylor-mode layer on the GPU "
                            "(linear layers and smooth activations): backend=\"numpy\" trains it")
    walk(model)
    return out


class _Taylor:
    """The network on a card in Taylor mode: stacks of (1 + 2 d) blocks of N rows."""

    def __init__(self, o, model, params, X, compensated=True):
        from . import gpu_train as GT
        self.o, self.GT = o, GT
        self.plan = taylor_plan(model)
        self.N, self.d = X.shape
        self.R = (1 + 2 * self.d) * self.N
        lin = [l_ for kind, l_ in self.plan if kind == "linear"]
        p_lin = _linear_params(model, params)
        if len(p_lin) != len(lin):
            raise PinnError("could not match the parameters to the linear layers")
        Ws = [np.asarray(ops.raw(pl.W), np.float32) for pl in p_lin]
        bs = [np.asarray(ops.raw(pl.b), np.float32) if getattr(l_, "bias", True) and hasattr(pl, "b") else None
              for l_, pl in zip(lin, p_lin)]
        off, self.offsets = 0, []
        for W in Ws:                                     # the matrices first (weight decay), then the biases
            self.offsets.append([off, W.shape, None])
            off += W.size
        self.n_decay = off
        for i, b in enumerate(bs):
            if b is not None:
                self.offsets[i][2] = off
                off += b.size
        self.n = off
        self.leaves = list(zip(p_lin, Ws, bs))
        flat = np.concatenate([W.reshape(-1) for W in Ws] + [b.reshape(-1) for b in bs if b is not None])
        flat = flat.astype(np.float32)
        self.P, self.G = o.alloc(self.n), o.alloc(self.n)
        o.put(self.P, flat)
        o.fill((self.G, 0), self.n, 0.0)
        self.M1, self.M2 = o.alloc(self.n), o.alloc(self.n)
        o.fill((self.M1, 0), self.n, 0.0)
        o.fill((self.M2, 0), self.n, 0.0)
        self.SQ = o.alloc(1)
        self.COMP = None
        if compensated:
            self.COMP = o.alloc(self.n)
            o.fill((self.COMP, 0), self.n, 0.0)
        # the input stack: x, one unit vector per input, zeros
        S0 = np.zeros((1 + 2 * self.d, self.N, self.d), np.float32)
        S0[0] = X
        for i in range(self.d):
            S0[1 + i, :, i] = 1.0
        self.S0 = o.alloc(S0.size)
        o.put(self.S0, S0.reshape(-1))
        self.bufs = []                                   # per step of the plan: its output stack
        width = self.d
        self.widths = []
        for kind, l_ in self.plan:
            if kind == "linear":
                width = l_.n_out
            self.widths.append(width)
            self.bufs.append(o.alloc(self.R * width))
        self.grads = [o.alloc(self.R * w) for w in self.widths]
        self.m = width

    def close(self):
        o = self.o
        for b in [self.P, self.G, self.M1, self.M2, self.SQ, self.S0] + self.bufs + self.grads + \
                ([self.COMP] if self.COMP is not None else []):
            o.free(b)

    def forward(self):
        o, R, N, d = self.o, self.R, self.N, self.d
        x, w_in = (self.S0, 0), self.d
        li = 0
        self.inputs = []
        for (kind, l_), out, w in zip(self.plan, self.bufs, self.widths):
            self.inputs.append((x, w_in))
            if kind == "linear":
                woff, shape, boff = self.offsets[li]
                o.gemm(x, (self.P, woff), (out, 0), R, w, w_in, w_in, 1, w, 1, w, passes=3)
                if boff is not None:
                    o.rowbias((out, 0), (self.P, boff), N, w)
                li += 1
            else:
                o.taylor_f((self._pre(kind_index=len(self.inputs) - 1), 0), (out, 0), N, w, d, l_)
            x, w_in = (out, 0), w
        flat = o.get(self.bufs[-1], R * self.m).reshape(1 + 2 * d, N, self.m)
        return flat[0], flat[1:1 + d].transpose(1, 0, 2), flat[1 + d:].transpose(1, 0, 2)

    def _pre(self, kind_index):
        return self.inputs[kind_index][0][0]

    def backward(self, gu, gdu, gd2u):
        o, R, N, d = self.o, self.R, self.N, self.d
        stack = np.concatenate([gu[None], np.transpose(gdu, (1, 0, 2)), np.transpose(gd2u, (1, 0, 2))], 0)
        o.put(self.grads[-1], stack.astype(np.float32).reshape(-1))
        li = sum(1 for k_, _ in self.plan if k_ == "linear") - 1
        for idx in range(len(self.plan) - 1, -1, -1):
            kind, l_ = self.plan[idx]
            (x, w_in), w = self.inputs[idx], self.widths[idx]
            dy = (self.grads[idx], 0)
            dx = (self.grads[idx - 1], 0) if idx > 0 else None
            if kind == "linear":
                woff, shape, boff = self.offsets[li]
                o.gemm(x, dy, (self.G, woff), w_in, w, R, 1, w_in, w, 1, w, beta=True, passes=3)
                if boff is not None:
                    o.colsum(dy, None, None, None, None, (self.G, boff), N, w, 0)
                if dx is not None:
                    o.gemm(dy, (self.P, woff), dx, R, w_in, w, w, 1, 1, w, w_in, passes=3)
                li -= 1
            elif dx is not None:
                o.taylor_b(x, dy, dx, N, w, d, l_)

    def params_back(self, params, model):
        flat = self.o.get(self.P, self.n)
        got = {}
        for (pl, W, b), (woff, shape, boff) in zip(self.leaves, self.offsets):
            got[id(pl.W)] = flat[woff:woff + W.size].reshape(W.shape)
            if b is not None:
                got[id(pl.b)] = flat[boff:boff + b.size].reshape(b.shape)
        return tree_map(lambda a: got.get(id(a), a) if isinstance(a, np.ndarray) else a, params)


def _linear_params(model, params):
    """The parameter records of the linear layers, in order."""
    from . import layers as L
    out = []

    def walk(l_, p):
        if isinstance(l_, L.Sequential):
            for x, q in zip(l_.layers, p):
                walk(x, q)
        elif isinstance(l_, L.Linear):
            out.append(p)
    walk(model, params)
    return out


# ============================================================================== training
def fit(model, params, points, loss, epochs=2000, opt=None, backend="auto", device="cuda", verbose=True,
        schedule=None, clip=None, every=None, ops_=None, compensated=True, graph=None):
    """Train so that loss(x, u, du, d2u) gets small; see the module's description."""
    from .stdlib import Adam, value_and_grad
    X = np.ascontiguousarray(np.asarray(ops.raw(points), np.float64))
    if X.ndim == 1:
        X = X[:, None]
    opt = opt or Adam(1e-3)
    E = int(epochs)
    every = int(every or max(1, E // 10))
    which = str(backend or "auto").lower()
    history = []
    if which in ("auto", "gpu"):
        try:
            taylor_plan(model)
            if graph is not None:
                raise PinnError("graph models train with backend=\"numpy\"")
            if ops_ is None:
                from . import gpu_train as GT
                ops_, why = GT.device(device)
                if ops_ is None:
                    raise PinnError(f"no GPU: {why}")
        except PinnError:
            if which == "gpu":
                raise
            which = "numpy"
        else:
            which = "gpu"
    t0 = time.perf_counter()
    if which == "gpu":
        if not isinstance(opt, Adam):
            raise PinnError("the GPU path takes adam / adamw")
        from .stdlib import AdamW
        wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
        net = _Taylor(ops_, model, params, X.astype(np.float32), compensated)
        o = ops_
        try:
            t = int(getattr(opt, "t", 0))
            for ep in range(E):
                u, du, d2u = net.forward()
                lv, g = value_and_grad(lambda a, b_, c: loss(X, a, b_, c), wrt=(0, 1, 2))(
                    u.astype(np.float64), du.astype(np.float64), d2u.astype(np.float64))
                lv = float(ops.raw(lv))
                if not math.isfinite(lv):
                    raise PinnError(f"the loss became {lv} at step {ep} (a smaller learning rate)")
                history.append(lv)
                net.backward(*[np.asarray(ops.raw(x), np.float64) for x in g])
                t += 1
                lr = float(ops.raw(schedule(ep))) if schedule is not None else float(opt.lr)
                o.fill((net.SQ, 0), 1, 0.0)
                if clip:
                    o.sumsq((net.G, 0), net.n, (net.SQ, 0))
                o.adam((net.P, 0), (net.G, 0), (net.M1, 0), (net.M2, 0), (net.SQ, 0), net.n, net.n_decay, lr, opt.b1,
                       opt.b2, opt.eps, 1 - opt.b1 ** t, 1 - opt.b2 ** t, wd, float(clip or 0.0),
                       (net.COMP, 0) if net.COMP is not None else None)
                if verbose and (ep % every == 0 or ep == E - 1):
                    print(f"step {ep:5}  loss {lv:.4e}  (GPU, Taylor mode)")
            opt.t = t
            out = net.params_back(params, model)
        finally:
            net.close()
        return Record(params=out, history=history, backend="gpu", seconds=round(time.perf_counter() - t0, 3))
    # the processor: Apsi's autodiff of any order
    ctx = {"graph": graph} if graph is not None else {}

    def objective(p):
        u, du, d2u = _jets(model, p, X, ctx)
        return loss(X, u, du, d2u)
    p = params
    for ep in range(E):
        lv, g = value_and_grad(objective)(p)
        lv = float(ops.raw(lv))
        if not math.isfinite(lv):
            raise PinnError(f"the loss became {lv} at step {ep} (a smaller learning rate)")
        history.append(lv)
        if schedule is not None:
            opt.lr = float(ops.raw(schedule(ep)))
        if clip:
            norm = math.sqrt(sum(float(np.sum(np.asarray(ops.raw(a)) ** 2)) for a in tree_leaves(g)))
            if norm > clip:
                g = tree_map(lambda a: a * (clip / norm), g)
        p = opt.step(p, g)
        if verbose and (ep % every == 0 or ep == E - 1):
            print(f"step {ep:5}  loss {lv:.4e}  (processor)")
    return Record(params=p, history=history, backend="numpy", seconds=round(time.perf_counter() - t0, 3))
