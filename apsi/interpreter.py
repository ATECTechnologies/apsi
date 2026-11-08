"""Tree-walking reference interpreter for Apsi."""
import operator
import os
import sys
import time
from fractions import Fraction

import numpy as np

from . import ast as A
from . import ops
from . import stdlib
from .autodiff import Node
from .errors import (ApsiError, RuntimeErr, NameErr, TypeCheckError, UnitError, ShapeError,
                     AssertionFailed, CheckFailed)
from .fmt import fmt, type_name
from .parser import parse
from .units import Quantity, DIMLESS, DIM_NAMES, dim_name, unit_string, make
from .values import (Function, BoundMethod, StructType, StructInstance, Record, RangeVal,
                     Namespace, Dtype, UnitNamespace)

np.seterr(all="ignore")
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


class ControlSignal(BaseException):
    pass


class ReturnSignal(ControlSignal):
    def __init__(self, value):
        self.value = value


class BreakSignal(ControlSignal):
    pass


class ContinueSignal(ControlSignal):
    pass


class Env:
    __slots__ = ("vars", "consts", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.consts = set()
        self.parent = parent

    def lookup(self, name):
        e = self
        while e is not None:
            v = e.vars
            if name in v:
                return v[name]
            e = e.parent
        raise KeyError(name)

    def define(self, name, value, const=False):
        self.vars[name] = value
        if const:
            self.consts.add(name)
        else:
            self.consts.discard(name)

    def assign(self, name, value):
        e = self
        while e is not None:
            if name in e.vars:
                if name in e.consts:
                    raise RuntimeErr(f"cannot assign to '{name}': it is a constant "
                                     f"(declared with let or a parameter); use var")
                e.vars[name] = value
                return
            e = e.parent
        raise NameErr(f"'{name}' is not defined; declare it first with let or var")


def _eq(a, b):
    r = a == b
    return r


BINOPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "//": operator.floordiv, "%": operator.mod, "@": operator.matmul, "**": operator.pow,
    "==": _eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
    ">=": operator.ge, "~=": lambda a, b: ops.approx(a, b),
    "in": lambda a, b: a in b, "not in": lambda a, b: a not in b,
    "±": lambda a, b: ops.plusminus(a, b),
}

TENSOR_TYPES = {"Tensor", "Vec", "Mat", "Matrix", "Vec2", "Vec3", "Vec4"}
DTYPE_NAMES = {"f16", "f32", "f64", "i32", "i64", "bool", "bf16", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2",
               "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"}


def wrap_python_error(e):
    if isinstance(e, ZeroDivisionError):
        return RuntimeErr("division by zero")
    msg = str(e)
    if isinstance(e, ValueError) and any(k in msg for k in ("broadcast", "shape", "matmul", "aligned", "dimension")):
        return ShapeError(f"shape mismatch: {msg}")
    if isinstance(e, ValueError) and "truth value" in msg:
        return RuntimeErr("a tensor with several elements cannot be used as true/false; "
                          "use any(...) / all(...) style reductions")
    if isinstance(e, IndexError):
        return RuntimeErr(f"index out of range: {msg}")
    if isinstance(e, KeyError):
        return RuntimeErr(f"key not found: {msg}")
    if isinstance(e, TypeError):
        return RuntimeErr(f"type error: {msg}")
    if isinstance(e, RecursionError):
        return RuntimeErr("maximum recursion depth exceeded")
    return RuntimeErr(f"{type(e).__name__}: {msg}")


class Interpreter:
    def __init__(self, permissions=None, policy=None):
        from .security import Policy
        self.policy = policy if policy is not None else Policy.from_set(
            set(permissions) if permissions is not None else {"fs"})
        self.testing = False
        self.globals = Env()
        core = stdlib.core_builtins()
        self.globals.vars.update(core)
        self.globals.vars["testing"] = lambda: self.testing
        self.modules = stdlib.modules(core)
        from . import interop, crypto, hardware, sci, precision, data, concurrency, web, models
        interop.install(self)
        models.install(self)
        web.install(self)
        data.install(self)
        concurrency.install(self)
        self.modules["sci"] = sci.module()
        from . import layers
        self.modules["layers"] = layers.module(self)
        self.modules["precision"] = precision.module()
        for k, v in self.modules["precision"].items():
            if k not in ("f16", "f32", "f64"):
                self.globals.vars.setdefault(k, v)
        self.globals.vars["einsum"] = sci.einsum
        self.modules["crypto"] = crypto.module()
        self.modules["sys"] = hardware.module()
        self.modules["gpu"] = self.modules["sys"]
        self.exact = False
        self.deriv_stack = []
        self.tests = []
        self.module_cache = {}
        self.importing = []          # files being imported right now (detects import cycles)
        self.dir_stack = [os.getcwd()]
        self._ev = {}
        self._ex = {}
        for cls in A.ALL:
            m = getattr(self, "ev_" + cls.__name__, None)
            if m:
                self._ev[cls] = m
            m = getattr(self, "ex_" + cls.__name__, None)
            if m:
                self._ex[cls] = m

    # ================================================================ entry
    engine = "compile"

    @property
    def permissions(self):
        return self.policy.as_set()

    def run(self, source, filename="<main>", check=True, env=None, engine=None):
        prog = parse(source, cache=not filename.startswith("<"))
        engine = engine or self.engine
        if env is None and engine == "compile":
            return self._run_compiled(prog, filename, check)
        if check:
            from .checker import Checker
            errs = Checker(self, os.path.dirname(os.path.abspath(filename))
                           if not filename.startswith("<") else os.getcwd()).check(prog)
            if errs:
                raise CheckFailed(errs)
        env = env or Env(self.globals)
        base = os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd()
        self.dir_stack.append(base)
        try:
            self.exec_block(prog.body, env)
        except ReturnSignal:
            raise RuntimeErr("'return' outside of a function")
        except (BreakSignal, ContinueSignal):
            raise RuntimeErr("'break'/'continue' outside of a loop")
        finally:
            self.dir_stack.pop()
        return env

    def _check(self, prog, filename):
        from .checker import Checker
        base = os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd()
        errs = Checker(self, base).check(prog)
        if errs:
            raise CheckFailed(errs)
        return base

    def _run_compiled(self, prog, filename, check):
        from .codegen import Compiled
        base = self._check(prog, filename) if check else (
            os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd())
        self.dir_stack.append(base)
        try:
            comp = Compiled(self, prog, filename, base, self.tests)
            self.last_compiled = comp
            comp.run()
        finally:
            self.dir_stack.pop()
        env = Env(self.globals)
        env.vars.update(comp.exports())
        return env

    def _enter_import(self, path):
        if path in self.importing:
            chain = self.importing[self.importing.index(path):] + [path]
            raise NameErr("circular import: " + " -> ".join(os.path.basename(p) for p in chain)
                          + "; move the shared code into a third file that both import")
        self.importing.append(path)

    def import_compiled(self, path):
        path = os.path.normpath(path)
        if path not in self.module_cache:
            if not os.path.exists(path):
                raise NameErr(f"cannot find file '{path}'")
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
            saved = self.tests
            self.tests = []
            self._enter_import(path)
            try:
                menv = self.run(src, path, check=True)
            except ApsiError as e:
                if e.filename is None:
                    e.filename, e.source = path, src
                raise
            finally:
                self.tests = saved
                self.importing.pop()
            self.module_cache[path] = menv
        return {k: v for k, v in self.module_cache[path].vars.items() if not k.startswith("_")}

    def run_tests(self, report=print):
        passed, failed = 0, []
        for item in self.tests:
            if len(item) == 3:
                name, fn, line = item
                t0 = time.perf_counter()
                try:
                    try:
                        fn()
                    except ApsiError:
                        raise
                    except Exception as e:
                        raise self.last_compiled.translate(e, sys.exc_info()[2]) from None
                    report(f"  ✓ {name}  ({(time.perf_counter() - t0) * 1000:.1f} ms)")
                    passed += 1
                except ApsiError as e:
                    report(f"  ✗ {name}\n      {e.kind}: {e.msg}" + (f" (line {e.line})" if e.line else ""))
                    failed.append((name, e))
                continue
            name, body, env, line = item
            t0 = time.perf_counter()
            try:
                self.exec_block(body, Env(env))
                dt = (time.perf_counter() - t0) * 1000
                report(f"  ✓ {name}  ({dt:.1f} ms)")
                passed += 1
            except ApsiError as e:
                report(f"  ✗ {name}\n      {e.kind}: {e.msg}" + (f" (line {e.line})" if e.line else ""))
                failed.append((name, e))
        return passed, failed

    # ================================================================ statements
    def exec_block(self, stmts, env):
        ex = self._ex
        for s in stmts:
            try:
                ex[s.__class__](s, env)
            except ControlSignal:
                raise
            except ApsiError as e:
                if e.line is None:
                    e.line, e.col = s.line, s.col
                raise
            except Exception as e:
                err = wrap_python_error(e)
                err.line, err.col = s.line, s.col
                raise err from None

    def ex_ExprStmt(self, s, env):
        v = self.eval(s.expr, env)
        if getattr(self, "repl_echo", False) and v is not None and env.parent is self.globals:
            print(fmt(v))
            env.define("_", v)

    def ex_Let(self, s, env):
        v = self.eval(s.value, env)
        if len(s.targets) == 1:
            if s.type is not None:
                v = self.check_type(v, s.type, {}, f"'{s.targets[0]}'", env)
            env.define(s.targets[0], v, const=not s.mutable)
        else:
            vals = self._unpack(v, len(s.targets))
            for n, x in zip(s.targets, vals):
                env.define(n, x, const=not s.mutable)

    def _unpack(self, v, n):
        try:
            vals = list(v)
        except TypeError:
            raise RuntimeErr(f"cannot unpack {type_name(v)} into {n} names")
        if len(vals) != n:
            raise RuntimeErr(f"expected {n} values to unpack, got {len(vals)}")
        return vals

    def ex_Assign(self, s, env):
        self.assign(s.target, self.eval(s.value, env), env)

    def assign(self, t, v, env):
        if isinstance(t, A.Name):
            env.assign(t.id, v)
        elif isinstance(t, A.TupleLit):
            vals = self._unpack(v, len(t.items))
            for sub, x in zip(t.items, vals):
                self.assign(sub, x, env)
        elif isinstance(t, A.Attr):
            obj = self.eval(t.obj, env)
            self.set_attr(obj, t.name, v)
        elif isinstance(t, A.Index):
            obj = self.eval(t.obj, env)
            idx = self.eval_index(t.index, env)
            self.set_index(obj, idx, v)
        else:
            raise RuntimeErr("invalid assignment target")

    def set_index(self, obj, idx, v):
        if isinstance(obj, np.ndarray) and isinstance(v, Quantity):
            raise UnitError(f"cannot store {dim_name(v.dim)} into a dimensionless tensor")
        if isinstance(obj, np.ndarray) and isinstance(v, Fraction):
            v = float(v)
        obj[idx] = v

    def set_attr(self, obj, name, v):
        if isinstance(obj, StructInstance):
            if name not in obj.fields:
                raise RuntimeErr(f"{obj.type.name} has no field '{name}'")
            obj.fields[name] = v
        elif isinstance(obj, Record):
            obj.fields[name] = v
        else:
            raise RuntimeErr(f"cannot set attribute '{name}' on {type_name(obj)}")

    def ex_AugAssign(self, s, env):
        fn = BINOPS[s.op]
        t = s.target
        rhs = self.eval(s.value, env)
        if isinstance(t, A.Name):
            env.assign(t.id, fn(self.lookup(t.id, env, t), rhs))
        elif isinstance(t, A.Attr):
            obj = self.eval(t.obj, env)
            self.set_attr(obj, t.name, fn(self.get_attr(obj, t.name), rhs))
        else:
            obj = self.eval(t.obj, env)
            idx = self.eval_index(t.index, env)
            self.set_index(obj, idx, fn(obj[idx], rhs))

    def ex_FnDef(self, s, env):
        f = Function(self, s.name, s.params, s.ret, s.body, s.is_expr, env, s.shape_params)
        for d in reversed(s.decorators):
            deco = self.eval(d, env)
            f = self.call_value(deco, [f], {})
        env.define(s.name, f)

    def ex_StructDef(self, s, env):
        st = StructType(self, s.name, [(p.name, p.type, p.default) for p in s.fields], {}, env)
        env.define(s.name, st)
        for m in s.methods:
            f = Function(self, m.name, m.params, m.ret, m.body, m.is_expr, env, m.shape_params)
            for d in reversed(m.decorators):
                f = self.call_value(self.eval(d, env), [f], {})
            st.methods[m.name] = f

    def ex_If(self, s, env):
        for cond, body in s.branches:
            if self.truthy(self.eval(cond, env)):
                self.exec_block(body, Env(env))
                return
        if s.else_body is not None:
            self.exec_block(s.else_body, Env(env))

    def truthy(self, v):
        if isinstance(v, (np.ndarray, Node)) and np.size(ops.raw(v)) != 1:
            raise RuntimeErr("a tensor with several elements cannot be used as a condition")
        return bool(v)

    def ex_While(self, s, env):
        while self.truthy(self.eval(s.cond, env)):
            try:
                self.exec_block(s.body, Env(env))
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def ex_For(self, s, env):
        it = self.eval(s.iter, env)
        if isinstance(it, (int, float)) and not isinstance(it, bool):
            raise RuntimeErr("cannot iterate over a number; use 0..n or range(n)")
        if isinstance(it, Record):
            it = list(it.fields.items())
        n = len(s.targets)
        for item in it:
            loop_env = Env(env)
            if n == 1:
                loop_env.define(s.targets[0], item, const=True)
            else:
                for name, x in zip(s.targets, self._unpack(item, n)):
                    loop_env.define(name, x, const=True)
            try:
                self.exec_block(s.body, loop_env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def ex_Return(self, s, env):
        raise ReturnSignal(self.eval(s.value, env) if s.value is not None else None)

    def ex_Break(self, s, env):
        raise BreakSignal()

    def ex_Continue(self, s, env):
        raise ContinueSignal()

    def ex_Pass(self, s, env):
        pass

    def ensure_module(self, name, base=None):
        """Load an installed .apsil library the first time `use NAME` names it. True when NAME is available."""
        if name in self.modules:
            return True
        from . import apsil
        try:
            entry = apsil.resolve_use(name, base or self.dir_stack[-1])
        except apsil.PackageError as e:
            raise NameErr(str(e))
        if not entry:
            return False
        with open(entry, encoding="utf-8-sig") as fh:
            src = fh.read()
        saved = self.tests
        self.tests = []
        self._enter_import(entry)
        try:
            menv = self.run(src, entry, check=True, env=Env(self.globals))
        except ApsiError as e:
            if e.filename is None:
                e.filename, e.source = entry, src
            raise
        finally:
            self.importing.pop()
            self.tests = saved
        self.modules[name] = {k: v for k, v in menv.vars.items() if not k.startswith("_")}
        return True

    def ex_Use(self, s, env):
        if s.path not in self.modules and not self.ensure_module(s.path):
            raise NameErr(f"unknown module '{s.path}'. Available: {', '.join(sorted(self.modules))}")
        members = self.modules[s.path]
        if s.alias:
            env.define(s.alias, Namespace(s.path, members))
        else:
            for k, v in members.items():
                env.define(k, v)
            if "." not in s.path and s.path not in members and s.path not in env.vars and s.path not in self.globals.vars:
                env.define(s.path, Namespace(s.path, members))      # `use mylib` also allows mylib.name(...)

    def ex_Import(self, s, env):
        path = s.path
        if not os.path.isabs(path):
            path = os.path.join(self.dir_stack[-1], path)
        path = os.path.normpath(path)
        if path not in self.module_cache:
            if not os.path.exists(path):
                raise NameErr(f"cannot find file '{s.path}'")
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
            self._enter_import(path)
            try:
                menv = self.run(src, path, check=True, env=Env(self.globals))
            except ApsiError as e:
                if e.filename is None:
                    e.filename, e.source = path, src
                raise
            finally:
                self.importing.pop()
            self.module_cache[path] = menv
        menv = self.module_cache[path]
        public = {k: v for k, v in menv.vars.items() if not k.startswith("_")}
        if s.alias:
            env.define(s.alias, Namespace(os.path.basename(path), public))
        else:
            for k, v in public.items():
                env.define(k, v)

    def ex_Test(self, s, env):
        self.tests.append((s.name, s.body, env, s.line))

    def ex_Assert(self, s, env):
        v = self.eval(s.cond, env)
        if isinstance(v, (np.ndarray, Quantity, Node)):
            ok = bool(np.all(ops.raw(v)))
        else:
            ok = bool(v)
        if not ok:
            msg = fmt(self.eval(s.msg, env)) if s.msg is not None else self._assert_detail(s.cond, env)
            raise AssertionFailed(msg, s.line, s.col)

    def _assert_detail(self, cond, env):
        if isinstance(cond, A.Binary) and cond.op in ("==", "!=", "<", "<=", ">", ">=", "~="):
            try:
                a, b = self.eval(cond.left, env), self.eval(cond.right, env)
                return f"assertion failed: {fmt(a)} {cond.op} {fmt(b)}"
            except Exception:
                pass
        return "assertion failed"

    def ex_Deriv(self, s, env):
        if not self.deriv_stack:
            raise RuntimeErr(f"{s.name}' = ... is only allowed inside an evolve block")
        self.deriv_stack[-1][s.name] = self.eval(s.expr, env)

    # ---------------------------------------------------------------- evolve
    def ex_Evolve(self, s, env):
        names = s.states
        state = [self.lookup(n, env, s) for n in names]
        rng = self.eval(s.range, env)
        if not isinstance(rng, RangeVal):
            raise RuntimeErr("evolve needs a time range like 0[s]..10[s]")
        dt = self.eval(s.step, env)
        t0, t1 = rng.start, rng.end
        span = t1 - t0
        try:
            nsteps = span / dt
        except UnitError:
            raise UnitError(f"step has units {dim_name(ops.dim_of(dt))} but the time range has "
                            f"{dim_name(ops.dim_of(t0))}")
        if isinstance(nsteps, Quantity):
            raise UnitError("time range and step must have the same units")
        nsteps = int(round(float(nsteps)))
        if nsteps <= 0:
            raise RuntimeErr("evolve: the time range must be longer than one step")
        every = int(self.eval(s.every, env)) if s.every is not None else 1
        tdim = ops.dim_of(dt)

        body, tvar = s.body, s.tvar
        checked = [False]

        def deriv(t, st):
            e = Env(env)
            e.vars[tvar] = t
            e.consts.add(tvar)
            for n, v in zip(names, st):
                e.vars[n] = v
                e.consts.add(n)
            d = {}
            self.deriv_stack.append(d)
            try:
                self.exec_block(body, e)
            finally:
                self.deriv_stack.pop()
            out = []
            for n in names:
                if n not in d:
                    raise RuntimeErr(f"evolve: no derivative for '{n}' (add a line {n}' = ...)", s.line, s.col)
                out.append(d[n])
            if not checked[0]:
                checked[0] = True
                for n, v, dv in zip(names, st, out):
                    want = ops.dim_of(v) / tdim
                    got = ops.dim_of(dv)
                    if got is not None and got != want:
                        raise UnitError(f"{n}' must have units of {dim_name(want)} "
                                        f"({n} per unit time), but got {dim_name(got)}", s.line, s.col)
            return out

        # ---- fast path: units checked on the first evaluation, then erased
        state_dims = [ops.dim_of(v) for v in state]
        fast = None
        if all(d is not None for d in state_dims) and tdim is not None:
            from .compiler import ErasedCompiler, Unsupported
            try:
                comp = ErasedCompiler(self)
                code = comp.block(body)
                # one compiled block per state that skips the other states' derivatives
                partial = {n: comp.block([b for b in body if not (isinstance(b, A.Deriv) and b.name != n)])
                           for n in names}
                deriv(t0, state)          # full, unit-checked evaluation

                def fast_deriv(t, st, only=None):
                    e = Env(env)
                    e.vars[tvar] = t
                    for n, v in zip(names, st):
                        e.vars[n] = v
                    d = {}
                    self.deriv_stack.append(d)
                    try:
                        (code if only is None else partial[only])(e)
                    finally:
                        self.deriv_stack.pop()
                    if only is not None:
                        return d[only]
                    return [d[n] for n in names]
                raw_state = [ops.raw(v) if not isinstance(v, Quantity) else v.v for v in state]
                fast_deriv(ops.raw(t0) if isinstance(t0, Quantity) else t0, raw_state)   # probe
                fast = fast_deriv
            except Unsupported:
                fast = None
            except ApsiError:
                raise
        if fast is not None:
            deriv = fast
            st_wrap = state_dims
            state = [v.v if isinstance(v, Quantity) else v for v in state]
            t0 = t0.v if isinstance(t0, Quantity) else t0
            dt = dt.v if isinstance(dt, Quantity) else dt
        else:
            st_wrap = None

        def axpy(xs, ds, h):
            return [x + h * d for x, d in zip(xs, ds)]

        rec_t, rec = [], {n: [] for n in names}

        def record(t, st):
            rec_t.append(t)
            for n, v in zip(names, st):
                rec[n].append(v.copy() if isinstance(v, np.ndarray) else
                              (Quantity(v.v.copy(), v.dim) if isinstance(v, Quantity) and isinstance(v.v, np.ndarray) else v))

        def one(t, st, i):
            if fast is not None:
                return deriv(t, st, names[i])
            return deriv(t, st)[i]

        method = s.method
        if method == "verlet" and len(names) != 2:
            raise RuntimeErr("verlet needs exactly two states: position, velocity")
        st = list(state)
        if s.into:
            record(t0, st)
        if method == "rk45":
            from .integrators import rk45
            tol = float(self.eval(s.tol, env)) if s.tol is not None else 1e-8
            counter = [0]

            def on_step(t, cur):
                counter[0] += 1
                if s.into and counter[0] % every == 0:
                    record(t, cur)
            t_end = t0 + nsteps * dt
            st, _, _ = rk45(deriv, t0, t_end, dt, st, tol, on_step)
            if s.into and counter[0] % every != 0:
                record(t_end, st)
            nsteps = 0
        for k in range(nsteps):
            t = t0 + k * dt
            if method == "euler":
                st = axpy(st, deriv(t, st), dt)
            elif method == "midpoint":
                k1 = deriv(t, st)
                st = axpy(st, deriv(t + 0.5 * dt, axpy(st, k1, 0.5 * dt)), dt)
            elif method == "rk4":
                k1 = deriv(t, st)
                k2 = deriv(t + 0.5 * dt, axpy(st, k1, 0.5 * dt))
                k3 = deriv(t + 0.5 * dt, axpy(st, k2, 0.5 * dt))
                k4 = deriv(t + dt, axpy(st, k3, dt))
                st = [x + (dt / 6.0) * (a + 2 * b + 2 * c + d)
                      for x, a, b, c, d in zip(st, k1, k2, k3, k4)]
            elif method == "symplectic":
                for i in reversed(range(len(st))):
                    st[i] = st[i] + dt * one(t, st, i)
            elif method == "verlet":
                x, v = st
                a0 = one(t, [x, v], 1)
                vh = v + (0.5 * dt) * a0
                x = x + dt * one(t, [x, vh], 0)
                v = vh + (0.5 * dt) * one(t + dt, [x, vh], 1)
                st = [x, v]
            if s.into and ((k + 1) % every == 0 or k == nsteps - 1):
                record(t0 + (k + 1) * dt, st)
        if st_wrap is not None:
            st = [make(v, d) for v, d in zip(st, st_wrap)]
        for n, v in zip(names, st):
            env.assign(n, v)
        if s.into:
            ts = ops.stack(rec_t)
            fields = {"t": make(ts, tdim) if st_wrap is not None else ts}
            for n, d in zip(names, state_dims):
                arr = ops.stack(rec[n])
                fields[n] = make(arr, d) if st_wrap is not None else arr
            env.define(s.into, Record(**fields))

    # ================================================================ expressions
    def eval(self, node, env):
        return self._ev[node.__class__](node, env)

    def ev_Num(self, n, env):
        if n.unit is not None:
            text, scale, dim = n.unit
            if dim.dimensionless:
                return n.value * scale
            return Quantity(n.value * scale, dim, (text, scale))
        if self.exact:
            return Fraction(n.text)
        return n.value

    def ev_Str(self, n, env):
        return n.value

    def ev_FStr(self, n, env):
        out = []
        for p in n.parts:
            if isinstance(p, str):
                out.append(p)
            else:
                e, spec = p
                out.append(fmt(self.eval(e, env), True, spec))
        return "".join(out)

    def ev_Bool(self, n, env):
        return n.value

    def ev_NoneLit(self, n, env):
        return None

    def lookup(self, name, env, node):
        try:
            return env.lookup(name)
        except KeyError:
            raise NameErr(f"'{name}' is not defined", node.line, node.col)

    def ev_Name(self, n, env):
        e = env
        name = n.id
        while e is not None:
            v = e.vars
            if name in v:
                return v[name]
            e = e.parent
        raise NameErr(f"'{name}' is not defined", n.line, n.col)

    def ev_Unary(self, n, env):
        v = self.eval(n.operand, env)
        if n.op == "-":
            return -v
        if n.op == "+":
            return +v
        return not self.truthy(v)

    def ev_Binary(self, n, env):
        a = self.eval(n.left, env)
        b = self.eval(n.right, env)
        try:
            if n.op == "**" and isinstance(a, Fraction) and isinstance(b, Fraction) and b.denominator != 1:
                return float(a) ** float(b)
            return BINOPS[n.op](a, b)
        except ControlSignal:
            raise
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            if isinstance(e, TypeError):
                err = RuntimeErr(f"cannot apply '{n.op}' to {type_name(a)} and {type_name(b)}")
            err.line, err.col = n.line, n.col
            raise err from None

    def ev_Logic(self, n, env):
        a = self.eval(n.left, env)
        if n.op == "and":
            return self.eval(n.right, env) if self.truthy(a) else a
        return a if self.truthy(a) else self.eval(n.right, env)

    def ev_IfExpr(self, n, env):
        return self.eval(n.a, env) if self.truthy(self.eval(n.cond, env)) else self.eval(n.b, env)

    def ev_Call(self, n, env):
        f = self.eval(n.func, env)
        args = [self.eval(a, env) for a in n.args]
        kwargs = {k: self.eval(v, env) for k, v in n.kwargs}
        try:
            return self.call_value(f, args, kwargs)
        except ControlSignal:
            raise
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            err.line, err.col = n.line, n.col
            raise err from None

    def call_value(self, f, args, kwargs):
        if isinstance(f, Function):
            return self.call_function(f, args, kwargs)
        if callable(f):
            return f(*args, **kwargs)
        raise RuntimeErr(f"{type_name(f)} is not callable")

    def call_function(self, fn, args, kwargs):
        env = Env(fn.closure)
        bindings = {}
        params = fn.params
        if len(args) > len(params):
            raise RuntimeErr(f"{fn.name}() takes {len(params)} argument(s) but {len(args)} were given")
        kwargs = dict(kwargs)
        prec = fn.precision
        for i, p in enumerate(params):
            if i < len(args):
                v = args[i]
            elif p.name in kwargs:
                v = kwargs.pop(p.name)
            elif p.default is not None:
                v = self.eval(p.default, fn.closure)
            else:
                raise RuntimeErr(f"{fn.name}() is missing argument '{p.name}'")
            if prec in ("f16", "f32", "f64"):
                v = _cast(v, prec)
            if p.type is not None:
                v = self.check_type(v, p.type, bindings, f"argument '{p.name}' of {fn.name}()", fn.closure)
            env.vars[p.name] = v
            env.consts.add(p.name)
        if kwargs:
            raise RuntimeErr(f"{fn.name}() got unexpected argument(s): {', '.join(kwargs)}")
        for sym, val in bindings.items():
            if sym not in env.vars:
                env.vars[sym] = val
                env.consts.add(sym)
        old_exact = self.exact
        if prec == "exact":
            self.exact = True
        elif prec is not None:
            self.exact = False
        try:
            if fn.is_expr:
                result = self.eval(fn.body, env)
            else:
                try:
                    self.exec_block(fn.body, env)
                    result = None
                except ReturnSignal as r:
                    result = r.value
        except ApsiError as e:
            e.trace.append(f"{fn.name}()")
            raise
        finally:
            self.exact = old_exact
        if prec in ("f16", "f32", "f64"):
            result = _cast(result, prec)
        if fn.ret is not None:
            result = self.check_type(result, fn.ret, bindings, f"return value of {fn.name}()", fn.closure)
        return result

    def construct(self, st, args, kwargs):
        if len(args) > len(st.fields):
            raise RuntimeErr(f"{st.name}() takes {len(st.fields)} field(s) but {len(args)} were given")
        kwargs = dict(kwargs)
        values = {}
        bindings = {}
        for i, (name, t, default) in enumerate(st.fields):
            if i < len(args):
                v = args[i]
            elif name in kwargs:
                v = kwargs.pop(name)
            elif default is not None:
                v = self.eval(default, st.env)
            else:
                raise RuntimeErr(f"{st.name}() is missing field '{name}'")
            if t is not None:
                v = self.check_type(v, t, bindings, f"field '{name}' of {st.name}", st.env)
            values[name] = v
        if kwargs:
            raise RuntimeErr(f"{st.name} has no field(s): {', '.join(kwargs)}")
        return StructInstance(st, values)

    def ev_Index(self, n, env):
        obj = self.eval(n.obj, env)
        idx = self.eval_index(n.index, env)
        try:
            if isinstance(obj, (list, tuple, str)) and isinstance(idx, float) and idx.is_integer():
                idx = int(idx)
            return obj[idx]
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            err.line, err.col = n.line, n.col
            raise err from None

    def eval_index(self, node, env):
        if isinstance(node, A.Slice):
            return slice(*(self.eval(x, env) if x is not None else None
                           for x in (node.lo, node.hi, node.step)))
        if isinstance(node, A.TupleLit):
            return tuple(self.eval_index(x, env) for x in node.items)
        v = self.eval(node, env)
        if isinstance(v, list) and v and all(isinstance(x, (int, np.integer)) for x in v):
            return np.asarray(v)
        return v

    def ev_Slice(self, n, env):
        return self.eval_index(n, env)

    def ev_Attr(self, n, env):
        obj = self.eval(n.obj, env)
        try:
            return self.get_attr(obj, n.name)
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise

    def get_attr(self, obj, name):
        if isinstance(obj, StructInstance):
            if name in obj.fields:
                return obj.fields[name]
            m = obj.type.methods.get(name)
            if m is not None:
                return BoundMethod(m, obj)
            raise RuntimeErr(f"{obj.type.name} has no field or method '{name}'")
        if isinstance(obj, Record):
            if name in obj.fields:
                return obj.fields[name]
            raise RuntimeErr(f"record has no field '{name}' (fields: {', '.join(obj.fields)})")
        if isinstance(obj, Namespace):
            if name in obj.members:
                return obj.members[name]
            raise NameErr(f"module '{obj.name}' has no member '{name}'")
        if isinstance(obj, UnitNamespace):
            return obj.get(name)
        if name.startswith("_"):
            raise RuntimeErr(f"cannot access private attribute '{name}'")
        if isinstance(obj, np.ndarray) and name == "shape":
            return tuple(obj.shape)
        try:
            return getattr(obj, name)
        except AttributeError:
            raise RuntimeErr(f"{type_name(obj)} has no attribute '{name}'")

    def ev_ListLit(self, n, env):
        return [self.eval(x, env) for x in n.items]

    def ev_TupleLit(self, n, env):
        return tuple(self.eval(x, env) for x in n.items)

    def ev_DictLit(self, n, env):
        return {self.eval(k, env): self.eval(v, env) for k, v in n.pairs}

    def ev_ListComp(self, n, env):
        out = []
        it = self.eval(n.iter, env)
        k = len(n.targets)
        for item in it:
            e = Env(env)
            if k == 1:
                e.vars[n.targets[0]] = item
            else:
                for name, x in zip(n.targets, self._unpack(item, k)):
                    e.vars[name] = x
            if n.cond is None or self.truthy(self.eval(n.cond, e)):
                out.append(self.eval(n.elem, e))
        return out

    def ev_DictComp(self, n, env):
        out = {}
        k = len(n.targets)
        for item in self.eval(n.iter, env):
            e = Env(env)
            if k == 1:
                e.vars[n.targets[0]] = item
            else:
                for name, x in zip(n.targets, self._unpack(item, k)):
                    e.vars[name] = x
            if n.cond is None or self.truthy(self.eval(n.cond, e)):
                out[self.eval(n.key, e)] = self.eval(n.value, e)
        return out

    def ev_Lambda(self, n, env):
        return Function(self, "<lambda>", n.params, None, n.body, True, env)

    def ev_Range(self, n, env):
        return RangeVal(self.eval(n.start, env), self.eval(n.end, env), n.inclusive)

    def ev_As(self, n, env):
        v = self.eval(n.expr, env)
        text, scale, dim = n.unit
        have = ops.dim_of(v)
        if have is None:
            raise RuntimeErr(f"cannot convert {type_name(v)} to [{text}]")
        if have != dim:
            raise UnitError(f"cannot express {dim_name(have)} in [{text}] ({dim_name(dim)})",
                            n.line, n.col)
        if dim.dimensionless:
            return v / scale
        return Quantity(v.v, v.dim, (text, scale))

    # ================================================================ types
    def check_type(self, v, t, bindings, what, env):
        if isinstance(t, A.TUnit):
            have = ops.dim_of(v)
            if have != t.dim:
                raise UnitError(f"{what} must be in [{t.text}] ({dim_name(t.dim)}), "
                                f"got {self._describe(v)}")
            return v
        name = t.name
        if name == "Any":
            return v
        if name in ("Float", "Number", "Real"):
            if isinstance(v, Quantity):
                raise UnitError(f"{what} must be a dimensionless Float, got {self._describe(v)}")
            if isinstance(v, (int, float, Fraction, np.number, Node)) and not isinstance(v, bool):
                return v
            raise TypeCheckError(f"{what} must be a Float, got {type_name(v)}")
        if name == "Int":
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                return v
            raise TypeCheckError(f"{what} must be an Int, got {type_name(v)}")
        if name == "Bool":
            if isinstance(v, (bool, np.bool_)):
                return v
            raise TypeCheckError(f"{what} must be a Bool, got {type_name(v)}")
        if name == "Str":
            if isinstance(v, str):
                return v
            raise TypeCheckError(f"{what} must be a Str, got {type_name(v)}")
        simple = {"List": list, "Tuple": tuple, "Dict": dict, "Record": Record}
        if name in simple:
            if isinstance(v, simple[name]):
                return v
            raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
        if name == "Fn":
            if callable(v):
                return v
            raise TypeCheckError(f"{what} must be a function, got {type_name(v)}")
        if name in DIM_NAMES:
            have = ops.dim_of(v)
            if have is None:
                raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
            if have != DIM_NAMES[name]:
                raise UnitError(f"{what} must be {dim_name(DIM_NAMES[name])}, got {self._describe(v)}")
            return v
        if name in TENSOR_TYPES:
            return self._check_tensor(v, t, bindings, what)
        try:
            st = env.lookup(name)
        except KeyError:
            st = None
        if isinstance(st, StructType):
            if isinstance(v, StructInstance) and v.type is st:
                return v
            raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
        raise TypeCheckError(f"unknown type '{name}'")

    def _describe(self, v):
        d = ops.dim_of(v)
        if isinstance(v, Quantity) and np.ndim(ops.raw(v)) == 0:
            return f"{dim_name(d)} ({fmt(v)})"
        return dim_name(d) if d is not None else type_name(v)

    def _check_tensor(self, v, t, bindings, what):
        specs, dim, dtype = [], None, None
        if t.name in ("Vec2", "Vec3", "Vec4"):
            specs.append(int(t.name[3]))
        for a in t.args:
            if isinstance(a, int):
                specs.append(a)
            elif isinstance(a, A.TUnit):
                dim = a.dim
            elif a.name in DIM_NAMES:
                dim = DIM_NAMES[a.name]
            elif a.name in DTYPE_NAMES:
                dtype = a.name
            else:
                specs.append(a.name)
        r = ops.raw(v)
        if not isinstance(r, np.ndarray):
            if isinstance(v, list):
                raise TypeCheckError(f"{what} must be a Tensor, got a List; wrap it with tensor([...])")
            if not (np.ndim(r) == 0 and not specs and isinstance(r, (int, float, np.number))):
                raise TypeCheckError(f"{what} must be a Tensor, got {type_name(v)}")
        shape = tuple(np.shape(r))
        if specs:
            if len(shape) != len(specs):
                want = ", ".join(str(s) for s in specs)
                raise ShapeError(f"{what} must have shape [{want}] (rank {len(specs)}), got shape {shape}")
            for s, actual in zip(specs, shape):
                if isinstance(s, int):
                    if s != actual:
                        raise ShapeError(f"{what}: expected size {s}, got {actual} (shape {shape})")
                elif s == "_":
                    continue
                elif s in bindings:
                    if bindings[s] != actual:
                        raise ShapeError(f"{what}: dimension {s} is {actual}, but {s} was already "
                                         f"bound to {bindings[s]} (shape {shape})")
                else:
                    bindings[s] = actual
        if dim is not None:
            have = ops.dim_of(v)
            if have != dim:
                raise UnitError(f"{what} must hold {dim_name(dim)}, got {dim_name(have)}")
        elif isinstance(v, Quantity) and t.name not in ("Vec2", "Vec3", "Vec4", "Vec"):
            pass
        if dtype is not None and isinstance(r, np.ndarray):
            v = _cast(v, dtype)
        return v


def _cast(v, prec):
    if prec not in ("f16", "f32", "f64", "i32", "i64", "bool"):
        from .precision import round_to
        return round_to(v, prec) if np.ndim(ops.raw(v)) else v
    npd = {"f16": np.float16, "f32": np.float32, "f64": np.float64,
           "i32": np.int32, "i64": np.int64, "bool": np.bool_}[prec]
    if isinstance(v, np.ndarray):
        return v.astype(npd) if v.dtype != npd else v
    if isinstance(v, Quantity) and isinstance(v.v, np.ndarray):
        return Quantity(v.v.astype(npd), v.dim, v.disp)
    return v


def run_source(source, filename="<main>", check=True):
    return Interpreter().run(source, filename, check)


def run_file(path, check=True):
    with open(path, encoding="utf-8-sig") as fh:
        return Interpreter().run(fh.read(), path, check)
