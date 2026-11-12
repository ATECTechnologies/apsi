"""`use dataframe` — a fast, low-memory columnar table engine (filter, select, group by, join, sort, window).

    use dataframe as df
    let sales = df.read_csv("sales.csv")                         # typed columns, parsed once
    let big = sales.filter((df.col("amount") > 100) & (df.col("region") == "EU"))
    let by = big.group_by("customer").agg(total=df.col("amount").sum(), n=df.count(), avg=df.col("amount").mean())
    let top = by.sort("total", descending=true).head(10)
    let full = top.join(customers, on="customer", how="left")
    print(full)
    full.write_csv("top.csv")

Columns are contiguous typed arrays (int64, float64, bool, datetime64, or dictionary-encoded strings), so a column of
a million numbers takes 8 MB — no per-cell objects. Operations are whole-column kernels: group-by factorises the keys
once and reduces with bincount / reduceat, joins use hash-free sorted matching on the factorised keys, filters are
boolean masks, and strings are compared through their dictionary codes. Expressions (`col("a") * 2 + col("b")`)
are evaluated column-wise without Python loops.
"""
import csv
import io

import numpy as np

from .errors import RuntimeErr


# ======================================================================================== columns
class Strings:
    """A dictionary-encoded string column: codes (int32) into a list of unique values (fast compare/group/join)."""
    __slots__ = ("codes", "values", "_index")

    def __init__(self, codes, values):
        self.codes = codes
        self.values = values
        self._index = None

    @staticmethod
    def from_list(items):
        index, values = {}, []
        codes = np.empty(len(items), dtype=np.int32)
        get = index.get
        for i, s in enumerate(items):
            c = get(s)
            if c is None:
                c = index[s] = len(values)
                values.append(s)
            codes[i] = c
        out = Strings(codes, values)
        out._index = index
        return out

    def index(self):
        if self._index is None:
            self._index = {v: i for i, v in enumerate(self.values)}
        return self._index

    def __len__(self):
        return len(self.codes)

    def take(self, idx):
        return Strings(self.codes[idx], self.values)

    def to_numpy(self):
        vals = np.empty(len(self.values) + 1, dtype=object)
        vals[:len(self.values)] = self.values
        vals[-1] = None
        return vals[self.codes]

    def tolist(self):
        v = self.values
        return [v[c] if c >= 0 else None for c in self.codes.tolist()]

    def compact(self):
        used = np.unique(self.codes[self.codes >= 0]) if len(self.codes) else np.array([], np.int32)
        if len(used) == len(self.values):
            return self
        remap = np.full(len(self.values) + 1, -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        return Strings(remap[self.codes], [self.values[i] for i in used.tolist()])


def _col(x, n=None):
    """Any column-like value as a column: numpy array or Strings."""
    if isinstance(x, Strings):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype.kind == "U":
            return _infer(list(x))
        return x
    if n is not None and not isinstance(x, (list, tuple, range)):
        return _col([x] * n)
    return _infer(list(x))


def _infer(items):
    """A typed column from Python values (ints → int64, floats → float64, bools, strings → dictionary)."""
    kinds = set()
    for v in items:
        if v is None or (isinstance(v, float) and v != v):
            kinds.add("null")
        elif isinstance(v, bool):
            kinds.add("bool")
        elif isinstance(v, (int, np.integer)):
            kinds.add("int")
        elif isinstance(v, (float, np.floating)):
            kinds.add("float")
        else:
            kinds.add("str")
        if len(kinds) > 2:
            break
    k = kinds - {"null"}
    if k == {"bool"} and "null" not in kinds:
        return np.array(items, dtype=bool)
    if k == {"int"} and "null" not in kinds:
        return np.array(items, dtype=np.int64)
    if k and k <= {"int", "float", "bool"}:
        return np.array([np.nan if v is None else v for v in items], dtype=np.float64)
    if not k:
        return np.full(len(items), np.nan)
    s = Strings.from_list([None if v is None else str(v) for v in items])
    if None in s.index():
        nul = s.index()[None]
        codes = s.codes.copy()
        codes[codes == nul] = -1
        return Strings(codes, s.values).compact()
    return s


def _len(c):
    return len(c)


def _take(c, idx):
    return c.take(idx) if isinstance(c, Strings) else c[idx]


def _dtype_name(c):
    if isinstance(c, Strings):
        return "str"
    k = c.dtype.kind
    return {"i": "int", "u": "int", "f": "float", "b": "bool", "M": "datetime", "m": "duration"}.get(k, str(c.dtype))


# ======================================================================================== expressions
class Expr:
    """A column expression, evaluated against a DataFrame: col("a") * 2 > col("b")."""

    def __init__(self, fn, name, agg=None):
        self.fn = fn
        self.name = name
        self.agg = agg                       # (kind, inner Expr) for aggregations

    def eval(self, df):
        return self.fn(df)

    def alias(self, name):
        return Expr(self.fn, name, self.agg and (self.agg[0], self.agg[1]))

    # arithmetic / comparison
    def _bin(self, other, op, sym):
        o = other if isinstance(other, Expr) else lit(other)

        def f(df):
            a, b = self.eval(df), o.eval(df)
            return _binop(a, b, op, sym)
        return Expr(f, self.name)

    def __add__(self, o): return self._bin(o, np.add, "+")
    def __radd__(self, o): return lit(o)._bin(self, np.add, "+")
    def __sub__(self, o): return self._bin(o, np.subtract, "-")
    def __rsub__(self, o): return lit(o)._bin(self, np.subtract, "-")
    def __mul__(self, o): return self._bin(o, np.multiply, "*")
    def __rmul__(self, o): return lit(o)._bin(self, np.multiply, "*")
    def __truediv__(self, o): return self._bin(o, np.true_divide, "/")
    def __rtruediv__(self, o): return lit(o)._bin(self, np.true_divide, "/")
    def __floordiv__(self, o): return self._bin(o, np.floor_divide, "//")
    def __mod__(self, o): return self._bin(o, np.mod, "%")
    def __pow__(self, o): return self._bin(o, np.power, "**")
    def __eq__(self, o): return self._bin(o, np.equal, "==")
    def __ne__(self, o): return self._bin(o, np.not_equal, "!=")
    def __lt__(self, o): return self._bin(o, np.less, "<")
    def __le__(self, o): return self._bin(o, np.less_equal, "<=")
    def __gt__(self, o): return self._bin(o, np.greater, ">")
    def __ge__(self, o): return self._bin(o, np.greater_equal, ">=")
    def __and__(self, o): return self._bin(o, np.logical_and, "&")
    def __or__(self, o): return self._bin(o, np.logical_or, "|")
    def __invert__(self): return Expr(lambda df: ~_bool(self.eval(df)), self.name)
    def __neg__(self): return Expr(lambda df: -self.eval(df), self.name)
    __hash__ = object.__hash__

    def is_in(self, values):
        vals = list(values)

        def f(df):
            a = self.eval(df)
            if isinstance(a, Strings):
                idx = a.index()
                codes = np.array([idx[v] for v in vals if v in idx], dtype=np.int32)
                return np.isin(a.codes, codes)
            return np.isin(a, vals)
        return Expr(f, self.name)

    def is_null(self):
        def f(df):
            a = self.eval(df)
            if isinstance(a, Strings):
                return a.codes < 0
            return np.isnan(a) if a.dtype.kind == "f" else np.zeros(len(a), bool)
        return Expr(f, self.name)

    def fill_null(self, v):
        def f(df):
            a = self.eval(df)
            if isinstance(a, Strings):
                return _infer([v if x is None else x for x in a.tolist()])
            if a.dtype.kind == "f":
                return np.where(np.isnan(a), v, a)
            return a
        return Expr(f, self.name)

    def map(self, fn):
        """Apply a function to every value (Python speed; prefer column operations)."""
        return Expr(lambda df: _infer([fn(x) for x in _tolist(self.eval(df))]), self.name)

    def abs(self): return Expr(lambda df: np.abs(self.eval(df)), self.name)
    def sqrt(self): return Expr(lambda df: np.sqrt(self.eval(df)), self.name)
    def log(self): return Expr(lambda df: np.log(self.eval(df)), self.name)
    def exp(self): return Expr(lambda df: np.exp(self.eval(df)), self.name)
    def round(self, d=0): return Expr(lambda df: np.round(self.eval(df), d), self.name)
    def cast(self, t):
        dt = {"int": np.int64, "float": np.float64, "bool": bool}.get(t)

        def f(df):
            a = self.eval(df)
            if t == "str":
                return _infer([None if x is None else str(x) for x in _tolist(a)])
            if isinstance(a, Strings):
                return np.array([dt(x) for x in a.tolist()], dtype=dt)
            return a.astype(dt)
        return Expr(f, self.name)

    # string helpers (work on the dictionary: cost ~ number of distinct values)
    def _str(self, fn, result="str"):
        def f(df):
            a = self.eval(df)
            if not isinstance(a, Strings):
                raise RuntimeErr("string operation on a non-string column")
            mapped = [fn(v) for v in a.values]
            if result == "bool":
                lut = np.array(mapped + [False], dtype=bool)
                return lut[a.codes]
            if result == "num":
                lut = np.array(mapped + [np.nan], dtype=np.float64)
                return lut[a.codes]
            return Strings(a.codes, mapped).compact() if len(set(mapped)) < len(mapped) else Strings(a.codes, mapped)
        return Expr(f, self.name)

    def str_lower(self): return self._str(str.lower)
    def str_upper(self): return self._str(str.upper)
    def str_strip(self): return self._str(str.strip)
    def str_len(self): return self._str(len, "num")
    def str_contains(self, sub): return self._str(lambda v: sub in v, "bool")
    def str_startswith(self, p): return self._str(lambda v: v.startswith(p), "bool")
    def str_endswith(self, p): return self._str(lambda v: v.endswith(p), "bool")
    def str_replace(self, a, b): return self._str(lambda v: v.replace(a, b))

    # aggregations (used in group_by().agg() and select())
    def _agg(self, kind, name=None):
        e = Expr(lambda df: _reduce_all(kind, self.eval(df)), name or self.name, (kind, self))
        return e

    def sum(self): return self._agg("sum")
    def mean(self): return self._agg("mean")
    def min(self): return self._agg("min")
    def max(self): return self._agg("max")
    def std(self): return self._agg("std")
    def var(self): return self._agg("var")
    def count(self): return self._agg("count")
    def first(self): return self._agg("first")
    def last(self): return self._agg("last")
    def n_unique(self): return self._agg("n_unique")
    def median(self): return self._agg("median")

    # window helpers
    def cumsum(self): return Expr(lambda df: np.cumsum(self.eval(df)), self.name)
    def shift(self, k=1):
        def f(df):
            a = self.eval(df).astype(np.float64)
            out = np.full_like(a, np.nan)
            if k >= 0:
                out[k:] = a[:len(a) - k]
            else:
                out[:k] = a[-k:]
            return out
        return Expr(f, self.name)

    def rolling_mean(self, w):
        def f(df):
            a = self.eval(df).astype(np.float64)
            c = np.cumsum(np.insert(a, 0, 0.0))
            out = np.full(len(a), np.nan)
            out[w - 1:] = (c[w:] - c[:-w]) / w
            return out
        return Expr(f, self.name)

    def __repr__(self):
        return f"<expr {self.name}>"


def _bool(a):
    return a if isinstance(a, np.ndarray) and a.dtype == bool else np.asarray(a, dtype=bool)


def _binop(a, b, op, sym):
    sa, sb = isinstance(a, Strings), isinstance(b, Strings)
    if sa or sb:
        if sym in ("==", "!="):
            if sa and not sb and np.ndim(b) == 0:
                code = a.index().get(b, -2)
                r = a.codes == code
                return r if sym == "==" else ~r
            if sa and sb:
                r = np.array(a.tolist(), dtype=object) == np.array(b.tolist(), dtype=object)
                return r if sym == "==" else ~r
        if sym in ("<", "<=", ">", ">=") and sa and np.ndim(b) == 0:
            lut = op(np.array(a.values, dtype=object), b).astype(bool)
            return np.append(lut, False)[a.codes]
        if sym == "+":
            la = a.tolist() if sa else [a] * len(b) if np.ndim(a) == 0 else list(a)
            lb = b.tolist() if sb else [b] * len(la) if np.ndim(b) == 0 else list(b)
            return _infer([None if x is None or y is None else str(x) + str(y) for x, y in zip(la, lb)])
        raise RuntimeErr(f"operator {sym} is not defined for string columns")
    return op(a, b)


def _reduce_all(kind, a):
    if isinstance(a, Strings):
        if kind == "count":
            return int(np.count_nonzero(a.codes >= 0))
        if kind == "n_unique":
            return int(len(np.unique(a.codes[a.codes >= 0])))
        if kind in ("first", "last"):
            return a.values[a.codes[0 if kind == "first" else -1]] if len(a) else None
        vals = sorted(v for v in a.tolist() if v is not None)
        return (vals[0] if kind == "min" else vals[-1]) if vals else None
    if kind == "count":
        return int(np.count_nonzero(~np.isnan(a))) if a.dtype.kind == "f" else len(a)
    if kind == "n_unique":
        return int(len(np.unique(a)))
    if kind == "first":
        return a[0].item() if len(a) else None
    if kind == "last":
        return a[-1].item() if len(a) else None
    f = {"sum": np.nansum, "mean": np.nanmean, "min": np.nanmin, "max": np.nanmax, "median": np.nanmedian,
         "std": lambda x: np.nanstd(x, ddof=1), "var": lambda x: np.nanvar(x, ddof=1)}[kind]
    return f(a).item() if len(a) else None


def col(name):
    return Expr(lambda df: df._cols[name] if name in df._cols else _missing(df, name), name)


def _missing(df, name):
    raise RuntimeErr(f"no column '{name}' (columns: {', '.join(df._cols)})")


def lit(v):
    return Expr(lambda df: v, "literal")


def count():
    return Expr(lambda df: df.height, "count", ("len", None))


def _tolist(a):
    return a.tolist() if isinstance(a, (Strings, np.ndarray)) else list(a)


# ======================================================================================== factorize
def _codes(c):
    """Integer codes 0..k-1 for the values of one column, plus k."""
    if isinstance(c, Strings):
        if (c.codes < 0).any():
            codes = c.codes.astype(np.int64)
            codes[codes < 0] = len(c.values)
            return codes, len(c.values) + 1
        return c.codes.astype(np.int64), len(c.values)
    if c.dtype.kind in "iub" and len(c):
        lo, hi = int(c.min()), int(c.max())
        if hi - lo < 4 * len(c) + 1024:                    # small integer range: no sort needed
            return (c.astype(np.int64) - lo), hi - lo + 1
    u, inv = np.unique(c, return_inverse=True)
    return inv.astype(np.int64), len(u)


def _dense(gid, k):
    """Dense group ids in order of first appearance for ids in [0, k) — O(n + k), no sorting of the rows."""
    n = len(gid)
    first_pos = np.full(k, n, dtype=np.int64)
    first_pos[gid[::-1]] = np.arange(n - 1, -1, -1)        # the last write is the smallest row index
    present = np.flatnonzero(first_pos < n)
    order = present[np.argsort(first_pos[present], kind="stable")]
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(len(order))
    return remap[gid], first_pos[order]


def _group_ids(cols):
    """One integer id per row for the combination of several key columns (dense, in first-appearance order)."""
    gid, k = None, 1
    for c in cols:
        codes, n = _codes(c)
        gid = codes if gid is None else gid * n + codes
        k *= n
        if k > 1 << 62:
            gid = np.unique(gid, return_inverse=True)[1].astype(np.int64)
            k = int(gid.max()) + 1 if len(gid) else 1
    if gid is None:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if k <= 4 * len(gid) + 4096:
        return _dense(gid, k)
    uniq, first, inv = np.unique(gid, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv], first[order]


# ======================================================================================== DataFrame
class DataFrame:
    def __init__(self, data=None, columns=None):
        self._cols = {}
        if data is None:
            data = {}
        if isinstance(data, DataFrame):
            self._cols = dict(data._cols)
            return
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys = list(dict.fromkeys(k for r in data for k in r))
            data = {k: [r.get(k) for r in data] for k in keys}
        elif isinstance(data, list) and columns:
            data = {c: [r[i] for r in data] for i, c in enumerate(columns)}
        n = None
        for k, v in data.items():
            c = _col(v)
            if n is not None and len(c) != n:
                raise RuntimeErr(f"column '{k}' has {len(c)} values, the others {n}")
            n = len(c)
            self._cols[str(k)] = c

    @staticmethod
    def _from(cols):
        d = DataFrame()
        d._cols = cols
        return d

    # ---- shape and access
    @property
    def height(self):
        return len(next(iter(self._cols.values()))) if self._cols else 0

    @property
    def width(self):
        return len(self._cols)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def columns(self):
        return list(self._cols)

    @property
    def dtypes(self):
        return {k: _dtype_name(v) for k, v in self._cols.items()}

    def __len__(self):
        return self.height

    def __getitem__(self, k):
        if isinstance(k, str):
            c = self._cols.get(k)
            if c is None:
                _missing(self, k)
            return c.to_numpy() if isinstance(c, Strings) else c
        if isinstance(k, list):
            return self.select(*k)
        if isinstance(k, slice) or isinstance(k, (int, np.integer)):
            idx = np.arange(self.height)[k]
            if np.ndim(idx) == 0:
                return self.row(int(idx))
            return self._from({n: _take(c, idx) for n, c in self._cols.items()})
        if isinstance(k, np.ndarray) and k.dtype == bool:
            return self.filter(k)
        if isinstance(k, Expr):
            return self.filter(k)
        raise RuntimeErr(f"cannot index a DataFrame with {type(k).__name__}")

    def __setitem__(self, k, v):
        self._cols[str(k)] = _col(v, self.height)

    def __getattr__(self, k):
        cols = self.__dict__.get("_cols")
        if cols is not None and k in cols:
            return self[k]
        raise AttributeError(k)

    def row(self, i):
        return {k: (c.values[c.codes[i]] if c.codes[i] >= 0 else None) if isinstance(c, Strings) else c[i].item()
                for k, c in self._cols.items()}

    def rows(self):
        lists = [c.tolist() for c in self._cols.values()]
        return [tuple(r) for r in zip(*lists)]

    def to_dicts(self):
        names = list(self._cols)
        return [dict(zip(names, r)) for r in self.rows()]

    def to_dict(self):
        return {k: c.tolist() for k, c in self._cols.items()}

    def to_numpy(self, *cols):
        names = cols or [k for k, c in self._cols.items() if not isinstance(c, Strings)]
        return np.column_stack([self._cols[n].astype(np.float64) for n in names])

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame({k: (c.to_numpy() if isinstance(c, Strings) else c) for k, c in self._cols.items()})

    # ---- transforms
    def _eval(self, e, n=None):
        if isinstance(e, Expr):
            v = e.eval(self)
        elif isinstance(e, str):
            v = self._cols[e]
        else:
            v = e
        if isinstance(v, Strings):
            return v
        if np.ndim(v) == 0:
            return _col([v.item() if hasattr(v, "item") else v] * (n if n is not None else self.height))
        return _col(v)

    def select(self, *exprs, **named):
        out = {}
        for e in exprs:
            if isinstance(e, (list, tuple)):
                for x in e:
                    out[x if isinstance(x, str) else x.name] = self._eval(x)
                continue
            out[e if isinstance(e, str) else e.name] = self._eval(e)
        for k, e in named.items():
            out[k] = self._eval(e)
        if out and all(len(v) == 1 for v in out.values()) and any(isinstance(e, Expr) and e.agg for e in
                                                                  list(exprs) + list(named.values())):
            return self._from(out)
        return self._from(out)

    def with_columns(self, *exprs, **named):
        cols = dict(self._cols)
        for e in exprs:
            cols[e.name] = self._eval(e)
        for k, e in named.items():
            cols[k] = self._eval(e)
        return self._from(cols)

    with_column = with_columns

    def drop(self, *names):
        return self._from({k: v for k, v in self._cols.items() if k not in names})

    def rename(self, mapping=None, **kw):
        m = dict(mapping or {}, **kw)
        return self._from({m.get(k, k): v for k, v in self._cols.items()})

    def filter(self, cond):
        mask = cond.eval(self) if isinstance(cond, Expr) else cond
        mask = _bool(mask)
        if mask.shape != (self.height,):
            raise RuntimeErr("filter needs one true/false per row")
        idx = np.flatnonzero(mask)
        return self._from({k: _take(c, idx) for k, c in self._cols.items()})

    where = filter

    def head(self, n=5):
        return self[:n]

    def tail(self, n=5):
        return self[max(0, self.height - n):]

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.height, size=min(n, self.height), replace=False)
        return self._from({k: _take(c, idx) for k, c in self._cols.items()})

    def sort(self, by, descending=False):
        keys = [by] if isinstance(by, (str, Expr)) else list(by)
        desc = descending if isinstance(descending, (list, tuple)) else [descending] * len(keys)
        arrays = []
        for k, d in zip(reversed(keys), reversed(desc)):
            c = self._eval(k)
            if isinstance(c, Strings):
                rank = np.argsort(np.argsort(np.array(c.values + [""], dtype=object), kind="stable"))
                key = rank[c.codes]
            else:
                key = c
            if d:
                key = -key.astype(np.float64) if key.dtype.kind != "b" else ~key
            arrays.append(key)
        idx = np.lexsort(arrays) if len(arrays) > 1 else np.argsort(arrays[0])
        return self._from({k: _take(c, idx) for k, c in self._cols.items()})

    def unique(self, subset=None):
        keys = [self._cols[k] for k in (subset or self.columns)]
        _, first = _group_ids(keys)
        idx = np.sort(first)
        return self._from({k: _take(c, idx) for k, c in self._cols.items()})

    def group_by(self, *keys):
        keys = [k for ks in keys for k in (ks if isinstance(ks, (list, tuple)) else [ks])]
        return GroupBy(self, keys)

    groupby = group_by

    def join(self, other, on=None, left_on=None, right_on=None, how="inner", suffix="_right"):
        left_on = [left_on or on] if isinstance(left_on or on, str) else list(left_on or on)
        right_on = [right_on or on] if isinstance(right_on or on, str) else list(right_on or on)
        lk = [self._cols[k] for k in left_on]
        rk = [other._cols[k] for k in right_on]
        # factorise both sides together so equal keys get equal ids
        ids = []
        for a, b in zip(lk, rk):
            if isinstance(a, Strings) and isinstance(b, Strings):
                idx = a.index()
                nl = len(a.values)
                extra = {}
                lut = np.empty(len(b.values) + 1, dtype=np.int64)
                for j, v in enumerate(b.values):                 # cost: distinct right keys, not rows
                    c = idx.get(v)
                    if c is None:
                        c = extra.setdefault(v, nl + len(extra))
                    lut[j] = c
                null_id = nl + len(extra)
                lut[-1] = null_id + 1                             # nulls never match
                la = a.codes.astype(np.int64)
                la[la < 0] = null_id
                ids.append((la, lut[b.codes], null_id + 2))
            elif isinstance(a, Strings) or isinstance(b, Strings):
                la, lb = a.tolist(), b.tolist()
                both = Strings.from_list([str(x) for x in la + lb])
                codes = both.codes.astype(np.int64)
                ids.append((codes[:len(la)], codes[len(la):], len(both.values)))
            else:
                u, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
                ids.append((inv[:len(a)], inv[len(a):], len(u)))
        lid = rid = None
        k = 1
        for a, b, n in ids:
            lid = a if lid is None else lid * n + a
            rid = b if rid is None else rid * n + b
            k *= n
        if k <= 4 * (len(lid) + len(rid)) + 4096 and len(rid) and np.bincount(rid, minlength=k).max() <= 1:
            # the right side has unique keys (a lookup table): direct addressing, no sort
            pos = np.full(k, -1, dtype=np.int64)
            pos[rid] = np.arange(len(rid))
            hit = pos[lid]
            cnt = (hit >= 0).astype(np.int64)
            li = np.flatnonzero(hit >= 0)
            ri = hit[li]
        else:
            order = np.argsort(rid, kind="stable")
            rs = rid[order]
            lo = np.searchsorted(rs, lid, "left")
            hi = np.searchsorted(rs, lid, "right")
            cnt = hi - lo
            li = np.repeat(np.arange(len(lid)), cnt)
            starts = np.repeat(lo, cnt)
            offs = np.arange(len(li)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            ri = order[starts + offs]
        if how in ("left", "outer", "full"):
            miss = np.flatnonzero(cnt == 0)
            li = np.concatenate([li, miss])
            ri = np.concatenate([ri, np.full(len(miss), -1)])
            o = np.argsort(li, kind="stable")
            li, ri = li[o], ri[o]
        if how in ("right", "outer", "full"):
            matched = np.zeros(len(rid), bool)
            matched[ri[ri >= 0]] = True
            extra = np.flatnonzero(~matched)
            li = np.concatenate([li, np.full(len(extra), -1)])
            ri = np.concatenate([ri, extra])
        out = {}
        for k, c in self._cols.items():
            out[k] = _take_null(c, li)
        if how in ("right", "outer", "full"):
            for lkname, rkname in zip(left_on, right_on):
                fill = li < 0
                if fill.any():
                    v = _coalesce(out[lkname], _take_null(other._cols[rkname], ri), fill)
                    lc, rc = self._cols[lkname], other._cols[rkname]
                    if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not isinstance(lc, Strings) and \
                            not isinstance(rc, Strings) and lc.dtype.kind in "iu" and rc.dtype.kind in "iu":
                        v = v.astype(np.int64)                 # every row has a key: integer keys stay integers
                    out[lkname] = v
        for k, c in other._cols.items():
            if k in right_on and right_on == left_on:
                continue
            name = k if k not in out else k + suffix
            out[name] = _take_null(c, ri)
        return self._from(out)

    def concat(self, *others):
        frames = [self] + list(others)
        out = {}
        for k in self.columns:
            parts = [f._cols[k] for f in frames]
            if any(isinstance(p, Strings) for p in parts):
                out[k] = _infer([v for p in parts for v in p.tolist()])
            else:
                out[k] = np.concatenate(parts)
        return self._from(out)

    def value_counts(self, column, sort=True):
        r = self.group_by(column).agg(count=count())
        return r.sort("count", descending=True) if sort else r

    def describe(self):
        rows = []
        for k, c in self._cols.items():
            if isinstance(c, Strings):
                rows.append({"column": k, "type": "str", "count": int((c.codes >= 0).sum()),
                             "unique": len(c.values), "mean": None, "std": None, "min": None, "max": None})
            else:
                a = c.astype(np.float64)
                rows.append({"column": k, "type": _dtype_name(c), "count": int((~np.isnan(a)).sum()),
                             "unique": None, "mean": float(np.nanmean(a)) if len(a) else None,
                             "std": float(np.nanstd(a, ddof=1)) if len(a) > 1 else None,
                             "min": float(np.nanmin(a)) if len(a) else None,
                             "max": float(np.nanmax(a)) if len(a) else None})
        return DataFrame(rows)

    def write_csv(self, path, interp=None):
        it = interp or _INTERP[0]
        if it is not None:
            from .security import check_path
            path = check_path(it, path, "write", "write_csv()")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            w.writerows(self.rows())

    def write_jsonl(self, path, interp=None):
        import json
        it = interp or _INTERP[0]
        if it is not None:
            from .security import check_path
            path = check_path(it, path, "write", "write_jsonl()")
        with open(path, "w", encoding="utf-8") as f:
            for r in self.to_dicts():
                f.write(json.dumps(r, ensure_ascii=False, default=_jdefault) + "\n")

    def write_json(self, path, interp=None):
        import json
        it = interp or _INTERP[0]
        if it is not None:
            from .security import check_path
            path = check_path(it, path, "write", "write_json()")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, ensure_ascii=False, default=_jdefault)

    def to_json(self):
        import json
        return json.dumps(self.to_dicts(), ensure_ascii=False, default=_jdefault)

    def memory_bytes(self):
        tot = 0
        for c in self._cols.values():
            if isinstance(c, Strings):
                tot += c.codes.nbytes + sum(len(v) for v in c.values) + 8 * len(c.values)
            else:
                tot += c.nbytes
        return tot

    def __iter__(self):
        return iter(self.to_dicts())

    def __repr__(self):
        n = self.height
        show = list(range(min(n, 10))) if n <= 20 else list(range(5)) + [-1] + list(range(n - 5, n))
        names = self.columns
        cells = [names, [self.dtypes[k] for k in names]]
        for i in show:
            if i < 0:
                cells.append(["…"] * len(names))
                continue
            r = self.row(i)
            cells.append([_fmt(r[k]) for k in names])
        widths = [max(len(str(row[j])) for row in cells) for j in range(len(names))]
        line = lambda row: "│ " + " │ ".join(str(v).rjust(w) if j else str(v).ljust(w)
                                              for j, (v, w) in enumerate(zip(row, widths))) + " │"
        sep = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        out = [f"shape: ({n}, {len(names)})", "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐", line(cells[0]),
               line(cells[1]), sep] + [line(r) for r in cells[2:]] + ["└" + "┴".join("─" * (w + 2) for w in widths) + "┘"]
        return "\n".join(out)

    __str__ = __repr__


def _jdefault(o):
    if isinstance(o, float) and o != o:
        return None
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _fmt(v):
    if isinstance(v, float):
        return "null" if v != v else f"{v:.6g}"
    return "null" if v is None else str(v)


def _take_null(c, idx):
    """Take rows; index -1 gives a null (NaN for numbers, None for strings)."""
    if (idx >= 0).all():
        return _take(c, idx)
    if isinstance(c, Strings):
        codes = np.where(idx >= 0, c.codes[np.maximum(idx, 0)], -1).astype(np.int32)
        return Strings(codes, c.values)
    a = c[np.maximum(idx, 0)].astype(np.float64) if c.dtype.kind in "iub" else c[np.maximum(idx, 0)].copy()
    a[idx < 0] = np.nan if a.dtype.kind == "f" else a[idx < 0]
    return a


def _coalesce(a, b, mask):
    if isinstance(a, Strings) or isinstance(b, Strings):
        la, lb = a.tolist(), b.tolist()
        return _infer([y if m else x for x, y, m in zip(la, lb, mask.tolist())])
    return np.where(mask, b, a)


class GroupBy:
    def __init__(self, df, keys):
        self.df, self.keys = df, keys
        self.gid, self.first = _group_ids([df._cols[k] for k in keys])
        self.k = len(self.first)

    def agg(self, *exprs, **named):
        df, gid, k = self.df, self.gid, self.k
        out = {key: _take(df._cols[key], self.first) for key in self.keys}
        items = [(e.name, e) for e in exprs] + list(named.items())
        order = None
        for name, e in items:
            if isinstance(e, str):
                e = col(e).sum()
            kind, inner = e.agg if e.agg else ("first", e)
            if kind == "len":
                out[name] = np.bincount(gid, minlength=k).astype(np.int64)
                continue
            a = inner.eval(df)
            if isinstance(a, Strings):
                if kind == "count":
                    out[name] = np.bincount(gid, weights=(a.codes >= 0), minlength=k).astype(np.int64)
                elif kind == "n_unique":
                    pair = np.unique(gid * (len(a.values) + 1) + a.codes)
                    out[name] = np.bincount(pair // (len(a.values) + 1), minlength=k).astype(np.int64)
                else:
                    if order is None:
                        order = np.argsort(gid, kind="stable")
                    sg = gid[order]
                    starts = np.searchsorted(sg, np.arange(k))
                    ends = np.searchsorted(sg, np.arange(k), "right")
                    pick = order[starts] if kind == "first" else order[ends - 1]
                    out[name] = a.take(pick)
                continue
            a = np.asarray(a)
            if kind in ("sum", "mean", "count", "var", "std"):
                valid = ~np.isnan(a) if a.dtype.kind == "f" else None
                x = np.where(valid, a, 0.0) if valid is not None else a
                n = np.bincount(gid, weights=valid, minlength=k) if valid is not None else \
                    np.bincount(gid, minlength=k).astype(np.float64)
                if kind == "count":
                    out[name] = n.astype(np.int64)
                    continue
                s = np.bincount(gid, weights=x, minlength=k)
                if kind == "sum":
                    out[name] = s.astype(np.int64) if a.dtype.kind in "iub" and np.abs(s).max(initial=0) < 2 ** 52 \
                        else s
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = s / n
                    if kind == "mean":
                        out[name] = mean
                        continue
                    d = (x - mean[gid])
                    if valid is not None:
                        d = np.where(valid, d, 0.0)
                    ss = np.bincount(gid, weights=d * d, minlength=k)
                    var = ss / (n - 1)
                out[name] = var if kind == "var" else np.sqrt(var)
                continue
            if kind in ("min", "max"):
                f = np.fmin if kind == "min" else np.fmax
                init = np.inf if kind == "min" else -np.inf
                r = np.full(k, init)
                f.at(r, gid, a.astype(np.float64))
                out[name] = r.astype(a.dtype) if a.dtype.kind in "iu" else r
                continue
            if kind == "first":
                out[name] = a[self.first]
                continue
            if order is None:
                order = np.argsort(gid, kind="stable")
            sg = gid[order]
            sa = a[order]
            starts = np.searchsorted(sg, np.arange(k))
            if kind == "first":
                out[name] = sa[starts]
            elif kind == "last":
                out[name] = sa[np.searchsorted(sg, np.arange(k), "right") - 1]
            elif kind == "median":
                ends = np.searchsorted(sg, np.arange(k), "right")
                out[name] = np.array([np.nanmedian(sa[s:e]) for s, e in zip(starts, ends)])
            elif kind == "n_unique":
                u = np.unique(np.stack([sg, sa]), axis=1)
                out[name] = np.bincount(u[0].astype(np.int64), minlength=k)
            else:
                raise RuntimeErr(f"unknown aggregation '{kind}'")
        return DataFrame._from(out)

    def count(self):
        return self.agg(count=count())

    def sum(self, *cols):
        return self.agg(**{c: col(c).sum() for c in cols or self._value_cols()})

    def mean(self, *cols):
        return self.agg(**{c: col(c).mean() for c in cols or self._value_cols()})

    def _value_cols(self):
        return [k for k, c in self.df._cols.items() if k not in self.keys and not isinstance(c, Strings)]


# ======================================================================================== IO
_INTERP = [None]


def _convert_text_column(vals):
    """Typed column from CSV text: int64, float64, bool or dictionary strings ("" → null)."""
    arr = np.array(vals, dtype=object)
    try:
        if all(v and (v.lstrip("-").isdigit()) for v in vals[:1000]):
            return np.array(vals, dtype=np.int64)
    except (ValueError, OverflowError):
        pass
    try:
        f = np.array([v if v != "" else "nan" for v in vals], dtype=np.float64)
        return f
    except ValueError:
        pass
    low = {v.lower() for v in vals[:1000] if v}
    if low and low <= {"true", "false"}:
        return np.array([v.lower() == "true" for v in vals], dtype=bool)
    s = Strings.from_list(vals)
    if "" in s.index():
        nul = s.index()[""]
        codes = s.codes.copy()
        codes[codes == nul] = -1
        return Strings(codes, s.values).compact()
    _ = arr
    return s


def read_csv(path, sep=",", columns=None, interp=None):
    it = interp or _INTERP[0]
    if it is not None:
        from .security import check_path
        path = check_path(it, path, "read", "read_csv()")
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    if len(data) > 1 << 16:
        fast = _fast_csv(data, sep, columns)
        if fast is not None:
            return fast
    return from_csv_text(data.decode("utf-8"), sep, columns)


def _fast_csv(data, sep, columns):
    """The C path: index every cell once, parse number columns in C, strings through the dictionary."""
    from . import fastc
    lib = fastc.load("fastcsv")
    if lib is None or len(sep) != 1:
        return None
    import ctypes
    n = len(data)
    cap = max(16, data.count(b"\n") + 2) * (data.split(b"\n", 1)[0].count(sep.encode()) + 1) + 16
    starts = np.empty(cap, dtype=np.int64)
    ends = np.empty(cap, dtype=np.int64)
    quoted = np.empty(cap, dtype=np.uint8)
    rows = ctypes.c_long(0)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.csv_index.restype = ctypes.c_long
    cells = lib.csv_index(ctypes.c_char_p(data), ctypes.c_long(n), ctypes.c_char(sep.encode()), P(starts), P(ends),
                          P(quoted), ctypes.c_long(cap), ctypes.byref(rows))
    if cells < 0:
        return None
    first_end = data.find(b"\n")
    header = next(csv.reader([data[:first_end if first_end >= 0 else n].decode("utf-8-sig").rstrip("\r")],
                             delimiter=sep))
    ncols = len(header)
    if cells % ncols:
        return None                                           # ragged rows: let the csv module report them
    nrows = cells // ncols - 1
    st, en, qt = starts[ncols:cells], ends[ncols:cells], quoted[ncols:cells]
    lib.parse_doubles.restype = lib.parse_longs.restype = ctypes.c_long
    out = {}
    for j, name in enumerate(header):
        if columns and name not in columns:
            continue
        if not qt[j::ncols].any():
            iv = np.empty(nrows, dtype=np.int64)
            if lib.parse_longs(ctypes.c_char_p(data), P(st), P(en), ctypes.c_long(nrows), ctypes.c_long(ncols),
                               ctypes.c_long(j), P(iv)) == 0:
                out[name] = iv
                continue
            fv = np.empty(nrows, dtype=np.float64)
            if lib.parse_doubles(ctypes.c_char_p(data), P(st), P(en), ctypes.c_long(nrows), ctypes.c_long(ncols),
                                 ctypes.c_long(j), P(fv)) == 0:
                out[name] = fv
                continue
        if not qt[j::ncols].any() and nrows:
            tsize = 1 << max(4, (2 * nrows).bit_length())
            codes = np.empty(nrows, dtype=np.int32)
            first = np.empty(nrows, dtype=np.int64)
            table = np.empty(tsize, dtype=np.int64)
            lib.dict_encode.restype = ctypes.c_long
            k = lib.dict_encode(ctypes.c_char_p(data), P(st), P(en), ctypes.c_long(nrows), ctypes.c_long(ncols),
                                ctypes.c_long(j), P(codes), P(first), P(table), ctypes.c_long(tsize))
            if k <= nrows // 2 or k < 100000:              # repetitive text: keep it dictionary-encoded
                cs = (first[:k] * ncols + j).tolist()
                values = [data[st[c]:en[c]].decode("utf-8") for c in cs]
                col_ = Strings(codes, values)
                if "" in values:
                    nul = values.index("")
                    cc = codes.copy()
                    cc[cc == nul] = -1
                    col_ = Strings(cc, values).compact()
                out[name] = col_
                continue
        s_, e_, q_ = st[j::ncols].tolist(), en[j::ncols].tolist(), qt[j::ncols].tolist()
        vals = [data[a:b].decode("utf-8") if not q else data[a:b].decode("utf-8").replace('""', '"')
                for a, b, q in zip(s_, e_, q_)]
        out[name] = _convert_text_column(vals)
    return DataFrame._from(out)


def from_csv_text(text, sep=",", columns=None):
    if isinstance(text, str) and len(text) > 1 << 16:
        fast = _fast_csv(text.encode("utf-8"), sep, columns)
        if fast is not None:
            return fast
    rows = csv.reader(io.StringIO(text), delimiter=sep)
    header = next(rows)
    cols = list(zip(*rows)) if text else []
    if not cols:
        return DataFrame({h: [] for h in header})
    out = {}
    for name, vals in zip(header, cols):
        if columns and name not in columns:
            continue
        out[name] = _convert_text_column(list(vals))
    return DataFrame._from(out)


def read_jsonl(path, interp=None):
    """One JSON object per line (JSON Lines / NDJSON) → a DataFrame (missing keys become nulls)."""
    import json
    it = interp or _INTERP[0]
    if it is not None:
        from .security import check_path
        path = check_path(it, path, "read", "read_jsonl()")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return DataFrame(rows) if rows else DataFrame()


def read_json(path, interp=None):
    """A JSON array of objects, or an object of columns → a DataFrame."""
    import json
    it = interp or _INTERP[0]
    if it is not None:
        from .security import check_path
        path = check_path(it, path, "read", "read_json()")
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return DataFrame(data)


def from_records(rows, columns=None):
    return DataFrame(rows, columns)


def concat(frames):
    frames = list(frames)
    return frames[0].concat(*frames[1:])


def module(interp=None):
    _INTERP[0] = interp
    return {"DataFrame": DataFrame, "frame": DataFrame, "read_csv": read_csv, "from_csv_text": from_csv_text,
            "read_jsonl": read_jsonl, "read_json": read_json,
            "from_records": from_records, "col": col, "lit": lit, "count": count, "concat": concat, "Expr": Expr}
