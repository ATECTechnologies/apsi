"""`use collections` — data structures beyond list, dict and set, and functional helpers.

Classes
    Counter(xs=none)        counts of hashable values: c.add(x, n=1), c[x] (0 when absent), c.most_common(k),
                            c.total(), c + d, c - d (counts below 1 removed), c.items(), c.to_dict()
    DefaultDict(make)       a dict whose missing keys are created by make(): DefaultDict(fn() => [])
    Deque(xs=none, max=none) double-ended queue: push / push_left / pop / pop_left / peek / peek_left,
                            rotate(n); with max it keeps the newest max items (a ring buffer)
    Heap(xs=none, key=none, max_heap=false)
                            priority queue: push(x), pop() (smallest first, or largest with max_heap),
                            peek(), push_pop(x), len(h); ties come out in insertion order
    LRUCache(capacity)      get(k, default) / set(k, v) / has(k) / delete(k); least recently used leave first;
                            stats() with hits and misses
    OrderedSet(xs=none)     a set that remembers insertion order: add, discard, has, |, &, -, list(s)
    BiMap(pairs=none)       a one-to-one map with lookups both ways: m.set(k, v), m.get(k), m.key_of(v)

Functions (also built in, without `use`)
    first(xs, pred=none, default=none)   last(xs, pred=none, default=none)
    partition(xs, pred)     (matching, not matching)
    take_while(xs, pred)    drop_while(xs, pred)
    scan(xs, f, init=none)  running results of f: scan([1, 2, 3], fn(a, b) => a + b) == [1, 3, 6]
    fold(xs, f, init)       one result: fold(xs, fn(acc, x) => acc + x, 0)
    flat_map(xs, f)         f(x) for every x, the lists joined
    zip_longest(a, b, ..., fill=none)    interleave(a, b, ...)
    distinct_by(xs, key)    the first x for every key(x), in order
    min_by(xs, key)  max_by(xs, key)  sum_by(xs, key)
    frequencies(xs)         {value: count}, most common first
    compose(f, g, ...)      compose(f, g)(x) == f(g(x))
    partial(f, *args, **kwargs)
"""
import collections
import heapq
import itertools
import threading

_MISSING = object()


def _items(xs):
    if xs is None:
        return []
    if isinstance(xs, dict):
        return list(xs.items())
    return list(xs)


# =========================================================================== classes
class Counter:
    """Counts of hashable values."""

    def __init__(self, xs=None):
        self._c = collections.Counter()
        if isinstance(xs, (dict, Counter)):
            for k, v in (xs._c if isinstance(xs, Counter) else xs).items():
                self._c[k] += int(v)
        elif xs is not None:
            self._c.update(xs)

    def add(self, x, n=1):
        self._c[x] += int(n)
        if self._c[x] <= 0:
            del self._c[x]
        return self

    def update(self, xs):
        self._c.update(xs)
        return self

    def __getitem__(self, x):
        return self._c.get(x, 0)

    def __setitem__(self, x, n):
        if int(n) <= 0:
            self._c.pop(x, None)
        else:
            self._c[x] = int(n)

    def __contains__(self, x):
        return x in self._c

    def __len__(self):
        return len(self._c)

    def __iter__(self):
        return iter(self._c)

    def most_common(self, k=None):
        return [tuple(p) for p in self._c.most_common(k)]

    def total(self):
        return sum(self._c.values())

    def items(self):
        return list(self._c.items())

    def keys(self):
        return list(self._c.keys())

    def values(self):
        return list(self._c.values())

    def to_dict(self):
        return dict(self._c)

    def __add__(self, other):
        return Counter(dict(self._c + other._c))

    def __sub__(self, other):
        return Counter(dict(self._c - other._c))

    def __eq__(self, other):
        return isinstance(other, Counter) and self._c == other._c

    def __repr__(self):
        body = ", ".join(f"{k!r}: {v}" for k, v in self._c.most_common())
        return f"Counter({{{body}}})"


class DefaultDict(dict):
    """A dict whose missing keys are created by make() on first access."""

    def __init__(self, make, pairs=None):
        super().__init__(_items(pairs))
        if not callable(make):
            raise TypeError("DefaultDict(make): make must be a function, e.g. fn() => []")
        self.make = make

    def __missing__(self, key):
        v = self[key] = self.make()
        return v

    def __repr__(self):
        return f"DefaultDict({dict.__repr__(self)})"


class Deque:
    """Double-ended queue; with max it keeps only the newest max items."""

    def __init__(self, xs=None, max=None):
        self._d = collections.deque(xs or [], maxlen=None if max is None else int(max))

    def push(self, x):
        self._d.append(x)
        return self

    def push_left(self, x):
        self._d.appendleft(x)
        return self

    def pop(self):
        if not self._d:
            raise IndexError("pop from an empty Deque")
        return self._d.pop()

    def pop_left(self):
        if not self._d:
            raise IndexError("pop_left from an empty Deque")
        return self._d.popleft()

    def peek(self, default=None):
        return self._d[-1] if self._d else default

    def peek_left(self, default=None):
        return self._d[0] if self._d else default

    def rotate(self, n=1):
        self._d.rotate(int(n))
        return self

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(list(self._d))

    def __getitem__(self, i):
        return self._d[i]

    def to_list(self):
        return list(self._d)

    def __repr__(self):
        return f"Deque({list(self._d)!r})"


class Heap:
    """Priority queue (binary heap). Smallest first, or largest with max_heap=true; key picks the priority.
    Equal priorities come out in the order they went in."""

    def __init__(self, xs=None, key=None, max_heap=False):
        self.key, self.max_heap = key, bool(max_heap)
        self._h = []
        self._n = itertools.count()
        for x in xs or []:
            self.push(x)

    def _entry(self, x):
        p = self.key(x) if self.key is not None else x
        if self.max_heap:
            p = _Neg(p)
        return (p, next(self._n), x)

    def push(self, x):
        heapq.heappush(self._h, self._entry(x))
        return self

    def pop(self):
        if not self._h:
            raise IndexError("pop from an empty Heap")
        return heapq.heappop(self._h)[2]

    def peek(self, default=None):
        return self._h[0][2] if self._h else default

    def push_pop(self, x):
        """Push x, then pop the best item (faster than push followed by pop)."""
        return heapq.heappushpop(self._h, self._entry(x))[2]

    def __len__(self):
        return len(self._h)

    def to_list(self):
        """All items, best first (the heap itself is unchanged)."""
        return [e[2] for e in sorted(self._h)]

    def __repr__(self):
        return f"Heap({self.to_list()!r})"


class _Neg:
    """Reverses the order of any comparable value (for max-heaps of non-numbers too)."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        return other.v < self.v

    def __eq__(self, other):
        return self.v == other.v


class LRUCache:
    """A bounded map: when full, the least recently used entry leaves. Thread-safe."""

    def __init__(self, capacity=128):
        self.capacity = max(1, int(capacity))
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, k, default=None):
        with self._lock:
            v = self._d.get(k, _MISSING)
            if v is _MISSING:
                self.misses += 1
                return default
            self._d.move_to_end(k)
            self.hits += 1
            return v

    def set(self, k, v):
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
        return v

    def has(self, k):
        return k in self._d

    __contains__ = has

    def delete(self, k):
        with self._lock:
            self._d.pop(k, None)

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)

    def stats(self):
        n = self.hits + self.misses
        return {"size": len(self._d), "capacity": self.capacity, "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0}

    def __repr__(self):
        return f"<LRUCache {len(self._d)}/{self.capacity}>"


class OrderedSet:
    """A set that keeps insertion order."""

    def __init__(self, xs=None):
        self._d = dict.fromkeys(xs or [])

    def add(self, x):
        self._d[x] = None
        return self

    def discard(self, x):
        self._d.pop(x, None)
        return self

    def has(self, x):
        return x in self._d

    __contains__ = has

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(list(self._d))

    def __or__(self, other):
        return OrderedSet(list(self) + list(other))

    def __and__(self, other):
        o = set(other)
        return OrderedSet(x for x in self._d if x in o)

    def __sub__(self, other):
        o = set(other)
        return OrderedSet(x for x in self._d if x not in o)

    def __eq__(self, other):
        return isinstance(other, OrderedSet) and list(self._d) == list(other._d)

    def to_list(self):
        return list(self._d)

    def __repr__(self):
        return f"OrderedSet({list(self._d)!r})"


class BiMap:
    """One-to-one map: every key has one value and every value one key."""

    def __init__(self, pairs=None):
        self._f, self._b = {}, {}
        for k, v in _items(pairs):
            self.set(k, v)

    def set(self, k, v):
        if k in self._f:
            self._b.pop(self._f[k], None)
        if v in self._b:
            self._f.pop(self._b[v], None)
        self._f[k], self._b[v] = v, k
        return self

    def get(self, k, default=None):
        return self._f.get(k, default)

    def key_of(self, v, default=None):
        return self._b.get(v, default)

    def delete(self, k):
        if k in self._f:
            self._b.pop(self._f.pop(k), None)

    def __contains__(self, k):
        return k in self._f

    def __len__(self):
        return len(self._f)

    def items(self):
        return list(self._f.items())

    def __repr__(self):
        return f"BiMap({self._f!r})"


# =========================================================================== functions
def first(xs, pred=None, default=None):
    for x in xs:
        if pred is None or pred(x):
            return x
    return default


def last(xs, pred=None, default=None):
    out = default
    for x in xs:
        if pred is None or pred(x):
            out = x
    return out


def partition(xs, pred):
    yes, no = [], []
    for x in xs:
        (yes if pred(x) else no).append(x)
    return yes, no


def take_while(xs, pred):
    return list(itertools.takewhile(pred, xs))


def drop_while(xs, pred):
    return list(itertools.dropwhile(pred, xs))


def scan(xs, f, init=_MISSING):
    it = iter(xs)
    out = []
    if init is _MISSING or init is None:
        try:
            acc = next(it)
        except StopIteration:
            return out
    else:
        acc = init
    out.append(acc)
    for x in it:
        acc = f(acc, x)
        out.append(acc)
    return out


def fold(xs, f, init):
    acc = init
    for x in xs:
        acc = f(acc, x)
    return acc


def flat_map(xs, f):
    out = []
    for x in xs:
        out.extend(f(x))
    return out


def zip_longest(*xss, fill=None):
    return [tuple(t) for t in itertools.zip_longest(*xss, fillvalue=fill)]


def interleave(*xss):
    out = []
    for t in itertools.zip_longest(*xss, fillvalue=_MISSING):
        out.extend(x for x in t if x is not _MISSING)
    return out


def distinct_by(xs, key):
    seen, out = set(), []
    for x in xs:
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def min_by(xs, key, default=None):
    xs = list(xs)
    return min(xs, key=key) if xs else default


def max_by(xs, key, default=None):
    xs = list(xs)
    return max(xs, key=key) if xs else default


def sum_by(xs, key):
    total = 0
    for x in xs:
        total = total + key(x)
    return total


def frequencies(xs):
    return dict(collections.Counter(xs).most_common())


def compose(*fs):
    if not fs:
        raise TypeError("compose() needs at least one function")

    def composed(*args, **kwargs):
        r = fs[-1](*args, **kwargs)
        for f in reversed(fs[:-1]):
            r = f(r)
        return r
    return composed


def partial(f, *args, **kwargs):
    def bound(*more, **kw):
        return f(*args, *more, **dict(kwargs, **kw))
    return bound


FUNCTIONS = {"first": first, "last": last, "partition": partition, "take_while": take_while,
             "drop_while": drop_while, "scan": scan, "fold": fold, "flat_map": flat_map, "zip_longest": zip_longest,
             "interleave": interleave, "distinct_by": distinct_by, "min_by": min_by, "max_by": max_by,
             "sum_by": sum_by, "frequencies": frequencies, "compose": compose, "partial": partial}
CLASSES = {"Counter": Counter, "DefaultDict": DefaultDict, "Deque": Deque, "Heap": Heap, "LRUCache": LRUCache,
           "OrderedSet": OrderedSet, "BiMap": BiMap}


def module():
    return dict(CLASSES, **FUNCTIONS)
