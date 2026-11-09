"""Python's object model end to end: everything a large real program uses."""
import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import heapq
import itertools
import json
import math
import operator
import re
import threading
from collections import Counter, OrderedDict, defaultdict, deque, namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Color(enum.Enum):
    RED = 1
    GREEN = 2

    @property
    def lower(self):
        return self.name.lower()


class Shape(abc.ABC):
    registry: Dict[str, type] = {}
    count = 0

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        Shape.registry[cls.__name__] = cls

    def __init__(self, name):
        self._name = name
        Shape.count += 1

    @abc.abstractmethod
    def area(self) -> float: ...

    def __repr__(self):
        return f"{type(self).__name__}({self._name!r}, area={self.area():.3f})"

    def __lt__(self, other):
        return self.area() < other.area()

    @classmethod
    def make(cls, *a):
        return cls(*a)

    @staticmethod
    def unit():
        return 1.0


class Circle(Shape):
    __slots__ = ("r",)

    def __init__(self, r):
        super().__init__("circle")
        self.r = r

    def area(self):
        return math.pi * self.r ** 2


class Rect(Shape):
    def __init__(self, w, h=None):
        super().__init__("rect")
        self.w, self.h = w, (w if h is None else h)

    def area(self):
        return self.w * self.h

    @property
    def ratio(self):
        return self.w / self.h

    @ratio.setter
    def ratio(self, v):
        self.w = self.h * v


@dataclass(order=True, frozen=True)
class Point:
    x: float
    y: float = 0.0
    tags: tuple = ()

    def __add__(self, o):
        return Point(self.x + o.x, self.y + o.y)

    def norm(self):
        return math.hypot(self.x, self.y)


@dataclass
class Bag:
    items: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None


class Vec(list):
    def __add__(self, o):
        return Vec(a + b for a, b in zip(self, o))

    def __matmul__(self, o):
        return sum(a * b for a, b in zip(self, o))


class AppError(Exception):
    def __init__(self, msg, code=1):
        super().__init__(msg)
        self.code = code


class Timer(contextlib.ContextDecorator):
    log = []

    def __enter__(self):
        Timer.log.append("enter")
        return self

    def __exit__(self, et, ev, tb):
        Timer.log.append(f"exit:{et.__name__ if et else None}")
        return et is KeyError


@contextlib.contextmanager
def tag(name, out):
    out.append(f"<{name}>")
    try:
        yield name.upper()
    finally:
        out.append(f"</{name}>")


def counter():
    n = 0
    def inc(step=1):
        nonlocal n
        n += step
        return n
    return inc


TOTAL = 0
def add_total(*vals, scale=1, **extra):
    global TOTAL
    TOTAL += sum(vals) * scale + len(extra)
    return TOTAL


def gen_pipeline(xs):
    acc = yield "ready"
    for x in xs:
        acc = yield (x, acc)
    return "done"


def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


@functools.lru_cache(maxsize=None)
def ways(n):
    return 1 if n <= 1 else ways(n - 1) + ways(n - 2)


shapes = [Circle(1.5), Rect(2, 3), Rect.make(2), Circle.make(0.5)]
print(sorted(shapes), Shape.count, sorted(Shape.registry), Shape.unit())
r = shapes[1]
r.ratio = 4
print(r.w, r.h, r.ratio, isinstance(r, Shape), issubclass(Circle, abc.ABC))
try:
    Shape("x")
except TypeError as e:
    print("abstract:", "abstract" in str(e))
p = Point(3, 4) + Point(1, 2, ("a",))
print(p, p.norm(), Point(1) < Point(2), dataclasses.asdict(p), hash(Point(1, 2)) == hash(Point(1, 2)))
try:
    p.x = 9
except dataclasses.FrozenInstanceError as e:
    print("frozen", type(e).__name__)
b1, b2 = Bag(), Bag()
b1.items.append(1)
print(b1, b2, b1 == Bag([1]), Color.RED, Color(2).lower, [c.name for c in Color], Color["GREEN"].value)
v = Vec([1, 2, 3]) + Vec([4, 5, 6])
print(v, type(v).__name__, v @ Vec([1, 1, 1]), len(v), v[::-1])
try:
    try:
        raise KeyError("inner")
    except KeyError as e:
        raise AppError("outer", code=7) from e
except AppError as e:
    print(e, e.code, repr(e.__cause__), e.args)
out = []
with tag("b", out) as t, tag("i", out):
    out.append(t)
print(out)
with Timer() as tm:
    raise KeyError("swallowed")
print(Timer.log, type(tm).__name__)


@Timer()
def decorated(x):
    return x * 2
print(decorated(21), Timer.log[-2:])
c = counter()
c(); c(5)
print(c(), add_total(1, 2, 3, scale=2, a=1), add_total(4), TOTAL)
g = gen_pipeline([10, 20])
print(next(g), g.send("a"), g.send("b"))
try:
    g.send("c")
except StopIteration as e:
    print("returned", e.value)
print(list(itertools.islice(fib(), 12)), ways(30), ways.cache_info().hits > 0)
words = re.findall(r"\w+", "the cat and the hat and the bat")
cnt = Counter(words)
dd = defaultdict(list)
for i, w in enumerate(words):
    dd[w[0]].append(i)
Pair = namedtuple("Pair", "a b")
dq = deque(maxlen=3)
for i in range(5):
    dq.append(i)
print(cnt.most_common(2), dict(dd), Pair(1, b=2), Pair(1, 2)._replace(a=9), list(dq), OrderedDict(a=1, b=2))
h = []
for x in [5, 1, 4, 2, 3]:
    heapq.heappush(h, (-x, str(x)))
print([heapq.heappop(h)[1] for _ in range(3)], functools.reduce(operator.mul, range(1, 7)),
      functools.partial(int, base=2)("1011"))
data = {"k": [1, 2.5, None, True, {"z": "ü"}]}
s = json.dumps(data, sort_keys=True, ensure_ascii=False)
print(s, json.loads(s) == data)
m = {**{"a": 1}, "b": 2, **{"a": 3}}
st = {x % 3 for x in range(10)}
first, *mid, last = range(6)
(a, b), c2 = (1, 2), 3
print(m, st, first, mid, last, a, b, c2, 1 < 2 <= 2 != 3, 5 ^ 3, 6 | 1, 6 & 3, ~6, 1 << 10, -7 // 2, -7 % 3)
if (n := len(words)) > 5:
    print(f"{n=} {n!r:>4} {math.pi:{8}.{3}f} {'x':^5}|{12345.678:,.1f}|{255:#x}|{0.1+0.2!s}")
lock = threading.Lock()
results = []
def work(i):
    with lock:
        results.append(i * i)
ts = [threading.Thread(target=work, args=(i,)) for i in range(5)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(sorted(results), any(x > 10 for x in results), all(x >= 0 for x in results))
for i in range(3):
    pass
else:
    print("for-else", i)
k = 0
while k < 3:
    k += 1
    if k == 10:
        break
else:
    print("while-else", k)
del results[0], m["b"]
print(results, m, sorted({"b": 1, "a": 2}.items(), key=lambda kv: kv[1], reverse=True))
print(str(None), repr("q'uote\""), [None, True, 1.0, "s"], (1,), {1: (2,)}, b"\x00\xff", 1e100, 1e-7, 2**70)
print(round(2.675, 2), round(0.5), round(1.5), int("-42"), float("1e3"), divmod(-7, 2), 7 / 2, 2 ** -1)
