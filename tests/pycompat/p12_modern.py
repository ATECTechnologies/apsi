# apsi-flags: --allow-python=all
import dataclasses, enum, functools, itertools, collections, contextlib, asyncio, re, math, operator
from dataclasses import dataclass, field
from typing import NamedTuple

# --- match
@dataclass
class Point:
    x: int
    y: int = 0

class Color(enum.Enum):
    RED = 1
    GREEN = 2

def describe(v):
    match v:
        case None:
            return "none"
        case True | False as b:
            return f"bool {b}"
        case 0 | 1:
            return "small"
        case int(n) if n < 0:
            return f"neg {n}"
        case int() | float():
            return "num"
        case "a" | "b":
            return "ab"
        case str(s):
            return "str " + s
        case [x, y, *rest]:
            return f"seq {x} {y} {rest}"
        case [x]:
            return f"one {x}"
        case {"k": k, **others}:
            return f"map {k} {sorted(others)}"
        case Point(x=0, y=yy):
            return f"yaxis {yy}"
        case Point(x, y) if x == y:
            return f"diag {x}"
        case Point():
            return "point"
        case Color.RED:
            return "red"
        case (1, (2, z)):
            return f"nested {z}"
        case _:
            return "other"

Point.__match_args__ = ("x", "y")
for v in [None, True, 0, -5, 3.5, "a", "zz", [1, 2, 3, 4], [9], {"k": 1, "m": 2, "a": 0}, Point(0, 4), Point(2, 2), Point(1, 3), Color.RED, Color.GREEN, (1, (2, 7)), ()]:
    print(repr(v), "->", describe(v))

# --- except*
def eg():
    try:
        raise ExceptionGroup("g", [ValueError("v"), TypeError("t"), KeyError("k")])
    except* ValueError as e:
        print("val", [str(x) for x in e.exceptions])
    except* (TypeError, KeyError) as e:
        print("tk", sorted(type(x).__name__ for x in e.exceptions))
eg()

# computed except kinds
KINDS = (ValueError, ZeroDivisionError)
def comp(i):
    try:
        if i == 0: 1 / 0
        if i == 1: int("x")
        if i == 2: [][1]
    except KINDS as e:
        return "kinds " + type(e).__name__
    except LookupError:
        return "lookup"
    return "ok"
print([comp(i) for i in range(4)])

# --- clash names
def f(match, case, type, var=3, let=4, fn=5):
    print(match, case, type, var, let, fn)
f(1, 2, 3)
var = 10; let = var + 1
print(var, let)

# --- comprehension targets
class Box: pass
b = Box()
d = {}
print([b.v for b.v in range(3)], b.v)
print([d["k"] for d["k"] in "xy"], d)

# --- async comprehension
async def agen(n):
    for i in range(n):
        await asyncio.sleep(0)
        yield i * i
async def main():
    xs = [x async for x in agen(5) if x % 2 == 0]
    ys = {x: x + 1 async for x in agen(3)}
    zs = [await asyncio.sleep(0, result=i) for i in range(3)]
    print(xs, ys, zs)
asyncio.run(main())

# --- type alias and generics
type Pair = tuple[int, int]
print(Pair.__name__, Pair.__value__)
def first[T](xs: list[T]) -> T:
    return xs[0]
print(first([7, 8]))
class Stack[T]:
    def __init__(self): self.items = []
    def push(self, x: T): self.items.append(x); return self
print(Stack().push(1).push(2).items)

# --- misc python
a_none = 1
print(1 < 2 < 3, 3 > 2 > 2, a_none is not None)
print(divmod(-7, 2), -7 // 2, -7 % 2, 7 ^ 3, ~5, 1 << 70, 2 ** -1)
print(f"{3.14159:.2f} {'x'!r} {10:>5}|{255:#x} {1e100}")
a, *b_, c = range(6); print(a, b_, c)
x = y = [1]; x.append(2); print(y)
print((n := 5) + n)
print(sorted({3, 1, 2}), {i % 3 for i in range(10)}, list(itertools.accumulate([1, 2, 3])))
print(functools.reduce(operator.mul, range(1, 10)))
print(collections.Counter("abracadabra").most_common(2))
for i in range(3):
    if i == 5: break
else:
    print("loop else")
w = 0
while w < 2:
    w += 1
else:
    print("while else", w)
with contextlib.suppress(KeyError):
    {}["x"]
print("suppressed")
s = "héllo wörld"
print(s.encode(), b"\x00\xff".hex(), bytes(range(3)), bytearray(b"ab") + b"c")
print(round(2.5), round(3.5), round(1.005, 2), 0.1 + 0.2, float("nan") == float("nan"), math.inf)
print(re.sub(r"(\w)(\w*)", lambda m: m[2] + m[1], "hello world"))
class P(NamedTuple):
    a: int
    b: str = "z"
print(P(1), P(1)._replace(b="q"))
lam = lambda *a, **k: (a, k)
print(lam(1, 2, z=3))
def gen():
    r = yield 1
    print("got", r)
    yield from range(2)
g = gen(); print(next(g), g.send("hi"), list(g))
print(dataclasses.asdict(Point(1, 2)))
print("%s-%d-%.3f" % ("a", 5, 1/3))
print(int.from_bytes(b"\x01\x02", "big"), (1024).to_bytes(2, "little"))
x = [1, 2, 3, 4, 5]; x[1:3] = [9]; del x[0]; print(x, x[::-1], x[-2:])
global_counter = 0
def inc():
    global global_counter
    global_counter += 1
inc(); inc(); print(global_counter)
def outer():
    n = 0
    def inner():
        nonlocal n
        n += 1
        return n
    inner(); return inner()
print(outer())
print(str(1/3), repr(1e-30), 1e-30 * 1e-300, 10**30 + 1)
