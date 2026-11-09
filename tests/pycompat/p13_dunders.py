import abc, json, heapq, bisect, fractions, decimal, string, textwrap, struct, copy, weakref, sys, functools, typing
from collections import defaultdict, OrderedDict, deque, ChainMap

class Meta(type):
    registry = []
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns)
        mcls.registry.append(name)
        cls.tag = kw.get("tag", "none")
        return cls

class Base(metaclass=Meta, tag="base"):
    __slots__ = ("v",)
    def __init__(self, v): self.v = v
    def __repr__(self): return f"{type(self).__name__}({self.v!r})"
    def __eq__(self, o): return isinstance(o, Base) and o.v == self.v
    def __hash__(self): return hash(self.v)
    def __lt__(self, o): return self.v < o.v
    def __add__(self, o): return type(self)(self.v + (o.v if isinstance(o, Base) else o))
    __radd__ = __add__
    def __len__(self): return 3
    def __getitem__(self, i):
        if i >= 3: raise IndexError(i)
        return self.v * i
    def __contains__(self, x): return x == self.v
    def __call__(self, *a): return sum(a) + self.v
    def __bool__(self): return self.v != 0

class Child(Base, tag="child"):
    __slots__ = ()
    def __repr__(self): return "Child:" + super().__repr__()

print(Meta.registry, Base.tag, Child.tag)
c = Child(2)
print(c, c + 1, 5 + c, sorted([Child(3), Child(1)]), list(c), 2 in c, c(1, 2), bool(Child(0)), {c, Child(2)})
try:
    c.zz = 1
except AttributeError as e:
    print("slots:", type(e).__name__)

class Shape(abc.ABC):
    @abc.abstractmethod
    def area(self): ...
    @classmethod
    def make(cls, *a): return cls(*a)
    @staticmethod
    def unit(): return "m2"
class Sq(Shape):
    def __init__(self, s): self._s = s
    @property
    def side(self): return self._s
    @side.setter
    def side(self, v):
        if v < 0: raise ValueError("neg")
        self._s = v
    def area(self): return self._s ** 2
try:
    Shape()
except TypeError as e:
    print("abstract ok")
q = Sq.make(3); q.side = 4; print(q.area(), Sq.unit(), q.side)

class Lazy:
    def __init__(self): self.d = {"a": 1}
    def __getattr__(self, n):
        if n in self.d: return self.d[n]
        raise AttributeError(n)
l = Lazy(); print(l.a, getattr(l, "b", "dflt"), hasattr(l, "zz"))

class Desc:
    def __set_name__(self, owner, name): self.name = "_" + name
    def __get__(self, obj, typ=None): return 0 if obj is None else getattr(obj, self.name, 0) * 10
    def __set__(self, obj, v): setattr(obj, self.name, v)
class UsesDesc:
    x = Desc()
u = UsesDesc(); u.x = 4; print(u.x, UsesDesc.x)

class Ctx:
    def __enter__(self): print("enter"); return self
    def __exit__(self, et, ev, tb):
        print("exit", et.__name__ if et else None); return True
with Ctx() as cc, Ctx():
    raise KeyError("x")
print("after with")

class Counter:
    def __init__(self, n): self.n = n
    def __iter__(self): return self
    def __next__(self):
        if self.n <= 0: raise StopIteration
        self.n -= 1; return self.n
print(list(Counter(4)))

class MyErr(Exception):
    def __init__(self, code, msg="bad"):
        super().__init__(msg); self.code = code
try:
    try:
        raise MyErr(42)
    except MyErr as e:
        raise RuntimeError("wrapped") from e
except RuntimeError as e:
    print(e, type(e.__cause__).__name__, e.__cause__.code, e.__cause__.args)
finally:
    print("finally")

def deco(tag):
    def wrap(f):
        @functools.wraps(f)
        def inner(*a, **k): return f"{tag}<{f(*a, **k)}>"
        return inner
    return wrap
@deco("b")
@deco("i")
def hello(name, /, greeting="hi", *, punct="!"): return f"{greeting} {name}{punct}"
print(hello("x"), hello("y", "yo", punct="?"), hello.__name__)

@functools.lru_cache(maxsize=None)
def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(90), fib.cache_info().hits)

dd = defaultdict(list)
for w in "apple bat avocado cherry banana".split(): dd[w[0]].append(w)
print(dict(dd), OrderedDict(b=1, a=2), deque([1, 2, 3], maxlen=2), ChainMap({"a": 1}, {"a": 2, "b": 3})["b"])
h = [5, 1, 4]; heapq.heapify(h); heapq.heappush(h, 0); print([heapq.heappop(h) for _ in range(4)])
print(bisect.bisect_left([1, 3, 5], 4), fractions.Fraction(3, 9) + 1, decimal.Decimal("0.1") * 3)
print(json.dumps({"b": [1, 2.5, None, True], "a": "é"}, sort_keys=True), json.loads('{"x": [1, {"y": null}]}'))
print(json.dumps({"k": 1}, indent=2))
print(string.ascii_lowercase[:5], textwrap.fill("a b c d e f g", 5), struct.pack("<iH", 1, 2), struct.unpack("<d", struct.pack("<d", 1.5)))
x = [[1, 2], [3]]; y = copy.deepcopy(x); y[0].append(9); print(x, y)
print("  Abc  ".strip().lower().center(7, "*"), "a,b,,c".split(","), "-".join(map(str, range(4))), "abc".zfill(5), "x".join("12"))
print("{:,}".format(1234567), "{:e}".format(12345.678), "{:08.3f}".format(-3.14159), "{:%}".format(0.25), f"{'left':<8}|", f"{42:b}")
print(complex(1, 2) * 1j, abs(3 + 4j), 7 / 2, 7 // -2, 2 ** 100, pow(3, 200, 7), int("ff", 16), bin(10), oct(8), hex(255))
print(sorted(["b", "A", "c"], key=str.lower), sorted([(1, "b"), (1, "a"), (0, "z")], reverse=True), max("abc", key=ord), min([], default=7))
print(list(zip("ab", [1, 2], strict=True)), list(enumerate("xy", 1)), any([]), all([]), list(reversed(range(3))), list(filter(None, [0, 1, 2])))
print(isinstance(True, int), issubclass(bool, int), type(1.0).__name__, callable(len), id(None) == id(None))
d = {"a": 1}; d |= {"b": 2}; print(d, d.get("z"), d.setdefault("c", 3), d.pop("a"), list(d.items()))
s = {1, 2}; s |= {3}; s &= {2, 3, 4}; print(s, frozenset([1]) | {2}, {1, 2} ^ {2, 3}, {1} <= {1, 2})
ba = bytearray(b"abc"); ba[0] = 65; print(ba, bytes(ba).decode(), memoryview(b"xy")[0], b"a b".split())
r = weakref.ref(q); print(r() is q)
print(sys.maxsize > 2**31, typing.get_type_hints(hello) == {}, len(sys.argv))
def kw(**k): return sorted(k.items())
print(kw(**{"a": 1}, b=2), [*range(2), *"ab"], {**{"a": 1}, "b": 2}, print.__name__)
print(chr(9731), ord("€"), "ß".upper(), "İ".lower().encode(), "a\tb".expandtabs(4), repr("q'\"\n"))
it = iter([1, 2, 3]); print(next(it), list(it), next(it, "end"))
print([[i * j for j in range(3)] for i in range(3)], {k: v for k, v in zip("ab", "cd")}, sum(x * x for x in range(10)))
