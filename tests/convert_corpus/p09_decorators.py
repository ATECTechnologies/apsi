import functools
def deco(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        print("call", fn.__name__)
        return fn(*a, **k)
    return w
@deco
def add(a, b):
    return a + b
print(add(1, 2))
@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(60))
def repeat(k):
    def d(fn):
        def w():
            return [fn() for _ in range(k)]
        return w
    return d
@repeat(3)
def one(): return 1
print(one())
from dataclasses import dataclass, field
@dataclass
class Pt:
    x: int
    y: int = 0
    tags: list = field(default_factory=list)
p = Pt(1); p.tags.append("a"); print(p, p == Pt(1, 0, ["a"]))
