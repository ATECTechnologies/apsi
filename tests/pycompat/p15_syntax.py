# apsi-flags: --allow-python=all
import asyncio, contextlib
def g2():
    r = yield from sub()
    print("sub returned", r)
def sub():
    yield "a"
    return 42
print(list(g2()))
class Arr:
    def __getitem__(self, k): return k
    def __class_getitem__(cls, k): return f"generic {k}"
a = Arr()
print(a[1:2, ...], a[...], a[::2], Arr[int], a[*(1, 2)], a[1, *(2, 3)])
@contextlib.asynccontextmanager
async def actx():
    print("in"); yield 5; print("out")
class AIt:
    def __init__(self): self.i = 0
    def __aiter__(self): return self
    async def __anext__(self):
        self.i += 1
        if self.i > 3: raise StopAsyncIteration
        return self.i
async def amain():
    async with actx() as v:
        print("v", v)
    async for x in AIt():
        print("x", x)
    else:
        print("aloop done")
    res = await asyncio.gather(*(asyncio.sleep(0, result=i) for i in range(3)))
    print(res)
asyncio.run(amain())
print(0o17, 0b101, 0xff, 1_000_000, 3j, 1.5e-3, .5, 5., 1e309)
s = ("implicit " "concat "
     'mixed')
print(s, r"raw\n", b"%d-%s" % (5, b"x"), """multi
line""", 'é\N{BULLET}')
x = 5
print(f"{x!r:>10}|{'a' + 'b'}|{x if x else 0}|{[1, 2][0]}|{ {'k': 1}['k'] }|{x:{'>'}{4}}|{x=}|{3.0:g}")
print(f"{'quote'!r}", f'{"dq"}', f"{x:#06b}", f"{{literal}}", f"{1_000:_}")
y = [1, 2, 3, 4]; z = type("Z", (), {})(); z.c = 1
del y[1:2], z.c
print(y, hasattr(z, "c"))
def side(v): print("side", v); return v
print(side(1) < side(2) < side(0) < side(5))
print(1 if 0 else 2 if 0 else 3, not 1 in [1], 2 not in [1], None is None)
f = lambda x, y=lambda: 7: x + y()
print(f(1))
try:
    pass
except Exception:
    print("no")
else:
    print("else runs")
print({k: v for k, v in enumerate("abc") if k != 1}, sorted({c for c in "hello" if c > "f"}))
print(...,  Ellipsis is ..., NotImplemented)
if __name__ == "__main__":
    print("main guard", __name__)
for i, (a1, *b1) in enumerate([(1, 2, 3), (4,)]):
    print(i, a1, b1)
matrix = [[1, 2], [3, 4]]
print([v for row in matrix for v in row if v % 2], list(map(lambda r: sum(r), matrix)))
print(sorted(set("mississippi")), "".join(sorted("cab")), "%5.1f|%-4d|%x" % (3.14159, 7, 255))
print(int(True), float(" 2.5 "), str(12).rjust(4, "0"), "a b  c".split(None, 1), "ab".partition("b"))
class Num:
    def __init__(self, v): self.v = v
    def __iadd__(self, o): self.v += o; return self
    def __matmul__(self, o): return self.v * o.v
    def __neg__(self): return Num(-self.v)
    def __format__(self, spec): return f"Num<{self.v:{spec}}>"
    def __index__(self): return self.v
n = Num(2); n += 3; print(n @ Num(4), (-n).v, f"{n:03d}", [10, 20, 30, 40, 50, 60][n], bin(n))
print(globals().get("x") if False else "skip", type(None).__name__)
