import sys
fs = [lambda: i for i in range(3)]
print([f() for f in fs])
fs = [lambda i=i: i for i in range(3)]
print([f() for f in fs])
x = "global"
class K:
    x = "class"
    y = [x for _ in range(1)]
    def m(self): return x
print(K.y, K().m())
def f():
    try:
        return "try"
    finally:
        print("fin runs")
print(f())
def g():
    for i in range(3):
        try:
            if i == 1: continue
            if i == 2: return i
        finally:
            print("f", i)
print(g())
def unbound():
    try:
        print(z)
        z = 1
    except UnboundLocalError as e:
        print("unbound", type(e).__name__)
unbound()
a = 1
del a
try:
    a
except NameError as e:
    print("deleted:", e)
def shadow():
    if False:
        y = 1
    return "y" in locals() if False else "ok"
print(shadow())
i = 99
for i in range(3): pass
print(i)
[j for j in range(5)]
try: j
except NameError: print("comp var not leaked")
def gen():
    try:
        yield 1
        yield 2
    finally:
        print("gen closed")
gg = gen(); next(gg); gg.close()
def nested():
    v = []
    def add(n):
        v.append(n)
        return v
    return add
ad = nested(); ad(1); print(ad(2))
def kwonly(a, b=2, *args, c, d=4, **kw): return a, b, args, c, d, kw
print(kwonly(1, c=3), kwonly(1, 2, 3, 4, c=5, e=6))
def mut(l=[]):
    l.append(1); return len(l)
print(mut(), mut(), mut([]))
t = (1, [2]); t[1].append(3); print(t)
try:
    t[0] = 5
except TypeError as e: print(e)
print(0.1 + 0.7, 1/3*3, 2**0.5, int(-2.7), round(-2.5), 10 / 4, -0.0, 1e16 + 1, float("1e-30"))
print(True + True, "%r" % (None,), str(b"x"), [] == [], [] is [], (1,) * 3, "ab" * 2, [0] * 3)
print(sys.getrecursionlimit() > 100)
def rec(n): return 0 if n == 0 else 1 + rec(n - 1)
print(rec(500))
n = 0
while True:
    n += 1
    if n % 2: continue
    if n > 6: break
print(n)
print(*[1, 2], sep="-", end="!\n")
print("a", file=sys.stderr)
class A:
    def __init__(self): self.__p = 5
    def get(self): return self.__p
print(A().get(), hasattr(A(), "_A__p"))
class B:
    count = 0
    def __init__(self): B.count += 1; type(self).count += 0
B(); B(); print(B.count)
s = "x"
s += "y"; s *= 2; print(s)
l = [3, 1, 2]; l.sort(key=lambda v: -v); print(l, l.index(2), l.count(9))
print(divmod(7.5, 2), 7.5 % -2, abs(-0.0), max(1, 2.0), min("b", "a"))
assert 1 == 1, "never"
try:
    assert 1 == 2, "msg"
except AssertionError as e: print("assert", e)
print(eval.__name__ if False else "no eval")
