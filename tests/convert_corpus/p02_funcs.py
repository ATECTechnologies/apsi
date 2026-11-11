def f(a, b=2, *args, c=3, **kw):
    return (a, b, args, c, sorted(kw.items()))
print(f(1), f(1, 5, 6, 7, c=9, z=1, y=2))
def outer():
    n = 0
    def inc():
        nonlocal n
        n += 1
        return n
    return inc
g = outer(); g(); print(g())
sq = lambda v: v * v
print(list(map(sq, range(5))), [i for i in range(10) if i % 3 == 0])
print({k: v for k, v in zip("abc", range(3))}, {i % 3 for i in range(10)})
total = 0
def add(v):
    global total
    total += v
add(3); add(4); print(total)
