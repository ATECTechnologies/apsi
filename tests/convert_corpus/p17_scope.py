x = 1
def f():
    print(x)
f()
def g():
    y = 0
    for i in range(3):
        y += i
    if y > 1:
        z = "big"
    else:
        z = "small"
    return z, i
print(g())
fns = [lambda i=i: i for i in range(3)]
print([h() for h in fns])
fns2 = [lambda: i for i in range(3)]
print([h() for h in fns2])
class K:
    v = [1]
    w = [a * 2 for a in range(2)]
print(K.v, K.w)
def deflt(a=[]):
    a.append(1); return len(a)
print(deflt(), deflt())
