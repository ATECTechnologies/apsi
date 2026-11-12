fns2 = [lambda: i for i in range(3)]
print([h() for h in fns2])
class K:
    w = [a * 2 for a in range(2)]
print(K.w)
def deflt(a=[]):
    a.append(1); return len(a)
print(deflt(), deflt())
