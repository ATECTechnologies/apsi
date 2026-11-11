for i in range(3):
    for j in range(3):
        if j == 2: break
        if j == 0: continue
        print(i, j)
else:
    print("no break")
n = 0
while n < 10:
    n += 3
else:
    print("while else", n)
x = 7
r = "big" if x > 5 else "small"
print(r, [y for y in range(20) if y % 2 if y % 3], any(v > 3 for v in [1, 5]), all([]))
if (k := len("abcd")) > 3:
    print("walrus", k)
a = b = 3
a += 1
print(a, b, 1 < 2 < 3, 1 < 3 < 2, not [], None is None, [] == [])
for i, (p, q) in enumerate(zip("ab", [1, 2]), start=1):
    print(i, p, q)
