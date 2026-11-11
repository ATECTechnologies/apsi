x = 5
y = x * 2 + 3
print(x, y, x / 2, x // 2, x % 3, -7 // 2, -7 % 3, 2 ** 10, 7.5 // 2)
s = "hello"
print(s[1:4], s[::-1], s.upper(), len(s), s * 2, "ell" in s)
a, *b, c = [1, 2, 3, 4, 5]
print(a, b, c)
print(f"{x:>5}|{3.14159:.2f}|{y!r}|{'q'!s}|{x=}")
print("%d-%s-%.1f" % (1, "a", 2.25), "{} {name}".format(1, name="n"))
