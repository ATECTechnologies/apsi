from collections import OrderedDict
print(0.1 + 0.2, 1 / 3, 2 / 3 * 3, 1e20, 1e-7, 123456789.123456789, float("inf"), -0.0)
print(round(2.5), round(3.5), round(-2.5), round(2.675, 2), -7 // 2, -7 % 3, 7 % -3, divmod(-7, 2))
print(int("ff", 16), int("0b101", 2), hex(255), oct(8), f"{255:08b}", f"{1234567:,}", f"{0.000123:e}", f"{12.5:+.1f}")
print("%r %s %5.2f %-4d| %x" % ("a", "b", 3.14159, 42, 255), "{!r} {:>5} {:^7}".format("q", 3, "mid"))
s = "Hello, World, Again"
print(s.partition(", "), s.rsplit(", ", 1), s.ljust(22, "."), s.rjust(22), s.swapcase(), " ".isspace())
xs = list(range(10))
xs[1:3] = [99]
del xs[0]
print(xs, xs[::-2], xs[-3:], xs[:-7], list(range(10, 0, -3)), list(reversed(range(3))))
a, b = 1, 2
a, b = b, a
(c, d), e = (3, 4), 5
print(a, b, c, d, e, [*xs[:2], *[7, 8]], {**{"k": 1}, **{"j": 2}})
print(1 < 2 <= 2 < 3, 2 ** 100, 10 ** 20 + 1, abs(-3), max([], default=0), min(3, 1, 2), sum([0.1] * 10))
d = {"b": 2, "a": 1, "c": 3}
print(sorted(d, key=d.get), list(d.items())[:2], "a" in d, d.get("z") is None, len(d))
def make_adder(n, k=10):
    def add(x):
        return x + n + k
    return add
total = 0
def bump():
    global total
    total += 1
bump(); bump()
print(make_adder(1)(2), total, list(map(str, [1, 2])), "-".join(map(str, range(3))), True + True, bool([]))
i = 0
while True:
    i += 1
    if i > 3:
        break
print(i, isinstance(True, int), type(3).__name__, type("s").__name__, str(None), repr("x"), [1, [2, [3]]])
od = OrderedDict([("x", 1), ("y", 2)])
print(list(od), ord("€"), chr(0x1F600) == "😀", "ß".upper(), "İ".lower() == "i̇", "a\tb".expandtabs(4))
print(f"{3.0}", f"{1e16}", f"{2.0 ** 0.5}", 7 / 2, 8 / 2, 1e300 * 1e10, 5e-324 / 2, (-8) ** (1 / 3) if False else 0)
