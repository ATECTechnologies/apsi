print({4, 5, 6}, {10, 3}, {100, 2, 37, 8, 64}, set(range(20, 0, -3)), {-1, -5, 7}, {1 << 40, 3})
s = set()
for i in [33, 1, 17, 9, 25, 41, 49, 57, 65, 73, 81]:
    s.add(i)
print(s, sorted(s) == sorted(list(s)), [x for x in {9, 1, 17}])
print("%5.2f|%-6s|%03d|%x|%r|%s|%+d|%e|%g|%%|%c" % (3.14159, "ab", 7, 255, "q", None, 5, 12345.678, 0.0001, 65))
print("%(a)s-%(b)d" % {"a": "x", "b": 2}, "%s" % "solo", "%.3s" % "abcdef", "%10.4f|" % 2.5)
print(type(3) == int, type("a") is str, type(3.0).__name__, type([]), int, len, type(None).__name__ if False else "NoneType")
print(isinstance(True, int), str(True), repr(None), [None, True, 'x'], ('a',), {'k': (1, 'v')})
