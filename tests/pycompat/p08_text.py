import re
from collections import namedtuple, deque, OrderedDict

Point = namedtuple("Point", ["x", "y"])
p = Point(1, 2)
print(p.x + p.y, p)
q = deque([1, 2, 3])
q.appendleft(0); q.append(4); q.popleft()
print(list(q))
log = """2024-01-01 ERROR disk full
2024-01-02 INFO ok
2024-01-03 ERROR cpu hot"""
errors = [m.group(1) for m in re.finditer(r"ERROR (.+)", log)]
print(errors, re.sub(r"\d", "#", "a1b22"), bool(re.match(r"^\d{4}", log)))
def wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    return out
print(wrap("the quick brown fox jumps over the lazy dog", 12))
def caesar(s, k):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.isalpha() else c for c in s.lower())
print(caesar("Hello World", 3))
od = OrderedDict()
od["z"] = 1; od["a"] = 2
print(list(od.keys()))
matrix = [[1, 2, 3], [4, 5, 6]]
print([list(r) for r in zip(*matrix)])
print("{:>6}|{:<6}|{:^6}".format("r", "l", "c"))
print(f"{1234567.891:,.2f}", f"{0.25:.0%}", f"{255:x}", f"{7:03d}")
