import time
from dataclasses import dataclass
class Node:
    __slots__ = ("val", "left", "right")
    def __init__(self, v, l=None, r=None): self.val, self.left, self.right = v, l, r
def make(d): return Node(d) if d == 0 else Node(d, make(d - 1), make(d - 1))
def check(n): return 1 if n.left is None else 1 + check(n.left) + check(n.right)
@dataclass
class P:
    x: float
    y: float
    def norm(self): return (self.x * self.x + self.y * self.y) ** 0.5
t = time.perf_counter()
total = sum(check(make(14)) for _ in range(8))
ps = [P(i, i + 1) for i in range(200000)]
s = sum(p.norm() for p in ps)
d = {}
for i in range(300000): d[f"k{i % 1000}"] = d.get(f"k{i % 1000}", 0) + i
words = " ".join(f"w{i % 97}" for i in range(200000)).split()
from collections import Counter
c = Counter(words).most_common(3)
print("objs", total, round(s, 3), len(d), c[0], f"{time.perf_counter() - t:.3f}s")
