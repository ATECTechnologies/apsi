from dataclasses import dataclass
import math

@dataclass
class Point:
    x: float
    y: float
    def norm(self):
        return math.hypot(self.x, self.y)

class Shape:
    def area(self):
        raise NotImplementedError
class Square(Shape):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def area(self):
        return self.s ** 2
    def __repr__(self):
        return f"Square({self.s})"

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

pts = [Point(3, 4), Point(1, 1)]
print([round(p.norm(), 3) for p in pts])
print(Square(3), Square(3).area(), isinstance(Square(2), Shape), issubclass(Square, Shape))
print(fib(50), sorted({"b": 2, "a": 1}.items()), frozenset([1, 2, 2]))
print({k: v for k, v in zip("abc", range(3))}, divmod(17, 5), 7 // 2, -7 // 2, 2 ** 10)
try:
    Shape().area()
except NotImplementedError:
    print("abstract")
