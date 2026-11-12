from itertools import islice, count, chain, permutations
from dataclasses import dataclass, field

def naturals():
    n = 0
    while True:
        yield n
        n += 1

def make_counter():
    c = 0
    def inc():
        nonlocal c
        c += 1
        return c
    return inc

class Vec:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y)
    def __mul__(self, k):
        return Vec(self.x * k, self.y * k)
    def __eq__(self, o):
        return self.x == o.x and self.y == o.y
    def __repr__(self):
        return f"Vec({self.x}, {self.y})"
    def __len__(self):
        return 2

class InsufficientFunds(Exception):
    pass

class Account:
    def __init__(self, balance=0):
        self.balance = balance
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFunds(f"need {amount - self.balance} more")
        self.balance -= amount
        return self.balance

@dataclass
class Inventory:
    items: list = field(default_factory=list)
    def add(self, x):
        self.items.append(x)

print(list(islice(naturals(), 5)))
inc = make_counter()
inc(); inc()
print(inc())
v = Vec(1, 2) + Vec(3, 4) * 2
print(v, v == Vec(7, 10), len(v))
acc = Account(50)
try:
    acc.withdraw(80)
except InsufficientFunds as e:
    print("error:", e)
print(acc.withdraw(20))
inv = Inventory()
inv.add("apple")
print(inv.items, Inventory().items)
d = {}
d.setdefault("a", []).append(1)
d.update({"b": 2})
print(d, d.pop("b"), d)
xs = [1, 2, 3]
xs.insert(0, 0); xs.extend([4, 5]); xs.remove(3)
print(xs, xs.pop(), xs[-2:], list(reversed(xs)))
for i in range(3):
    if i == 10:
        break
else:
    print("loop finished")
print(list(map(lambda t: t * 2, [1, 2])), list(filter(None, [0, 1, 2])))
print(1 < 2 < 3, "ab" * 3, 2 ** 100)
people = [("bob", 25), ("amy", 30), ("cat", 25)]
print(sorted(people, key=lambda p: (p[1], p[0])))
print(list(chain([1], [2, 3])), len(list(permutations("abc"))))
if (n := len(xs)) > 3:
    print("walrus", n)
print("%s is %d years, %.1f%%" % ("bob", 25, 12.345))
total = 0
for k, val in enumerate(["a", "b"], start=1):
    total += k
print(total, sum([1, 2], 10), [(i, j) for i in range(2) for j in range(2) if i != j])
