import math
from typing import List

class Account:
    rate = 0.05
    count = 0

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self._balance = balance
        Account.count += 1

    @property
    def balance(self):
        return self._balance

    @staticmethod
    def fee(x):
        return round(x * 0.01, 2)

    @classmethod
    def empty(cls, owner):
        return cls(owner)

    def deposit(self, amt):
        if amt <= 0:
            raise ValueError(f"bad amount {amt}")
        self._balance += amt
        return self

    def __repr__(self):
        return f"Account({self.owner!r}, {self._balance:.2f})"

    def __lt__(self, other):
        return self._balance < other._balance

    def __eq__(self, other):
        return isinstance(other, Account) and self._balance == other._balance

def gen(n):
    for i in range(n):
        if i % 2:
            yield i * i

def stats(*xs, **opts):
    total = sum(xs)
    scale = opts.get("scale", 1)
    return total * scale, len(xs), sorted(opts)

counter = 0
def bump():
    global counter
    counter += 1
    return counter

def outer():
    n = 0
    def inner():
        nonlocal n
        n += 2
        return n
    inner(); inner()
    return n

accts = [Account("ann", 50), Account("bob"), Account.empty("cy")]
accts[1].deposit(20).deposit(5.5)
try:
    accts[2].deposit(-1)
except ValueError as e:
    print("error:", e)
finally:
    print("checked")
print(sorted(accts), max(accts).owner, accts[0].balance, Account.fee(123.456), Account.count)
print(list(gen(7)), stats(1, 2, 3, scale=2, z=1), bump(), bump(), outer())
words = ["banana", "Apple", "cherry", "date"]
print(sorted(words, key=str.lower), sorted(words, key=len, reverse=True), [w.upper()[:3] for w in words if "a" in w])
print({w: len(w) for w in words}, {len(w) for w in words}, list(enumerate(words, 1))[:2], list(zip(words, range(3))))
print("%s has %d items, %.3f avg" % ("cart", 3, 2.0 / 3), "{:>8}|{:<5}|{:^7}".format("r", "l", "c"), "-".join(map(str, range(4))))
s = "  Hello, World  "
print(s.strip().lower(), s.split(","), s.replace("l", "L"), s.find("W"), s.count("l"), s.strip().startswith("He"), "x".join("abc"))
print("abc"[::-1], "a,b,,c".split(","), " ".join(reversed("one two".split())), "Title case".title(), "42".zfill(5), "ab" * 3)
d = {"a": 1, "b": 2}
d.update(c=3); d.setdefault("a", 9); d.pop("b")
print(d, list(d.items()), d.get("z", 0), "a" in d, sorted(d.values(), reverse=True))
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(nums[1:5], nums[::2], nums[-3:], sum(nums) / len(nums), min(nums), any(n > 8 for n in nums), all(n > 0 for n in nums))
print(math.sqrt(16), math.pi, 10 ** 20, 2 ** 0.5, round(2.675, 2), round(3.5), round(4.5), int("7") + float("1.5"), abs(-3))
matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
print(matrix, [list(r) for r in zip(*matrix)], [x for row in matrix for x in row if x % 2 == 0])
a, *rest = [1, 2, 3, 4]
print(a, rest, (lambda x, y=2: x * y)(5), isinstance(3, (int, float)), type(3.0).__name__, str(None), bool([]))
with open("tmp_p3.txt", "w") as fh:
    fh.write("line1\nline2\n")
with open("tmp_p3.txt") as fh:
    print([l.rstrip() for l in fh])
