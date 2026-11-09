import sys
import decimal
from decimal import Decimal, getcontext
sys.setrecursionlimit(5000)
getcontext().prec = 50
print(Decimal(1) / Decimal(7))
decimal.getcontext().prec = 30
print(Decimal(2).sqrt())
sys.stdout.write("no newline|")
sys.stdout.write("done\n")
print(sys.maxsize > 2**62, sys.float_info.epsilon < 1e-15)
def depth(n):
    return 0 if n == 0 else 1 + depth(n - 1)
print(depth(3000))
vals = [1, 2, 5]
print([v * 10 ** d for d in (-1, 0, 1) for v in vals])
print({(a, b) for a in range(2) for b in range(2)} == {(0, 0), (0, 1), (1, 0), (1, 1)})
print(hash("x") == hash("x"), isinstance(3, (int, float)), callable(len))
print("a", "b", sep="-", end="!\n")
