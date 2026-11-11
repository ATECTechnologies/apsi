import math
print(math.sqrt(2), math.floor(-2.5), math.ceil(2.1), round(2.5), round(3.5), round(2.675, 2), abs(-3))
print(divmod(17, 5), pow(3, 4, 5), 0.1 + 0.2, 1e300 * 10, int("ff", 16), bin(10), hex(255), int(3.9))
print(max([3, 1, 2]), min("bca"), sum([0.1] * 10), sorted([3, 1, 2], reverse=True), 10 ** 20, 2 ** 100)
print(True + True, 7 // -2, -7 % -3, float("inf") > 1e308, math.isnan(float("nan")), 1 / 3)
from fractions import Fraction
from decimal import Decimal
print(Fraction(1, 3) + Fraction(1, 6), Decimal("0.1") + Decimal("0.2"))
print(complex(1, 2) * 2, (1+2j).real, isinstance(3, (int, float)), 3 == 3.0, hash(1) == hash(1.0))
