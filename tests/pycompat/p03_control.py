def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n % 2 == 0:
        return "even"
    else:
        return "odd"

def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    finally:
        pass

def gen_squares(n):
    for i in range(n):
        yield i * i

def collatz(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def varargs(*args, **kwargs):
    return sum(args), sorted(kwargs.items())

print([classify(x) for x in (-1, 0, 3, 4)])
print(safe_div(1, 2), safe_div(1, 0))
print(list(gen_squares(5)))
print(collatz(27))
print(varargs(1, 2, 3, a=1, b=2))
first, *rest = [1, 2, 3, 4]
print(first, rest)
x = 10
x += 5; x *= 2; x -= 1; x //= 3
print(x, 7 % 3, 2 ** 10, -7 // 2, 7 / 2)
matrix = [[i * j for j in range(3)] for i in range(3)]
print(matrix, [row[1] for row in matrix])
pairs = list(zip("abc", [1, 2, 3]))
print(pairs, dict(pairs))
for i, (a, b) in enumerate(pairs):
    if i == 2:
        break
    print(i, a, b)
print(any(v > 2 for _, v in pairs), all(v > 0 for _, v in pairs))
s = {1, 2, 3}
s.add(4)
print(sorted(s | {9}), sorted(s & {2, 3, 7}), len(s))
t = (1, 2, 3)
print(t[1], len(t), t + (4,))
print(round(3.14159, 2), int("42") + 1, float("2.5"), str(12) + "!", bool(0))
print(divmod(17, 5), pow(2, 8), hex(255), bin(5), chr(65), ord("A"))
