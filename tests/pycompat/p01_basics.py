import math
from collections import defaultdict, Counter

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]

words = "the quick brown fox jumps over the lazy dog the end".split()
counts = Counter(words)
groups = defaultdict(list)
for w in words:
    groups[len(w)].append(w)

print(fib(30))
print(primes(50))
print(counts.most_common(2))
print(sorted(groups.items()))
print({k: v for k, v in sorted(counts.items()) if v > 1})
total = sum(x * x for x in range(10))
print(f"total={total:5d} pi={math.pi:.3f}")
s = "Hello, World"
print(s.lower(), s.upper(), s.replace("World", "Apsi"), s[::-1], s[2:5])
print(", ".join(str(i) for i in range(5)))
print(max([3, 1, 4, 1, 5], key=lambda x: -x), min(3, 7), abs(-2.5))
nums = [5, 3, 8, 1]
nums.sort(reverse=True)
print(nums, len(nums), nums.index(8), 3 in nums)
