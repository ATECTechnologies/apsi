import time
def primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
def strings(n):
    d = {}
    for i in range(n):
        k = "k" + str(i % 1000)
        d[k] = d.get(k, 0) + len(k)
    return sum(d.values())
class V:
    def __init__(self, x, y):
        self.x = x; self.y = y
    def add(self, o):
        return V(self.x + o.x, self.y + o.y)
def objs(n):
    v = V(0, 0)
    for i in range(n):
        v = v.add(V(i, 1))
    return v.x + v.y
t = time.perf_counter(); a = len(primes(200000)); t1 = time.perf_counter() - t
t = time.perf_counter(); b = fib(20); t2 = time.perf_counter() - t
t = time.perf_counter(); c = strings(50000); t3 = time.perf_counter() - t
t = time.perf_counter(); d = objs(50000); t4 = time.perf_counter() - t
print(a, b, c, d)
import sys
print(f"sieve {t1:.3f}s fib {t2:.3f}s dict/str {t3:.3f}s objects {t4:.3f}s", file=sys.stderr)
