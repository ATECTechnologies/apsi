def gen(n):
    for i in range(n):
        yield i * i
def chain():
    yield from gen(3)
    yield 100
print(list(chain()), sum(x for x in gen(5)))
import itertools
c = itertools.count(1)
print([next(c) for _ in range(3)], list(itertools.islice(gen(100), 2, 5)))
it = iter([1, 2])
print(next(it), next(it), next(it, "end"))
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
print(list(itertools.takewhile(lambda v: v < 50, fib())))
