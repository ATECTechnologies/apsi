import math, sys
from collections import defaultdict, Counter, deque, namedtuple, OrderedDict
from functools import reduce, lru_cache, partial
from itertools import combinations, permutations, product, chain, accumulate, groupby, count, islice
from typing import List, Dict, Optional
import copy

Pair = namedtuple("Pair", ["a", "b"])

@lru_cache(maxsize=None)
def fact(n):
    return 1 if n < 2 else n * fact(n - 1)

def main() -> None:
    words = "the quick brown fox jumps over the lazy dog the end".split()
    c = Counter(words)
    print(c.most_common(2), c["the"], len(c))
    d = defaultdict(list)
    for w in words:
        d[len(w)].append(w)
    print(sorted(d.items())[:2])
    q = deque([1, 2, 3])
    q.appendleft(0); q.append(4); q.popleft()
    print(list(q), Pair(1, 2), Pair(1, 2).b)
    print(reduce(lambda x, y: x * y, range(1, 6)), fact(20), partial(pow, 2)(10))
    print(list(combinations([1, 2, 3], 2)), list(permutations("ab")), list(product([0, 1], repeat=2)))
    print(list(chain([1], [2, 3])), list(accumulate([1, 2, 3])), [(k, len(list(g))) for k, g in groupby("aabbbc")])
    print(list(islice(count(5), 3)), math.gcd(12, 18), math.isclose(0.1 + 0.2, 0.3), math.floor(2.7), math.inf)
    nested = [[1, 2], [3]]
    cp = copy.deepcopy(nested); cp[0].append(9)
    print(nested, cp, sys.maxsize > 0)
    od = OrderedDict(); od["x"] = 1; od["y"] = 2
    print(list(od.keys()), {1, 2} | {3}, None, True and not False, "a\tb", repr("it's"))
    x: Optional[int] = None
    print(x is None, f"{3.14159:.2f}", "%d items" % 5 if False else "ok", divmod(7, 2), 7 / 2, 7 // 2, -7 % 3)

if __name__ == "__main__":
    main()
