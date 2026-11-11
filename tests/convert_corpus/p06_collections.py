from collections import Counter, defaultdict, deque, namedtuple, OrderedDict
c = Counter("abracadabra")
print(c.most_common(2), c["a"])
d = defaultdict(list)
for w in ["ab", "ac", "b"]:
    d[w[0]].append(w)
print(dict(d))
q = deque([1, 2, 3]); q.appendleft(0); q.rotate(1); print(q, q.pop())
P = namedtuple("P", "x y")
p = P(1, 2)
print(p, p.x + p.y, p._replace(x=5), tuple(p))
import heapq
h = [5, 1, 4]; heapq.heapify(h); heapq.heappush(h, 0); print([heapq.heappop(h) for _ in range(4)])
print(sorted({"b": 2, "a": 3}.items(), key=lambda kv: -kv[1]))
