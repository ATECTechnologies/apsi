import heapq
from functools import lru_cache

def dijkstra(graph, start):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

@lru_cache(maxsize=None)
def ways(n):
    if n <= 1:
        return 1
    return ways(n - 1) + ways(n - 2)

def quicksort(xs):
    if len(xs) <= 1:
        return xs
    p = xs[len(xs) // 2]
    return quicksort([x for x in xs if x < p]) + [x for x in xs if x == p] + quicksort([x for x in xs if x > p])

def binary_search(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        elif xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

g = {"a": [("b", 1), ("c", 4)], "b": [("c", 2), ("d", 5)], "c": [("d", 1)], "d": []}
print(sorted(dijkstra(g, "a").items()))
print(ways(40))
data = [9, 4, 7, 1, 8, 2, 2, 6]
srt = quicksort(data)
print(srt, binary_search(srt, 7), binary_search(srt, 5))
stack = []
for c in "(()[]{})":
    if c in "([{":
        stack.append(c)
    else:
        stack.pop()
print(len(stack) == 0)
