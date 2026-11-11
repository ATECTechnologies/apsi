def quicksort(xs):
    if len(xs) <= 1:
        return xs
    p, *rest = xs
    return quicksort([x for x in rest if x < p]) + [p] + quicksort([x for x in rest if x >= p])
print(quicksort([5, 3, 8, 1, 9, 2]))
def bfs(g, s):
    seen, order, q = {s}, [], [s]
    while q:
        u = q.pop(0); order.append(u)
        for v in g.get(u, []):
            if v not in seen:
                seen.add(v); q.append(v)
    return order
print(bfs({1: [2, 3], 2: [4], 3: [4], 4: []}, 1))
grid = [[0] * 3 for _ in range(3)]
grid[1][1] = 5
print(grid, [row[:] for row in grid][1], list(zip(*grid))[1])
memo = {}
def ways(n):
    if n in memo: return memo[n]
    r = 1 if n <= 1 else ways(n - 1) + ways(n - 2)
    memo[n] = r
    return r
print(ways(80))
print(sorted(["bb", "a", "ccc"], key=len), sorted([(2, "b"), (1, "z"), (2, "a")]))
