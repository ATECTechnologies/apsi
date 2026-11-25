import heapq, bisect, string, operator, json, time, os, statistics
from abc import ABC, abstractmethod
h = []
for x in [5, 1, 8, 3, 9, 2]:
    heapq.heappush(h, x)
print(h, heapq.heappop(h), h, heapq.nlargest(2, [4, 7, 1]), heapq.nsmallest(2, [4, 7, 1]))
data = [9, 4, 7, 1]
heapq.heapify(data); print(data)
a = [1, 3, 3, 7]
print(bisect.bisect_left(a, 3), bisect.bisect(a, 3), bisect.bisect_right(a, 8))
bisect.insort(a, 4); print(a)
print(string.ascii_uppercase[:5], string.digits, string.capwords("hello big world"))
print(sorted([(1, "b"), (0, "z")], key=operator.itemgetter(1)), operator.add(2, 3), list(map(operator.mul, [1, 2], [3, 4])))
obj = {"name": "x", "vals": [1, 2.5, None, True], "nested": {"k": "é"}}
s = json.dumps(obj)
print(s, json.loads(s) == obj, json.dumps(obj, indent=2, sort_keys=True), json.dumps([1, 2], separators=(",", ":")))
t0 = time.time(); print(time.time() >= t0, isinstance(time.perf_counter(), float))
print(os.path.join("a", "b", "c.txt"), os.path.splitext("dir/f.tar.gz"), os.path.basename("/x/y.py"), os.path.dirname("/x/y.py"), os.path.exists("nope_x"))
print(statistics.mean([1, 2, 3, 4]), statistics.median([3, 1, 2]), statistics.median([1, 2, 3, 4]), round(statistics.stdev([1, 2, 3, 4]), 10), statistics.mode([1, 2, 2, 3]), round(statistics.pvariance([1, 2, 3, 4]), 10))
class Shape(ABC):
    @abstractmethod
    def area(self): ...
class Sq(Shape):
    def __init__(self, s): self.s = s
    def area(self): return self.s ** 2
print(Sq(3).area(), os.environ.get("NOPE_VAR_X", "dflt"))
