"""`a ?? b`, `a?.b`, `a?[i]` (none-safe operators) and `use collections`, in both engines and the formatter."""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                       # noqa: E402
from apsi.formatter import format_source            # noqa: E402

NULLSAFE = r'''
struct P:
    name: Str
    next: Any

let cfg = {"port": 8080}
let a = P("a", none)
let b = P("b", a)
var calls = 0
fn tick():
    calls += 1
    return none
let xs = none
print(cfg.get("port") ?? 80, cfg.get("host") ?? "localhost")
print(b?.next?.name, a?.next?.name, a.next?.name ?? "end")
print(xs?[0], [1, 2]?[1], xs?[0] ?? -1)
print(tick() ?? tick() ?? "both none", calls)
print(none ?? false, 0 ?? 1, "" ?? "x")
print((fn(s) => s?.upper())(none), (fn(s) => s?.upper())("hi"))
let nested = {"a": {"b": [10, 20]}}
print(nested.get("a")?.get("b")?[1], nested.get("z")?.get("b")?[1])
'''

EXPECTED = ['8080 localhost', 'a none end', 'none 2 -1', 'both none 2', 'false 0 ', 'none HI', '20 none']

COLLECTIONS = r'''
use collections
let c = Counter("mississippi")
print(c.most_common(2), c["s"], c["z"], c.total())
let g = DefaultDict(fn() => [])
g["a"].append(1)
print(g["a"], len(g))
let q = Deque([1, 2, 3], max=3)
q.push(4)
print(q.to_list(), q.pop_left())
let h = Heap(key=fn(t) => t[0])
h.push((3, "c"))
h.push((1, "a"))
h.push((1, "b"))
print(h.pop(), h.pop(), len(h))
print(Heap([5, 1, 9], max_heap=true).pop())
let lru = LRUCache(2)
lru.set("a", 1)
lru.set("b", 2)
lru.get("a")
lru.set("c", 3)
print(lru.has("b"), lru.get("a"))
print(OrderedSet([3, 1, 3, 2]).to_list(), BiMap({"en": "hi"}).key_of("hi"))
print(first([1, 4, 6], fn(x) => x > 2), last([1, 4, 6, 7], fn(x) => x % 2 == 0), partition([1, 2, 3, 4], fn(x) => x % 2 == 0))
print(scan([1, 2, 3, 4], fn(a, b) => a + b), fold([1, 2, 3], fn(a, x) => a * x, 1), flat_map([1, 2], fn(x) => [x, x * 10]))
print(zip_longest([1, 2], ["a"], fill="-"), interleave([1, 2, 3], ["a"]), distinct_by(["aa", "ab", "b"], fn(s) => s[0]))
print(min_by(["ccc", "a"], len), max_by(["ccc", "a"], len), sum_by([1, 2, 3], fn(x) => x * x), frequencies("aab"))
print(compose(fn(x) => x * 2, fn(x) => x + 1)(3), partial(fn(a, b) => a - b, 10)(3))
'''

EXPECTED_C = ['[("i", 4), ("s", 4)] 4 0 11', '[1] 1', '[2, 3, 4] 2', '(1, "a") (1, "b") 1', '9', 'false 1',
              '[3, 1, 2] en', '4 6 ([2, 4], [1, 3])', '[1, 3, 6, 10] 6 [1, 10, 2, 20]',
              '[(1, "a"), (2, "-")] [1, "a", 2, 3] ["aa", "b"]', 'a ccc 14 {"a": 2, "b": 1}', '8 7']


def run(src, engine):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        Interpreter().run(src, "<t>", engine=engine)
    return out.getvalue().splitlines()


class NoneSafe(unittest.TestCase):
    def test_both_engines(self):
        for engine in ("compile", "interp"):
            self.assertEqual(run(NULLSAFE, engine), EXPECTED, engine)

    def test_formatter_keeps_meaning(self):
        src = "let x = a??b\nlet y = c ?.d\nlet z = e?[1]\n"
        self.assertEqual(format_source(src), "let x = a ?? b\nlet y = c?.d\nlet z = e?[1]\n")


class Collections(unittest.TestCase):
    def test_both_engines(self):
        for engine in ("compile", "interp"):
            self.assertEqual(run(COLLECTIONS, engine), EXPECTED_C, engine)


if __name__ == "__main__":
    unittest.main()
