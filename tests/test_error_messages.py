"""Error messages that say what is wrong, where, and how to fix it: the closest real name, the caret on the failing
part of the line, and the usual habit from another language translated into Apsi."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    # (program, words the message must contain, the column the caret must point at or None)
    ("fn add(a, b):\n    return a + b\nprint(ad(1, 2))\n", ["'ad' is not defined", "did you mean 'add'"], 7),
    ('let s = "abc"\nprint(s.uper())\n', ["has no attribute 'uper'", "did you mean 'upper'"], 9),
    ('class Dog:\n    fn __init__(self, name):\n        self.name = name\nlet d = Dog("x")\nprint(d.nmae)\n',
     ["did you mean 'name'"], 9),
    ('let d = {"a": 1}\nprint(d["b"])\n', ["key not found: 'b'", "d.get(key, default)"], 9),
    ("let xs = [1, 2, 3]\nprint(xs[5])\n", ["index out of range", "positions start at 0"], 9),
    ('print("a" + 1)\n', ["cannot join Str and Int", "str(x)"], 11),
    ("print(10 / 0)\n", ["division by zero"], 10),
    ('let x = int("abc")\n', ["'abc' is not a whole number"], 9),
    ("let t = (1, 2)\nt[0] = 5\n", ["tuples cannot change"], 2),
    ("let a = [1, 2\nprint(a)\n", ["'[' is not closed", "end of line 1"], None),
    ("let x = 5\nif x = 5:\n    print(x)\n", ["to compare two values write =="], None),
    ("def f(x):\n    return x\n", ["Apsi defines functions with 'fn'"], None),
    ("let x = 1\nif x > 0:\n    print(1)\nelse if x < 0:\n    print(2)\n", ["write 'elif'"], None),
    ('print "hi"\n', ["print is a function"], None),
    ("var x = 1\nx++\n", ["write x += 1"], None),
    ("let a = True\n", ["Apsi writes 'true'"], None),
    ("let b = null\n", ["Apsi writes 'none'"], None),
    ("let a = 1\nif a > 0 && a < 3:\n    print(1)\n", ["Apsi writes 'and'"], None),
    ("let v = 3[m]\nprint(v + 1)\n", ["give the number a unit too, e.g. 1[m]"], None),
    ("fn f(a, b=1, c):\n    return a\n", ["parameter 'c' needs a default value"], None),
    ('let d = {"a": 1}\nfor k, v in d:\n    print(k)\n', ["d.items()"], None),
    ("import os\n", ["use os"], None),
]


class Messages(unittest.TestCase):
    def test_messages(self):
        with tempfile.TemporaryDirectory() as td:
            for i, (src, words, col) in enumerate(CASES):
                path = os.path.join(td, f"e{i}.apsi")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)
                r = subprocess.run([sys.executable, "-m", "apsi", "run", path], capture_output=True, text=True,
                                   cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
                out = r.stdout + r.stderr
                with self.subTest(program=src.splitlines()[-1]):
                    self.assertNotIn("Traceback", out)
                    self.assertNotEqual(r.returncode, 0)
                    for w in words:
                        self.assertIn(w, out)
                    if col is not None:
                        first = out.splitlines()[0]
                        self.assertIn(f":{col}:", first.split(": ")[0] + ":", first)


if __name__ == "__main__":
    unittest.main()
