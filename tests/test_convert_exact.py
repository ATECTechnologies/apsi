"""`apsi convert` against real Python: every program in convert_corpus/ is run by Python and, converted, by Apsi;
the output must be the same. With --idiomatic a file that uses something the idiomatic form cannot express exactly
(match, async, metaclasses, loop variables read after the loop ...) is converted with the exact form instead."""
import ast
import glob
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = sorted(glob.glob(os.path.join(ROOT, "tests", "convert_corpus", "*.py")))
ENV = {**os.environ, "PYTHONPATH": ROOT, "APSI_QUIET": "1"}


def apsi(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "apsi", *args], capture_output=True, text=True, cwd=cwd, env=ENV,
                          timeout=120)


class Exact(unittest.TestCase):
    def test_corpus_runs_like_python(self):
        with tempfile.TemporaryDirectory() as td:
            for src in CORPUS:
                name = os.path.basename(src)
                with self.subTest(program=name):
                    want = subprocess.run([sys.executable, src], capture_output=True, text=True, cwd=td,
                                          timeout=60).stdout
                    dst = os.path.join(td, name[:-3] + ".apsi")
                    apsi("convert", src, "-o", dst, "--force", "--no-check")
                    got = apsi("run", "--allow-python=all", "--allow-read", "--allow-write", dst, cwd=td)
                    self.assertEqual(want, got.stdout, got.stderr[-2000:])


class IdiomaticGate(unittest.TestCase):
    def blockers(self, code):
        from apsi.convert import idiomatic_blockers
        return [why for _, why in idiomatic_blockers(ast.parse(code))]

    def test_finds_what_idiomatic_cannot_express(self):
        cases = {
            "match x:\n    case 1:\n        pass\n": "match",
            "async def f():\n    pass\n": "async",
            "for i in range(3):\n    pass\nprint(i)\n": "loop variable 'i'",
            "fs = [lambda: i for i in range(3)]\n": "loop variable later",
            "class A(B, C):\n    pass\n": "several base classes",
            "class E(ValueError):\n    pass\n": "extends ValueError",
            "class C:\n    def __exit__(self, *a):\n        return True\n": "__exit__",
            "def f(a=[]):\n    return a\n": "default value",
            "x = 1j\n": "complex",
            "class C:\n    @classmethod\n    def m(cls):\n        pass\n": "@classmethod",
        }
        for code, word in cases.items():
            with self.subTest(code=code):
                self.assertTrue(any(word in b for b in self.blockers(code)), self.blockers(code))

    def test_plain_code_stays_idiomatic(self):
        code = ("import math\nfn = None\ndef area(r):\n    return math.pi * r ** 2\n"
                "for i in range(3):\n    print(area(i))\nfs = [lambda v, i=i: v + i for i in range(3)]\n")
        self.assertEqual(self.blockers(code), [])

    def test_falls_back_to_exact(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "m.py")
            with open(src, "w") as fh:
                fh.write("def kind(x):\n    match x:\n        case 0:\n            return 'zero'\n"
                         "        case _:\n            return 'other'\nprint(kind(0), kind(2))\n")
            r = apsi("convert", src, "--idiomatic", "--force")
            self.assertIn("kept Python's exact semantics", r.stdout + r.stderr)
            with open(src[:-3] + ".apsi") as fh:
                self.assertIn("use pycompat", fh.read())
            self.assertEqual(apsi("run", src[:-3] + ".apsi").stdout, "zero other\n")


if __name__ == "__main__":
    unittest.main()
