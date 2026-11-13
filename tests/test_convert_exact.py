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


class Scopes(unittest.TestCase):
    """One file and what it imports (the default), only the files named (--only), the whole project (--project),
    --exclude and --plan."""

    def project(self, td):
        os.makedirs(os.path.join(td, "pkg"))
        os.makedirs(os.path.join(td, "tests"))
        files = {"pyproject.toml": "[project]\nname = 'x'\n", "main.py": "import helper\nfrom pkg import util\n"
                 "print(helper.double(util.three()))\n", "helper.py": "def double(x):\n    return 2 * x\n",
                 "pkg/__init__.py": "", "pkg/util.py": "def three():\n    return 3\n",
                 "tests/test_h.py": "import helper\n"}
        for k, v in files.items():
            with open(os.path.join(td, k), "w") as f:
                f.write(v)
        return os.path.join(td, "main.py")

    def converted(self, td):
        return sorted(os.path.relpath(os.path.join(r, f), td).replace(os.sep, "/") for r, _d, fs in os.walk(td)
                      for f in fs if f.endswith(".apsi"))

    def test_follow_imports_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            main = self.project(td)
            self.assertEqual(apsi("convert", main).returncode, 0)
            self.assertEqual(self.converted(td), ["helper.apsi", "main.apsi", "pkg/__init__.apsi", "pkg/util.apsi"])
            self.assertEqual(apsi("run", main[:-3] + ".apsi").stdout, "6\n")

    def test_only_the_file(self):
        with tempfile.TemporaryDirectory() as td:
            main = self.project(td)
            r = apsi("convert", main, "--only")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("kept as Python imports (--only): helper, pkg", r.stdout)
            self.assertEqual(self.converted(td), ["main.apsi"])
            self.assertEqual(apsi("run", "--allow-python=all", main[:-3] + ".apsi", cwd=ROOT).stdout, "6\n")

    def test_whole_project_with_exclude_and_plan(self):
        with tempfile.TemporaryDirectory() as td:
            main = self.project(td)
            plan = apsi("convert", os.path.join(td, "pkg", "util.py"), "--project", "--plan")
            self.assertIn("would write 5 files", plan.stdout)
            self.assertEqual(self.converted(td), [])                      # --plan writes nothing
            self.assertEqual(apsi("convert", main, "--project", "--exclude", "tests/*").returncode, 0)
            self.assertEqual(self.converted(td), ["helper.apsi", "main.apsi", "pkg/__init__.apsi", "pkg/util.apsi"])


class Libraries(unittest.TestCase):
    """Patterns of real libraries (datasets, fsspec, multiprocess, huggingface_hub): a name re-exported by a package,
    `from .. import x` served by the package at run time, `import pkg.sub` inside pkg, import cycles, fields under an
    if of a class body, globals(), `with (a / b).open() as f`, a bare return in an async generator."""

    def test_library_patterns_run_like_python(self):
        files = {
            "pkg/__init__.py": "from . import ctx\nfrom .util import helper\nglobals().update({'AuthError': ctx.AuthError})\n"
                               "def __getattr__(name):\n    if name == 'lazy':\n        from .sub import lazy\n        return lazy\n"
                               "    raise AttributeError(name)\n",
            "pkg/ctx.py": "class AuthError(Exception):\n    pass\n",
            "pkg/util.py": "def helper():\n    return 'h'\n",
            "pkg/sub/__init__.py": "def lazy():\n    return 'lazy'\n",
            "pkg/conn.py": ("import os, pathlib, asyncio\nimport pkg.util\nfrom . import AuthError, util\nfrom . import lazy\n"
                            "from typing import TYPE_CHECKING, ClassVar, Dict\n"
                            "class M:\n    if TYPE_CHECKING:\n        f: ClassVar[Dict[str, int]] = {}\n"
                            "    if os.name == 'nt':\n        w = 1\n    else:\n        w = 2\n    h: int = 1\n"
                            "def _hidden():\n    return 5\n"
                            "async def agen(n):\n    for i in range(n):\n        if i == 2:\n            return\n        yield i\n"
                            "async def collect():\n    return [x async for x in agen(5)]\n"
                            "def go(tmp):\n    p = pathlib.Path(tmp)\n    with (p / 'x.txt').open('w') as fh:\n        fh.write('ok')\n"
                            "    try:\n        raise AuthError('e')\n    except AuthError as e:\n        caught = str(e)\n"
                            "    return [caught, M.w > 0, M.h, globals()['_hidden'](), pkg.util.helper(), util.helper(), lazy(),\n"
                            "            asyncio.run(collect()), (p / 'x.txt').read_text()]\n"),
            "main.py": "import sys, tempfile\nfrom pkg import conn\nprint(conn.go(tempfile.mkdtemp()))\n",
        }
        with tempfile.TemporaryDirectory() as td:
            for k, v in files.items():
                os.makedirs(os.path.dirname(os.path.join(td, k)), exist_ok=True)
                with open(os.path.join(td, k), "w") as f:
                    f.write(v)
            want = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, cwd=td).stdout
            self.assertIn("lazy", want)
            r = apsi("convert", os.path.join(td, "main.py"))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            got = apsi("run", "--allow-all", os.path.join(td, "main.apsi"), cwd=td)
            self.assertEqual(got.stdout, want, got.stderr)


if __name__ == "__main__":
    unittest.main()
