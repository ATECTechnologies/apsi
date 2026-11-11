"""apsi convert turns the libraries a program imports that are written in Python into Apsi (apsi_packages/), so the
result runs without the Python environment it came from; compiled libraries stay Python imports with a note."""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from apsi.cli import main

LIB = {
    "mylib/__init__.py": "from .core import greet, Counter\nfrom . import util\n__all__ = ['greet', 'Counter', 'util']\n",
    "mylib/core.py": ("import json\nfrom other import shout\n\nclass Counter:\n    def __init__(self):\n"
                      "        self.n = 0\n    def add(self, k=1):\n        self.n += k\n        return self\n\n"
                      "def greet(name):\n    return shout(f'hello {name}') + json.dumps({'n': len(name)})\n"),
    "mylib/util.py": "import importlib\n\ndef double(x):\n    return [i * 2 for i in x]\n\n"
                     "def by_name():\n    return importlib.import_module('other').shout('dyn')\n",
    "mylib/data.txt": "payload\n",
    "other.py": "def shout(s):\n    return s.upper() + '!'\n",
    "fastlib/__init__.py": "",
    "fastlib/_core.cpython-399-x86_64-linux-gnu.so": "",
}
PROGRAM = """import mylib
from mylib.util import double, by_name
from mylib import Counter
from importlib.resources import files
print(mylib.greet("apsi"))
print(double([1, 2, 3]), Counter().add(2).add(3).n, by_name())
print(files("mylib").joinpath("data.txt").read_text().strip())
"""


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = main(list(args))
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class Packages(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        sp = os.path.join(self.dir, ".venv", "Lib", "site-packages")      # a venv of another Python version
        for rel, text in LIB.items():
            p = os.path.join(sp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(text)
        with open(os.path.join(self.dir, ".venv", "pyvenv.cfg"), "w") as fh:
            fh.write("home = C:\\\\Python399\nversion = 3.99.0\n")
        self.prog = os.path.join(self.dir, "main.py")
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        for k in [m for m in sys.modules if m.split(".")[0] in ("mylib", "other")]:
            del sys.modules[k]
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_pure_libraries_are_converted_and_run_without_the_venv(self):
        with open(self.prog, "w") as fh:
            fh.write(PROGRAM)
        rc, out, err = run("convert", self.prog)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("library mylib: 3 modules converted", out)
        vend = os.path.join(self.dir, "apsi_packages")
        self.assertTrue(os.path.exists(os.path.join(vend, "other.apsi")))
        self.assertTrue(os.path.exists(os.path.join(vend, "mylib", "data.txt")))       # data files come along
        with open(os.path.join(self.dir, "main.apsi")) as fh:
            self.assertIn('import "apsi_packages/mylib/__init__.apsi" as mylib', fh.read())
        shutil.rmtree(os.path.join(self.dir, ".venv"))                                  # Apsi alone from here
        os.chdir(self.dir)
        rc, out, err = run("run", "main.apsi", "--allow-all")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.splitlines(), ['HELLO APSI!{"n": 4}', "[2, 4, 6] 5 DYN!", "payload"])

    def test_compiled_libraries_stay_python_with_a_note(self):
        with open(self.prog, "w") as fh:
            fh.write("import fastlib\nimport mylib\n")
        rc, out, err = run("convert", self.prog)
        self.assertIn("'fastlib' is a compiled (C/C++) library", out)
        with open(os.path.join(self.dir, "main.apsi")) as fh:
            text = fh.read()
        self.assertIn('pyimport("fastlib")', text)
        self.assertIn('import "apsi_packages/mylib/__init__.apsi"', text)

    def test_no_packages_keeps_python_imports(self):
        with open(self.prog, "w") as fh:
            fh.write("import mylib\n")
        run("convert", self.prog, "--no-packages")
        with open(os.path.join(self.dir, "main.apsi")) as fh:
            self.assertIn('pyimport("mylib")', fh.read())
        self.assertFalse(os.path.exists(os.path.join(self.dir, "apsi_packages")))


if __name__ == "__main__":
    unittest.main()
