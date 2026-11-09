"""Real third-party-sized code: twelve pure-Python standard-library modules (difflib, statistics, ipaddress, ...)
are copied next to a driver, converted with `apsi convert`, and must print exactly what CPython prints."""
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ["difflib", "textwrap", "fractions", "statistics", "shlex", "colorsys", "calendar", "pprint",
           "ipaddress", "graphlib", "heapq", "bisect"]


class StdlibConvert(unittest.TestCase):
    def test_twelve_modules(self):
        lib = sysconfig.get_paths()["stdlib"]
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED="0")
        with tempfile.TemporaryDirectory() as td:
            for m in MODULES:
                shutil.copy(os.path.join(lib, m + ".py"), td)
            shutil.copy(os.path.join(HERE, "stdlib_convert", "drive.py"), td)
            want = subprocess.run([sys.executable, "drive.py"], cwd=td, env=env, capture_output=True,
                                  text=True, timeout=120).stdout
            subprocess.run([sys.executable, "-m", "apsi", "convert", "drive.py"], cwd=td, env=env,
                           capture_output=True, text=True, timeout=300, check=True)
            got = subprocess.run([sys.executable, "-m", "apsi", "run", "drive.apsi", "--allow-python=all"], cwd=td,
                                 env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(want, got.stdout, got.stderr[-2000:])

    def test_big_modules(self):
        """_pydecimal (6,000+ lines), _pydatetime, argparse, configparser, tomllib, html.parser, urllib.parse."""
        lib = sysconfig.get_paths()["stdlib"]
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED="0")
        with tempfile.TemporaryDirectory() as td:
            for m in ["_pydecimal", "_pydatetime", "argparse", "configparser", "fnmatch", "base64", "string"]:
                shutil.copy(os.path.join(lib, m + ".py"), td)
            shutil.copytree(os.path.join(lib, "tomllib"), os.path.join(td, "tomllib"))
            shutil.copy(os.path.join(lib, "html", "parser.py"), os.path.join(td, "hparser.py"))
            shutil.copy(os.path.join(lib, "urllib", "parse.py"), os.path.join(td, "uparse.py"))
            shutil.copy(os.path.join(HERE, "stdlib_convert", "drive2.py"), os.path.join(td, "drive.py"))
            want = subprocess.run([sys.executable, "drive.py"], cwd=td, env=env, capture_output=True,
                                  text=True, timeout=120).stdout
            subprocess.run([sys.executable, "-m", "apsi", "convert", "drive.py"], cwd=td, env=env,
                           capture_output=True, text=True, timeout=300, check=True)
            got = subprocess.run([sys.executable, "-m", "apsi", "run", "drive.apsi", "--allow-python=all"], cwd=td,
                                 env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(want, got.stdout, got.stderr[-2000:])

    def test_introspective_modules(self):
        """dataclasses, enum, pickle, json (a package), random, _py_abc, contextlib: metaclasses, descriptors,
        frames-free introspection and package-relative imports."""
        lib = sysconfig.get_paths()["stdlib"]
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED="0")
        with tempfile.TemporaryDirectory() as td:
            for m in ["dataclasses", "enum", "copy", "operator", "contextlib", "_py_abc", "pickle", "reprlib",
                      "keyword", "textwrap", "random", "fractions", "numbers"]:
                shutil.copy(os.path.join(lib, m + ".py"), td)
            shutil.copytree(os.path.join(lib, "json"), os.path.join(td, "json"))
            shutil.copy(os.path.join(HERE, "stdlib_convert", "drive3.py"), os.path.join(td, "drive.py"))
            want = subprocess.run([sys.executable, "drive.py"], cwd=td, env=env, capture_output=True,
                                  text=True, timeout=120).stdout
            subprocess.run([sys.executable, "-m", "apsi", "convert", "drive.py"], cwd=td, env=env,
                           capture_output=True, text=True, timeout=300, check=True)
            got = subprocess.run([sys.executable, "-m", "apsi", "run", "drive.apsi", "--allow-python=all"], cwd=td,
                                 env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(want, got.stdout, got.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
