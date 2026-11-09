"""Python-level tests for the Apsi toolchain.  Run:  python -m unittest discover tests"""
import io
import os
import subprocess
import sys
import contextlib
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                                  # noqa: E402
from apsi.errors import CheckFailed, UnitError, ShapeError, ApsiError  # noqa: E402
from apsi.lexer import tokenize                               # noqa: E402
from apsi.units import parse_unit, D, Quantity                # noqa: E402
from apsi.interop import PermissionErr                        # noqa: E402


def tearDownModule():
    # tests/apsi/models.apsi writes safetensors files next to the working folder; leave no trace
    import shutil
    for d in (os.path.join(ROOT, "_tmp_models"), os.path.join(os.getcwd(), "_tmp_models")):
        shutil.rmtree(d, ignore_errors=True)


def run(src, perms=None, check=True):
    it = Interpreter(permissions=perms)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        env = it.run(src, "<test>", check=check)
    return env, out.getvalue()


def check_errors(src):
    try:
        run(src)
    except CheckFailed as e:
        return [x.msg for x in e.errors]
    return []


class Lexer(unittest.TestCase):
    def test_unit_literals(self):
        toks = tokenize("2kg + 9.81[m/s^2] + 1e3 + 2eV + 3.5e-2m")
        nums = [(t.value, t.unit) for t in toks if t.type == "NUMBER"]
        self.assertEqual(nums, [("2", "kg"), ("9.81", "m/s^2"), ("1e3", None), ("2", "eV"), ("3.5e-2", "m")])

    def test_range_not_float(self):
        toks = tokenize("0..10")
        self.assertEqual([t.value for t in toks[:3]], ["0", "..", "10"])

    def test_indentation(self):
        types = [t.type for t in tokenize("if x:\n    y\nz\n")]
        self.assertIn("INDENT", types)
        self.assertIn("DEDENT", types)

    def test_unknown_unit_is_syntax_error(self):
        with self.assertRaises(ApsiError):
            tokenize("3furlongs")


class Units(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_unit("N"), (1.0, D(m=1, kg=1, s=-2)))
        s, d = parse_unit("km/h")
        self.assertAlmostEqual(s, 1000 / 3600)
        self.assertEqual(d, D(m=1, s=-1))
        self.assertEqual(parse_unit("J/(mol K)")[1], D(m=2, kg=1, s=-2, mol=-1, K=-1))
        self.assertEqual(parse_unit("m²")[1], D(m=2))
        self.assertEqual(parse_unit("s^-1")[1], D(s=-1))

    def test_quantity_algebra(self):
        a = Quantity(2.0, D(m=1))
        b = Quantity(4.0, D(s=1))
        self.assertEqual((a / b).dim, D(m=1, s=-1))
        self.assertEqual(a / a, 1.0)
        with self.assertRaises(UnitError):
            a + b


class Checker(unittest.TestCase):
    def test_catches_unit_errors(self):
        errs = check_errors("let x = 5m + 2s")
        self.assertTrue(any("cannot add Length" in e for e in errs))

    def test_catches_bad_argument_and_return(self):
        errs = check_errors("fn f(m: Mass) -> Energy = m\nf(2s)")
        self.assertTrue(any("must be Mass" in e for e in errs))
        self.assertTrue(any("should return Energy" in e for e in errs))

    def test_catches_names_and_constants(self):
        errs = check_errors("let a = 1\na = 2\nprint(nope)")
        self.assertTrue(any("declared with let" in e for e in errs))
        self.assertTrue(any("'nope' is not defined" in e for e in errs))

    def test_evolve_rules(self):
        errs = check_errors("let x = 1m\nevolve x over t in 0[s]..1[s] step 1[ms]:\n    x' = 3m\n")
        self.assertTrue(any("must be declared with var" in e for e in errs))
        self.assertTrue(any("units of Velocity" in e for e in errs))

    def test_valid_program_passes(self):
        self.assertEqual(check_errors("fn f(x) = x * 2\nlet y = f(3)\nprint(y)"), [])

    def test_forward_references_in_functions(self):
        self.assertEqual(check_errors("fn a() = b()\nfn b() = 1\nprint(a())"), [])


class Runtime(unittest.TestCase):
    def test_shape_generics(self):
        src = "fn f[N](a: Tensor[N], b: Tensor[N]) = a + b\nf(zeros(3), zeros(4))"
        with self.assertRaises(ShapeError):
            run(src)

    def test_runtime_unit_error_has_location(self):
        for engine in ("compile", "interp"):
            it = Interpreter()
            it.engine = engine
            try:
                it.run("fn f(x) = x + 1m\nf(2s)", "<t>", check=False)
            except UnitError as e:
                self.assertEqual((e.line, e.col), (1, 13), engine)
                self.assertIn("f()", e.trace, engine)
                continue
            self.fail("expected UnitError")

    def test_fast_evolve_matches_checked_path(self):
        src = """
var x = 1m
var v = 0[m/s]
evolve x, v over t in 0[s]..2[s] step 1[ms] using rk4:
    x' = v
    v' = -4[1/s^2] * x
"""
        env, _ = run(src)
        x = env.vars["x"]
        self.assertAlmostEqual(x.v, np.cos(4.0), places=9)

    def test_print_formatting(self):
        _, out = run('print(9.0[J], 5[km] as [km], [1, 2], true, none)')
        self.assertEqual(out.strip(), "9.0 J 5.0 km [1, 2] true none")


class Platform(unittest.TestCase):
    def test_windows_line_endings_and_bom(self):
        src = "\ufefflet x = 2m\r\nif true:\r\n    print(x * 2)\r\n"
        _, out = run(src)
        self.assertEqual(out.strip(), "4.0 m")

    def test_library_name_resolution(self):
        from apsi.interop import library_candidates
        names = [os.path.basename(c) for c in library_candidates("native/vecmath")]
        if sys.platform.startswith("win"):
            self.assertEqual(names[0], "vecmath.dll")
        elif sys.platform != "darwin":
            self.assertEqual(names[0], "libvecmath.so")
        self.assertEqual(library_candidates("x/libfoo.so"), ["x/libfoo.so"])

    def test_platform_builtin(self):
        _, out = run("print(platform())")
        self.assertIn(out.strip(), ("windows", "linux", "macos"))

    def test_doctor_runs(self):
        from apsi.hardware import info
        i = info()
        self.assertIn("os_meets_target", i)
        self.assertIsInstance(i["threads"], int)

    def test_formatter_is_idempotent_and_safe(self):
        from apsi.formatter import format_checked
        src = "fn   f(a,b=10)->Float=a+b\nlet x=f(1,b =2)  #c\nif x>1:\n      print( x)\n"
        once, changed = format_checked(src)
        self.assertTrue(changed)
        twice, changed2 = format_checked(once)
        self.assertFalse(changed2)
        self.assertEqual(once, twice)


class Permissions(unittest.TestCase):
    def test_python_denied_by_default(self):
        with self.assertRaises(PermissionErr):
            run('pyimport("os")')

    def test_python_allowed(self):
        env, _ = run('let m = pyimport("math")\nlet r = m.sqrt(16.0)', perms={"python"})
        self.assertEqual(env.vars["r"], 4.0)

    def test_sandbox_blocks_files(self):
        with self.assertRaises(PermissionErr):
            run('read_text("x.txt")', perms=set())


class CLI(unittest.TestCase):
    def _apsi(self, *args):
        return subprocess.run([sys.executable, "-m", "apsi", *args], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8")

    def test_examples_tests_pass(self):
        r = self._apsi("test", "examples/01_hello.apsi", "examples/05_orbit.apsi",
                       "examples/08_neural_net.apsi", "tests/apsi")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("0 failed", r.stdout)

    def test_both_engines_pass_language_suite(self):
        for engine in ("compile", "interp"):
            r = self._apsi("test", "tests/apsi", "--engine", engine)
            self.assertEqual(r.returncode, 0, engine + r.stdout + r.stderr)

    def test_compiled_is_much_faster_than_tree_walker(self):
        import time
        src = "var s = 0.0\nfor i in 0..100000:\n    s += i * 0.5\n"
        times = {}
        for engine in ("compile", "interp"):
            it = Interpreter()
            it.engine = engine
            t0 = time.perf_counter()
            it.run(src, "<bench>")
            times[engine] = time.perf_counter() - t0
        self.assertLess(times["compile"] * 3, times["interp"])

    def test_check_reports_errors(self):
        path = os.path.join(ROOT, "tests", "_bad.apsi")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("let x = 1kg + 1m\n")
        try:
            r = self._apsi("check", path)
            self.assertEqual(r.returncode, 1)
            self.assertIn("UnitError", r.stderr)
        finally:
            os.remove(path)

    def test_version(self):
        self.assertIn("Apsi", self._apsi("version").stdout)


class LanguageServer(unittest.TestCase):
    def test_diagnostics_hover_definition(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lsp_client import Client
        c = Client(ROOT)
        try:
            caps = c.request("initialize", {"capabilities": {}})["capabilities"]
            self.assertTrue(caps["hoverProvider"])
            uri = "file:///tmp/apsi_lsp_test.apsi"
            src = ("fn kinetic(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2\n"
                   "let speed = 3[m/s]\nlet e = kinetic(2kg, speed)\nlet bad = speed + 1kg\n")
            d = c.notify("textDocument/didOpen", {"textDocument": {"uri": uri, "languageId": "apsi",
                                                                    "version": 1, "text": src}},
                         expect="textDocument/publishDiagnostics")
            self.assertEqual(len(d["diagnostics"]), 1)
            self.assertIn("cannot add Velocity", d["diagnostics"][0]["message"])
            h = c.request("textDocument/hover", {"textDocument": {"uri": uri},
                                                 "position": {"line": 2, "character": 22}})
            self.assertIn("Velocity", h["contents"]["value"])
            h = c.request("textDocument/hover", {"textDocument": {"uri": uri},
                                                 "position": {"line": 2, "character": 10}})
            self.assertIn("-> Energy", h["contents"]["value"])
            loc = c.request("textDocument/definition", {"textDocument": {"uri": uri},
                                                        "position": {"line": 2, "character": 22}})
            self.assertEqual(loc["range"]["start"]["line"], 1)
            fixed = c.notify("textDocument/didChange", {"textDocument": {"uri": uri, "version": 2},
                                                        "contentChanges": [{"text": "let x = 1m\n"}]},
                             expect="textDocument/publishDiagnostics")
            self.assertEqual(fixed["diagnostics"], [])
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main()
