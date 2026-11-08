"""Tests for the command line, the REPL, projects and the runner.  Run:  python -m unittest discover tests"""
import argparse
import contextlib
import json
import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi.repl import Repl, needs_more                      # noqa: E402
from apsi.cli import _make_interp                           # noqa: E402
from apsi.project import read_toml, scaffold, TEMPLATES     # noqa: E402

ENV = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")


def apsi(*args, cwd=None, stdin=None, timeout=120):
    r = subprocess.run([sys.executable, "-m", "apsi", *args], cwd=cwd, input=stdin, capture_output=True,
                       text=True, env=ENV, timeout=timeout, encoding="utf-8")
    return r.returncode, r.stdout, r.stderr


def repl_args():
    return argparse.Namespace(profile=None, allow_read=None, allow_write=None, allow_net=None, allow_env=None,
                              allow_python=None, allow_ffi=None, allow_all=False, sandbox=False, engine="interp")


def repl(script):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):          # program output goes to sys.stdout
        Repl(repl_args(), _make_interp, stdin=io.StringIO(script), stdout=out).loop()
    return out.getvalue()


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="apsi-cli-")
        self.old = os.getcwd()

    def tearDown(self):
        os.chdir(self.old)
        shutil.rmtree(self.dir, ignore_errors=True)


class ReplInput(unittest.TestCase):
    def test_needs_more(self):
        self.assertTrue(needs_more("if x > 0:"))
        self.assertTrue(needs_more("let xs = [1,"))
        self.assertTrue(needs_more('let s = """abc'))
        self.assertTrue(needs_more("fn f(x) ="))
        self.assertTrue(needs_more("let y = 1 +"))
        self.assertFalse(needs_more("let y = 1 + 2"))
        self.assertFalse(needs_more('print("a:")'))
        self.assertFalse(needs_more("x  # comment ending with :"))


class ReplSession(TempDir):
    def test_values_blocks_and_last_result(self):
        out = repl("let x = 3[m/s]\nx * 2s\n_ + 1m\nfn sq(a) =\n    a * a\nfor i in 0..3:\n    print(sq(i))\n\n")
        self.assertIn("6.0 m", out)
        self.assertIn("7.0 m", out)
        self.assertIn("4", out)

    def test_shell_words(self):
        sub = os.path.join(self.dir, "sub dir")
        os.makedirs(sub)
        with open(os.path.join(sub, "hello.apsi"), "w", encoding="utf-8") as fh:
            fh.write('print("hi from file")\n')
        out = repl(f'cd "{sub}"\npwd\nls\nrun hello.apsi\ncd -\n')
        self.assertIn(os.path.realpath(sub), os.path.realpath(out.splitlines()[0].strip()) + "\n" + out)
        self.assertIn("hello.apsi", out)
        self.assertIn("hi from file", out)

    def test_windows_style_path_is_not_a_syntax_error(self):
        out = repl("cd C:\\Users\\nobody\\nowhere\n")
        self.assertNotIn("SyntaxError", out)
        self.assertIn("cd:", out)

    def test_variables_shadow_shell_words(self):
        out = repl("let dir = 5\ndir + 1\n")
        self.assertIn("6", out)

    def test_commands(self):
        out = repl("let a = 2kg\n:vars\n:type a\n:doc sqrt\n:time 1 + 1\n:nope\n")
        self.assertIn("Mass", out)
        self.assertIn("sqrt", out)
        self.assertIn("unknown command", out)

    def test_errors_do_not_end_the_session(self):
        out = repl("nope\n5m + 2s\nexit(3)\nprint(\"still here\")\n")
        self.assertIn("NameError", out)
        self.assertIn("UnitError", out)
        self.assertIn("still here", out)

    def test_save_and_load(self):
        os.chdir(self.dir)
        repl("let k = 41\nfn inc(x) = x + 1\n:save s.apsi\n")
        out = repl(":load s.apsi\ninc(k)\n")
        self.assertIn("42", out)


class CommandLine(TempDir):
    def test_shortcuts(self):
        p = os.path.join(self.dir, "a.apsi")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("print(args())\n")
        rc, out, _ = apsi(p, "x", "y")
        self.assertEqual(rc, 0)
        self.assertIn('["x", "y"]', out)
        rc, out, _ = apsi("-e", "sqrt(4[m^2])")
        self.assertIn("2.0 m", out)
        rc, out, _ = apsi(stdin="print(1 + 1)\n")
        self.assertIn("2", out)
        rc, _, err = apsi("tset")
        self.assertEqual(rc, 2)
        self.assertIn("did you mean", err)

    def test_permission_flags_do_not_swallow_the_file(self):
        p = os.path.join(self.dir, "p.apsi")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('print("ran")\n')
        for argv in (["--allow-python", p], ["run", "--allow-read", p], ["--allow-net", p, "x"],
                     ["run", "--allow-read", self.dir, p], ["--allow-python", "all", p]):
            rc, out, err = apsi(*argv)
            self.assertEqual(rc, 0, f"{argv}: {err}")
            self.assertIn("ran", out)
        rc, out, _ = apsi("--allow-python", "-e", "1 + 1")
        self.assertIn("2", out)

    def test_input_and_exit(self):
        p = os.path.join(self.dir, "b.apsi")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('let n = input("name? ")\nprint(f"hi {n}")\nexit(3)\n')
        rc, out, _ = apsi(p, stdin="bob\n")
        self.assertEqual(rc, 3)
        self.assertIn("hi bob", out)

    def test_deep_recursion_is_a_clean_error(self):
        p = os.path.join(self.dir, "r.apsi")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("fn r(n: Int) -> Int = 0 if n == 0 else r(n - 1)\nprint(r(5000))\nprint(r(10000000))\n")
        for engine in ("compile", "interp"):
            rc, out, err = apsi(p, "--engine", engine)
            self.assertEqual(rc, 1, err[-500:])
            self.assertIn("0", out)
            self.assertIn("maximum recursion depth", err)
            self.assertLess(len(err.splitlines()), 30, "repeated frames are folded")

    def test_circular_import(self):
        a, b = os.path.join(self.dir, "a.apsi"), os.path.join(self.dir, "b.apsi")
        with open(a, "w", encoding="utf-8") as fh:
            fh.write('import "b.apsi"\nlet x = 1\n')
        with open(b, "w", encoding="utf-8") as fh:
            fh.write('import "a.apsi"\nlet y = 2\n')
        rc, _, err = apsi(a)
        self.assertEqual(rc, 1)
        self.assertIn("circular import", err)

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_ctrl_c_stops_a_running_program(self):
        p = os.path.join(self.dir, "loop.apsi")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("var i = 0\nwhile true:\n    i += 1\n")
        proc = subprocess.Popen([sys.executable, "-m", "apsi", "run", p], env=ENV, stderr=subprocess.PIPE, text=True,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=20)
        self.assertEqual(proc.returncode, 130)
        self.assertIn("interrupted", err)

    def test_doc_and_completion(self):
        rc, out, _ = apsi("doc", "nn", "adam")
        self.assertIn("adam", out)
        for shell in ("bash", "zsh", "fish", "powershell"):
            rc, out, _ = apsi("completion", shell)
            self.assertEqual(rc, 0)
            self.assertIn("run", out)

    def test_check_and_fmt_accept_folders(self):
        with open(os.path.join(self.dir, "ok.apsi"), "w", encoding="utf-8") as fh:
            fh.write("let x = 1\n")
        rc, out, _ = apsi("check", "-q", self.dir)
        self.assertEqual(rc, 0)
        rc, _, _ = apsi("fmt", "--check", self.dir)
        self.assertEqual(rc, 0)


class Editors(TempDir):
    def test_install_and_uninstall_in_every_found_editor(self):
        home = os.path.join(self.dir, "home")
        win = os.name == "nt"
        sublime = "AppData/Roaming/Sublime Text/Packages" if win else ".config/sublime-text/Packages"
        nvim = "AppData/Local/nvim" if win else ".config/nvim"
        vim = "vimfiles" if win else ".vim"
        for d in (".cursor", ".vscode/extensions", sublime, nvim, vim):
            os.makedirs(os.path.join(home, *d.split("/")))
        with open(os.path.join(home, ".vscode", "extensions", "extensions.json"), "w") as fh:
            fh.write("[]")
        env = dict(ENV, HOME=home, USERPROFILE=home, APPDATA=os.path.join(home, "AppData", "Roaming"),
                   LOCALAPPDATA=os.path.join(home, "AppData", "Local"), XDG_CONFIG_HOME="", XDG_DATA_HOME="",
                   PATH=os.path.dirname(sys.executable))
        env.pop("XDG_CONFIG_HOME"), env.pop("XDG_DATA_HOME")
        run = lambda *a: subprocess.run([sys.executable, "-m", "apsi", "editors", *a, "--root", ROOT],
                                        capture_output=True, text=True, env=env, timeout=120)
        r = run("install", "--json")
        res = {x["id"]: x for x in json.loads(r.stdout)["results"]}
        for ed in ("vscode", "cursor", "sublime", "vim"):
            self.assertTrue(res[ed]["ok"], res[ed])
        self.assertTrue(os.path.isfile(os.path.join(home, ".cursor", "extensions", "apsi.apsi-language-1.10.0", "package.json")))
        self.assertTrue(os.path.isfile(os.path.join(home, vim, "pack", "apsi", "start", "apsi", "syntax", "apsi.vim")))
        with open(os.path.join(home, ".vscode", "extensions", "extensions.json")) as fh:
            self.assertEqual(json.load(fh)[0]["identifier"]["id"], "apsi.apsi-language")
        r = run("list", "--json")
        listed = {x["id"]: x for x in json.loads(r.stdout)}
        self.assertTrue(listed["cursor"]["installed"])
        run("uninstall")
        left = [os.path.join(r, n) for r, ds, fs in os.walk(home) for n in ds + fs if "apsi" in n.lower()]
        self.assertEqual(left, [])


PY_PROGRAM = """
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Ball:
    mass: float
    speed: float = 0.0

    def energy(self):
        return 0.5 * self.mass * self.speed ** 2


class Counter:
    def __init__(self, start=0):
        self.count = start

    def add(self, n=1):
        self.count += n
        return self.count


class Loud(Counter):
    def __init__(self, start):
        super().__init__(start * 10)


def squares(n):
    for i in range(n):
        if i % 2 == 0:
            yield i * i


def stats(values):
    total = 0
    for v in values:
        total += v
    mean = total / len(values)
    return mean, max(values, key=lambda v: abs(v - mean))


def main():
    print("energy", Ball(2.0, speed=3.0).energy())
    c = Counter(1)
    c.add(4)
    print("count", c.count, Loud(2).add())
    print("squares", list(squares(7)))
    m, far = stats([1, 2, 9])
    print("stats", m, far)
    x = np.linspace(0, 1, 3)
    print("sum", float(np.sum(x ** 2)), int(np.argmax(x)))
    pairs = [(1, (2, 3))]
    for k, (a, b) in pairs:
        print("nested", k + a + b)
    first, *rest = [1, 2, 3]
    print("star", first, len(rest))
    mask = (x > 0.2) & (x < 0.8)
    print("mask", int(np.sum(mask)))
    print(f"fmt {math.sqrt(2):.3f} {'ok'}")


if __name__ == "__main__":
    main()
"""


class Convert(TempDir):
    def test_python_program_runs_the_same(self):
        src = os.path.join(self.dir, "prog.py")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(PY_PROGRAM)
        py = subprocess.run([sys.executable, src], capture_output=True, text=True, timeout=60).stdout.splitlines()
        rc, out, err = apsi("convert", src, "--run")
        self.assertEqual(rc, 0, out + err)
        lines = out.splitlines()
        for want in py:
            want = want.replace("True", "true").replace("False", "false")
            self.assertIn(want, lines, f"missing {want!r} in Apsi output:\n{out}")
        self.assertTrue(os.path.exists(os.path.join(self.dir, "prog.apsi")))

    def test_folders_and_libraries(self):
        os.makedirs(os.path.join(self.dir, "pkg"))
        with open(os.path.join(self.dir, "pkg", "a.py"), "w") as fh:
            fh.write("import json\nimport torch.nn as nn\nm = nn.Sequential(nn.Linear(2, 3), nn.ReLU())\n"
                     "print(json.dumps({'k': 1}))\n")
        with open(os.path.join(self.dir, "pkg", "b.py"), "w") as fh:
            fh.write("import requests\nr = requests.get('https://example.com')\n")
        rc, out, err = apsi("convert", os.path.join(self.dir, "pkg"), "-o", os.path.join(self.dir, "out"))
        self.assertIn("converted 2 files", out)
        with open(os.path.join(self.dir, "out", "a.apsi")) as fh:
            a = fh.read()
        self.assertIn("layers.sequential([layers.linear(2, 3), layers.activation(\"relu\")])", a)
        self.assertIn("json_str(", a)
        with open(os.path.join(self.dir, "out", "b.apsi")) as fh:
            self.assertIn('pyimport("requests")', fh.read())
        self.assertIn("--allow-python", out)


class BuildAndCompile(TempDir):
    def setUp(self):
        super().setUp()
        self.env = dict(ENV, APSI_HOME=os.path.join(self.dir, "home"))
        os.makedirs(os.path.join(self.dir, "lib"))
        with open(os.path.join(self.dir, "lib", "util.apsi"), "w", encoding="utf-8") as fh:
            fh.write('fn greet(n) = f"hello {n}"\n')
        self.app = os.path.join(self.dir, "app.apsi")
        with open(self.app, "w", encoding="utf-8") as fh:
            fh.write('import "lib/util.apsi"\nprint(greet("world"), args())\nprint(3[m] / 2[s])\n')

    def run_apsi(self, *a):
        r = subprocess.run([sys.executable, "-m", "apsi", *a], capture_output=True, text=True, env=self.env,
                           encoding="utf-8", timeout=120)
        return r.returncode, r.stdout, r.stderr

    @unittest.skipIf(os.name == "nt", "the Windows .exe launcher is exercised by running it on Windows")
    def test_build_runs_anywhere_with_its_imports_and_arguments(self):
        out = os.path.join(self.dir, "out")
        rc, so, err = self.run_apsi("build", self.app, "-o", out, "--allow-read")
        self.assertEqual(rc, 0, err)
        self.assertIn("--allow-read", so)
        exe = os.path.join(out, "app")
        env = dict(self.env)
        env.pop("PYTHONPATH", None)                  # the executable carries its own copy of Apsi
        r = subprocess.run([exe, "a", "--flag"], capture_output=True, text=True, cwd=tempfile.gettempdir(),
                           env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('hello world ["a", "--flag"]', r.stdout)
        self.assertIn("1.5 m/s", r.stdout)

    def test_build_refuses_programs_with_errors(self):
        bad = os.path.join(self.dir, "bad.apsi")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("print(nope)\n")
        rc, _, err = self.run_apsi("build", bad, "-o", os.path.join(self.dir, "out"))
        self.assertEqual(rc, 1)
        self.assertIn("NameError", err)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out", "bad")))

    def test_compile_cache(self):
        rc, so, err = self.run_apsi("compile", self.dir)
        self.assertEqual(rc, 0, err)
        self.assertIn("compiled 2/2", so)
        rc, so, _ = self.run_apsi("compile", "--stats")
        self.assertIn("2 programs", so)
        rc, so, _ = self.run_apsi(self.app)
        self.assertIn("hello world", so)
        rc, so, _ = self.run_apsi("compile", "--clear")
        self.assertIn("cleared (2 entries)", so)

    def test_tampered_cache_entries_are_never_loaded(self):
        from apsi import compile_cache
        old = os.environ.get("APSI_HOME")
        os.environ["APSI_HOME"] = self.env["APSI_HOME"]
        try:
            from apsi.parser import parse
            src = "let unique_name_for_cache_test = 41\n"
            parse(src, cache=True)
            compile_cache._MEMO.clear()
            files = [os.path.join(r, f) for r, _, fs in os.walk(compile_cache.cache_dir()) for f in fs]
            self.assertEqual(len(files), 1)
            with open(files[0], "r+b") as fh:
                fh.seek(40)
                fh.write(b"\x00\x01\x02")
            self.assertIsNone(compile_cache.load(src))
            self.assertFalse(os.path.exists(files[0]), "a bad entry is removed")
            self.assertEqual(type(parse(src, cache=True)).__name__, "Program")
        finally:
            compile_cache._MEMO.clear()
            if old is None:
                os.environ.pop("APSI_HOME", None)
            else:
                os.environ["APSI_HOME"] = old


class Projects(TempDir):
    def test_every_template_checks_and_tests(self):
        for name in TEMPLATES:
            root = os.path.join(self.dir, name)
            os.makedirs(root)
            scaffold(root, name, name)
            self.assertEqual(read_toml(os.path.join(root, "apsi.toml"))["project"]["main"], "src/main.apsi")
            rc, out, err = apsi("check", "-q", "src", "tests", cwd=root)
            self.assertEqual(rc, 0, f"{name}: {out}{err}")
            rc, out, err = apsi("test", cwd=root, timeout=300)
            self.assertEqual(rc, 0, f"{name}: {out}{err}")

    def test_run_and_tasks(self):
        rc, _, err = apsi("new", "demo", cwd=self.dir)
        self.assertEqual(rc, 0, err)
        root = os.path.join(self.dir, "demo")
        self.assertTrue(os.path.exists(os.path.join(root, ".vscode", "tasks.json")))
        rc, out, _ = apsi("run", cwd=root)
        self.assertIn("Hello, demo!", out)
        rc, out, _ = apsi("task", cwd=root)
        self.assertIn("start", out)
        rc, out, _ = apsi("start", cwd=root)
        self.assertIn("Hello, demo!", out)
        rc, _, _ = apsi("new", "demo", cwd=self.dir)
        self.assertEqual(rc, 1, "refuses to overwrite a non-empty folder")

    def test_toml_reader(self):
        p = os.path.join(self.dir, "apsi.toml")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('[project]\nname = "x"  # comment\nn = 3\n\n[tasks]\nci = [\n  "apsi check",\n  "apsi test",\n]\n')
        d = read_toml(p)
        self.assertEqual(d["project"], {"name": "x", "n": 3})
        self.assertEqual(d["tasks"]["ci"], ["apsi check", "apsi test"])


if __name__ == "__main__":
    unittest.main()
