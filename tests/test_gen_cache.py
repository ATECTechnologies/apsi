"""The per-file cache of generated code: reused when nothing changed, regenerated when the file, a file it
imports, or what an import resolves to changes; a tampered entry is ignored."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GenCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.env = dict(os.environ, APSI_HOME=os.path.join(self.dir, "home"), PYTHONPATH=ROOT)
        self.env.pop("APSI_NO_CACHE", None)
        self.write("geom.apsi", """
            struct V2:
                x: Float
                y: Float
            fn norm2(v: V2) -> Float = v.x * v.x + v.y * v.y
            def twice(n: int) -> int:
                return n * 2
            """)
        self.write("main.apsi", """
            from geom import V2, norm2, twice
            import geom
            print(norm2(V2(3.0, 4.0)), twice(4), geom.twice(1))
            """)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, src):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(src).lstrip())

    def run_main(self, name="main.apsi"):
        r = subprocess.run([sys.executable, "-m", "apsi", "run", name], cwd=self.dir, env=self.env,
                           capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()

    def entries(self):
        return glob.glob(os.path.join(self.dir, "home", "cache", "compiled", "gen", "*", "*.gen"))

    def test_reused_and_refreshed(self):
        self.assertEqual(self.run_main(), "25.0 8 2")
        first = set(self.entries())
        self.assertEqual(len(first), 2)
        self.assertEqual(self.run_main(), "25.0 8 2")          # from the cache, same result
        self.assertEqual(set(self.entries()), first)
        self.write("geom.apsi", """
            struct V2:
                x: Float
                y: Float
            fn norm2(v: V2) -> Float = v.x + v.y
            def twice(n: int) -> int:
                return n * 3
            """)
        self.assertEqual(self.run_main(), "7.0 12 3")            # the imported file changed
        self.assertEqual(len(set(self.entries()) - first), 2)

    def test_type_checks_survive_the_cache(self):
        self.write("bad.apsi", "from geom import twice\nlet f = twice\nprint(f(\"x\"))\n")
        for _ in range(2):
            self.assertIn("must be an Int", self.run_main("bad.apsi"))

    def test_tampered_entry_is_ignored(self):
        self.assertEqual(self.run_main(), "25.0 8 2")
        for path in self.entries():
            with open(path, "r+b") as fh:
                data = bytearray(fh.read())
                data[-5] ^= 0xFF
                fh.seek(0)
                fh.write(data)
        self.assertEqual(self.run_main(), "25.0 8 2")

    def test_local_file_appearing_changes_the_import(self):
        self.write("j.apsi", "import jsonx\nprint(jsonx.VALUE)\n")
        self.assertIn("unknown module", self.run_main("j.apsi"))
        self.write("jsonx.apsi", "VALUE = 42\n")
        self.assertEqual(self.run_main("j.apsi"), "42")


if __name__ == "__main__":
    unittest.main()
