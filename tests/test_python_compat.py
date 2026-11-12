"""Python programs converted with `apsi convert` must print exactly what Python prints, byte for byte (the
converted files start with `use pycompat`, which gives them Python's semantics)."""
import glob
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _norm(t):
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "none", t)
    return t.replace("'", '"').strip()


class PythonCompat(unittest.TestCase):
    pass


def _make(path):
    def test(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        want = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=120).stdout
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "prog.apsi")
            subprocess.run([sys.executable, "-m", "apsi", "convert", path, "-o", out], capture_output=True,
                           text=True, env=env, timeout=120, check=True)
            with open(path, encoding="utf-8") as f:
                head = f.readline()
            flags = head.split(":", 1)[1].split() if head.startswith("# apsi-flags:") else []
            got = subprocess.run([sys.executable, "-m", "apsi", "run", out, *flags], capture_output=True, text=True,
                                 env=env, timeout=120)
        self.assertEqual(want, got.stdout, got.stderr)
    return test


for _p in sorted(glob.glob(os.path.join(HERE, "pycompat", "*.py"))):
    setattr(PythonCompat, "test_" + os.path.basename(_p)[:-3], _make(_p))

if __name__ == "__main__":
    unittest.main()
