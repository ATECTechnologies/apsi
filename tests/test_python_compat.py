"""Python programs converted with `apsi convert` must print what Python prints (Apsi spells true/false/none
in lower case and quotes strings with "…", which is normalised before comparing)."""
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
            got = subprocess.run([sys.executable, "-m", "apsi", "run", out], capture_output=True, text=True,
                                 env=env, timeout=120)
        self.assertEqual(_norm(want), _norm(got.stdout), got.stderr)
    return test


for _p in sorted(glob.glob(os.path.join(HERE, "pycompat", "*.py"))):
    setattr(PythonCompat, "test_" + os.path.basename(_p)[:-3], _make(_p))

if __name__ == "__main__":
    unittest.main()
