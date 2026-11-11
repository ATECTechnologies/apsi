"""A virtual environment of another Python version is not mixed into pyimport() (its compiled modules cannot load)."""
import os
import sys
import tempfile
import unittest

from apsi.envfile import venv_mismatch


def make(cfg):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "pyvenv.cfg"), "w") as fh:
        fh.write(cfg)
    return d


class Venv(unittest.TestCase):
    here = f"{sys.version_info.major}.{sys.version_info.minor}"

    def test_same_version_is_used(self):
        self.assertIsNone(venv_mismatch(make(f"home = /x\nversion = {self.here}.1\nexecutable = /x/python{self.here}\n")))

    def test_other_version_or_free_threaded_is_refused(self):
        why = venv_mismatch(make("home = C:\\\\P\nversion = 3.99.0\nexecutable = C:\\\\P\\\\python.exe\n"))
        self.assertIn("3.99", why)
        if getattr(sys, "_is_gil_enabled", lambda: True)():
            why = venv_mismatch(make(f"version = {self.here}.0\nexecutable = C:\\\\P\\\\python{self.here}t.exe\n"))
            self.assertIn("free-threaded", why)

    def test_unreadable_config_is_not_an_error(self):
        self.assertIsNone(venv_mismatch(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
