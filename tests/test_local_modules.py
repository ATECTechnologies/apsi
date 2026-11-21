"""`use NAME` for a program's own files, and `from M import a, b as c` (both engines, and the checker)."""
import os
import shutil
import tempfile
import unittest

from apsi.errors import ApsiError, CheckFailed
from apsi.interpreter import Interpreter


FILES = {
    "geom.apsi": "struct V2:\n    x: float\n    y: float\n    def add(self, o) = V2(self.x + o.x, self.y + o.y)\n"
                 "fn _hidden() = 1\nlet ORIGIN = V2(0.0, 0.0)\n",
    "shapes/circle.apsi": "fn area(r) = 3 * r * r\nclass Square:\n    def __init__(self, s):\n        self.s = s\n",
    "main.apsi": (
        "use geom\n"
        "use shapes.circle as sc\n"
        "from geom import V2 as Vec, ORIGIN\n"
        "from shapes.circle import (\n    area,\n    Square,\n)\n"
        "from \"geom.apsi\" import V2 as Other\n"
        "from math import sqrt, pi as PI\n"
        "let v = geom.V2(1.0, 2.0) + Vec(1.0, 1.0) + ORIGIN + Other(0.0, 1.0)\n"
        "let result = [v.x, v.y, sc.area(2), area(1), Square(5).s, sqrt(9.0), PI > 3]\n"
    ),
}


def _text(e):
    return " ".join(str(x) for x in getattr(e, "errors", [])) + " " + str(e)


class LocalModules(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for rel, src in FILES.items():
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_file(self, name, src=None, engine="compile"):
        path = os.path.join(self.dir, name)
        if src is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
        with open(path, encoding="utf-8") as fh:
            return Interpreter().run(fh.read(), path, engine=engine)

    def test_both_engines(self):
        for engine in ("compile", "interp"):
            env = self.run_file("main.apsi", engine=engine)
            self.assertEqual(list(env.vars["result"]), [2.0, 4.0, 12, 3, 5, 3.0, True], engine)

    def test_private_and_missing_names(self):
        for engine in ("compile", "interp"):
            for src, msg in (("from geom import _hidden\n", "no public name '_hidden'"),
                             ("from geom import ORIGN\n", "did you mean 'ORIGIN'"),
                             ("from math import sqrtt\n", "no name 'sqrtt'"),
                             ("use nowhere\n", "put nowhere.apsi next to this file")):
                with self.assertRaises((ApsiError, CheckFailed)) as cm:
                    self.run_file("bad.apsi", src, engine)
                self.assertIn(msg, _text(cm.exception), (engine, src))

    def test_standard_modules_win_over_local_files(self):
        with open(os.path.join(self.dir, "math.apsi"), "w") as fh:
            fh.write("fn sqrt(x) = -1\n")
        env = self.run_file("m.apsi", "use math\nlet r = sqrt(4.0)\n")
        self.assertEqual(env.vars["r"], 2.0)

    def test_from_import_star_is_explained(self):
        with self.assertRaises((ApsiError, CheckFailed)) as cm:
            self.run_file("s.apsi", "from geom import *\n")
        self.assertIn("use geom", _text(cm.exception))

    def test_from_is_still_a_name(self):
        env = self.run_file("n.apsi", "let from_ = 1\nvar d = {\"from\": 2}\nlet r = d[\"from\"] + from_\n")
        self.assertEqual(env.vars["r"], 3)


if __name__ == "__main__":
    unittest.main()
