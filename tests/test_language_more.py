"""enum declarations and match with built-in types."""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(src):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.apsi")
        open(p, "w", encoding="utf-8").write(textwrap.dedent(src))
        r = subprocess.run([sys.executable, "-m", "apsi", "run", p], capture_output=True, text=True, cwd=ROOT)
        c = subprocess.run([sys.executable, "-m", "apsi", "check", p], capture_output=True, text=True, cwd=ROOT)
        return r, c


class Enums(unittest.TestCase):
    def test_declarations(self):
        r, c = run("""
            enum Color:
                RED
                GREEN = 5
                BLUE
            enum Dir: N, E, S, W
            enum Mode: FAST = "fast", SAFE = "safe"
            print(Color.RED.value, Color.GREEN.value, Color.BLUE.value, Color.BLUE.name)
            print([d.name for d in Dir], Dir(2).name, Color["GREEN"].value, Mode("safe").name)
            let c = Color.GREEN
            match c:
                case Color.GREEN:
                    print("green")
                case _:
                    print("other")
            print(c == Color.GREEN, c == Color.RED, len(Dir), {Color.RED: 1}[Color.RED])
            let enum = 3
            print(enum + 1)
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split("\n")[:5], ["0 5 6 BLUE", '["N", "E", "S", "W"] S 5 SAFE', "green",
                                                    "true false 4 1", "4"])
        self.assertEqual(c.returncode, 0, c.stdout + c.stderr)

    def test_errors(self):
        r, _ = run("""
            enum Bad: A, A
        """)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("twice", r.stdout + r.stderr)


class Match(unittest.TestCase):
    def test_builtin_types_in_patterns(self):
        r, c = run("""
            fn kind(v):
                match v:
                    case str() as s:
                        return "text " + s
                    case int(n) if n > 100:
                        return "big int"
                    case [x, *rest]:
                        return f"list starting {x} ({len(rest)} more)"
                    case {"kind": "point", "x": px}:
                        return f"point {px}"
                    case _:
                        return "other"
            print(kind("hi"), kind(500), kind([1, 2, 3]), kind({"kind": "point", "x": 3}), kind(3.5))
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "text hi big int list starting 1 (2 more) point 3 other")
        self.assertEqual(c.returncode, 0, c.stdout + c.stderr)


if __name__ == "__main__":
    unittest.main()
