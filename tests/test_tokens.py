"""`use tokens`: every transform must give back exactly what went in."""
import json
import random
import unittest

from apsi import tokens as T


class Tokens(unittest.TestCase):
    def test_round_trips_are_exact(self):
        rnd = random.Random(7)
        for _ in range(200):
            rows = [{"id": i, "s": rnd.choice(["ann", "§1", "a,b", "q\"\n", "ü"]), "f": rnd.random(),
                     "b": rnd.random() < 0.5, "n": None, "l": [1, {"x": 2}]} for i in range(rnd.randint(2, 9))]
            self.assertEqual(T.unpack(T.pack(rows)), rows)
            text = "\n".join(rnd.choice(["§1", "§§", "a long line that repeats again", "x", ""]) for _ in range(40))
            self.assertEqual(T.expand(T.dedupe(text, min_chars=5)), text)
            new = "\n".join(rnd.choice("abcd") for _ in range(25))
            self.assertEqual(T.patch(text, T.diff(text, new)), new)
        for v in [{"a": 1}, [1, 2], "x", [{"a": 1}, {"b": 2}]]:
            text, how = T.best(v)
            self.assertEqual(T.restore(text, how), v)

    def test_tables_save_tokens(self):
        rows = [{"order_id": 1000 + i, "customer": f"c{i % 9}", "total": i * 1.5, "paid": True} for i in range(100)]
        s = T.savings(json.dumps(rows, indent=2), T.pack(rows))
        self.assertGreater(s.percent, 40)


if __name__ == "__main__":
    unittest.main()
