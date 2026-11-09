"""`use memory`: RAM budget with SSD spill, out-of-core chunks/map, the two-tier store, OOM-safe batches."""
import os
import tempfile
import unittest

import numpy as np


class Memory(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        os.environ["APSI_SPILL_DIR"] = self.td.name
        from apsi import memory
        self.m = memory
        self.mod = memory.module()
        self.saved = memory._BUDGET

    def tearDown(self):
        self.m._BUDGET = self.saved
        os.environ.pop("APSI_SPILL_DIR", None)
        self.td.cleanup()

    def test_spills_beyond_budget(self):
        self.m.budget(0.001)                       # 1 MB
        small = self.m.array(1000, "float64")
        big = self.m.array((1000, 1000), "float32")
        self.assertFalse(self.m.on_disk(small))
        self.assertTrue(self.m.on_disk(big))
        big[5, 5] = 2.0
        self.assertEqual(float(big.sum()), 2.0)
        path = self.m._SPILLED[id(big)]
        self.m.free(big)
        self.assertFalse(os.path.exists(path))

    def test_map_and_chunks(self):
        a = np.arange(100_000, dtype=np.float64).reshape(-1, 4)
        out = self.m.map(lambda x: x * 2, a, rows=1000)
        np.testing.assert_array_equal(out, a * 2)
        total = sum(float(b.sum()) for b in self.m.chunks(out, rows=777))
        self.assertEqual(total, float((a * 2).sum()))

    def test_store_tiers(self):
        st = self.mod["store"]("t", ram_mb=0.01, disk_gb=0.001)
        st.set("a", np.ones((100, 100)))
        st.set(("k", 1), {"x": [1, 2]})
        self.assertEqual(st.get(("k", 1)), {"x": [1, 2]})
        self.assertEqual(st.get("a").shape, (100, 100))
        self.assertIsNone(st.get("missing"))
        fresh = self.mod["store"]("t")          # a new process sees the SSD tier
        self.assertEqual(fresh.get(("k", 1)), {"x": [1, 2]})
        key = self.m._key_of(("k", 1))
        p = fresh._file(key, ".pkl")
        with open(p, "r+b") as fh:            # tampering: never unpickled
            fh.seek(40)
            b = fh.read(1)
            fh.seek(40)
            fh.write(bytes([b[0] ^ 0xFF]))
        self.assertIsNone(self.mod["store"]("t").get(("k", 1)))

    def test_cached_and_fit_batch(self):
        calls = []

        @self.mod["cached"]
        def sq(n):
            calls.append(n)
            return n * n
        sq.store.clear()
        self.assertEqual((sq(9), sq(9)), (81, 81))
        self.assertEqual(calls, [9])

        def work(b):
            if b > 100:
                raise MemoryError
            return b
        self.assertEqual(self.m.fit_batch(work, 1000), (62, 62))
        with self.assertRaises(ValueError):
            self.m.fit_batch(lambda b: (_ for _ in ()).throw(ValueError("x")), 8)

    def test_status(self):
        s = self.m.status()
        self.assertGreater(s.ram_total_gb, 0)
        self.assertIn("RAM", str(s))
        self.assertEqual(self.m.best_device(1e9), "cpu")


if __name__ == "__main__":
    unittest.main()
