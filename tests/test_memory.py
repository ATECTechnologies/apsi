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


class Smarter(unittest.TestCase):
    """Container limits, memory pressure, buffer pools, flat-memory streaming, store accounting."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        os.environ["APSI_SPILL_DIR"] = self.td.name
        from apsi import memory
        self.m = memory
        self.saved = (memory.CGROUP_FILE, memory.CGROUP_ROOT, memory._BUDGET)

    def tearDown(self):
        self.m.CGROUP_FILE, self.m.CGROUP_ROOT, self.m._BUDGET = self.saved
        self.m._CG[1] = None
        self.m._LEVEL[0] = 0.0
        os.environ.pop("APSI_SPILL_DIR", None)
        self.td.cleanup()

    def fake(self, files, cgroup_line):
        root = os.path.join(self.td.name, "cg")
        for rel, text in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(text)
        cf = os.path.join(self.td.name, "self-cgroup")
        with open(cf, "w") as fh:
            fh.write(cgroup_line + "\n")
        self.m.CGROUP_FILE, self.m.CGROUP_ROOT = cf, root
        self.m._CG[1] = None

    def test_cgroup_v2_limit_is_what_counts(self):
        self.fake({"kubepods/pod1/memory.max": str(512 * 2**20), "kubepods/pod1/memory.current": str(200 * 2**20),
                   "kubepods/pod1/memory.stat": "anon 1\ninactive_file %d\n" % (40 * 2**20),
                   "kubepods/memory.max": "max"}, "0::/kubepods/pod1")
        lim = self.m.limits()
        self.assertTrue(lim.source.startswith("container"))
        self.assertAlmostEqual(lim.ram_total_gb, 0.5, places=3)
        self.assertAlmostEqual(lim.ram_free_gb, (512 - 200 + 40) / 1024, places=3)

    def test_cgroup_v1_and_no_limit(self):
        self.fake({"memory/docker/abc/memory.limit_in_bytes": str(2**30),
                   "memory/docker/abc/memory.usage_in_bytes": str(2**29)}, "4:memory:/docker/abc")
        self.assertAlmostEqual(self.m.limits().ram_total_gb, 1.0, places=3)
        self.fake({"memory.max": "max"}, "0::/")
        self.assertEqual(self.m.limits().source, "machine")

    def test_pressure_levels_and_reactions(self):
        self.fake({"box/memory.max": str(1000 * 2**20), "box/memory.current": str(970 * 2**20)}, "0::/box")
        self.assertEqual(self.m.pressure().level, "critical")
        store = self.m.Store(os.path.join(self.td.name, "s"), ram_mb=64)
        for k in range(20):
            store.ram[f"k{k}"] = (k, 1000)
            store.ram_bytes += 1000
        self.m._REACTED[0] = 0.0
        self.m.array(2**21, "uint8")                    # a big allocation under pressure: caches give RAM back
        self.assertEqual(store.ram_bytes, 0)
        big = self.m.array(2**21, "uint8")              # ... and new big arrays go to the SSD
        self.assertTrue(self.m.on_disk(big))
        self.fake({"box/memory.max": str(1000 * 2**20), "box/memory.current": str(100 * 2**20)}, "0::/box")
        self.m._LEVEL[0] = 0.0
        self.assertIn(self.m.pressure().level, ("ok", "moderate"))

    def test_watcher_calls_back(self):
        self.fake({"box/memory.max": str(1000 * 2**20), "box/memory.current": str(990 * 2**20)}, "0::/box")
        seen = []
        w = self.m.on_pressure(lambda p: seen.append(p.level), level="high", every=0.05, repeat=60)
        import time
        for _ in range(100):
            if seen:
                break
            time.sleep(0.05)
        w.stop()
        self.assertEqual(seen[:1], ["critical"])

    def test_pool_reuses_buffers(self):
        p = self.m.pool(max_gb=0.1)
        a = p.take((100, 100))
        a[:] = 3.0
        addr = a.ctypes.data
        p.give(a)
        b = p.take((50, 200), zero=True)
        self.assertEqual(b.ctypes.data, addr)
        self.assertEqual(float(b.sum()), 0.0)
        self.assertEqual(p.stats()["reused"], 1)
        with self.assertRaises(ValueError):
            p.give(np.zeros(3))
        p.give(b)
        self.m.relieve("critical")
        self.assertEqual(p.stats()["spare_mb"], 0)

    def test_streaming_a_spilled_array_keeps_the_data(self):
        self.m.budget(0.0001)
        a = self.m.array((300_000, 4), "float64")
        self.assertTrue(self.m.on_disk(a))
        a[:] = np.arange(1_200_000, dtype=np.float64).reshape(-1, 4)
        total = sum(float(b.sum()) for b in self.m.chunks(a, rows=50_000))   # pages dropped behind
        self.assertEqual(total, float(np.arange(1_200_000, dtype=np.float64).sum()))
        self.assertEqual(float(a[123_456, 2]), 123_456 * 4 + 2.0)            # read back from the file
        out = self.m.map(lambda x: x + 1.0, a, rows=40_000)
        self.assertEqual(float(out[299_999, 3]), 1_199_999 + 1.0)
        rep = self.m.report()
        self.assertTrue(any(r.where == "SSD" and r.shape == [300_000, 4] for r in rep))

    def test_store_keeps_a_running_disk_total(self):
        s = self.m.Store(os.path.join(self.td.name, "t"), ram_mb=1, disk_gb=0.0002)    # about 200 KB
        v = np.zeros(1000)                                                              # 8 KB each
        for k in range(60):
            s.set(f"k{k}", v)
        st = s.stats()
        self.assertLessEqual(st["disk_mb"] * 2**20, 0.0002 * 2**30 + 1)
        self.assertEqual(s.disk_bytes, int(round(st["disk_mb"] * 2**20)))
        self.assertIn("k59", s)
        s.delete("k59")
        self.assertEqual(s.disk_bytes, sum(f[1] for f in s._files()))
