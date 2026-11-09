"""use dataframe: results must equal pandas on random data (nulls, several keys, every join kind)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi import dataframe as D  # noqa: E402

try:
    import pandas as pd
except ImportError:                 # pragma: no cover
    pd = None


@unittest.skipUnless(pd is not None, "pandas is the reference")
class DataFrameTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 3000
        self.raw = {"k": [f"k{i}" for i in rng.integers(0, 40, n)], "g": rng.integers(0, 5, n).tolist(),
                    "x": np.round(rng.normal(size=n), 6).tolist(), "y": rng.integers(-50, 50, n).tolist()}
        self.raw["x"][::17] = [None] * len(self.raw["x"][::17])
        self.a = D.DataFrame(self.raw)
        self.p = pd.DataFrame({k: [np.nan if v is None else v for v in vs] for k, vs in self.raw.items()})

    def test_group_by(self):
        col = D.col
        r = self.a.group_by("k", "g").agg(s=col("x").sum(), m=col("x").mean(), c=col("x").count(),
                                          lo=col("y").min(), hi=col("y").max(), sd=col("x").std()).sort(["k", "g"])
        p = self.p.groupby(["k", "g"]).agg(s=("x", "sum"), m=("x", "mean"), c=("x", "count"), lo=("y", "min"),
                                           hi=("y", "max"), sd=("x", "std")).reset_index()
        self.assertEqual(r["k"].tolist(), p.k.tolist())
        for c in ("s", "m", "sd"):
            self.assertTrue(np.allclose(r[c], p[c].values, equal_nan=True), c)
        for c in ("c", "lo", "hi"):
            self.assertTrue((r[c] == p[c].values).all(), c)

    def test_filter_sort_expressions(self):
        col = D.col
        r = self.a.filter((col("y") > 10) & (col("k").is_in(["k1", "k2", "k3"]))).with_columns(
            z=col("y") * 2 + 1).sort(["k", "y"], descending=[False, True])
        p = self.p[(self.p.y > 10) & self.p.k.isin(["k1", "k2", "k3"])].assign(z=lambda d: d.y * 2 + 1)
        p = p.sort_values(["k", "y"], ascending=[True, False], kind="stable")
        self.assertEqual(r["k"].tolist(), p.k.tolist())
        self.assertEqual(r["y"].tolist(), p.y.tolist())
        self.assertEqual(r["z"].tolist(), p.z.tolist())

    def test_joins(self):
        right = {"k": [f"k{i}" for i in range(0, 60, 2)], "w": list(range(30))}
        ra, rp = D.DataFrame(right), pd.DataFrame(right)
        for how in ("inner", "left", "right", "outer"):
            r = self.a.join(ra, on="k", how=how)
            p = self.p.merge(rp, on="k", how=how)
            self.assertEqual(r.height, len(p), how)
            got = sorted(zip(r["k"].tolist(), np.nan_to_num(r["w"].astype(float), nan=-1).tolist()))
            want = sorted(zip(p.k.tolist(), np.nan_to_num(p.w.astype(float), nan=-1).tolist()))
            self.assertEqual(got, want, how)

    def test_csv_roundtrip_and_fast_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            big = D.DataFrame({"k": [f"name, {i % 7}" for i in range(30000)], "v": list(range(30000)),
                               "f": [i / 3 for i in range(30000)], "s": [f"s{i % 5}" for i in range(30000)]})
            big.write_csv(path)
            back = D.read_csv(path)
            self.assertEqual(back.dtypes, {"k": "str", "v": "int", "f": "float", "s": "str"})
            self.assertEqual(back["k"].tolist()[:3], big["k"].tolist()[:3])
            self.assertTrue(np.allclose(back["f"], big["f"]))
            self.assertEqual(back["s"].tolist(), big["s"].tolist())


if __name__ == "__main__":
    unittest.main()
