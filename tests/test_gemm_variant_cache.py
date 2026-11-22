"""The build of a tiled product found again for a shape seen before: the cached answer is the full choice's."""
import os
import unittest
from unittest import mock

import numpy as np

from apsi import gpu_train as GT


def _fake(fn):
    g = object.__new__(GT.Gpu)
    g.fn = {n: None for n in fn}
    g._only16, g._twin, g._wrange, g._choice, g._saved = set(), {}, None, {}, {}
    g.profile, g._w16_gen, g._wgen, g._choice_file = None, -1, 0, None
    return g


def _args(rng, M, N, K, pa, pb, nb2=1):
    return [("p", pa), ("p", pb), ("p", 4096), ("p", 0), ("u", M), ("u", N), ("u", K), ("u", K), ("u", 1),
            ("u", N), ("u", 1), ("u", N), ("u", nb2), ("u", 0), ("u", 0), ("u", 0), ("u", 0), ("u", 0), ("u", 0),
            ("f", 1.0), ("u", 0), ("u", 0), ("u", K), ("p", 0), ("u", 0), ("u", 0), ("p", 0), ("u", 0),
            ("p", 8), ("p", 4096), ("u", 0), ("u", 0), ("u", 0), ("u", 0)]


class VariantCacheTest(unittest.TestCase):
    def test_cached_answer_is_the_full_choice(self):
        rng = np.random.default_rng(0)
        names = ["gemm_tkn", "gemm_tkn#x", "gemm_tkn#h", "gemm_tkn#v", "to_bf16"]
        for force in ("a", "x", "h", "v"):
            with mock.patch.dict(os.environ, {"APSI_GEMM_VARIANT": force}):
                g = _fake(names)
                calls = []
                full = g._variant_full

                def spy(*a, **k):
                    calls.append(1)
                    return full(*a, **k)
                g._variant_full = spy
                for _ in range(200):
                    M, N, K = int(rng.integers(1, 3000)), int(rng.choice([264, 512, 99])), int(rng.choice([264, 64]))
                    pa = int(rng.integers(1, 1 << 20)) * 16 + int(rng.choice([0, 4, 8]))
                    pb = int(rng.integers(1, 1 << 20)) * 16
                    a = _args(rng, M, N, K, pa, pb)
                    got = g._variant("gemm_tkn", (1, 1, 1), a, M, N, N, 1)
                    self.assertEqual(got, full("gemm_tkn", (1, 1, 1), a, M, N, N, 1), (force, M, N, K, pa % 16))
                self.assertLess(len(calls), 200)            # shapes seen again: no full choice


if __name__ == "__main__":
    unittest.main()
