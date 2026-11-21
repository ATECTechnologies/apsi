"""`apsi gpu test` and `apsi gpu bench max` on the Emulator (the real card runs the same code)."""
import io
import os
import shutil
import tempfile
import unittest

from apsi import gpu_bench, gpu_suite, gpu_train as GT


class GpuSuite(unittest.TestCase):
    def setUp(self):
        self._device = GT.device
        GT.device = lambda name="cuda": (GT.Emulator(), "")

    def tearDown(self):
        GT.device = self._device

    def test_the_apsi_program_part(self):
        out = io.StringIO()
        self.assertEqual(gpu_suite._program(out), 0, out.getvalue())
        text = out.getvalue()
        self.assertIn("ALL OK", text)
        self.assertIn("stopped and resumed", text)

    def test_largest_model_ladder(self):
        ladder = list(gpu_bench.MAX_LADDER)
        try:
            gpu_bench.MAX_LADDER[:] = [("tiny", dict(d=32, heads=2, layers=1)), ("small", dict(d=64, heads=4, layers=2))]
            out = io.StringIO()
            rc, best = gpu_bench.largest(out=out, context=16, batch=2, vocab=50)
        finally:
            gpu_bench.MAX_LADDER[:] = ladder
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual(best["size"], "small")

    def test_report_without_a_card(self):
        GT.device = lambda name="cuda": (None, "no NVIDIA driver")
        d = tempfile.mkdtemp()
        try:
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                rc = gpu_suite.run(os.path.join(d, "r"), quick=True)
            self.assertEqual(rc, 1)
            with open(os.path.join(d, "r", "report.txt"), encoding="utf-8") as fh:
                self.assertIn("RESULT: 0 of 1 parts passed", fh.read())
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
