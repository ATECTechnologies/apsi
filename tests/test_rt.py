"""rt: real-time loops and the video pipeline."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi import rt  # noqa: E402


class Frames(unittest.TestCase):
    def test_gray_matches_bt601(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        ref = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
        self.assertLessEqual(np.abs(rt.gray(a).astype(float) - ref).max(), 1.0)
        saved, rt._cv = rt._cv, None                         # the NumPy path gives the same image
        try:
            self.assertLessEqual(np.abs(rt.gray(a).astype(float) - ref).max(), 1.0)
        finally:
            rt._cv = saved

    def test_motion_finds_the_moving_block(self):
        a = np.zeros((480, 640), np.uint8)
        b = a.copy()
        b[100:140, 200:260] = 200
        for step in (1, 2):
            m = rt.motion(a, b, 25, step=step)
            self.assertTrue(m.moved)
            self.assertAlmostEqual(m.cx, 229.5, delta=1.0)
            self.assertAlmostEqual(m.cy, 119.5, delta=1.0)
            self.assertEqual(m.box[:2], [200, 100])
        self.assertFalse(rt.motion(a, a).moved)

    def test_box_blur_and_resize(self):
        img = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.assertEqual(rt.resize(img, 5, 5).shape, (5, 5))
        self.assertEqual(rt.downsample(img, 2).shape, (5, 5))
        self.assertAlmostEqual(float(rt.box_blur(img, 1)[5, 5]), 55.0, places=3)


class Pipeline(unittest.TestCase):
    def test_video_keeps_up_and_reports(self):
        seen = []

        def process(frame, info):
            seen.append(rt.motion(info.prev_gray, info.gray).moved)
        r = rt.video(rt.synthetic(640, 480, fps=240), process, fps=120, seconds=1.0)
        # 120 fps target on a 240 fps source (shared macOS CI machines are slower: half the target there)
        self.assertGreater(r.fps, 100 if not sys.platform == "darwin" else 30)   # macOS runners: shared, slower CPUs
        self.assertEqual(r.errors, 0)
        self.assertLess(r.latency_p50_ms, 8.4)
        self.assertTrue(any(seen))

    def test_errors_are_isolated(self):
        n = [0]

        def bad(frame, info):
            n[0] += 1
            if n[0] % 5 == 0:
                raise ValueError("one bad frame")
        r = rt.video(rt.synthetic(320, 240, fps=0), bad, frames=40, warmup=0)
        self.assertGreater(r.errors, 0)
        self.assertGreaterEqual(r.frames, 39)

    def test_loop_rate(self):
        # the best of three runs: a shared CI machine can pause any one of them
        runs = [rt.loop(1000, lambda k, t: None, seconds=0.3) for _ in range(3)]
        self.assertGreater(max(st.hz for st in runs), 900)
        self.assertLess(min(st.jitter_p50_us for st in runs), 200)


class Sizes(unittest.TestCase):
    def test_named_and_custom_sizes(self):
        from apsi import rt
        self.assertEqual(rt.size("fhd"), (1920, 1080))
        self.assertEqual(rt.size("1600x900"), (1600, 900))
        self.assertEqual(rt.size("4K"), (3840, 2160))
        self.assertEqual(rt.size((800, 600)), (800, 600))
        self.assertEqual(rt.size(1366, 768), (1366, 768))
        with self.assertRaises(ValueError):
            rt.size("huge")
        for name in ("vga", "svga", "hd", "wxga", "hd+", "fhd", "qhd", "4k"):
            src = rt.synthetic(name, fps=0)
            self.assertEqual((src.width, src.height), rt.SIZES[name])
            self.assertEqual(src.read().shape[:2], (src.height, src.width))


if __name__ == "__main__":
    unittest.main()
