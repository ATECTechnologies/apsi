"""The automatic CPU/GPU choice of @gpu functions (apsi/native/gpu.py _Choice), without a GPU."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apsi.native import gpu  # noqa: E402


def drive(choice, cost, calls):
    sides = []
    for _ in range(calls):
        s = choice.next()
        choice.record(s, cost[s])
        sides.append(s)
    return sides


class Choice(unittest.TestCase):
    def test_warm_up_is_not_counted(self):
        c = gpu._Choice()
        c.next()
        c.record("gpu", 10.0)                  # first launch loads the kernel: slow, ignored
        self.assertEqual(c.gpu, [])
        self.assertEqual(c.next(), "gpu")

    def test_faster_side_wins_and_loser_is_rechecked(self):
        sides = drive(gpu._Choice(), {"gpu": 1.0, "cpu": 2.0}, 300)
        self.assertEqual(sides[:3], ["gpu", "gpu", "cpu"])
        self.assertGreater(sides.count("gpu") / len(sides), 0.97)
        self.assertGreater(sides[3:].count("cpu"), 2)          # re-timed now and then

    def test_cpu_wins_small_work(self):
        sides = drive(gpu._Choice(), {"gpu": 3.0, "cpu": 1.0}, 300)
        self.assertGreater(sides.count("cpu") / len(sides), 0.95)

    def test_choice_follows_a_change_of_load(self):
        c = gpu._Choice()
        drive(c, {"gpu": 1.0, "cpu": 0.5}, 100)                # CPU faster at first
        sides = drive(c, {"gpu": 1.0, "cpu": 5.0}, 300)        # then the CPU gets busy
        self.assertGreater(sides[-100:].count("gpu"), 95)


if __name__ == "__main__":
    unittest.main()
