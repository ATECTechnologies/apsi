"""`use scale` / `apsi plan`: the large-model planner follows the published accounting exactly."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import scale          # noqa: E402
from apsi import Interpreter    # noqa: E402


class Planner(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(scale.number("21B"), 21e9)
        self.assertEqual(scale.number("1.5T"), 1.5e12)
        self.assertEqual(scale.number(7e9), 7e9)

    def test_flops_and_time_match_hand_calculation(self):
        p = scale.plan("70B", tokens="1.4T", gpu="H100", gpus=1024, layers=80, width=8192, heads=64)
        f = p.fields
        want = 6 * 70e9 * 1.4e12 + 12 * 80 * 8192 * 4096 * 1.4e12
        self.assertAlmostEqual(f["flops"] / want, 1.0, places=9)
        ideal_days = want / (1024 * 989e12 * 0.40) / 86400
        self.assertGreaterEqual(f["days"], ideal_days * 0.999)          # recompute/bubble only add time
        self.assertLess(f["days"], ideal_days * 1.5)
        self.assertEqual(f["tensor_parallel"] * f["pipeline_parallel"] * f["data_parallel"], 1024)
        self.assertLessEqual(f["memory_per_gpu_gb"], 0.9 * 80 + 0.1)

    def test_model_states_are_16_bytes_per_parameter(self):
        f = scale.plan("21B", gpus=64).fields
        self.assertAlmostEqual(f["model_states_gb"], round(21e9 * 16 / 2 ** 30, 1))

    def test_moe_uses_active_parameters_for_compute(self):
        dense = scale.plan("1T", tokens="10T", gpus=8192).fields
        moe = scale.plan("1T", active="32B", tokens="10T", gpus=8192).fields
        self.assertLess(moe["flops"], dense["flops"] / 20)
        self.assertEqual(moe["model_states_gb"], dense["model_states_gb"])

    def test_every_size_from_21b_to_2t_gets_a_plan(self):
        for size, kw in [("21B", {}), ("500B", {}), ("800B", {"active": "40B"}), ("1T", {"active": "32B"}),
                         ("2T", {"gpu": "B200", "precision": "fp8"}), ("2T", {})]:
            f = scale.plan(size, **kw).fields
            self.assertGreater(f["gpus"], 0)
            self.assertLessEqual(f["gpus"], 200_000)
            self.assertLessEqual(f["memory_per_gpu_gb"], 0.9 * f["gpu_memory_gb"] + 0.1, size)

    def test_impossible_requests_say_so(self):
        with self.assertRaises(ValueError):
            scale.plan("21B", gpu="GTX1660", gpus=4)
        f = scale.plan("2T").fields
        self.assertTrue(any("not reachable" in n for n in f["notes"]))

    def test_inference_memory(self):
        f = scale.inference("70B", precision="fp8").fields
        self.assertAlmostEqual(f["weights_gb"], round(70e9 / 2 ** 30, 1))
        self.assertGreaterEqual(f["gpus"], 1)

    def test_use_scale_in_apsi(self):
        env = Interpreter().run('use scale\nlet p = plan("21B", gpus=64)\nlet n = p.gpus\n', "<t>")
        self.assertEqual(env.lookup("n"), 64)


if __name__ == "__main__":
    unittest.main()
