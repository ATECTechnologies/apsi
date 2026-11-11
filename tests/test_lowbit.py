"""Low-bit training (fp8, fp6, fp4, int8, int4): the rounding matches the GPU's, models still learn, and PyTorch
rounds bit for bit like NumPy."""
import unittest

import numpy as np

from apsi import layers, lowbit, hub_gpu
from apsi.stdlib import Adam

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None


def task(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 40)).astype(np.float32)
    w = rng.normal(size=(40,))
    y = ((X @ w) > 0).astype(np.int64) + 2 * (X[:, 0] > 1).astype(np.int64)
    return X, y


class LowBit(unittest.TestCase):
    def test_names(self):
        self.assertEqual(lowbit.parse("MXFP4"), "fp4")
        self.assertEqual(lowbit.parse("e2m3"), "fp6_e2m3")
        self.assertEqual(lowbit.parse("fp8"), "fp8")
        self.assertIsNone(lowbit.parse("bf16"))
        self.assertIsNone(lowbit.parse({"store": "float32"}))

    def test_rounding_is_the_gpu_layout(self):
        rng = np.random.default_rng(1)
        W = rng.normal(size=(64, 96)).astype(np.float32)          # [in, out]: blocks of 32 along "in"
        for kind in lowbit.KINDS:
            got = lowbit.round_weights(W, kind)
            q, sc = hub_gpu.qnumpy(W.T, kind)
            want = (q.reshape(96, 2, 32) * sc[:, :, None]).reshape(96, 64).T
            np.testing.assert_array_equal(got, want, err_msg=kind)
            self.assertEqual(got.dtype, W.dtype)
        odd = rng.normal(size=(3, 3, 5, 7)).astype(np.float32)      # any size: the last block is padded
        self.assertEqual(lowbit.round_weights(odd, "fp4").shape, odd.shape)
        b = rng.normal(size=(7,))
        np.testing.assert_array_equal(lowbit.round_weights(b, "fp4"), b)          # biases stay as they are

    def test_accuracy_order(self):
        rng = np.random.default_rng(2)
        W = rng.normal(size=(256, 128)).astype(np.float32)
        err = {k: float(np.sqrt(((lowbit.round_weights(W, k) - W) ** 2).mean())) for k in lowbit.KINDS}
        self.assertLess(err["int8"], err["fp8"])
        self.assertLess(err["fp8"], err["fp6"])
        self.assertLess(err["fp6_e2m3"], err["fp6"])
        self.assertLess(err["fp6"], err["fp4"])

    def test_models_learn_in_every_format(self):
        X, y = task()
        net = layers.mlp([40, 64, 4])
        acc = {}
        for prec in ("fp32",) + lowbit.KINDS:
            r = layers.fit(net, net.init(seed=0), X, y, loss="cross_entropy", opt=Adam(1e-2), epochs=12,
                           batch_size=100, verbose=False, precision=prec)
            self.assertLess(r.history[-1], r.history[0] * 0.6, prec)
            pred = np.asarray(layers.predict(net, r.params, X, precision=prec)).argmax(-1)
            acc[prec] = float((pred == y).mean())
        for prec in lowbit.KINDS:
            self.assertGreater(acc[prec], acc["fp32"] - (0.06 if prec in ("fp4", "int4") else 0.03), (prec, acc))

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_torch_rounds_like_numpy(self):
        rng = np.random.default_rng(3)
        W = np.concatenate([rng.normal(size=(96, 50)), rng.uniform(-3, 3, size=(96, 50)) * 1e-3,
                            np.repeat(np.array([[0.0, 0.5, -0.25, 1.75, 2.5, 7.5, 28.0, 448.0]]).T, 50, 1)[:8]],
                           axis=0).astype(np.float32)
        for kind in lowbit.KINDS:
            got = lowbit.round_weights_torch(torch.from_numpy(W), kind).numpy()
            np.testing.assert_array_equal(got, lowbit.round_weights(W, kind), err_msg=kind)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_torch_training(self):
        X, y = task(4)
        net = layers.mlp([40, 64, 4])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for prec in ("fp8", "fp6_e2m3", "fp4", "int4"):
            r = layers.fit(net, net.init(seed=0), X, y, loss="cross_entropy", opt=Adam(1e-2), epochs=10,
                           batch_size=100, verbose=False, precision=prec, backend="torch", device=dev)
            self.assertLess(r.history[-1], r.history[0] * 0.6, prec)
            pred = np.asarray(layers.predict(net, r.params, X, precision=prec, backend="torch",
                                             device=dev)).argmax(-1)
            self.assertGreater((pred == y).mean(), 0.8, prec)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs PyTorch with CUDA (fp8 tensor cores through torch)")
    def test_fp8_tensor_cores_when_the_card_has_them(self):
        dev = torch.device("cuda")
        if not lowbit.fp8_matmul_available(dev):
            self.skipTest("this GPU has no fp8 tensor cores (compute capability below 8.9)")
        lowbit._HW["on"] = True
        try:
            x = torch.randn(40, 64, device=dev, requires_grad=True)
            w = torch.randn(64, 32, device=dev, requires_grad=True)
            y = lowbit.matmul(x, w)
            self.assertTrue(lowbit._HW["on"], lowbit._HW["why"])
            rel = ((y.float() - x @ w).norm() / (x @ w).norm()).item()
            self.assertLess(rel, 0.08)
            y.float().sum().backward()
            self.assertEqual(x.grad.shape, x.shape)
            self.assertEqual(w.grad.shape, w.shape)
        finally:
            lowbit._HW["on"] = False


if __name__ == "__main__":
    unittest.main()
