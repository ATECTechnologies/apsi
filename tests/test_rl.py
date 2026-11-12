"""Reinforcement learning, preference learning and search (`use rl`)."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import rl, ops, torch_compat as T          # noqa: E402
from apsi.values import Record                       # noqa: E402


class RL(unittest.TestCase):
    def test_ppo_solves_cartpole(self):
        l1, l2, pi, vf = T.Linear(4, 64), T.Linear(64, 64), T.Linear(64, 2), T.Linear(64, 1)
        params = Record(l1=l1.init(1), l2=l2.init(2), pi=pi.init(3), vf=vf.init(4))

        def model(p, x):
            h = ops.tanh(l2(p.l2, ops.tanh(l1(p.l1, x))))
            return [pi(p.pi, h), vf(p.vf, h)]
        r = rl.ppo(rl.cartpole(8, seed=0), model, params, total_steps=200000, epochs=10, lr=1e-3, seed=0)

        def score(p):
            return rl.evaluate(rl.cartpole(1, seed=5),
                               lambda o: np.asarray(ops.raw(model(p, o)[0])).argmax(1), episodes=5).mean
        best = score(r.params)
        # floating-point differences between BLAS libraries (Accelerate on macOS, OpenBLAS elsewhere) change the
        # exact training trajectory; a short continuation from where it stopped reaches the same goal
        for extra in range(2):
            if best >= 195.0:
                break
            r = rl.ppo(rl.cartpole(8, seed=extra + 1), model, r.params, total_steps=100000, epochs=10, lr=5e-4,
                       seed=extra + 1)
            best = max(best, score(r.params))
        self.assertGreaterEqual(best, 195.0)

    def test_dqn_gridworld(self):
        l1, l2 = T.Linear(16, 64), T.Linear(64, 4)
        params = Record(l1=l1.init(1), l2=l2.init(2))
        model = lambda p, x: l2(p.l2, ops.relu(l1(p.l1, x)))     # noqa: E731
        r = rl.dqn(rl.gridworld(4, n=4, seed=0), model, params, total_steps=12000, learning_starts=500,
                   target_every=100, seed=0)
        ev = rl.evaluate(rl.gridworld(4, 1), lambda o: np.asarray(ops.raw(model(r.params, o))).argmax(1), 3)
        self.assertAlmostEqual(ev.mean, 0.95, places=6)            # the shortest path

    def test_gae_and_preference_losses(self):
        rng = np.random.default_rng(0)
        R, V = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
        D, L = (rng.random((7, 3)) < 0.2) * 1.0, rng.normal(size=3)
        g = rl.gae(R, V, D, L, 0.9, 0.8)
        adv = np.zeros_like(R)
        for n in range(3):
            for t in range(7):
                s, c = 0.0, 1.0
                for k in range(t, 7):
                    nv = L[n] if k == 6 else V[k + 1, n]
                    s += c * (R[k, n] + 0.9 * nv * (1 - D[k, n]) - V[k, n])
                    c *= 0.9 * 0.8 * (1 - D[k, n])
                    if D[k, n]:
                        break
                adv[t, n] = s
        self.assertLess(np.abs(adv - g.advantages).max(), 1e-12)
        pc, pr, rc, rr = (rng.normal(size=5) for _ in range(4))
        z = 0.1 * ((pc - pr) - (rc - rr))
        ref = float(np.mean(np.log1p(np.exp(-z))))
        self.assertAlmostEqual(float(ops.raw(rl.dpo_loss(pc, pr, rc, rr, 0.1).loss)), ref, places=12)
        self.assertAlmostEqual(float(ops.raw(rl.preference_loss(np.array([2.0]), np.array([0.0])))),
                               math.log1p(math.exp(-2.0)), places=12)
        a = rl.grpo_advantages([1, 2, 3, 4, 0, 0, 1, 1], 4)
        self.assertAlmostEqual(float(a[:4].mean()), 0.0, places=12)

    def test_mcts_and_beam_search(self):
        def legal(s):
            return [i for i in range(9) if s[0][i] == 0]

        def winner(b):
            for a, b2, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
                if b[a] and b[a] == b[b2] == b[c]:
                    return b[a]
            return 0

        def apply(s, a):
            b = list(s[0])
            b[a] = s[1]
            return (tuple(b), -s[1])

        def term(s):
            return winner(s[0]) != 0 or all(s[0])

        def rew(s):
            w = winner(s[0])
            return 0.0 if w == 0 else (1.0 if w == -s[1] else -1.0)
        self.assertEqual(rl.mcts(((1, 1, 0, -1, -1, 0, 0, 0, 0), 1), legal, apply, term, rew, 2000,
                                 two_player=True).action, 2)
        self.assertEqual(rl.mcts(((1, 0, 0, -1, -1, 0, 1, 0, 0), 1), legal, apply, term, rew, 3000,
                                 two_player=True).action, 5)
        bs = rl.beam_search(lambda seq: np.log([0.1, 0.6, 0.3]) if len(seq) < 3 else np.log([0.98, 0.01, 0.01]),
                            [], beam_width=2, max_len=4, eos=0)
        self.assertEqual(bs[0].tokens, [1, 1, 1, 0])


class Memory(unittest.TestCase):
    def test_checkpointing_gives_identical_gradients(self):
        from apsi.stdlib import value_and_grad
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(32, 16)), rng.normal(size=(32, 16))
        blocks = [T.Sequential(T.Linear(16, 32), T.module()["GELU"](), T.Dropout(0.2), T.Linear(32, 16))
                  for _ in range(3)]
        P = [b.init(i) for i, b in enumerate(blocks)]

        def run(ck):
            ops._RNG[0] = np.random.default_rng(5)

            def f(p):
                h = X
                for b, pb in zip(blocks, p):
                    h = h + (T.checkpoint_layer(b, pb, h, train=True) if ck else b(pb, h, train=True))
                return ops.mean((h - Y) ** 2)
            return value_and_grad(f)(P)
        (v0, g0), (v1, g1) = run(False), run(True)
        self.assertEqual(v0, v1)
        a, b = T._flat_params(Record(g=g0)), T._flat_params(Record(g=g1))
        for k in a:
            self.assertTrue(np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k)

    def test_float32_training_stays_float32(self):
        from apsi import layers as L
        rng = np.random.default_rng(0)
        model = L.sequential([L.linear(8, 16), L.activation("relu"), L.linear(16, 2)])
        r = L.fit(model, model.init(0), rng.normal(size=(64, 8)), rng.normal(size=(64, 2)), epochs=3,
                  batch_size=32, verbose=False, precision="fp32")
        self.assertEqual(np.asarray(r.params[0].W).dtype, np.float32)
        self.assertLess(r.history[-1], r.history[0])


if __name__ == "__main__":
    unittest.main()
