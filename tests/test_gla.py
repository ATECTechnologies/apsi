"""GLA 3.0 in `use layers` (apsi/gla.py): the parameter count of every part against GLA's own formula, the layers of
GLA's brains on the GPU trainer against NumPy, and the whole model trained on NumPy, on one card and split across
cards (with the writer's tied embedding on two cards)."""
import unittest

import numpy as np

from apsi import gla as GL, graphs as GR, layers as L, gpu_train as GT
from apsi.stdlib import Adam


def hetero(seed=0, N=40):
    rng = np.random.default_rng(seed)
    g = L.graph(edges=rng.integers(0, N, (2 * N, 2)).tolist(), n=N, batch=np.repeat([0, 1, 2], [15, 15, N - 30]),
                types=rng.integers(0, 4, 2 * N), node_types=rng.integers(0, 5, N))
    return rng.normal(size=(N, 16)), g, rng


def same_training(tc, m, batches, loss, epochs=4, lr=0.02, **gpu):
    a = L.fit_batches(m, m.init(seed=0), batches, loss, opt=Adam(lr), epochs=epochs, backend="numpy", verbose=False)
    b = GT.fit_batches(m, m.init(seed=0), batches, loss, opt=Adam(lr), epochs=epochs, verbose=False,
                       precision="fp32", ops=gpu.pop("ops", None) or GT.Emulator(), **gpu)
    np.testing.assert_allclose(a.history, b.history, rtol=1e-5)
    return a, b


class Sizes(unittest.TestCase):
    def test_every_part_is_counted_as_gla_counts_it(self):
        parts = GL.gla_count("21b")
        self.assertEqual(parts["brain_a"], 5731233859)
        self.assertEqual(parts["writer"], 8214031382)
        self.assertEqual(sum(parts.values()), 20480386836)
        for size in ("tiny", "small"):
            m = GL.gla(size)
            p = m.init(seed=0)
            self.assertEqual(GL.gla_parts(m, p), GL.gla_count(m.config))
            self.assertEqual(L.count_params(p), sum(GL.gla_count(m.config).values()))
        for kw in (dict(tie=False), dict(split=False)):
            m = GL.gla("tiny", **kw)
            self.assertEqual(GL.gla_parts(m, m.init(seed=0)), GL.gla_count(m.config))

    def test_the_relations_of_each_brain(self):
        rels = GL.relation_table()
        a = GL.brain_relations(GL.CODE_TYPES)
        self.assertIn(("func", "rev_invokes", "call"), a)
        self.assertEqual(rels[len(a):], [("word", "next", "word"), ("word", "near", "word"),
                                          ("word", "rev_next", "word"), ("word", "rev_near", "word")])


class BrainLayers(unittest.TestCase):
    """Each brain owns some node types and relations; the GPU trainer's kernels give NumPy's training."""

    def test_owned_types_pass_the_others_through(self):
        X, g, _ = hetero()
        for lay in (GR.HGT(16, 4, types=[0, 2, 4], relations=[1, 3]), GR.TypedLinear(16, 16, [3, 1]),
                    GR.GlobalAttention(16, 4, ratio=0.5, learned_keys=True, bias=True, types=[1, 3])):
            y = np.asarray(lay(lay.init(1), X, graph=g))
            own = np.isin(g.ntype, [0, 2, 4] if isinstance(lay, GR.HGT) else [1, 3])
            if isinstance(lay, GR.TypedLinear):
                self.assertTrue(np.all(y[~own] == 0))
            else:
                np.testing.assert_array_equal(y[~own], X[~own])

    def test_global_attention_with_learned_keys_starts_as_a_no_op(self):
        X, g, _ = hetero()
        ga = GR.GlobalAttention(16, 4, max_keys=6, ratio=0.5, learned_keys=True, bias=True, zero_init=True)
        np.testing.assert_array_equal(np.asarray(ga(ga.init(0), X, graph=g)), X)
        self.assertEqual(ga.init(0).Wq.shape, (16, 8))

    def test_layers_on_the_card(self):
        X, g, rng = hetero()
        Y = rng.integers(0, 3, len(X))
        for lay in (GR.GlobalAttention(16, 4), GR.GlobalAttention(16, 4, max_keys=8),
                    GR.GlobalAttention(16, 4, max_keys=6, ratio=0.5, learned_keys=True, bias=True, types=[1, 3, 4]),
                    GR.HGT(16, 4, types=[0, 2, 4], relations=[1, 3]), GR.TypedNorm(16, [2, 0])):
            same_training(self, L.Sequential([lay, L.Linear(16, 3)]), [(X, Y, g)], "cross_entropy")

    def test_brains_and_typed_pool_on_the_card(self):
        X, g, rng = hetero()
        Y = rng.integers(0, 3, 3)
        for brain in (GR.HGTBrain(16, 4, 3, [0, 2, 4], [1, 3], global_every=2, max_keys=6),
                      GR.HGTBrain(16, 4, 2, [1, 3], [0, 2])):
            m = L.Sequential([brain, GR.TypedPool(brain.types, [2.0, 1.0, 0.5][:brain.T]), L.Linear(16, 3)])
            same_training(self, m, [(X, Y, g)], "cross_entropy", epochs=6)

    def test_gated_cross_attention_into_a_masked_memory(self):
        rng = np.random.default_rng(0)
        B, T, S, D = 3, 5, 4, 16
        keep = (rng.random((B, S)) > 0.4).astype(float)
        keep[:, 0] = 1
        mem = np.concatenate([rng.normal(size=(B, S, D)), keep[:, :, None]], 2)
        x, Y = rng.normal(size=(B, T, D)), rng.integers(0, 3, (B, T))
        att = L.MultiHeadAttention(D, 4, kv_heads=2, qk_norm=True, bias=False, gate=0.25, memory_mask=True)
        m = L.DAG([["a", att, ["x", "m"]], ["o", L.Linear(D, 3), "a"]], inputs=["x", "m"], output="o")
        p = m.init(0)
        y1 = np.asarray(m(p, [x, mem]))
        mem2 = mem.copy()
        mem2[:, :, :D][keep == 0] = 99.0                       # an empty slot is never read
        np.testing.assert_allclose(np.asarray(m(p, [x, mem2])), y1)
        same_training(self, m, [([x, mem], Y)], "cross_entropy", lr=0.05)


class WholeModel(unittest.TestCase):
    def test_gla_trains_the_same_on_numpy_one_card_and_many(self):
        """GLA tiny (three brains, router, bridge, graph memory, the writer with gated cross-attention and tied
        embeddings): NumPy, one card, 4 stages, 2 stages x 2 replicas, a cut between the writer's embedding and its
        tied output, a cut inside a brain - the same training."""
        m = GL.gla("tiny")
        batches = [GL.gla_batch(GL.gla_task("tiny", int(b), seed=s)) for s, b in enumerate([3, 2, 4, 3, 2, 3, 2, 4])]
        kw = dict(epochs=2, verbose=False)
        ref = L.fit_batches(m, m.init(seed=0), batches, GL.gla_loss, opt=Adam(0.003), backend="numpy", accumulate=4,
                            **kw)
        names = [n for n, *_ in m.nodes]
        k = names.index("w_emb")
        j = names.index("brain_a_l1")                   # a brain cut between its layers (its shared weights on both)
        for ops, acc, st in ((GT.Emulator(), 4, None), ([GT.Emulator() for _ in range(4)], 4, None),
                             ([GT.Emulator() for _ in range(4)], 4, 2),
                             ([GT.Emulator() for _ in range(3)], 4, [k + 1, 3, len(names) - k - 4]),
                             ([GT.Emulator() for _ in range(2)], 4, [j, len(names) - j])):
            r = GT.fit_batches(m, m.init(seed=0), batches, GL.gla_loss, opt=Adam(0.003), precision="fp32", ops=ops,
                               accumulate=acc, stages=st, **kw)
            np.testing.assert_allclose(ref.history, r.history, rtol=1e-5)
        from apsi.values import tree_leaves
        worst = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                    for a, b in zip(tree_leaves(ref.params), tree_leaves(r.params)))
        self.assertLess(worst, 1e-3)

    def test_gla_learns_its_task(self):
        m = GL.gla("tiny")
        train = [GL.gla_batch(GL.gla_task("tiny", 8, seed=s)) for s in range(12)]
        r = L.fit_batches(m, m.init(seed=0), train, GL.gla_loss, opt=Adam(0.003), epochs=6, backend="numpy",
                          verbose=False)
        self.assertLess(r.history[-1], 0.6 * r.history[0])


class WrittenInApsi(unittest.TestCase):
    def test_a_layer_written_in_apsi_trains_on_one_card_and_many(self):
        """A layer the GPU trainer has no kernel for, written in Apsi (its forward takes only what it needs: graph=,
        train=): NumPy, one card, three stages and two stages x two replicas give the same training."""
        from apsi.interpreter import run_source
        env = run_source('''
use layers
use nn
let mine = custom_layer(fn(seed=0) => record(W=randn(32, 32, seed=seed) * 0.1, g=ones(1) * 0.5),
                        fn(p, x, train=true, graph=none) => x + tanh(p.g[0]) * (segment_mean(x, graph.batch,
                                                                       graph.graphs)[graph.batch] @ p.W))
let plain = fn(a, train=true) => a
assert plain(1, train=false) == 1
''')
        mine = env.lookup("mine")
        rng = np.random.default_rng(0)

        def make(G):
            sizes = rng.integers(3, 9, G)
            src, dst, off = [], [], 0
            for s_ in sizes:
                e = rng.integers(0, s_, (2 * s_, 2)) + off
                src += list(e[:, 0])
                dst += list(e[:, 1])
                off += s_
            g = L.graph(edges=np.stack([src, dst], 1).tolist(), n=int(sizes.sum()), batch=np.repeat(np.arange(G), sizes))
            return rng.normal(size=(int(sizes.sum()), 6)), rng.integers(0, 3, G), g
        batches = [make(G) for G in (3, 2, 4, 3, 2, 3, 2, 4)]
        m = L.DAG([["a", L.Linear(6, 32), "x"], ["b", mine, "a"], ["c", GR.GlobalAttention(32, 4), "b"],
                   ["p", L.GraphPool("mean"), "c"], ["o", L.Linear(32, 3), "p"]], inputs=["x"], output="o")
        kw = dict(opt=None, epochs=3, verbose=False, accumulate=4)
        ref = L.fit_batches(m, m.init(seed=0), batches, "cross_entropy", backend="numpy", **dict(kw, opt=Adam(0.01)))
        for ops, st in ((GT.Emulator(), None), ([GT.Emulator() for _ in range(3)], None),
                        ([GT.Emulator() for _ in range(4)], 2)):
            r = GT.fit_batches(m, m.init(seed=0), batches, "cross_entropy", precision="fp32", ops=ops, stages=st,
                               **dict(kw, opt=Adam(0.01)))
            np.testing.assert_allclose(ref.history, r.history, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
