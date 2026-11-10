"""Long-range graph layers (apsi/graphs.py): reach, isolation between graphs, relations, and learning a task that
needs information from far away."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import graphs as G                         # noqa: E402
from apsi import layers as L                         # noqa: E402
from apsi.stdlib import Adam                         # noqa: E402


def chains(n_graphs, length, reverse=True):
    edges, batch = [], []
    for k in range(n_graphs):
        base = k * length
        edges += [[base + i, base + i + 1] for i in range(length - 1)]
        batch += [k] * length
    return L.graph(edges=edges, n=n_graphs * length, batch=batch, reverse=reverse)


def depends(layer, g, x, a, b):
    """Does output row a change when input row b changes?"""
    p = layer.init(3)
    x2 = x.copy()
    x2[b, 0] += 1.0
    return float(np.abs(np.asarray(layer(p, x2, graph=g))[a] - np.asarray(layer(p, x, graph=g))[a]).max()) > 1e-9


class Structure(unittest.TestCase):
    def test_hops_and_shortcuts(self):
        g = chains(1, 10)
        self.assertEqual(list(G.hop_distances(g, 0)), list(range(10)))
        self.assertEqual(G.eccentricity(g), 9)
        k = G.k_hop(g, 3)
        self.assertEqual(int(k.hop.max()), 3)
        self.assertTrue(((k.src == 0) & (k.dst == 3)).any())
        self.assertFalse(((k.src == 0) & (k.dst == 4)).any())

    def test_reverse_edges_are_their_own_relation(self):
        g = L.graph(edges=[[0, 1], [1, 2]], n=3, reverse=True)
        self.assertEqual(g.relations, 2)
        self.assertTrue(((g.src == 1) & (g.dst == 0) & (g.rel == 1)).any())


class Reach(unittest.TestCase):
    def setUp(self):
        self.g = chains(2, 8)
        self.x = np.random.default_rng(0).standard_normal((16, 8))

    def test_local_layer_reads_neighbours_only(self):
        layer = G.GraphTransformer(8, 2, relations=2)
        self.assertTrue(depends(layer, self.g, self.x, 0, 1))
        self.assertFalse(depends(layer, self.g, self.x, 0, 2))

    def test_global_attention_reaches_the_whole_graph_but_not_other_graphs(self):
        layer = G.GlobalAttention(8, 2)
        self.assertTrue(depends(layer, self.g, self.x, 0, 7))
        self.assertFalse(depends(layer, self.g, self.x, 0, 8))      # node 8 belongs to the second graph

    def test_landmark_keys_above_max_keys(self):
        layer = G.GlobalAttention(8, 2, max_keys=3)
        y = layer(layer.init(0), self.x, graph=self.g)
        self.assertEqual(np.shape(y), (16, 8))

    def test_k_hop_and_mix_hop_read_k_hops_in_one_layer(self):
        k3 = G.k_hop(self.g, 3)
        self.assertTrue(depends(G.GraphTransformer(8, 2), k3, self.x, 0, 3))
        self.assertFalse(depends(G.GraphTransformer(8, 2), k3, self.x, 0, 4))
        self.assertTrue(depends(G.MixHop(8, 4, (0, 1, 3)), self.g, self.x, 0, 3))


class Memory(unittest.TestCase):
    def test_top_k_salient_nodes_per_graph(self):
        g = chains(3, 5)
        x = np.random.default_rng(1).standard_normal((15, 8))
        layer = G.GraphMemory(8, k=4)
        p = layer.init(0)
        m = np.asarray(layer(p, x, graph=g))
        self.assertEqual(m.shape, (3, 4, 8))
        score = (x @ p.w + p.b).ravel()
        best = np.argsort(-score[5:10])[0] + 5                      # graph 1's most salient node comes first
        w = 1 / (1 + np.exp(-score[best]))
        self.assertTrue(np.allclose(m[1, 0], x[best] * w))
        small = np.asarray(G.GraphMemory(8, k=7)(p, x, graph=g))  # fewer nodes than k: the rest is zero
        self.assertTrue(np.allclose(small[:, 5:], 0))


class Learning(unittest.TestCase):
    def task(self, n_graphs=48, length=6, seed=0):
        """Node 0 of every chain must report the class written at the far end (5 hops away)."""
        rng = np.random.default_rng(seed)
        g = chains(n_graphs, length)
        x = np.zeros((n_graphs * length, 4))
        y = np.zeros(n_graphs * length, dtype=np.int64)
        mask = np.zeros(n_graphs * length, dtype=bool)
        for k in range(n_graphs):
            c = rng.integers(0, 2)
            x[k * length + length - 1, c] = 1.0
            x[k * length:(k + 1) * length, 2] = np.arange(length) / length       # position along the chain
            y[k * length] = c
            mask[k * length] = True
        return g, x, y, mask

    def accuracy(self, model, params, g, x, y, mask):
        pred = np.asarray(model(params, x, graph=g)).argmax(1)
        return float((pred[mask] == y[mask]).mean())

    def test_one_global_layer_solves_a_five_hop_task(self):
        g, x, y, mask = self.task()
        net = G.graph_transformer_net(4, 16, 2, layers=1, heads=2, relations=2, global_every=1)
        r = L.fit(net, net.init(0), x, y, loss="cross_entropy", opt=Adam(1e-2), epochs=150, graph=g, mask=mask,
                  verbose=False)
        self.assertGreaterEqual(self.accuracy(net, r.params, g, x, y, mask), 0.95)

    def test_message_passing_needs_enough_layers(self):
        g, x, y, mask = self.task()
        deep = G.graph_transformer_net(4, 16, 2, layers=6, heads=2, relations=2, global_every=0)
        r = L.fit(deep, deep.init(1), x, y, loss="cross_entropy", opt=Adam(3e-3), epochs=300, graph=g, mask=mask,
                  verbose=False)
        self.assertGreaterEqual(self.accuracy(deep, r.params, g, x, y, mask), 0.95)
        shallow = G.graph_transformer_net(4, 16, 2, layers=2, heads=2, relations=2, global_every=0)
        r = L.fit(shallow, shallow.init(1), x, y, loss="cross_entropy", opt=Adam(3e-3), epochs=300, graph=g,
                  mask=mask, verbose=False)
        self.assertLessEqual(self.accuracy(shallow, r.params, g, x, y, mask), 0.75)    # cannot see 5 hops away


if __name__ == "__main__":
    unittest.main()
