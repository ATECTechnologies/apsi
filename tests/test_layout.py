"""use layout: byte-exact C layouts, bit fields, register maps, C headers that compile, FFI buffers; base-16/2/8
literals."""
import ctypes
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from apsi import layout as Lo
from apsi.lexer import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fields():
    inner = Lo.c_struct("Inner", [("a", Lo.u8), ("b", Lo.f64)])
    return inner, [("id", Lo.u16), ("flags", Lo.bits(Lo.u8, [("ready", 1), ("mode", 3)])), ("v", Lo.f32),
                   ("name", Lo.array(Lo.char, 5)), ("inner", inner), ("xs", Lo.array(Lo.i16, 3)), ("big", Lo.u64)]


class _CI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_double)]


_CF = [("id", ctypes.c_uint16), ("flags", ctypes.c_uint8), ("v", ctypes.c_float), ("name", ctypes.c_char * 5),
       ("inner", _CI), ("xs", ctypes.c_int16 * 3), ("big", ctypes.c_uint64)]


class Layouts(unittest.TestCase):
    def test_offsets_match_the_c_compiler(self):
        """Natural alignment and pack(1, 2, 4) against ctypes (the platform C compiler's rules)."""
        _, f = _fields()
        for pack in (0, 1, 2, 4):
            S = Lo.c_struct("P", f, pack=pack)

            class C(ctypes.Structure):
                if pack:
                    _pack_ = pack
                _fields_ = _CF
            self.assertEqual(S.size, ctypes.sizeof(C), pack)
            for n, _ in _CF:
                self.assertEqual(S.offset(n), getattr(C, n).offset, (pack, n))
            self.assertEqual(S.numpy_dtype().itemsize, S.size)

    def test_pack_unpack_view_and_numpy(self):
        _, f = _fields()
        S = Lo.c_struct("P", f)
        vals = {"id": 7, "flags": {"ready": 1, "mode": 5}, "v": 1.5, "name": "abc", "inner": {"a": 3, "b": 2.25},
                "xs": [1, -2, 3], "big": 2 ** 63}
        b = S.pack(vals)
        c = type("C", (ctypes.Structure,), {"_fields_": _CF}).from_buffer_copy(b)
        self.assertEqual((c.id, c.flags, c.v, c.name, c.inner.b, list(c.xs), c.big),
                         (7, 1 | 5 << 1, 1.5, b"abc", 2.25, [1, -2, 3], 2 ** 63))
        r = S.unpack(b)
        self.assertEqual((r.flags.mode, r.name, list(r.xs), r.inner.a), (5, "abc", [1, -2, 3], 3))
        arr = np.frombuffer(b * 3, S.numpy_dtype())
        self.assertEqual(list(arr["inner"]["b"]), [2.25] * 3)
        buf = bytearray(b)
        v = S.view(buf)
        v.id, v.inner.b = 9, 4.5
        self.assertEqual((S.unpack(buf).id, S.unpack(buf).inner.b), (9, 4.5))
        big = Lo.c_struct("B", [("x", Lo.u32)], endian="big")
        self.assertEqual(big.pack({"x": 0x01020304}), b"\x01\x02\x03\x04")

    def test_refusals(self):
        S = Lo.c_struct("S", [("x", Lo.u8), ("f", Lo.bits(Lo.u8, [("a", 3)]))])
        with self.assertRaises(Lo.LayoutError):
            S.pack({"x": 256})
        with self.assertRaises(Lo.LayoutError):
            S.pack({"f": {"a": 8}})
        with self.assertRaises(Lo.LayoutError):
            Lo.bits(Lo.u8, [("a", 5), ("b", 4)])
        with self.assertRaises(Lo.LayoutError):
            Lo.c_struct("S", [("x", Lo.u8), ("x", Lo.u8)])
        with self.assertRaises(Lo.LayoutError):
            Lo.register_map("R", 0, [("A", 0, Lo.u32), ("B", 2, Lo.u16)])
        with self.assertRaises(Lo.LayoutError):
            S.view(bytes(4))

    def test_registers(self):
        R = Lo.register_map("UART0", 0x40001000, [("DR", 0, Lo.u32), ("SR", 4, Lo.u32, [("txe", 1), ("rxne", 1)]),
                                                  ("BRR", 8, Lo.u16)])
        img = bytearray(R.size)
        R.write(img, "SR", {"txe": 1, "rxne": 1})
        R.write(img, "BRR", 0x1A2B)
        r = R.read(img)
        self.assertEqual((r.SR.value, r.SR.rxne, r.BRR), (3, 1, 0x1A2B))
        self.assertIn("#define UART0_SR (*(volatile uint32_t *)(UART0_BASE + 0x4u))", R.c_header())

    @unittest.skipUnless(shutil.which("cc") or shutil.which("gcc"), "a C compiler")
    def test_headers_compile_and_c_writes_through_a_view(self):
        cc = shutil.which("cc") or shutil.which("gcc")
        _, f = _fields()
        # (Windows keeps a loaded library locked until the process ends: the folder is left for the system to clean)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            for name, pack in (("Nat", 0), ("Packed", 1), ("Four", 4)):
                open(os.path.join(d, f"{name}.h"), "w").write(Lo.c_struct(name, f, pack=pack).c_header())
            R = Lo.register_map("TIM", 0x40000000, [("CR", 0, Lo.u32, [("en", 1)]), ("CNT", 4, Lo.u32)])
            open(os.path.join(d, "tim.h"), "w").write(R.c_header())
            S = Lo.c_struct("Sample", [("id", Lo.u16), ("flags", Lo.bits(Lo.u8, [("ok", 1), ("mode", 3)])),
                                       ("value", Lo.f32), ("count", Lo.u64)])
            open(os.path.join(d, "sample.h"), "w").write(S.c_header())
            src = os.path.join(d, "dev.c")
            open(src, "w").write('#include "Nat.h"\n#include "Packed.h"\n#include "Four.h"\n#include "tim.h"\n'
                                 '#include "sample.h"\nvoid fill(Sample *s, uint16_t id) { s->id = id; '
                                 's->flags = SAMPLE_FLAGS_OK_MASK | (5u << SAMPLE_FLAGS_MODE_SHIFT); '
                                 's->value = 2.5f; s->count += 10; }\n')
            lib = os.path.join(d, "dev.dll" if os.name == "nt" else "libdev.so")
            r = subprocess.run([cc, "-std=c11", "-Wall", "-Werror", "-shared", "-fPIC", src, "-o", lib],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            from apsi.interop import CLibrary
            fill = CLibrary(lib).fn("fill", ["ptr_void", "u16"], "void")
            mem = bytearray(S.size * 2)
            v = S.view(mem, S.size)
            v.count = 32
            fill(v, 0xBEEF)
            self.assertEqual((v.id, v.flags.ok, v.flags.mode, v.value, v.count), (0xBEEF, 1, 5, 2.5, 42))
            self.assertEqual(bytes(mem[:S.size]), bytes(S.size))


class Literals(unittest.TestCase):
    def test_bases(self):
        toks = [t.value for t in tokenize("x = 0xFF + 0b1010_0101 + 0o17 + 0xdead_beef\n") if t.type == "NUMBER"]
        self.assertEqual(toks, ["255", "165", "15", str(0xDEADBEEF)])
        for bad in ("0x\n", "0xFG\n", "0b102\n", "0o8\n"):
            with self.assertRaises(Exception):
                tokenize(bad)


if __name__ == "__main__":
    unittest.main()


class Notebooks(unittest.TestCase):
    def test_ipynb_converts_and_runs(self):
        import json
        import sys
        nb = {"cells": [{"cell_type": "markdown", "source": ["# Demo\n"]},
                        {"cell_type": "code", "source": ["%time\n", "def sq(v):\n", "    return v * v\n"]},
                        {"cell_type": "code", "source": ["!ls\n", "print(sum(sq(i) for i in range(4)))\n"]}],
              "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "nb.ipynb")
            json.dump(nb, open(p, "w"))
            r = subprocess.run([sys.executable, "-m", "apsi", "convert", p, "--run"], capture_output=True, text=True,
                               cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("14", r.stdout)
            text = open(os.path.join(d, "nb.apsi"), encoding="utf-8").read()
            self.assertIn("# (notebook only) %time", text)
            self.assertIn("# --- cell 2", text)


class RaggedGraphs(unittest.TestCase):
    def test_batch_graphs_and_gine(self):
        """Graphs of different sizes with edge features batched; gine trains the same on NumPy and the GPU trainer."""
        from apsi import layers as L, gpu_train as GT
        from apsi.stdlib import Adam
        rng = np.random.default_rng(0)
        items = []
        for _ in range(12):
            n = int(rng.integers(3, 9))
            g = L.graph(edges=rng.integers(0, n, (2 * n, 2)).tolist(), n=n,
                        edge_features=rng.normal(size=(2 * n, 2)), undirected=True,
                        node_types=rng.integers(0, 2, n))
            items.append((rng.normal(size=(n, 4)), g, int(n % 3)))
        X, Y, g = L.batch_graphs(items[:4])
        self.assertEqual((len(X), g.n, g.graphs, len(Y)), (sum(len(i[0]) for i in items[:4]),) * 2 + (4, 4))
        self.assertEqual(g.efeat.shape, (g.edges, 2))
        self.assertEqual(list(np.bincount(g.batch)), [i[1].n for i in items[:4]])
        np.testing.assert_array_equal(g.ntype, np.concatenate([i[1].ntype for i in items[:4]]))
        batches = [L.batch_graphs(items[k:k + 4]) for k in range(0, 12, 4)]
        m = L.Sequential([L.GINE(4, 16, 2), L.GINE(16, 16, 2), L.GraphPool("mean"), L.Linear(16, 3)])
        a = L.fit_batches(m, m.init(seed=0), batches, "cross_entropy", opt=Adam(0.01), epochs=5, backend="numpy",
                          verbose=False)
        b = GT.fit_batches(m, m.init(seed=0), batches, "cross_entropy", opt=Adam(0.01), epochs=5, verbose=False,
                           precision="fp32", ops=GT.Emulator())
        np.testing.assert_allclose(a.history, b.history, rtol=1e-5)
        self.assertLess(a.history[-1], a.history[0])
        with self.assertRaises(Exception):
            L.GINE(4, 8, 2)(m.layers[0].init(), X, graph=L.graph(edges=[[0, 1]], n=len(X)))
