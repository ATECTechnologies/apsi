"""Native Hugging Face model runners: tiny random checkpoints in the real file layout (config.json,
model.safetensors with Hugging Face tensor names, tokenizer.json), checked against straightforward per-head
reference code, padding invariance, and key/value-cache generation against full recomputation. Set
APSI_HF_COMPARE=1 with transformers + torch installed to also compare against the reference library."""
import contextlib
import io
import json
import math
import os
import struct
import tempfile
import unittest

import numpy as np

from apsi import hub as H
from apsi import hub_models as M


def write_safetensors(path, tensors, dtype="F32"):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32)
        if dtype == "BF16":                                  # the format of most published checkpoints
            a = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def rnd(rng, *shape, s=0.3):
    return (rng.standard_normal(shape) * s).astype(np.float32)


def tiny_bert(folder, labels=("NEG", "POS")):
    rng = np.random.default_rng(0)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "good", "bad", "movie", "!", "the"]
    Hd, I, L, V = 8, 16, 2, len(vocab)
    W = {"bert.embeddings.word_embeddings.weight": rnd(rng, V, Hd), "bert.embeddings.position_embeddings.weight":
         rnd(rng, 32, Hd), "bert.embeddings.token_type_embeddings.weight": rnd(rng, 2, Hd),
         "bert.embeddings.LayerNorm.weight": 1 + rnd(rng, Hd, s=0.1), "bert.embeddings.LayerNorm.bias": rnd(rng, Hd, s=0.1)}
    for i in range(L):
        p = f"bert.encoder.layer.{i}."
        for n in ("query", "key", "value"):
            W[p + f"attention.self.{n}.weight"] = rnd(rng, Hd, Hd)
            W[p + f"attention.self.{n}.bias"] = rnd(rng, Hd, s=0.1)
        W[p + "attention.output.dense.weight"] = rnd(rng, Hd, Hd)
        W[p + "attention.output.dense.bias"] = rnd(rng, Hd, s=0.1)
        W[p + "attention.output.LayerNorm.weight"] = 1 + rnd(rng, Hd, s=0.1)
        W[p + "attention.output.LayerNorm.bias"] = rnd(rng, Hd, s=0.1)
        W[p + "intermediate.dense.weight"] = rnd(rng, I, Hd)
        W[p + "intermediate.dense.bias"] = rnd(rng, I, s=0.1)
        W[p + "output.dense.weight"] = rnd(rng, Hd, I)
        W[p + "output.dense.bias"] = rnd(rng, Hd, s=0.1)
        W[p + "output.LayerNorm.weight"] = 1 + rnd(rng, Hd, s=0.1)
        W[p + "output.LayerNorm.bias"] = rnd(rng, Hd, s=0.1)
    W["bert.pooler.dense.weight"] = rnd(rng, Hd, Hd)
    W["bert.pooler.dense.bias"] = rnd(rng, Hd, s=0.1)
    W["classifier.weight"] = rnd(rng, len(labels), Hd)
    W["classifier.bias"] = rnd(rng, len(labels), s=0.1)
    write_safetensors(os.path.join(folder, "model.safetensors"), W)
    cfg = {"model_type": "bert", "hidden_size": Hd, "num_attention_heads": 2, "num_hidden_layers": L,
           "intermediate_size": I, "hidden_act": "gelu", "layer_norm_eps": 1e-12, "max_position_embeddings": 32,
           "id2label": {str(i): l for i, l in enumerate(labels)}, "pad_token_id": 0}
    with open(os.path.join(folder, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    tok = {"model": {"type": "WordPiece", "vocab": {t: i for i, t in enumerate(vocab)}, "unk_token": "[UNK]",
                     "continuing_subword_prefix": "##"},
           "normalizer": {"type": "BertNormalizer", "lowercase": True},
           "pre_tokenizer": {"type": "BertPreTokenizer"},
           "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "[CLS]"}},
                              {"Sequence": {"id": "A"}}, {"SpecialToken": {"id": "[SEP]"}}], "pair": []},
           "added_tokens": [{"id": i, "content": t, "special": True} for i, t in enumerate(vocab[:5])]}
    with open(os.path.join(folder, "tokenizer.json"), "w") as fh:
        json.dump(tok, fh)
    return W, cfg


def reference_bert(W, cfg, ids):
    """Per-head, per-position reference (independent of the vectorised runner)."""
    from math import erf
    g = lambda n: W["bert." + n]                                # noqa: E731
    Hd, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    dh = Hd // nh

    def ln(v, w, b):
        m = sum(v) / len(v)
        var = sum((t - m) ** 2 for t in v) / len(v)
        return [(t - m) / math.sqrt(var + 1e-12) * w[i] + b[i] for i, t in enumerate(v)]

    def lin(v, w, b):
        return [sum(w[o][i] * v[i] for i in range(len(v))) + b[o] for o in range(len(w))]
    xs = [ln([g("embeddings.word_embeddings.weight")[t][k] + g("embeddings.position_embeddings.weight")[p][k]
              + g("embeddings.token_type_embeddings.weight")[0][k] for k in range(Hd)],
             g("embeddings.LayerNorm.weight"), g("embeddings.LayerNorm.bias")) for p, t in enumerate(ids)]
    for i in range(cfg["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        Q = [lin(x, g(p + "attention.self.query.weight"), g(p + "attention.self.query.bias")) for x in xs]
        K = [lin(x, g(p + "attention.self.key.weight"), g(p + "attention.self.key.bias")) for x in xs]
        V = [lin(x, g(p + "attention.self.value.weight"), g(p + "attention.self.value.bias")) for x in xs]
        new = []
        for a in range(len(xs)):
            ctx = []
            for h in range(nh):
                sl = slice(h * dh, (h + 1) * dh)
                sc = [sum(q * k for q, k in zip(Q[a][sl], K[b][sl])) / math.sqrt(dh) for b in range(len(xs))]
                mx = max(sc)
                e = [math.exp(s - mx) for s in sc]
                z = sum(e)
                ctx += [sum(e[b] / z * V[b][sl][d] for b in range(len(xs))) for d in range(dh)]
            o = lin(ctx, g(p + "attention.output.dense.weight"), g(p + "attention.output.dense.bias"))
            x = ln([o[k] + xs[a][k] for k in range(Hd)], g(p + "attention.output.LayerNorm.weight"),
                   g(p + "attention.output.LayerNorm.bias"))
            f = lin(x, g(p + "intermediate.dense.weight"), g(p + "intermediate.dense.bias"))
            f = [0.5 * t * (1 + erf(t / math.sqrt(2))) for t in f]
            f = lin(f, g(p + "output.dense.weight"), g(p + "output.dense.bias"))
            new.append(ln([f[k] + x[k] for k in range(Hd)], g(p + "output.LayerNorm.weight"),
                          g(p + "output.LayerNorm.bias")))
        xs = new
    return np.array(xs)


def bf16(a):
    """Round to bfloat16 (as float32): weights that a GPU copy holds exactly."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def tiny_llama(folder, kind="llama", ctx=64, window=None, exact=False, dtype="F32"):
    rng = np.random.default_rng(1)
    G = "Ġ"
    vocab = ["<s>", "</s>", "h", "e", "l", "o", G, "w", "r", "d", "he", "ll", "hell", "hello", G + "w", "or",
             G + "wor", "ld", G + "world", "!"]
    V, Hd, nh, kvh, I, L = len(vocab), 16, 4, 2, 32, 2
    dh = Hd // nh
    W = {"model.embed_tokens.weight": rnd(rng, V, Hd), "model.norm.weight": 1 + rnd(rng, Hd, s=0.1)}
    for i in range(L):
        p = f"model.layers.{i}."
        W[p + "input_layernorm.weight"] = 1 + rnd(rng, Hd, s=0.1)
        W[p + "post_attention_layernorm.weight"] = 1 + rnd(rng, Hd, s=0.1)
        W[p + "self_attn.q_proj.weight"] = rnd(rng, nh * dh, Hd)
        W[p + "self_attn.k_proj.weight"] = rnd(rng, kvh * dh, Hd)
        W[p + "self_attn.v_proj.weight"] = rnd(rng, kvh * dh, Hd)
        W[p + "self_attn.o_proj.weight"] = rnd(rng, Hd, nh * dh)
        if kind == "qwen3":
            W[p + "self_attn.q_norm.weight"] = 1 + rnd(rng, dh, s=0.1)
            W[p + "self_attn.k_norm.weight"] = 1 + rnd(rng, dh, s=0.1)
        if kind == "qwen2":
            for n in ("q", "k", "v"):
                W[p + f"self_attn.{n}_proj.bias"] = rnd(rng, W[p + f"self_attn.{n}_proj.weight"].shape[0], s=0.1)
        W[p + "mlp.gate_proj.weight"] = rnd(rng, I, Hd)
        W[p + "mlp.up_proj.weight"] = rnd(rng, I, Hd)
        W[p + "mlp.down_proj.weight"] = rnd(rng, Hd, I)
    if exact:
        W = {k: bf16(v) for k, v in W.items()}
    write_safetensors(os.path.join(folder, "model.safetensors"), W, dtype)
    cfg = {"model_type": kind, "hidden_size": Hd, "num_attention_heads": nh, "num_key_value_heads": kvh,
           "num_hidden_layers": L, "intermediate_size": I, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
           "max_position_embeddings": ctx, "eos_token_id": 1, "tie_word_embeddings": True, "hidden_act": "silu"}
    if window:
        cfg["sliding_window"] = window
        cfg["use_sliding_window"] = True
    with open(os.path.join(folder, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    merges = ["h e", "l l", "he ll", "hell o", f"{G} w", "o r", f"{G}w or", "l d", f"{G}wor ld"]
    tok = {"model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(vocab)}, "merges": merges},
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
           "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>"}},
                              {"Sequence": {"id": "A"}}], "pair": []},
           "added_tokens": [{"id": 0, "content": "<s>", "special": True}, {"id": 1, "content": "</s>", "special": True}]}
    with open(os.path.join(folder, "tokenizer.json"), "w") as fh:
        json.dump(tok, fh)
    return W, cfg


class Encoders(unittest.TestCase):
    def test_bert_matches_reference_and_heads(self):
        with tempfile.TemporaryDirectory() as td:
            W, cfg = tiny_bert(td)
            m = H.Model(td)
            ids = m.tokenizer.encode("hello good movie !").ids
            want = reference_bert(W, cfg, ids)
            got = m.net.hidden(np.array([ids]))[0]
            np.testing.assert_allclose(got, want, atol=1e-4)
            r = m.classify("good movie")
            self.assertIn(r["label"], ("NEG", "POS"))
            both = m.classify(["good movie", "bad movie"], top_k=2)
            self.assertAlmostEqual(sum(x["score"] for x in both[0]), 1.0, places=5)
            # padding does not change a sentence's embedding
            alone = m.embed("hello")
            batch = m.embed(["hello", "the good movie the bad movie hello world"])
            np.testing.assert_allclose(alone, batch[0], atol=1e-5)
            self.assertGreater(m.similarity("good movie", "good movie"), 0.999)


class Decoders(unittest.TestCase):
    def check_decoder(self, kind):
        with tempfile.TemporaryDirectory() as td:
            tiny_llama(td, kind)
            m = H.Model(td)
            ids = m.tokenizer.encode("hello world").ids
            # cached generation == recomputing the whole sequence every step
            out = m.net.generate(ids, max_new_tokens=6, eos=[])
            seq = list(ids)
            for _ in range(6):
                lg = m.net.forward(np.array([seq]), m.net.new_cache(), 0)[0, -1]
                seq.append(int(np.argmax(lg)))
            self.assertEqual(out, seq[len(ids):])
            text = m.generate("hello", max_new_tokens=4)
            self.assertIsInstance(text, str)
            pieces = []
            text = m.generate("hello", max_new_tokens=12, temperature=0.8, top_k=5, top_p=0.9, seed=1,
                              stream=pieces.append)
            self.assertEqual("".join(pieces), text)                # the streamed pieces add up to the answer
            # RoPE: a rotation, so it keeps vector lengths
            x = np.random.default_rng(3).standard_normal((1, 2, 5, m.net.dh)).astype(np.float32)
            r = m.net._rope(x, np.arange(5))
            np.testing.assert_allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)

    def test_llama(self):
        self.check_decoder("llama")

    def test_qwen2_biases(self):
        self.check_decoder("qwen2")


class FakeGpu:
    """hub_gpu.Runner on the processor: the same bfloat16 weights and float32 sums, to check everything around it."""

    class dev:
        index, memory = 0, 8 << 30

    name = "fake"

    def __init__(self):
        self.calls = 0
        self.closed = False

    def upload(self, w):
        from apsi import hub_gpu
        p = hub_gpu.to_bf16_pairs(np.asarray(w, np.float32)).view(np.uint16).astype(np.uint32) << 16
        return p.view(np.float32).reshape(w.shape)

    def matmul(self, x, m):
        self.calls += 1
        return (x.astype(np.float32) @ m.T).astype(np.float32)

    def close(self):
        self.closed = True


class DecoderOnGpu(unittest.TestCase):
    def run_kind(self, kind):
        from apsi import hub_gpu
        real = hub_gpu.runner
        fake = FakeGpu()
        hub_gpu.runner = lambda device="auto": (None, "device=cpu") if device == "cpu" else (fake, "")
        try:
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, kind)
                cpu = H.Model(td, device="cpu")
                gpu = H.Model(td, device="auto")
                self.assertEqual(cpu.device, "cpu")
                self.assertTrue(gpu.device.startswith("cuda:0"))
                ids = cpu.tokenizer.encode("hello world").ids
                a = cpu.net.forward(np.array([ids]), cpu.net.new_cache(), 0)
                b = gpu.net.forward(np.array([ids]), gpu.net.new_cache(), 0)
                self.assertGreater(fake.calls, 0)
                np.testing.assert_allclose(a, b, atol=0.05)          # bfloat16 weights
                # cached generation on the GPU path == recomputing everything on the GPU path
                out = gpu.net.generate(ids, max_new_tokens=5, eos=[])
                seq = list(ids)
                for _ in range(5):
                    seq.append(int(np.argmax(gpu.net.forward(np.array([seq]), gpu.net.new_cache(), 0)[0, -1])))
                self.assertEqual(out, seq[len(ids):])
                # a driver error part way falls back to the processor
                fake.matmul = lambda x, m: (_ for _ in ()).throw(RuntimeError("lost"))
                M._WARNED.clear()
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    again = gpu.net.generate(ids, max_new_tokens=5, eos=[])
                self.assertIn("processor", err.getvalue())
                self.assertEqual(again, cpu.net.generate(ids, max_new_tokens=5, eos=[]))
                self.assertIsNone(gpu.net.gpu)
                self.assertTrue(fake.closed)
                self.assertEqual(gpu.to("cpu"), "cpu")
        finally:
            hub_gpu.runner = real

    def test_llama(self):
        self.run_kind("llama")

    def test_qwen2(self):
        self.run_kind("qwen2")

    def test_bf16_rounding(self):
        from apsi import hub_gpu
        x = np.array([[1.0, 1.00390625, 1.01171875, -2.5]], np.float32)   # exact, tie to even, tie up, exact
        got = (hub_gpu.to_bf16_pairs(x).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.reshape(-1), [1.0, 1.0, 1.015625, -2.5])

    def test_ptx(self):
        from apsi import hub_gpu
        try:
            ptx = hub_gpu._build_ptx()
        except ImportError:
            self.skipTest("llvmlite")
        self.assertIn(".entry apsi_kernel", ptx)
        self.assertIn("shfl.sync.bfly", ptx)

    @unittest.skipIf(os.environ.get("APSI_NO_GPU"), "APSI_NO_GPU")
    def test_real_gpu(self):
        from apsi import hub_gpu
        r, why = hub_gpu.runner("auto")
        if r is None:
            self.skipTest(why)
        rng = np.random.default_rng(0)
        w = rng.standard_normal((1000, 384)).astype(np.float32)
        m = r.upload(w)
        wb = (hub_gpu.to_bf16_pairs(w).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(w.shape)
        for T in (1, 7):
            x = rng.standard_normal((T, 384)).astype(np.float32)
            np.testing.assert_allclose(r.matmul(x, m), x @ wb.T, rtol=1e-4, atol=1e-3)
        r.close()


def _on(runner_obj, fn):
    """Run fn with hub_gpu.runner handing out runner_obj (and nothing for device="cpu")."""
    from apsi import hub_gpu
    real = hub_gpu.runner
    hub_gpu.runner = lambda device="auto": (None, "device=cpu") if device == "cpu" else (runner_obj, "")
    try:
        return fn()
    finally:
        hub_gpu.runner = real


class WholeNetworkOnGpu(unittest.TestCase):
    """The whole-network GPU path, with the NumPy stand-in for the kernels (the GPU is checked against it below)."""

    def compare(self, kind, make_runner, window=None, ctx=600, tol=2e-3):
        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, kind, ctx=ctx, window=window, exact=True)
                cpu = H.Model(td, device="cpu")
                gpu = H.Model(td, device="auto")
                self.assertTrue(gpu.net.full, gpu.device_note)
                gpu.net.CHUNK = 128                                  # several passes over the prompt
                gpu.net.SCRATCH = 4 * gpu.net.nh * 96 * 300          # and passes that shrink as the context grows
                rng = np.random.default_rng(5)
                ids = list(rng.integers(0, 20, 300))
                # the whole sequence at once, in passes of 256 positions, against the processor
                a = cpu.net.forward(np.array([ids]), cpu.net.new_cache(), 0)
                b = gpu.net.forward(np.array([ids]), gpu.net.new_cache(), 0)
                np.testing.assert_allclose(a, b, atol=tol)
                # a prompt, then one token at a time: the cache grows from 256 to 512 positions on the way
                cc, gc = cpu.net.new_cache(), gpu.net.new_cache()
                cpu.net.forward(np.array([ids[:250]]), cc, 0)
                gpu.net.forward(np.array([ids[:250]]), gc, 0)
                for j in range(250, 262):
                    a = cpu.net.forward(np.array([[ids[j]]]), cc, j, last=True)
                    b = gpu.net.forward(np.array([[ids[j]]]), gc, j, last=True)
                    np.testing.assert_allclose(a, b, atol=tol)
                self.assertEqual(gc.cap, 512)
                # generation keeps working, also through a failing GPU
                out = gpu.net.generate(ids[:10], max_new_tokens=6, eos=[])
                self.assertEqual(len(out), 6)
                return gpu
        return _on(make_runner(), go)

    def test_llama(self):
        from apsi import hub_gpu
        self.compare("llama", hub_gpu.Emulator)

    def test_qwen2_biases(self):
        from apsi import hub_gpu
        self.compare("qwen2", hub_gpu.Emulator)

    def test_qwen3_norms(self):
        from apsi import hub_gpu
        self.compare("qwen3", hub_gpu.Emulator)

    def test_sliding_window(self):
        from apsi import hub_gpu
        self.compare("llama", hub_gpu.Emulator, window=40)

    def test_bf16_checkpoint_goes_to_the_gpu_as_stored(self):
        from apsi import hub_gpu

        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "qwen2", ctx=600, exact=True, dtype="BF16")
                cpu = H.Model(td, device="cpu")
                gpu = H.Model(td)
                self.assertTrue(gpu.net.full)
                ids = [3, 5, 7, 11, 2, 9]
                a = cpu.net.forward(np.array([ids]), cpu.net.new_cache(), 0)
                b = gpu.net.forward(np.array([ids]), gpu.net.new_cache(), 0)
                np.testing.assert_allclose(a, b, atol=2e-3)
                decoded = set(gpu.net.W._f32)                    # float32 copies made on the way
                self.assertFalse([k for k in decoded if k.endswith("proj.weight") or "embed" in k], decoded)
        _on(hub_gpu.Emulator(), go)

    def test_split_between_gpu_and_processor(self):
        from apsi import hub_gpu

        def go():
            # Windows keeps a memory-mapped file open until the mapping is collected: let cleanup skip it
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                tiny_llama(td, "qwen2", ctx=600, exact=True, dtype="BF16")
                cpu = H.Model(td, device="cpu")
                ids = [3, 5, 7, 11, 2, 9] * 50
                want = cpu.net.forward(np.array([ids]), cpu.net.new_cache(), 0)
                for memory in ("ram", "disk"):
                    half = H.Model(td, gpu_layers=1, memory=memory)
                    self.assertIn("1 of 2 layers", half.device)
                    got = half.net.forward(np.array([ids]), half.net.new_cache(), 0)
                    np.testing.assert_allclose(got, want, atol=2e-3)
                    self.assertEqual(half.generate("hello", max_new_tokens=6),
                                     cpu.generate("hello", max_new_tokens=6))
                # too small a card: "auto" splits by itself, "cuda" refuses
                small = hub_gpu.Emulator()
                per_layer = H.Model(td, device="cpu").net._layer_bytes(0)
                kv = 2 * 4 * 2 * 4 * 600 * 2
                small.free_memory = lambda: per_layer * 1.5 + kv + (384 << 20) + 2 * 20 * 16
                fit = _on(small, lambda: H.Model(td))
                self.assertIn("1 of 2 layers", fit.device)
                with self.assertRaises(M.ModelError):
                    _on(small, lambda: H.Model(td, device="cuda"))
        _on(hub_gpu.Emulator(), go)

    def test_sampling_on_the_gpu_matches_the_processor(self):
        from apsi import hub_gpu

        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "qwen2", exact=True)
                m = H.Model(td)
                net, r = m.net, m.net.gpu
                V = 20
                lg = np.array([4, 3.5, 3, 1, 0.5, -1] + [-6] * 14, np.float32)
                r.put(r.ws("log", V), lg)
                seen = np.zeros(V, np.float32)
                seen[1] = 1
                SEEN = r.ws("seen", V)
                r.put(SEEN, seen)
                for temp, k, p_ in ((0.7, 0, 0.9), (1.0, 3, 1.0), (1.3, 0, 0.5), (0.8, 2, 0.95), (1.0, 0, 1.0)):
                    rng = np.random.default_rng(0)
                    got = np.bincount([net._sample_gpu(SEEN, V, temp, k, p_, 1.3, rng) for _ in range(6000)],
                                      minlength=V) / 6000
                    pen = lg.copy()
                    pen[1] = pen[1] / 1.3
                    rng = np.random.default_rng(1)
                    want = np.bincount([M._sample(pen, temp, k, p_, rng) for _ in range(6000)], minlength=V) / 6000
                    np.testing.assert_allclose(got, want, atol=0.03, err_msg=f"t={temp} k={k} p={p_}")
                # a flat distribution has too many candidates: the processor takes over (None)
                r.put(r.ws("log", V), np.zeros(V, np.float32))
                net.CANDIDATES = 8
                self.assertIsNone(net._sample_gpu(SEEN, V, 1.0, 0, 0.9, 1.0, np.random.default_rng(0)))
                net.CANDIDATES = 4096
                # end to end: sampled generation on the GPU path gives as many tokens as asked
                self.assertEqual(len(net.generate([3, 4, 5], max_new_tokens=7, eos=[], temperature=0.8, top_p=0.9,
                                                  repetition_penalty=1.1, seed=3)), 7)
        _on(hub_gpu.Emulator(), go)

    def test_out_of_memory_retries_on_the_gpu(self):
        from apsi import hub_gpu

        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "qwen2", exact=True)
                cpu = H.Model(td, device="cpu")
                gpu = H.Model(td)
                ids = cpu.tokenizer.encode("hello world").ids
                calls = [0]
                real = gpu.net.gpu.attention

                def once(*a, **k):
                    calls[0] += 1
                    if calls[0] == 5:
                        raise RuntimeError("cuMemAlloc failed: CUDA error 2: out of memory")
                    return real(*a, **k)
                gpu.net.gpu.attention = once
                got = gpu.net.generate(ids, max_new_tokens=8, eos=[])
                self.assertIsNotNone(gpu.net.gpu)                  # still on the GPU
                self.assertEqual(got, cpu.net.generate(ids, max_new_tokens=8, eos=[]))
        _on(hub_gpu.Emulator(), go)

    def test_one_model_many_threads(self):
        from apsi import hub_gpu
        import threading

        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "llama", exact=True)
                gpu = H.Model(td)
                want = gpu.generate("hello", max_new_tokens=6)
                got = [None] * 6

                def run(i):
                    got[i] = gpu.generate("hello", max_new_tokens=6)
                ts = [threading.Thread(target=run, args=(i,)) for i in range(6)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                self.assertEqual(got, [want] * 6)
        _on(hub_gpu.Emulator(), go)

    def test_failure_falls_back(self):
        from apsi import hub_gpu

        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "qwen2", exact=True)
                cpu = H.Model(td, device="cpu")
                gpu = H.Model(td)
                ids = cpu.tokenizer.encode("hello world").ids
                calls = [0]
                real = gpu.net.gpu.attention

                def flaky(*a, **k):
                    calls[0] += 1
                    if calls[0] > 10:
                        raise RuntimeError("lost")
                    return real(*a, **k)
                gpu.net.gpu.attention = flaky
                want = cpu.net.generate(ids, max_new_tokens=8, eos=[])
                streamed = []
                with contextlib.redirect_stderr(io.StringIO()):
                    got = gpu.net.generate(ids, max_new_tokens=8, eos=[], stream=streamed.append)
                self.assertEqual(streamed, got)                     # every token reported once, none twice
                self.assertEqual(len(got), 8)
                self.assertEqual(got, want)
                self.assertIsNone(gpu.net.gpu)
                self.assertEqual(gpu.net.generate(ids, max_new_tokens=8, eos=[]), want)
        _on(hub_gpu.Emulator(), go)

    @unittest.skipIf(os.environ.get("APSI_NO_GPU"), "APSI_NO_GPU")
    def test_real_gpu_kernels(self):
        """Every kernel on a real GPU against the NumPy stand-in, at real model sizes and awkward ones."""
        from apsi import hub_gpu
        g, why = hub_gpu.runner("auto")
        if g is None:
            self.skipTest(why)
        e = hub_gpu.Emulator()
        rng = np.random.default_rng(7)

        def pair(n, scale=1.0):
            a = (rng.standard_normal(n) * scale).astype(np.float32)
            return g.vec(a), e.vec(a)

        def same(bg, be, n, tol, what):
            np.testing.assert_allclose(g.get(bg, n), e.get(be, n), rtol=tol, atol=tol, err_msg=what)
        try:
            # matrix products: one row, a few rows, many rows; widths for both vector paths; fused norm; bias; +=
            for O, I in ((96, 64), (100, 42), (130, 88), (896, 896)):
                w = (rng.standard_normal((O, I)) * 0.1).astype(np.float32)
                mg, me = g.matrix(w), e.matrix(w)
                lng, lne = pair(I, 0.2)
                bg_, be_ = pair(O)
                for T in (1, 3, 8, 37, 70):
                    xg, xe = pair(T * I)
                    for kw in ({}, {"bias": True}, {"ln": True}, {"acc": True, "bias": True}):
                        start_y = (rng.standard_normal(T * O)).astype(np.float32)
                        yg, ye = g.vec(start_y), e.vec(start_y)
                        args = dict(bias=bg_ if kw.get("bias") else None, acc=kw.get("acc", False))
                        ea = dict(bias=be_ if kw.get("bias") else None, acc=kw.get("acc", False))
                        if kw.get("ln"):
                            args.update(ln=lng, eps=1e-6, off=0.0)
                            ea.update(ln=lne, eps=1e-6, off=0.0)
                        e.linear(me, xe, ye, T, **ea)
                        want = e.get(ye, T * O)
                        sms = g.dev.sms
                        for fake_sms in (sms, 1):                   # both tile heights of the tiled product
                            yg = g.vec(e.get(ye, T * O) * 0 + start_y)
                            g.dev.sms = fake_sms
                            try:
                                g.linear(mg, xg, yg, T, **args)
                            finally:
                                g.dev.sms = sms
                            np.testing.assert_allclose(g.get(yg, T * O), want, rtol=2e-3, atol=2e-3,
                                                       err_msg=f"linear O={O} I={I} T={T} {kw} sms={fake_sms}")
            # q/k/v preparation and attention at real head sizes, grouped heads, a sliding window
            for nh, kvh, dh, rot, norm, start in ((14, 2, 64, 64, False, 120), (8, 8, 128, 64, True, 200),
                                                  (4, 2, 32, 32, False, 3), (16, 2, 128, 128, False, 2900)):
                cap = start + 100
                Kg, Ke = pair(kvh * cap * dh)
                Vg, Ve = pair(kvh * cap * dh)
                for T in (1, 9):
                    width = (nh + 2 * kvh) * dh
                    qg, qe = pair(T * width)
                    Qg, Qe = pair(T * nh * dh)
                    cs = rng.standard_normal(cap * rot // 2).astype(np.float32)          # tables by position
                    sn = rng.standard_normal(cap * rot // 2).astype(np.float32)
                    cg, ce = g.vec(cs), e.vec(cs)
                    sg, se = g.vec(sn), e.vec(sn)
                    qn = pair(dh) if norm else (None, None)
                    kn = pair(dh) if norm else (None, None)
                    g.qkv_prep(qg, Qg, Kg, Vg, cg, sg, qn[0], kn[0], T, nh, kvh, dh, rot, start, cap, 1e-6)
                    e.qkv_prep(qe, Qe, Ke, Ve, ce, se, qn[1], kn[1], T, nh, kvh, dh, rot, start, cap, 1e-6)
                    same(Qg, Qe, T * nh * dh, 1e-4, f"q nh={nh} dh={dh} T={T}")
                    same(Kg, Ke, kvh * cap * dh, 1e-4, f"k cache nh={nh} dh={dh} T={T}")
                    same(Vg, Ve, kvh * cap * dh, 1e-4, f"v cache nh={nh} dh={dh} T={T}")
                    for window in (0, 50):
                        Og, Oe = pair(T * nh * dh)
                        Sg, Se = pair(nh * T * (start + T))
                        for splits in (None, 1, 3, 7):
                            Og, Oe = pair(T * nh * dh)
                            g.attention(Qg, Kg, Vg, Og, Sg, T, nh, kvh, dh, start, cap, window, 0.125, splits)
                            e.attention(Qe, Ke, Ve, Oe, Se, T, nh, kvh, dh, start, cap, window, 0.125)
                            same(Og, Oe, T * nh * dh, 2e-3,
                                 f"attention nh={nh} dh={dh} T={T} window={window} splits={splits}")
            for rows, cols in ((20, 16), (1000, 896)):
                w = (rng.standard_normal((rows, cols))).astype(np.float32)
                mg, me = g.matrix(w), e.matrix(w)
                ids = rng.integers(0, rows, 9)
                xg, xe = pair(9 * cols)
                g.embed(mg, ids, xg, 1.5)
                e.embed(me, ids, xe, 1.5)
                same(xg, xe, 9 * cols, 1e-6, f"embed {rows}x{cols}")
            for V in (20, 151936):
                lg = (rng.standard_normal(V) * 3).astype(np.float32)
                sn = (rng.random(V) < 0.01).astype(np.float32)
                lgg, lge = g.vec(lg), e.vec(lg)
                sng, sne = g.vec(sn), e.vec(sn)
                a_ = g.sample_prep(lgg, sng, V, 1.2, 1 / 0.7, 16.0, 4096)
                b_ = e.sample_prep(lge, sne, V, 1.2, 1 / 0.7, 16.0, 4096)
                self.assertAlmostEqual(a_[0], b_[0], places=4)
                self.assertAlmostEqual(a_[1] / b_[1], 1.0, places=3)
                self.assertEqual(a_[4], b_[4])
                if a_[4] <= 4096:
                    self.assertEqual(sorted(a_[2].tolist()), sorted(b_[2].tolist()))
            for kind in (0, 1):
                gug, gue = pair(5 * 2 * 300)
                mg_, me_ = pair(5 * 300)
                g.act_mul(gug, mg_, 5, 300, kind)
                e.act_mul(gue, me_, 5, 300, kind)
                same(mg_, me_, 5 * 300, 1e-4, f"act kind={kind}")
        finally:
            g.close()

    @unittest.skipIf(os.environ.get("APSI_NO_GPU"), "APSI_NO_GPU")
    def test_real_gpu_whole_network(self):
        """The kernels on a real GPU against the NumPy stand-in, every model kind."""
        from apsi import hub_gpu
        r, why = hub_gpu.runner("auto")
        if r is None:
            self.skipTest(why)
        r.close()
        for kind, window in (("llama", None), ("qwen2", None), ("qwen3", None), ("llama", 40)):
            with self.subTest(kind=kind, window=window):
                self.compare(kind, lambda: hub_gpu.runner("auto")[0], window=window)
                # and directly: GPU against the emulator, much tighter than against float32 weights
                def both():
                    with tempfile.TemporaryDirectory() as td:
                        tiny_llama(td, kind, ctx=600, window=window)
                        e = _on(hub_gpu.Emulator(), lambda: H.Model(td))
                        g = _on(hub_gpu.runner("auto")[0], lambda: H.Model(td))
                        ids = [3, 5, 7, 11, 2, 9, 4, 13] * 40
                        a = e.net.forward(np.array([ids]), e.net.new_cache(), 0)
                        b = g.net.forward(np.array([ids]), g.net.new_cache(), 0)
                        np.testing.assert_allclose(a, b, atol=2e-3)
                        g.net.release()
                both()


class NativeProcessorPath(unittest.TestCase):
    def test_bf16_checkpoint_on_the_processor(self):
        from apsi import hub_cpu
        if not hub_cpu.available():
            self.skipTest(hub_cpu._why)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_llama(a, "qwen2", ctx=600, exact=True, dtype="BF16")
            tiny_llama(b, "qwen2", ctx=600, exact=True)                 # the same weights, stored as float32
            native = H.Model(a, device="cpu")
            plain = H.Model(b, device="cpu")
            self.assertTrue(native.net._cm)                             # bfloat16 products on all cores
            self.assertFalse(plain.net._cm)
            for n in (5, 40):                                           # one-row and many-row paths
                ids = [3, 5, 7, 11, 2] * (n // 5)
                x = native.net.forward(np.array([ids]), native.net.new_cache(), 0)
                y = plain.net.forward(np.array([ids]), plain.net.new_cache(), 0)
                np.testing.assert_allclose(x, y, atol=1e-4)
            self.assertEqual(native.generate("hello", max_new_tokens=8), plain.generate("hello", max_new_tokens=8))
            big = [k for k in native.net.W._f32 if k.endswith("proj.weight")]
            self.assertEqual(big, [])                                   # no float32 copies of the matrices


class ReusedCache(unittest.TestCase):
    """A prompt that starts like the last one (a conversation) reuses its cache: same tokens as from scratch."""

    def check(self, device):
        from apsi import hub_gpu
        with tempfile.TemporaryDirectory() as td:
            tiny_llama(td, "qwen2", ctx=600, exact=True)

            def go():
                m = H.Model(td, device=device)
                fresh = H.Model(td, device=device)
                rng = np.random.default_rng(2)
                a = [int(x) for x in rng.integers(2, 20, 40)]
                first = m.net.generate(a, max_new_tokens=10, eos=[])
                b = a + first + [int(x) for x in rng.integers(2, 20, 12)]
                calls = []
                real = m.net.forward
                m.net.forward = lambda ids, *k, **kw: (calls.append(np.asarray(ids).shape[1]), real(ids, *k, **kw))[1]
                again = m.net.generate(b, max_new_tokens=10, eos=[])
                m.net.forward = real
                self.assertLess(calls[0], 20)                        # only the new part of the prompt was computed
                fresh.net.release() if device != "cpu" else None
                want = H.Model(td, device=device).net.generate(b, max_new_tokens=10, eos=[])
                self.assertEqual(again, want)
                # a different prompt starts over
                self.assertEqual(m.net.generate(a[::-1], max_new_tokens=5, eos=[]),
                                 H.Model(td, device=device).net.generate(a[::-1], max_new_tokens=5, eos=[]))
            if device == "cpu":
                go()
            else:
                _on(hub_gpu.Emulator(), go)

    def test_processor(self):
        self.check("cpu")

    def test_gpu(self):
        self.check("auto")


class Chat(unittest.TestCase):
    def test_formats_and_conversation(self):
        with tempfile.TemporaryDirectory() as td:
            tiny_llama(td, "qwen2", ctx=600)
            m = H.Model(td, device="cpu")
            self.assertEqual(m.chat_format(), "none")
            msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "and?"}]
            m._fmt = "chatml"
            self.assertEqual(m.chat_prompt(msgs, system="S"),
                             "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                             "hello<|im_end|>\n<|im_start|>user\nand?<|im_end|>\n<|im_start|>assistant\n")
            m._fmt = "llama3"
            self.assertTrue(m.chat_prompt("q", system="S").endswith(
                "<|start_header_id|>user<|end_header_id|>\n\nq<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))
            m._fmt = "gemma"
            self.assertEqual(m.chat_prompt("q", system="S"),
                             "<bos><start_of_turn>user\nS\n\nq<end_of_turn>\n<start_of_turn>model\n")
            m._fmt = "mistral"
            self.assertEqual(m.chat_prompt(msgs), "<s>[INST] hi [/INST] hello</s>[INST] and? [/INST]")
            m._fmt = None
            with tempfile.TemporaryDirectory() as t2:
                with open(os.path.join(t2, "tokenizer_config.json"), "w") as fh:
                    json.dump({"chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}"}, fh)
                m.folder, old = t2, m.folder
                self.assertEqual(m.chat_format(), "chatml")
                m.folder, m._fmt = old, None
            # end to end: a reply, streamed pieces that add up to it, and turns that are remembered
            pieces = []
            r = m.chat("hello world", max_new_tokens=6, temperature=0, stream=pieces.append)
            self.assertEqual("".join(pieces).strip(), r)
            c = m.conversation(system="Be brief.", max_new_tokens=4, temperature=0)
            c.say("hello")
            c.say("world")
            self.assertEqual([x["role"] for x in c.history], ["user", "assistant", "user", "assistant"])
            c.reset()
            self.assertEqual(c.history, [])
            with self.assertRaises(H.HubError):
                m.chat([{"role": "robot", "content": "x"}])
            # a long conversation is cut to fit: oldest turns go first
            long = [{"role": "user" if i % 2 == 0 else "assistant", "content": "hello world " * 40} for i in range(41)]
            self.assertIsInstance(m.chat(long, max_new_tokens=4, temperature=0), str)


class UnigramTokenizer(unittest.TestCase):
    def test_viterbi(self):
        vocab = [["<unk>", 0.0], ["▁", -2.0], ["▁hello", -3.0], ["▁hell", -2.5], ["o", -2.0], ["▁wor", -3.0],
                 ["ld", -2.0], ["▁world", -6.5], ["h", -4.0], ["e", -4.0], ["l", -4.0]]
        t = H.Tokenizer({"model": {"type": "Unigram", "vocab": vocab, "unk_id": 0},
                         "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
                         "decoder": {"type": "Metaspace", "replacement": "▁"}})
        e = t.encode("hello world")
        self.assertEqual(e.tokens, ["▁hello", "▁wor", "ld"])           # -3 + -5 beats ▁hell+o (-4.5) and ▁world
        self.assertEqual(t.decode(e.ids), "hello world")
        self.assertEqual(t.encode("hello zz").tokens, ["▁hello", "▁", "<unk>"])  # unknown run merged


@unittest.skipUnless(os.environ.get("APSI_HF_COMPARE"), "set APSI_HF_COMPARE=1 with transformers and torch")
class AgainstTransformers(unittest.TestCase):
    def test_same_outputs(self):
        import torch
        from transformers import AutoModel
        with tempfile.TemporaryDirectory() as td:
            tiny_bert(td)
            m = H.Model(td)
            ids = m.tokenizer.encode("hello good movie !").ids
            ref = AutoModel.from_pretrained(td)(torch.tensor([ids])).last_hidden_state[0].detach().numpy()
            np.testing.assert_allclose(m.net.hidden(np.array([ids]))[0], ref, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
