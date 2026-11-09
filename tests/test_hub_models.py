"""Native Hugging Face model runners: tiny random checkpoints in the real file layout (config.json,
model.safetensors with Hugging Face tensor names, tokenizer.json), checked against straightforward per-head
reference code, padding invariance, and key/value-cache generation against full recomputation. Set
APSI_HF_COMPARE=1 with transformers + torch installed to also compare against the reference library."""
import json
import math
import os
import struct
import tempfile
import unittest

import numpy as np

from apsi import hub as H
from apsi import hub_models as M


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32)
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
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
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"))
    tok = {"model": {"type": "WordPiece", "vocab": {t: i for i, t in enumerate(vocab)}, "unk_token": "[UNK]",
                     "continuing_subword_prefix": "##"},
           "normalizer": {"type": "BertNormalizer", "lowercase": True},
           "pre_tokenizer": {"type": "BertPreTokenizer"},
           "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "[CLS]"}},
                              {"Sequence": {"id": "A"}}, {"SpecialToken": {"id": "[SEP]"}}], "pair": []},
           "added_tokens": [{"id": i, "content": t, "special": True} for i, t in enumerate(vocab[:5])]}
    json.dump(tok, open(os.path.join(folder, "tokenizer.json"), "w"))
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


def tiny_llama(folder, kind="llama"):
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
        if kind == "qwen2":
            for n in ("q", "k", "v"):
                W[p + f"self_attn.{n}_proj.bias"] = rnd(rng, W[p + f"self_attn.{n}_proj.weight"].shape[0], s=0.1)
        W[p + "mlp.gate_proj.weight"] = rnd(rng, I, Hd)
        W[p + "mlp.up_proj.weight"] = rnd(rng, I, Hd)
        W[p + "mlp.down_proj.weight"] = rnd(rng, Hd, I)
    write_safetensors(os.path.join(folder, "model.safetensors"), W)
    cfg = {"model_type": kind, "hidden_size": Hd, "num_attention_heads": nh, "num_key_value_heads": kvh,
           "num_hidden_layers": L, "intermediate_size": I, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
           "max_position_embeddings": 64, "eos_token_id": 1, "tie_word_embeddings": True, "hidden_act": "silu"}
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"))
    merges = ["h e", "l l", "he ll", "hell o", f"{G} w", "o r", f"{G}w or", "l d", f"{G}wor ld"]
    tok = {"model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(vocab)}, "merges": merges},
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
           "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>"}},
                              {"Sequence": {"id": "A"}}], "pair": []},
           "added_tokens": [{"id": 0, "content": "<s>", "special": True}, {"id": 1, "content": "</s>", "special": True}]}
    json.dump(tok, open(os.path.join(folder, "tokenizer.json"), "w"))
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
            m.generate("hello", max_new_tokens=3, temperature=0.8, top_k=5, top_p=0.9, seed=1, stream=pieces.append)
            # RoPE: a rotation, so it keeps vector lengths
            x = np.random.default_rng(3).standard_normal((1, 2, 5, m.net.dh)).astype(np.float32)
            r = m.net._rope(x, np.arange(5))
            np.testing.assert_allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)

    def test_llama(self):
        self.check_decoder("llama")

    def test_qwen2_biases(self):
        self.check_decoder("qwen2")


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
