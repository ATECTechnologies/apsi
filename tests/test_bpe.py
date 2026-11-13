"""data.train_tokenizer / bpe.train: byte-level BPE trained in Apsi, exact round trips, the tokenizer.json format."""
import glob
import json
import os
import tempfile
import unittest

from apsi import bpe
from apsi.hub import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corpus():
    return [open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))]


class BPE(unittest.TestCase):
    def test_round_trip_and_compression(self):
        tok = bpe.train(corpus(), vocab_size=2000, engine="apsi")
        self.assertEqual(tok.vocab_size, 2000)
        for s in ("Hello wörld 👋 — tabs\tand\nnew lines", "def f(x):\n    return x ** 2\n", "", "   ", "日本語のテキスト"):
            self.assertEqual(tok.decode(tok.encode(s)), s)
        text = corpus()[0]
        self.assertLess(len(tok.encode(text)), len(text.encode("utf-8")) / 2.5)

    def test_deterministic_and_saved(self):
        a = bpe.train(corpus(), vocab_size=1200, engine="apsi")
        b = bpe.train(corpus(), vocab_size=1200, engine="apsi")
        self.assertEqual(a.spec, b.spec)
        path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
        a.save(path)
        c = Tokenizer.from_file(path)
        s = "Training a tokenizer in Apsi."
        self.assertEqual(c.encode(s, add_special_tokens=False).ids, a.encode(s))
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["model"]["type"], "BPE")

    def test_the_merges_agree_with_huggingface(self):
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            self.skipTest("Hugging Face tokenizers is not installed")
        mine = bpe.train(corpus(), vocab_size=3000, engine="apsi")
        theirs = bpe.train(corpus(), vocab_size=3000, engine="rust")
        a, b = set(mine.spec["model"]["merges"]), set(
            m if isinstance(m, str) else " ".join(m) for m in theirs.spec["model"]["merges"])
        self.assertGreater(len(a & b) / len(b), 0.95)          # ties may be broken differently, nothing else
        text = corpus()[1]
        self.assertLess(abs(len(mine.encode(text)) - len(theirs.encode(text))) / len(theirs.encode(text)), 0.02)

    def test_text_dataset_takes_it(self):
        from apsi.textdata import text_dataset
        tok = bpe.train(corpus(), vocab_size=800, engine="apsi")
        ds = text_dataset(corpus()[:2], tokenizer=tok, seq_len=32)
        x, y = next(iter(ds))
        self.assertEqual(len(x), 32)
        self.assertLess(max(int(v) for v in x), tok.vocab_size)

    def test_in_apsi(self):
        from apsi.interpreter import run_source
        env = run_source('use data\nlet tok = data.train_tokenizer(["one two three two one"] * 50, vocab_size=270, '
                         'engine="apsi")\nlet back = tok.decode(tok.encode("two one"))\n')
        self.assertEqual(env.lookup("back"), "two one")


if __name__ == "__main__":
    unittest.main()
