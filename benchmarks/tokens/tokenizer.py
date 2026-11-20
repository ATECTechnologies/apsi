"""data.train_tokenizer: time to train a byte-level BPE tokenizer in Apsi and in Hugging Face's Rust trainer, the share
of merges they agree on, and how many tokens each needs for a held-out text.   python3 benchmarks/tokens/tokenizer.py"""
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from apsi import bpe      # noqa: E402

files = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")) + glob.glob(os.path.join(ROOT, "apsi", "*.py")))
held = os.path.join(ROOT, "README.md")
texts = [open(f, encoding="utf-8").read() for f in files]
mb = sum(len(t.encode()) for t in texts) / 1e6
test = open(held, encoding="utf-8").read()
print(f"corpus: {len(files)} files, {mb:.1f} MB; held out: README.md ({len(test.encode()):,} bytes)")
res = {}
for engine in ("apsi", "rust"):
    try:
        t0 = time.perf_counter()
        tok = bpe.train(texts, vocab_size=8000, engine=engine)
        s = time.perf_counter() - t0
    except ImportError:
        print(f"{engine}: not installed (pip install tokenizers)")
        continue
    n = len(tok.encode(test))
    assert tok.decode(tok.encode(test)) == test
    res[engine] = tok
    print(f"{engine:5} | 8,000 tokens | {s:.2f} s | README.md: {n:,} tokens ({len(test.encode()) / n:.2f} bytes a token) | round trip exact")
if len(res) == 2:
    a = set(res["apsi"].spec["model"]["merges"])
    b = {m if isinstance(m, str) else " ".join(m) for m in res["rust"].spec["model"]["merges"]}
    print(f"merges in common: {len(a & b):,} of {len(b):,} ({100 * len(a & b) / len(b):.1f} %)")
