"""tokens.focus on real files: tokens before and after, and whether unfold() gives the file back exactly.
    python3 benchmarks/tokens/focus.py"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from apsi import token_focus as F, tokens as T     # noqa: E402

CASES = [
    ("apsi/lexer.py", "how are string escapes handled?", 1500),
    ("apsi/gla.py", "how is the router balance loss computed?", 1500),
    ("apsi/gpu_train.py", "how does a checkpoint resume training?", 3000),
    ("docs/GPU.md", "the card is hot or loud: how do I lower its temperature?", 600),
    ("docs/TRAINING.md", "what happens if I turn the NaN guard off?", 500),
]
print(f"token counts: {'exact (tiktoken o200k)' if T.exact() else 'estimated'}")
for path, q, budget in CASES:
    text = open(os.path.join(ROOT, path), encoding="utf-8").read()
    t0 = time.perf_counter()
    f = F.focus(text, q, budget=budget)
    ms = (time.perf_counter() - t0) * 1000
    ok = F.unfold(f) == text
    print(f"{path} | {q} | {f.saved.before} | {f.tokens} | {100 * (1 - f.tokens / f.saved.before):.1f} % | "
          f"{len(f.kept)} whole, {len(f.outlined)} outlined of {len(f.units)} | {ms:.0f} ms | unfold exact: {ok}")
