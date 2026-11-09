import sys, time, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from apsi import dataframe as D
rng = np.random.default_rng(0); N = 1_000_000
pd.DataFrame({"id": np.arange(N), "customer": np.array([f"c{i}" for i in range(50_000)], dtype=object)[rng.integers(0, 50_000, N)], "amount": rng.gamma(2.0, 50.0, N).round(2), "qty": rng.integers(1, 10, N)}).to_csv("/tmp/apsi_big.csv", index=False)
s = time.perf_counter(); p = pd.read_csv("/tmp/apsi_big.csv"); tp = time.perf_counter() - s
s = time.perf_counter(); a = D.read_csv("/tmp/apsi_big.csv"); ta = time.perf_counter() - s
print(f"read_csv 1M rows: pandas {tp:.2f}s apsi {ta:.2f}s", a.dtypes, np.allclose(a["amount"], p.amount.values))
