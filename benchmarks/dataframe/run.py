import sys, time, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from apsi import dataframe as D
rng = np.random.default_rng(0)
N = 2_000_000
cust = np.array([f"c{i}" for i in range(50_000)], dtype=object)
data = {"id": np.arange(N), "customer": cust[rng.integers(0, 50_000, N)], "region": np.array(["EU", "US", "ASIA", "LATAM"], dtype=object)[rng.integers(0, 4, N)],
        "amount": rng.gamma(2.0, 50.0, N), "qty": rng.integers(1, 10, N)}
info = pd.DataFrame({"customer": cust, "tier": np.array(["gold", "silver", "bronze"], dtype=object)[rng.integers(0, 3, 50_000)]})
def t(f, n=3):
    best = 1e9
    for _ in range(n):
        s = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - s)
    return best, r
pdf = pd.DataFrame(data)
s = time.perf_counter(); adf = D.DataFrame({k: (list(v) if v.dtype == object else v) for k, v in data.items()}); build = time.perf_counter() - s
ainfo = D.DataFrame({"customer": list(info.customer), "tier": list(info.tier)})
print(f"build (from Python lists of strings): {build:.2f}s; memory apsi {adf.memory_bytes()/1e6:.0f} MB vs pandas {pdf.memory_usage(deep=True).sum()/1e6:.0f} MB")
col = D.col
cases = [
 ("filter", lambda: pdf[(pdf.amount > 100) & (pdf.region == "EU")], lambda: adf.filter((col("amount") > 100) & (col("region") == "EU"))),
 ("groupby 50k keys sum/mean/count", lambda: pdf.groupby("customer", sort=False).agg(total=("amount", "sum"), avg=("amount", "mean"), n=("qty", "count")),
   lambda: adf.group_by("customer").agg(total=col("amount").sum(), avg=col("amount").mean(), n=col("qty").count())),
 ("groupby 2 keys std/max", lambda: pdf.groupby(["region", "qty"], sort=False).agg(s=("amount", "std"), m=("amount", "max")),
   lambda: adf.group_by("region", "qty").agg(s=col("amount").std(), m=col("amount").max())),
 ("join 2M x 50k (left)", lambda: pdf.merge(info, on="customer", how="left"), lambda: adf.join(ainfo, on="customer", how="left")),
 ("sort by float", lambda: pdf.sort_values("amount"), lambda: adf.sort("amount")),
 ("with_columns arithmetic", lambda: pdf.assign(total=pdf.amount * pdf.qty + 1), lambda: adf.with_columns(total=col("amount") * col("qty") + 1)),
]
for name, fp, fa in cases:
    tp, rp = t(fp); ta, ra = t(fa)
    print(f"{name:34s} pandas {tp*1000:8.1f} ms   apsi {ta*1000:8.1f} ms   x{tp/ta:5.2f}  rows {len(rp)} / {len(ra)}")
# correctness
g = adf.group_by("customer").agg(total=col("amount").sum(), n=col("qty").count()).sort("customer")
gp = pdf.groupby("customer").agg(total=("amount", "sum"), n=("qty", "count")).reset_index().sort_values("customer")
print("groupby equal:", np.allclose(g["total"], gp.total.values), (g["n"] == gp.n.values).all(), list(g["customer"][:3]) == list(gp.customer[:3]))
j = adf.join(ainfo, on="customer", how="left")
jp = pdf.merge(info, on="customer", how="left")
print("join equal:", (np.array(j["tier"].tolist()) == jp.tier.values).all(), (j["id"] == jp.id.values).all())
