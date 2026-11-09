import time, re, json
t = time.perf_counter()
text = "\n".join(f"user{i}@example.com visited /page/{i % 50} at 2026-11-{i % 28 + 1:02d}" for i in range(100000))
emails = re.findall(r"[\w.]+@[\w.]+", text)
pages = {}
for line in text.splitlines():
    p = line.split()[2]
    pages[p] = pages.get(p, 0) + 1
blob = json.dumps([{"i": i, "s": str(i) * 3} for i in range(50000)])
back = json.loads(blob)
out = "".join(sorted(set(w[::-1].upper() for w in text.split()[:50000])))[:20]
print("strs", len(emails), len(pages), len(back), out, f"{time.perf_counter() - t:.3f}s")
