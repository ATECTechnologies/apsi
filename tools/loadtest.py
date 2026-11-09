"""Minimal HTTP/1.1 keep-alive load generator (no dependencies).

    python tools/loadtest.py http://127.0.0.1:8080/health --requests 20000 --concurrency 500
"""
import argparse
import asyncio
import time
from urllib.parse import urlparse


async def worker(host, port, path, n, lat, errors, body, method):
    reader, writer = await asyncio.open_connection(host, port)
    payload = body.encode() if body else b""
    req = (f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Length: {len(payload)}\r\n"
           f"Content-Type: application/json\r\n\r\n").encode() + payload
    for _ in range(n):
        t0 = time.perf_counter()
        writer.write(req)
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        if length:
            await reader.readexactly(length)
        lat.append(time.perf_counter() - t0)
        if status >= 400:
            errors.append(status)
    writer.close()


async def main(url, total, conc, body, method):
    u = urlparse(url)
    per = max(1, total // conc)
    lat, errors = [], []
    t0 = time.perf_counter()
    await asyncio.gather(*[worker(u.hostname, u.port or 80, u.path or "/", per, lat, errors, body, method)
                           for _ in range(conc)])
    dt = time.perf_counter() - t0
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] * 1000
    print(f"{len(lat)} requests, {conc} concurrent connections, {dt:.2f}s")
    print(f"  throughput : {len(lat) / dt:,.0f} req/s")
    print(f"  latency    : p50 {pct(.5):.1f} ms  p90 {pct(.9):.1f} ms  p99 {pct(.99):.1f} ms")
    print(f"  errors     : {len(errors)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--requests", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--body", default="")
    ap.add_argument("--method", default="GET")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.requests, a.concurrency, a.body, a.method))
