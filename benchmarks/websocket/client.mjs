const url = process.env.APSI_WS_URL || "ws://127.0.0.1:58080";
const open = (p) => new Promise((res, rej) => { const w = new WebSocket(url + p); w.binaryType = "arraybuffer"; w.q = []; w.wait = []; w.onmessage = (e) => { const f = w.wait.shift(); f ? f(e.data) : w.q.push(e.data); }; w.onopen = () => res(w); w.onerror = rej; });
const next = (w) => w.q.length ? Promise.resolve(w.q.shift()) : new Promise((res) => w.wait.push(res));
const t0 = performance.now();
const e = await open("/echo");
e.send("héllo wörld 👋 日本"); console.error("echo text:", await next(e));
e.send(new Uint8Array([0, 1, 255]).buffer); console.error("echo binary:", Array.from(new Uint8Array(await next(e))));
const big = "x".repeat(1 << 20); e.send(big); console.error("echo 1MB:", (await next(e)).length === big.length);
// throughput: N round trips sequentially, then a burst
let N = 5000, t = performance.now();
for (let i = 0; i < N; i++) { e.send("m" + i); await next(e); }
const rt = (performance.now() - t) / N;
console.error(`round trip: ${(rt * 1000).toFixed(0)} us avg over ${N}`);
let got = 0; t = performance.now();
const done = new Promise((res) => { e.onmessage = () => { if (++got === 20000) res(); }; });
for (let i = 0; i < 20000; i++) e.send("b" + i);
await done;
console.error(`burst: 20000 echoed in ${(performance.now() - t).toFixed(0)} ms = ${(20000 / ((performance.now() - t) / 1000)).toFixed(0)} msg/s`);
// rooms
const a = await open("/chat/r1"), b = await open("/chat/r1");
console.error("hello a:", JSON.parse(await next(a)).room, "hello b:", JSON.parse(await next(b)).room);
const pb = next(b), pa = next(a);
a.send("from a");
console.error("b got:", JSON.parse(await pb).text, "| a got:", JSON.parse(await pa).text);
a.close(); b.close(); e.close();
// many concurrent connections
const conns = await Promise.all(Array.from({ length: 1000 }, () => open("/echo")));
const replies = await Promise.all(conns.map((w, i) => { const p = next(w); w.send("c" + i); return p; }));
console.error("1000 concurrent connections all answered:", replies.every((r, i) => r === "c" + i));
conns.forEach((w) => w.close());
console.error("total", (performance.now() - t0).toFixed(0), "ms");
process.exit(0);
