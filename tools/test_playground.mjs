// Usage: npm install pyodide@0.27.7 && node tools/test_playground.mjs playground/apsi_bundle.js
// (set NUMPY_WHEEL=/path/to/numpy-...-wasm32.whl when offline)
// Runs the playground exactly like the browser page does, but in Node.
import { loadPyodide } from "pyodide";
import fs from "fs";
import vm from "vm";
const bundleSrc = fs.readFileSync(process.argv[2], "utf8");
const ctx = { window: {} };
vm.runInNewContext(bundleSrc, ctx);
const B = ctx.window.APSI_BUNDLE;
const py = await loadPyodide();
await py.loadPackage(process.env.NUMPY_WHEEL || "numpy");
py.FS.mkdirTree("/apsi_pkg/apsi");
for (const [p, src] of Object.entries(B.files)) py.FS.writeFile("/apsi_pkg/" + p, src);
py.runPython("import sys; sys.path.insert(0, '/apsi_pkg'); import apsi.playground");
const run = py.globals.get("apsi").playground.run;
console.log("Apsi", py.runPython("import apsi; apsi.__version__"), "on", py.runPython("import sys; sys.version.split()[0]"), "(wasm)");
let pass = 0, fail = 0;
for (const [name, src] of Object.entries(B.examples)) {
  const t0 = Date.now();
  const r = run(src).toJs({ dict_converter: Object.fromEntries });
  const ok = r.ok && (!r.tests || /\b0 failed/.test(r.tests));
  ok ? pass++ : fail++;
  console.log(`${ok ? "✓" : "✗"} ${name} (${Date.now() - t0} ms)` + (ok ? "" : "\n" + (r.error || r.tests)));
}
const bad = run("let x = 1m + 2s").toJs({ dict_converter: Object.fromEntries });
console.log("checker error shown:", /UnitError/.test(bad.error));
const sb = run('read_text("x")').toJs({ dict_converter: Object.fromEntries });
console.log("sandbox enforced:", /PermissionError|permission/.test(sb.error), "-", (sb.error || "").split("\n")[0]);
console.log(`${pass} examples passed, ${fail} failed`);
