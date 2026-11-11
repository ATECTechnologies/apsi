// A small payload in the real format for tests on machines without the built one (no network for pip):
// runtime/python/bin/python3 runs the system Python with this checkout on the path.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");

const REPO = path.resolve(__dirname, "..", "..");

function make(version) {
  const res = fs.mkdtempSync(path.join(os.tmpdir(), "apsi-res-"));
  const out = path.join(res, "payload");
  const bin = path.join(out, "runtime", "python", "bin");
  fs.mkdirSync(bin, { recursive: true });
  const py = fs.existsSync("/usr/bin/python3") ? "/usr/bin/python3" : "python3";
  fs.writeFileSync(path.join(bin, "python3"),
    `#!/bin/sh\nexport PYTHONPATH='${REPO}'\n` +
    `first=1\nfor a in "$@"; do if [ $first = 1 ]; then set --; first=0; fi; case "$a" in -I|-P) ;; *) set -- "$@" "$a";; esac; done\n` +
    `exec ${py} "$@"\n`, { mode: 0o755 });
  for (const d of ["examples", "docs", "brand", "vscode", "editors"]) fs.mkdirSync(path.join(out, d), { recursive: true });
  fs.writeFileSync(path.join(out, "examples", "01_hello.apsi"), fs.readFileSync(path.join(REPO, "examples", "01_hello.apsi")));
  fs.writeFileSync(path.join(out, "docs", "README.md"), "# Apsi\n");
  fs.copyFileSync(path.join(REPO, "README.md"), path.join(out, "README.md"));
  fs.cpSync(path.join(REPO, "assets", "brand"), path.join(out, "brand"), { recursive: true,
    filter: (src) => !/logo-source\.png$|\.bmp$|banner/.test(path.basename(src)) });
  const files = {};
  const walk = (d, rel) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(path.join(d, e.name), r);
      else files[r] = crypto.createHash("sha256").update(fs.readFileSync(path.join(d, e.name))).digest("hex");
    }
  };
  walk(out, "");
  const blob = JSON.stringify({ algorithm: "sha256", files });
  fs.writeFileSync(path.join(out, "payload-files.json"), blob);
  const size = (d) => { let t = 0; for (const e of fs.readdirSync(d, { withFileTypes: true })) t += e.isDirectory() ? size(path.join(d, e.name)) : fs.statSync(path.join(d, e.name)).size; return t; };
  const info = { name: "Apsi", version, platform: process.platform === "win32" ? "windows" : "linux", python: "3.12.11",
    components: { runtime: Math.max(size(path.join(out, "runtime")), 20e6), examples: size(path.join(out, "examples")), docs: size(path.join(out, "docs")), editors: 1 },
    files: Object.keys(files).length, sha256: crypto.createHash("sha256").update(blob).digest("hex") };
  fs.writeFileSync(path.join(out, "payload.json"), JSON.stringify(info, null, 2));
  return res;
}

module.exports = { make };
