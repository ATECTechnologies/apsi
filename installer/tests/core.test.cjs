// End-to-end test of the installer core on the current OS: install into a temporary HOME,
// run the installed `apsi`, then uninstall and check nothing is left behind.
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), "apsi-home-"));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.APSI_HOME = path.join(tmpHome, ".apsi");
if (process.platform === "win32") process.env.LOCALAPPDATA = path.join(tmpHome, "AppData", "Local");
const core = require("../electron/installer-core.cjs");

test("system check reads the payload and the machine", async () => {
  const sys = await core.detectSystem();
  assert.ok(sys.version);
  assert.ok(sys.requiredBytes > 10e6);
  assert.ok(sys.doctor, "apsi doctor --json from the bundled runtime");
  assert.strictEqual(sys.existing, null);
  const opts = core.defaultOptions(sys);
  assert.strictEqual(opts.mode, "install");
  assert.ok(opts.dir.startsWith(tmpHome));
});

test("install, use, upgrade and uninstall", { timeout: 300000 }, async () => {
  const sys = await core.detectSystem();
  const opts = core.defaultOptions(sys);
  opts.components.editors = false;
  opts.components.desktopShortcut = true;
  opts.security.profile = "strict";
  const progress = [];
  const res = await core.install(opts, { onProgress: (p) => progress.push(p.percent), onLog: () => {} });
  assert.ok(res.ok, res.log.join("\n"));
  assert.strictEqual(progress[progress.length - 1], 100);
  for (let i = 1; i < progress.length; i++) assert.ok(progress[i] >= progress[i - 1] - 1e-9, "progress never goes back");

  const bin = path.join(opts.dir, "bin", process.platform === "win32" ? "apsi.cmd" : "apsi");
  const v = spawnSync(bin, ["version"], { encoding: "utf8", shell: process.platform === "win32" });
  assert.match(v.stdout, /Apsi \d/);
  const cfg = JSON.parse(fs.readFileSync(path.join(tmpHome, ".apsi", "config.json"), "utf8"));
  assert.strictEqual(cfg.profile, "strict");
  if (process.platform !== "win32") {
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "applications", "apsi-repl.desktop")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "mime", "packages", "apsi.xml")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "icons", "hicolor", "256x256", "apps", "apsi.png")));
  }
  const again = await core.detectSystem();
  assert.ok(again.existing && again.existing.version === sys.version, "existing install detected");
  const opts2 = core.defaultOptions(again);
  assert.strictEqual(opts2.mode, "upgrade");
  opts2.components.examples = false;
  const res2 = await core.install(opts2, {});
  assert.ok(res2.ok, res2.log.join("\n"));
  assert.ok(!fs.existsSync(path.join(opts.dir, "examples")), "deselected component removed on modify");

  await core.uninstall(opts.dir);
  assert.ok(!fs.existsSync(opts.dir));
  if (process.platform !== "win32") {
    assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
    assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "share", "applications", "apsi-repl.desktop")));
    const prof = path.join(tmpHome, ".profile");
    if (fs.existsSync(prof)) assert.ok(!fs.readFileSync(prof, "utf8").includes(">>> apsi >>>"));
  }
});

test("a failed install rolls back completely", async () => {
  const sys = await core.detectSystem();
  const opts = core.defaultOptions(sys);
  const blocker = path.join(tmpHome, "not-a-dir");
  fs.writeFileSync(blocker, "x");
  opts.dir = path.join(blocker, "Apsi");          // cannot create a folder inside a file
  const res = await core.install(opts, {});
  assert.strictEqual(res.ok, false);
  assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
});

test("generated uninstall scripts only touch recorded paths", () => {
  const rec = { dir: "/opt/x", files: ["/a b/c'd"], dirs: [], registryKeys: ["HKCU\\Software\\Apsi"], profileFiles: [], pathEntry: "C:\\A" };
  const sh = core._internal.linuxUninstallScript(rec);
  assert.ok(sh.includes("rm -f '/a b/c'\\''d'"));
  const ps = core._internal.winUninstallScript(rec);
  assert.ok(ps.includes("Registry::HKCU\\Software\\Apsi"));
});

test("a tampered or extended payload is refused before anything is copied", async () => {
  const real = core.payloadDir();
  const fake = fs.mkdtempSync(path.join(os.tmpdir(), "apsi-payload-"));
  // small synthetic payload with the same format
  const crypto = require("crypto");
  fs.mkdirSync(path.join(fake, "runtime"));
  fs.writeFileSync(path.join(fake, "runtime", "a.txt"), "hello");
  const files = { "runtime/a.txt": crypto.createHash("sha256").update("hello").digest("hex") };
  const blob = JSON.stringify({ algorithm: "sha256", files });
  fs.writeFileSync(path.join(fake, "payload-files.json"), blob);
  const info = JSON.parse(fs.readFileSync(path.join(real, "payload.json"), "utf8"));
  info.sha256 = crypto.createHash("sha256").update(blob).digest("hex");
  fs.writeFileSync(path.join(fake, "payload.json"), JSON.stringify(info));
  assert.strictEqual(await core.verifyPayload(fake), 1);
  fs.writeFileSync(path.join(fake, "runtime", "a.txt"), "evil");
  await assert.rejects(core.verifyPayload(fake), /modified/);
  fs.writeFileSync(path.join(fake, "runtime", "a.txt"), "hello");
  fs.writeFileSync(path.join(fake, "runtime", "planted.dll"), "MZ");
  await assert.rejects(core.verifyPayload(fake), /Unexpected file/);
  fs.rmSync(path.join(fake, "runtime", "planted.dll"));
  fs.writeFileSync(path.join(fake, "payload-files.json"), blob.replace("\"sha256\"", "\"sha512\""));
  await assert.rejects(core.verifyPayload(fake), /does not match/);
});

test("install folder validation", () => {
  assert.ok(!core.validateDir("").ok);
  assert.ok(!core.validateDir(path.parse(process.cwd()).root).ok);
  assert.ok(!core.validateDir(os.homedir()).ok);
  if (process.platform !== "win32") assert.ok(!core.validateDir("/usr/local/apsi").ok);
  const nonEmpty = fs.mkdtempSync(path.join(os.tmpdir(), "apsi-ne-"));
  fs.writeFileSync(path.join(nonEmpty, "x.txt"), "x");
  assert.ok(!core.validateDir(nonEmpty).ok);
  assert.ok(core.validateDir(path.join(os.tmpdir(), "apsi-new-" + Date.now())).ok);
  assert.ok(core.validateDir(path.join(os.tmpdir(), "OneDrive", "Apsi")).warnings.length >= 1);
});
