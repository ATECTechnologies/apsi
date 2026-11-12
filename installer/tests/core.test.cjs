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
const fake = require("./fake-payload.cjs");

// the built payload when there is one, otherwise a small one in the same format
let RES;
try { core.payloadDir(); RES = undefined; } catch { RES = fake.make("2.4.4"); }
const hooksFor = (res, more = {}) => ({ resourcesPath: res, onLog: () => {}, ...more });
const bin = (dir) => path.join(dir, "bin", process.platform === "win32" ? "apsi.cmd" : "apsi");
const run = (exe, args, env = {}) => spawnSync(exe, args, { encoding: "utf8", shell: process.platform === "win32", env: { ...process.env, ...env } });

test("system check reads the payload and the machine", async () => {
  const sys = await core.detectSystem(RES);
  assert.ok(sys.version);
  assert.ok(sys.requiredBytes > 10e6);
  assert.ok(sys.doctor, "apsi doctor --json from the bundled runtime");
  assert.strictEqual(sys.existing, null);
  const opts = core.defaultOptions(sys);
  assert.strictEqual(opts.mode, "install");
  assert.ok(opts.dir.startsWith(tmpHome));
});

test("install, use, upgrade and uninstall", { timeout: 300000 }, async () => {
  const sys = await core.detectSystem(RES);
  const opts = core.defaultOptions(sys);
  opts.components.editors = false;
  opts.components.desktopShortcut = true;
  opts.security.profile = "strict";
  const progress = [];
  const res = await core.install(opts, hooksFor(RES, { onProgress: (p) => progress.push(p.percent) }));
  assert.ok(res.ok, res.log.join("\n"));
  assert.strictEqual(progress[progress.length - 1], 100);
  for (let i = 1; i < progress.length; i++) assert.ok(progress[i] >= progress[i - 1] - 1e-9, "progress never goes back");

  const vdir = path.join(opts.dir, sys.version);           // the root holds one folder per version
  assert.ok(fs.existsSync(path.join(vdir, "install.json")));
  const v = run(bin(vdir), ["version"]);
  assert.match(v.stdout, /Apsi \d/);
  assert.match(run(bin(opts.dir), ["version"]).stdout, /Apsi \d/, "shared launcher");
  assert.strictEqual(fs.readFileSync(path.join(opts.dir, "current.txt"), "utf8").trim(), sys.version);
  const cfg = JSON.parse(fs.readFileSync(path.join(tmpHome, ".apsi", "config.json"), "utf8"));
  assert.strictEqual(cfg.profile, "strict");
  if (process.platform !== "win32") {
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "applications", "apsi-repl.desktop")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "mime", "packages", "apsi.xml")));
    assert.ok(fs.existsSync(path.join(tmpHome, ".local", "share", "icons", "hicolor", "256x256", "apps", "apsi.png")));
  }
  const again = await core.detectSystem(RES);
  assert.ok(again.existing && again.existing.version === sys.version, "existing install detected");
  const opts2 = core.defaultOptions(again);
  assert.strictEqual(opts2.mode, "upgrade");
  opts2.components.examples = false;
  const res2 = await core.install(opts2, hooksFor(RES));
  assert.ok(res2.ok, res2.log.join("\n"));
  assert.ok(!fs.existsSync(path.join(vdir, "examples")), "deselected component removed on modify");

  await core.uninstall(vdir);
  assert.ok(!fs.existsSync(opts.dir));
  if (process.platform !== "win32") {
    assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
    assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "share", "applications", "apsi-repl.desktop")));
    const prof = path.join(tmpHome, ".profile");
    if (fs.existsSync(prof)) assert.ok(!fs.readFileSync(prof, "utf8").includes(">>> apsi >>>"));
  }
});

test("two versions side by side: no uninstall, each kept in its own folder", { timeout: 300000 }, async () => {
  if (process.platform === "win32") return;               // the Windows run is covered by the Windows build
  const resNew = fake.make("2.5.0");
  const sys = await core.detectSystem(RES);
  const opts = core.defaultOptions(sys);
  opts.components.editors = false;
  assert.ok((await core.install(opts, hooksFor(RES))).ok);
  const root = opts.dir;

  const sys2 = await core.detectSystem(resNew);
  assert.strictEqual(sys2.version, "2.5.0");
  assert.strictEqual(sys2.existing, null, "a new version is a new install, not a change of the old one");
  assert.deepStrictEqual(sys2.installed.versions, [sys.version], "the older version is found");
  assert.strictEqual(path.resolve(sys2.installed.root), path.resolve(root), "same Apsi folder offered");
  const opts2 = core.defaultOptions(sys2);
  assert.strictEqual(opts2.mode, "install");
  assert.strictEqual(path.resolve(opts2.dir), path.resolve(root));
  opts2.components.editors = false;
  const r2 = await core.install(opts2, hooksFor(resNew));
  assert.ok(r2.ok, r2.log && r2.log.join("\n"));
  for (const v of [sys.version, "2.5.0"]) assert.ok(fs.existsSync(path.join(root, v, "install.json")), v);
  assert.deepStrictEqual(core.installedVersions(root).map((x) => x.version).sort(core.compareVersions), [sys.version, "2.5.0"]);
  assert.strictEqual(fs.readFileSync(path.join(root, "current.txt"), "utf8").trim(), "2.5.0");
  assert.ok(fs.existsSync(path.join(root, "bin", `apsi-${sys.version}`)));
  assert.ok(fs.existsSync(path.join(root, "bin", "apsi-2.5.0")));
  // the shared launcher runs the current version; each per-version launcher tells it which one it is
  const pv = run(path.join(root, "bin", `apsi-${sys.version}`), ["version"]);
  assert.match(pv.stdout, /Apsi \d/);

  // removing the newer one switches back to the older one; removing the last removes the root
  const left = await core.uninstall(path.join(root, "2.5.0"));
  assert.deepStrictEqual(left.left, [sys.version]);
  assert.ok(!fs.existsSync(path.join(root, "2.5.0")));
  assert.strictEqual(fs.readFileSync(path.join(root, "current.txt"), "utf8").trim(), sys.version);
  assert.match(run(bin(root), ["version"]).stdout, /Apsi \d/);
  assert.ok(fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
  await core.uninstall(path.join(root, sys.version));
  assert.ok(!fs.existsSync(root), "root removed with the last version");
  assert.ok(!fs.existsSync(path.join(tmpHome, ".local", "bin", "apsi")));
});

test("a failed install rolls back completely", async () => {
  const sys = await core.detectSystem(RES);
  const opts = core.defaultOptions(sys);
  const blocker = path.join(tmpHome, "not-a-dir");
  fs.writeFileSync(blocker, "x");
  opts.dir = path.join(blocker, "Apsi");          // cannot create a folder inside a file
  const res = await core.install(opts, hooksFor(RES));
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
  const real = core.payloadDir(RES);
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
