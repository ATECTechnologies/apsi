// Apsi installer core — platform logic with no Electron dependency (unit-testable with node --test).
// Every change made to the system is recorded in install.json so that uninstall and
// rollback undo exactly what was done, nothing more.
"use strict";

const fs = require("fs");
const fsp = require("fs/promises");
const os = require("os");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const IS_WIN = process.platform === "win32";
const HOME = os.homedir();

// ------------------------------------------------------------------ helpers
function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    let out = "";
    let err = "";
    let child;
    try {
      child = spawn(cmd, args, { windowsHide: true, ...opts });
    } catch (e) {
      resolve({ code: -1, out: "", err: String(e) });
      return;
    }
    child.stdout && child.stdout.on("data", (d) => { out += d; if (opts.onLine) opts.onLine(String(d)); });
    child.stderr && child.stderr.on("data", (d) => { err += d; if (opts.onLine) opts.onLine(String(d)); });
    let timer = null;
    if (opts.timeoutMs) {
      timer = setTimeout(() => {
        try { child.kill(); } catch { /* already gone */ }
        resolve({ code: -2, out, err: err + `timed out after ${opts.timeoutMs} ms` });
      }, opts.timeoutMs);
    }
    child.on("error", (e) => { clearTimeout(timer); resolve({ code: -1, out, err: err + String(e) }); });
    child.on("close", (code) => { clearTimeout(timer); resolve({ code, out, err }); });
  });
}

// Run a .cmd/.bat on Windows safely even when paths contain spaces.
function runScript(file, args, opts = {}) {
  if (!IS_WIN) return run(file, args, opts);
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const line = [q(file), ...args.map(q)].join(" ");
  return run(process.env.ComSpec || "cmd.exe", ["/d", "/s", "/c", `"${line}"`], { ...opts, windowsVerbatimArguments: true });
}

function which(name) {
  const r = spawnSync(IS_WIN ? "where" : "which", [name], { encoding: "utf8", windowsHide: true });
  if (r.status === 0) return r.stdout.split(/\r?\n/).filter(Boolean)[0];
  return null;
}

function payloadDir(resourcesPath) {
  const candidates = [
    resourcesPath && path.join(resourcesPath, "payload"),
    process.execPath && path.join(path.dirname(process.execPath), "resources", "payload"),
    path.join(__dirname, "..", "..", "payload"),   // next to app.asar
    path.join(__dirname, "..", "payload"),         // development checkout
  ].filter(Boolean);
  for (const c of candidates) if (fs.existsSync(path.join(c, "payload.json"))) return c;
  if (resourcesPath && /[\\/]temp[\\/]/i.test(resourcesPath)) {
    throw new Error("Setup's files were removed while it was starting (usually because Setup was opened twice, or an " +
      "antivirus removed them). Close every Apsi Setup window, wait a few seconds and open ApsiSetup once.");
  }
  throw new Error("Installer payload not found. Build it with: npm run payload:win (or payload:linux)");
}

// ------------------------------------------------------------------ payload integrity
// payload.json pins the sha256 of payload-files.json, which lists the sha256 of every file.
// Every file is re-hashed before anything is copied; extra, missing or changed files abort Setup.
async function verifyPayload(payload, onFrac = () => {}) {
  const crypto = require("crypto");
  const info = readPayloadInfo(payload);
  const listPath = path.join(payload, "payload-files.json");
  if (!fs.existsSync(listPath)) throw new Error("Payload file list is missing — the installer is damaged.");
  const blob = await fsp.readFile(listPath);
  const listHash = crypto.createHash("sha256").update(blob).digest("hex");
  if (info.sha256 && listHash !== info.sha256) throw new Error("Payload file list does not match payload.json — the installer was modified.");
  const { files } = JSON.parse(blob.toString("utf8"));
  const names = Object.keys(files);
  let i = 0;
  for (const rel of names) {
    if (rel.includes("..") || path.isAbsolute(rel)) throw new Error(`Unsafe path in payload: ${rel}`);
    const full = path.join(payload, ...rel.split("/"));
    const h = crypto.createHash("sha256");
    await new Promise((res, rej) => fs.createReadStream(full).on("data", (d) => h.update(d)).on("end", res).on("error", rej))
      .catch(() => { throw new Error(`Payload file missing: ${rel}`); });
    if (h.digest("hex") !== files[rel]) throw new Error(`Payload file was modified: ${rel}`);
    i += 1;
    if (i % 100 === 0) onFrac(i / names.length);
  }
  // reject files that are not on the list (e.g. a planted DLL next to python.exe)
  const known = new Set(names.concat(["payload.json", "payload-files.json"]));
  const walk = async (d, rel = "") => {
    for (const e of await fsp.readdir(d, { withFileTypes: true })) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) await walk(path.join(d, e.name), r);
      else if (!e.isSymbolicLink() && !known.has(r)) throw new Error(`Unexpected file in payload: ${r}`);
    }
  };
  await walk(payload);
  return names.length;
}

// ------------------------------------------------------------------ install folder validation
function validateDir(dir) {
  const problems = [];
  const warnings = [];
  if (!dir || !dir.trim()) return { ok: false, problems: ["Choose a folder."], warnings };
  const d = path.resolve(dir);
  if (!path.isAbsolute(dir)) problems.push("Use a full path (for example C:\\Users\\you\\Apps\\Apsi).");
  if (/[<>"|?*\x00-\x1f]/.test(IS_WIN ? d.slice(2) : d)) problems.push("The path contains characters that are not allowed.");
  if (d.length > 180) problems.push("The path is too long (keep it under 180 characters).");
  if (path.parse(d).root === d) problems.push("Do not install into the root of a drive.");
  const lower = d.toLowerCase();
  const sysDirs = IS_WIN
    ? [process.env.SystemRoot || "C:\\Windows", process.env.ProgramData || "C:\\ProgramData"]
    : ["/bin", "/sbin", "/usr", "/etc", "/boot", "/lib", "/lib64", "/proc", "/sys", "/dev", "/var", "/root"];
  for (const s of sysDirs) {
    const sl = s.toLowerCase();
    if (lower === sl || lower.startsWith(sl + path.sep)) problems.push(`System folders such as ${s} cannot be used.`);
  }
  const home = os.homedir();
  if (lower === home.toLowerCase()) problems.push("Choose a folder inside your home folder, not the home folder itself.");
  if (IS_WIN && (lower.startsWith((process.env.ProgramFiles || "C:\\Program Files").toLowerCase()))) {
    warnings.push("Program Files needs administrator rights; the per-user default is recommended.");
  }
  if (/onedrive|dropbox|google drive/i.test(d)) warnings.push("Synced folders (OneDrive, Dropbox) slow Apsi down and may lock files.");
  try {
    if (fs.existsSync(d)) {
      const st = fs.lstatSync(d);
      if (st.isSymbolicLink()) problems.push("The folder is a symbolic link; choose a real folder.");
      else if (!st.isDirectory()) problems.push("A file with this name already exists.");
      else {
        const entries = fs.readdirSync(d);
        if (entries.length && !entries.includes("install.json")) problems.push("The folder is not empty and does not contain Apsi. Choose an empty or new folder.");
      }
    }
  } catch (e) {
    problems.push(`The folder cannot be read (${e.code || e.message}).`);
  }
  return { ok: problems.length === 0, problems, warnings, dir: d };
}

function readPayloadInfo(payload) {
  return JSON.parse(fs.readFileSync(path.join(payload, "payload.json"), "utf8"));
}

function pythonIn(root) {
  return IS_WIN ? path.join(root, "runtime", "python", "python.exe")
                : path.join(root, "runtime", "python", "bin", "python3");
}

function defaultInstallDir(allUsers) {
  if (IS_WIN) {
    if (allUsers) return path.join(process.env.ProgramFiles || "C:\\Program Files", "Apsi");
    return path.join(process.env.LOCALAPPDATA || path.join(HOME, "AppData", "Local"), "Programs", "Apsi");
  }
  return path.join(HOME, ".local", "share", "apsi");
}

function apsiHome() {
  return process.env.APSI_HOME || path.join(HOME, ".apsi");
}

async function isAdmin() {
  if (!IS_WIN) return typeof process.getuid === "function" && process.getuid() === 0;
  const r = await run("net", ["session"]);
  return r.code === 0;
}

async function diskFree(dir) {
  let d = dir;
  while (d && !fs.existsSync(d)) d = path.dirname(d);
  try {
    const s = await fsp.statfs(d || os.homedir());
    return s.bavail * s.bsize;
  } catch {
    return null;
  }
}

function xdgDir(name) {
  // reads ~/.config/user-dirs.dirs (XDG_TEMPLATES_DIR="$HOME/Templates")
  try {
    const txt = fs.readFileSync(path.join(HOME, ".config", "user-dirs.dirs"), "utf8");
    const m = new RegExp(`^XDG_${name}_DIR="([^"]*)"`, "m").exec(txt);
    if (m) return m[1].replace(/^\$HOME/, HOME);
  } catch { /* not configured */ }
  return name === "TEMPLATES" ? path.join(HOME, "Templates") : null;
}

const EXT_ID = "apsi.apsi-language";

// `apsi editors list --json` from the bundled runtime: VS Code, Cursor, Windsurf, VSCodium, Sublime, Vim, …
async function detectEditors(py) {
  const r = await run(py, ["-I", "-B", "-X", "utf8", "-m", "apsi", "editors", "list", "--json"],
    { timeoutMs: 30000, env: { ...process.env, PYTHONUTF8: "1", PYTHONNOUSERSITE: "1" } });
  if (r.code !== 0) return [];
  try { return JSON.parse(r.out.trim().split(/\r?\n/).pop()); } catch { return []; }
}

function findVSCode() {
  const c = which(IS_WIN ? "code.cmd" : "code");
  if (c) return c;
  if (IS_WIN) {
    const guesses = [
      path.join(process.env.LOCALAPPDATA || "", "Programs", "Microsoft VS Code", "bin", "code.cmd"),
      path.join(process.env.ProgramFiles || "", "Microsoft VS Code", "bin", "code.cmd"),
    ];
    for (const g of guesses) if (fs.existsSync(g)) return g;
  }
  return null;
}

function readExisting(dir) {
  const candidates = [dir, defaultInstallDir(false), defaultInstallDir(true)].filter(Boolean);
  if (IS_WIN) {
    const r = spawnSync("reg", ["query", "HKCU\\Software\\Apsi", "/v", "InstallDir"], { encoding: "utf8", windowsHide: true });
    const m = r.status === 0 && /InstallDir\s+REG_SZ\s+(.+)/.exec(r.stdout);
    if (m) candidates.unshift(m[1].trim());
  }
  for (const c of candidates) {
    const f = path.join(c, "install.json");
    if (fs.existsSync(f)) {
      try {
        return { dir: c, ...JSON.parse(fs.readFileSync(f, "utf8")) };
      } catch { /* ignore corrupt record */ }
    }
  }
  return null;
}

// ------------------------------------------------------------------ system check
async function detectSystem(resourcesPath) {
  const payload = payloadDir(resourcesPath);
  const info = readPayloadInfo(payload);
  const py = pythonIn(payload);
  let doctor = null;
  const r = await run(py, ["-I", "-B", "-X", "utf8", "-m", "apsi", "doctor", "--json"], { timeoutMs: 45000, env: { ...process.env, PYTHONUTF8: "1", PYTHONNOUSERSITE: "1" } });
  if (r.code === 0) {
    try { doctor = JSON.parse(r.out.trim().split(/\r?\n/).pop()); } catch { doctor = null; }
  }
  const required = Object.values(info.components).reduce((a, b) => a + b, 0);
  const dir = defaultInstallDir(false);
  return {
    platform: process.platform,
    arch: process.arch,
    version: info.version,
    payload: info,
    doctor,
    osName: doctor ? doctor.os : `${os.type()} ${os.release()}`,
    osOk: doctor ? !!doctor.os_meets_target : true,
    cpu: doctor ? doctor.cpu : (os.cpus()[0] || {}).model,
    cpuOk: doctor ? !!doctor.cpu_meets_target : true,
    threads: os.cpus().length,
    ramGB: os.totalmem() / 1e9,
    gpus: doctor ? doctor.gpus || [] : [],
    gpuOk: doctor ? !!doctor.gpu_meets_target : false,
    admin: await isAdmin(),
    editors: await detectEditors(py),
    existing: readExisting(null),
    defaultDir: dir,
    diskFree: await diskFree(dir),
    requiredBytes: required,
  };
}

function defaultOptions(sys) {
  return {
    mode: sys.existing ? "upgrade" : "install",
    dir: sys.existing ? sys.existing.dir : sys.defaultDir,
    allUsers: false,
    components: {
      runtime: true,
      examples: true,
      docs: true,
      editors: (sys.editors || []).some((e) => e.found && e.kind !== "manual"),
      addToPath: true,
      fileAssociation: true,
      startMenu: true,
      desktopShortcut: false,
    },
    packages: { gpu: false, onnx: false, torch: false, transformers: false },
    security: { profile: "standard", verify: true, scanOnRun: true, auditLog: true, guardLimit: 150 },
  };
}

// ------------------------------------------------------------------ copy with progress
async function countFiles(dir) {
  let n = 0;
  for (const e of await fsp.readdir(dir, { withFileTypes: true })) {
    if (e.isDirectory()) n += await countFiles(path.join(dir, e.name));
    else n += 1;
  }
  return n;
}

async function copyTree(src, dst, onFile) {
  await fsp.mkdir(dst, { recursive: true });
  for (const e of await fsp.readdir(src, { withFileTypes: true })) {
    const s = path.join(src, e.name);
    const d = path.join(dst, e.name);
    if (e.isDirectory()) {
      await copyTree(s, d, onFile);
    } else if (e.isSymbolicLink()) {
      const target = await fsp.readlink(s);
      await fsp.rm(d, { force: true });
      await fsp.symlink(target, d);
      onFile();
    } else {
      await fsp.copyFile(s, d);
      const st = await fsp.stat(s);
      await fsp.chmod(d, st.mode);
      onFile();
    }
  }
}

// ------------------------------------------------------------------ Windows registry / PATH
async function regAdd(key, name, value, type = "REG_SZ") {
  const args = ["add", key, "/f"];
  if (name === "") args.push("/ve"); else args.push("/v", name);
  args.push("/t", type, "/d", value);
  const r = await run("reg", args);
  if (r.code !== 0) throw new Error(`registry write failed for ${key}: ${r.err || r.out}`);
}

async function regDelete(key) {
  await run("reg", ["delete", key, "/f"]);
}

async function winUserPath(action, dir) {
  // [Environment]::SetEnvironmentVariable broadcasts WM_SETTINGCHANGE for us.
  const ps = action === "add"
    ? `$p=[Environment]::GetEnvironmentVariable('Path','User'); if(-not $p){$p=''}; ` +
      `$parts=$p.Split(';')|Where-Object{$_ -ne ''}; if($parts -notcontains '${dir}'){ ` +
      `[Environment]::SetEnvironmentVariable('Path',(($parts+'${dir}') -join ';'),'User') }`
    : `$p=[Environment]::GetEnvironmentVariable('Path','User'); if($p){ ` +
      `$parts=$p.Split(';')|Where-Object{$_ -ne '' -and $_ -ne '${dir}'}; ` +
      `[Environment]::SetEnvironmentVariable('Path',($parts -join ';'),'User') }`;
  const r = await run("powershell", ["-NoProfile", "-NonInteractive", "-Command", ps]);
  if (r.code !== 0) throw new Error(`could not update PATH: ${r.err}`);
}

// ------------------------------------------------------------------ Linux desktop integration
const PROFILE_BEGIN = "# >>> apsi >>>";
const PROFILE_END = "# <<< apsi <<<";

async function linuxPathBlock(action, binDir) {
  const files = [path.join(HOME, ".profile"), path.join(HOME, ".bashrc"), path.join(HOME, ".zshrc")];
  const block = `${PROFILE_BEGIN}\ncase ":$PATH:" in *":${binDir}:"*) ;; *) export PATH="${binDir}:$PATH" ;; esac\n${PROFILE_END}\n`;
  const touched = [];
  for (const f of files) {
    if (!fs.existsSync(f) && f.endsWith(".zshrc")) continue;
    let text = fs.existsSync(f) ? await fsp.readFile(f, "utf8") : "";
    const re = new RegExp(`\\n?${PROFILE_BEGIN}[\\s\\S]*?${PROFILE_END}\\n?`, "g");
    text = text.replace(re, "\n");
    if (action === "add") text = text.replace(/\n*$/, "\n") + "\n" + block;
    await fsp.writeFile(f, text);
    touched.push(f);
  }
  return touched;
}

// ------------------------------------------------------------------ launchers
function winLaunchers(dir) {
  const py = `"%~dp0..\\runtime\\python\\python.exe"`;
  const env = "chcp 65001 >nul\r\nset \"PYTHONUTF8=1\"\r\nset \"PYTHONNOUSERSITE=1\"\r\n";
  return {
    "apsi.cmd": `@echo off\r\nsetlocal\r\n${env}${py} -I -X utf8 -m apsi %*\r\nendlocal & exit /b %ERRORLEVEL%\r\n`,
    "apsi-repl.cmd": `@echo off\r\ntitle Apsi REPL\r\n${env}${py} -I -X utf8 -m apsi repl\r\n`,
    "apsi-run.cmd": `@echo off\r\n${env}if "%~1"=="--test" (\r\n  ${py} -I -X utf8 -m apsi test "%~2"\r\n) else (\r\n  pushd "%~dp1"\r\n  ${py} -I -X utf8 -m apsi run "%~1"\r\n  popd\r\n)\r\necho.\r\npause\r\n`,
  };
}

function linuxLaunchers(dir) {
  const py = path.join(dir, "runtime", "python", "bin", "python3");
  const env = `export PYTHONUTF8=1 PYTHONNOUSERSITE=1\n`;
  return {
    "apsi": `#!/bin/sh\n${env}exec "${py}" -I -X utf8 -m apsi "$@"\n`,
    "apsi-repl": `#!/bin/sh\n${env}exec "${py}" -I -X utf8 -m apsi repl\n`,
    "apsi-run": `#!/bin/sh\n${env}cd "$(dirname "$1")" && "${py}" -I -X utf8 -m apsi run "$1"\necho\nprintf "Press Enter to close..."; read _\n`,
  };
}

// ------------------------------------------------------------------ uninstall scripts
function winUninstallScript(record) {
  const q = (s) => s.replace(/'/g, "''");
  const lines = [
    "# Apsi uninstaller (generated). Removes exactly what the installer created.",
    "$ErrorActionPreference = 'Continue'",
  ];
  if (record.pathEntry) {
    lines.push(`$p=[Environment]::GetEnvironmentVariable('Path','User'); if($p){ $parts=$p.Split(';')|Where-Object{$_ -ne '' -and $_ -ne '${q(record.pathEntry)}'}; [Environment]::SetEnvironmentVariable('Path',($parts -join ';'),'User') }`);
  }
  for (const cli of record.extensionClis || []) lines.push(`& '${q(cli)}' --uninstall-extension ${EXT_ID} 2>$null | Out-Null`);
  for (const k of record.registryKeys || []) lines.push(`Remove-Item -Path 'Registry::${q(k)}' -Recurse -Force -ErrorAction SilentlyContinue`);
  for (const f of record.files || []) lines.push(`Remove-Item -LiteralPath '${q(f)}' -Force -ErrorAction SilentlyContinue`);
  for (const d of record.dirs || []) lines.push(`Remove-Item -LiteralPath '${q(d)}' -Recurse -Force -ErrorAction SilentlyContinue`);
  lines.push(`Remove-Item -LiteralPath '${q(record.dir)}' -Recurse -Force -ErrorAction SilentlyContinue`);
  lines.push("Write-Host 'Apsi has been removed.'");
  return lines.join("\r\n") + "\r\n";
}

function linuxUninstallScript(record) {
  const q = (s) => `'${s.replace(/'/g, "'\\''")}'`;
  const lines = ["#!/bin/sh", "# Apsi uninstaller (generated). Removes exactly what the installer created."];
  for (const f of record.profileFiles || []) {
    lines.push(`[ -f ${q(f)} ] && sed -i '/^# >>> apsi >>>$/,/^# <<< apsi <<<$/d' ${q(f)}`);
  }
  for (const cli of record.extensionClis || []) lines.push(`${q(cli)} --uninstall-extension ${EXT_ID} >/dev/null 2>&1 || true`);
  for (const f of record.files || []) lines.push(`rm -f ${q(f)}`);
  for (const d of record.dirs || []) lines.push(`rm -rf ${q(d)}`);
  if (record.mime) lines.push("command -v update-mime-database >/dev/null && update-mime-database \"$HOME/.local/share/mime\" >/dev/null 2>&1 || true");
  lines.push("command -v update-desktop-database >/dev/null && update-desktop-database \"$HOME/.local/share/applications\" >/dev/null 2>&1 || true");
  lines.push(`rm -rf ${q(record.dir)}`);
  lines.push('echo "Apsi has been removed."');
  return lines.join("\n") + "\n";
}

// ------------------------------------------------------------------ install
const STEPS = [
  ["integrity", "Verifying installer integrity", 8], ["prepare", "Preparing", 2], ["runtime", "Copying the Apsi runtime", 55], ["content", "Copying examples and documentation", 5],
  ["launchers", "Creating the apsi command", 2], ["path", "Adding Apsi to PATH", 2], ["assoc", "Registering .apsi files", 2],
  ["shortcuts", "Creating shortcuts", 2], ["editors", "Adding Apsi to your code editors", 6],
  ["packages", "Installing optional packages", 15], ["security", "Applying security settings", 2],
  ["uninstaller", "Registering the uninstaller", 3], ["verify", "Verifying the installation", 5],
];

async function install(options, hooks = {}) {
  const payload = payloadDir(hooks.resourcesPath);
  const info = readPayloadInfo(payload);
  const dir = path.resolve(options.dir);
  const c = options.components;
  const log = [];
  const say = (msg) => { log.push(msg); hooks.onLog && hooks.onLog(msg); };
  const total = STEPS.reduce((a, s) => a + s[2], 0);
  let done = 0;
  const step = (key, frac = 1) => {
    const s = STEPS.find((x) => x[0] === key);
    hooks.onProgress && hooks.onProgress({ step: key, label: s[1], percent: Math.min(100, ((done + s[2] * frac) / total) * 100) });
  };
  const finish = (key) => { done += STEPS.find((x) => x[0] === key)[2]; step(key, 0); };
  const record = {
    name: "Apsi", version: info.version, platform: process.platform, dir, installedAt: new Date().toISOString(),
    options, files: [], dirs: [], registryKeys: [], profileFiles: [], pathEntry: null, mime: false, extensionClis: [],
  };
  const createdDir = !fs.existsSync(dir);
  record.size = Object.values(info.components).reduce((a, b) => a + b, 0) / 1024;   // KB for Apps & Features

  try {
    // 0 integrity
    step("integrity", 0);
    if (!hooks.skipIntegrity) {
      const n = await verifyPayload(payload, (f) => step("integrity", f));
      say(`Verified ${n} payload files (sha256)`);
    }
    const v = validateDir(dir);
    if (!v.ok && !(fs.existsSync(path.join(dir, "install.json")))) throw new Error(v.problems.join(" "));
    finish("integrity");

    // 1 prepare
    step("prepare");
    say(`Installing Apsi ${info.version} to ${dir}`);
    if (options.mode === "upgrade" || options.mode === "repair") {
      await fsp.rm(path.join(dir, "runtime"), { recursive: true, force: true });
      say("Removed the previous runtime");
    }
    await fsp.mkdir(dir, { recursive: true });
    const probe = path.join(dir, ".write-test");
    await fsp.writeFile(probe, "ok");
    await fsp.rm(probe);
    finish("prepare");

    // 2 runtime
    const rsrc = path.join(payload, "runtime");
    const n = await countFiles(rsrc);
    let k = 0;
    await copyTree(rsrc, path.join(dir, "runtime"), () => { k += 1; if (k % 50 === 0) step("runtime", k / n); });
    await copyTree(path.join(payload, "brand"), path.join(dir, "brand"), () => {});
    for (const f of ["README.md", "CHANGELOG.md", "LICENSE", "payload.json"]) {
      if (fs.existsSync(path.join(payload, f))) await fsp.copyFile(path.join(payload, f), path.join(dir, f));
    }
    say(`Copied runtime (${n} files)`);
    finish("runtime");

    // 3 content
    step("content");
    for (const part of ["examples", "docs"]) {
      const target = path.join(dir, part);
      await fsp.rm(target, { recursive: true, force: true });
      if (c[part]) {
        await copyTree(path.join(payload, part), target, () => {});
        say(`Copied ${part}`);
      }
    }
    finish("content");

    // 4 launchers
    step("launchers");
    const bin = path.join(dir, "bin");
    await fsp.mkdir(bin, { recursive: true });
    const launchers = IS_WIN ? winLaunchers(dir) : linuxLaunchers(dir);
    for (const [name, text] of Object.entries(launchers)) {
      await fsp.writeFile(path.join(bin, name), text);
      if (!IS_WIN) await fsp.chmod(path.join(bin, name), 0o755);
    }
    say("Created the apsi command");
    finish("launchers");

    // 5 PATH
    step("path");
    if (c.addToPath) {
      if (IS_WIN) {
        await winUserPath("add", bin);
        record.pathEntry = bin;
      } else {
        const localBin = path.join(HOME, ".local", "bin");
        await fsp.mkdir(localBin, { recursive: true });
        for (const name of ["apsi", "apsi-repl"]) {
          const link = path.join(localBin, name);
          await fsp.rm(link, { force: true });
          await fsp.symlink(path.join(bin, name), link);
          record.files.push(link);
        }
        if (!(process.env.PATH || "").split(":").includes(localBin)) {
          record.profileFiles = await linuxPathBlock("add", localBin);
        }
      }
      say("Added Apsi to PATH (open a new terminal to use it)");
    }
    finish("path");

    // 6 file association
    step("assoc");
    if (c.fileAssociation) {
      if (IS_WIN) {
        const cls = "HKCU\\Software\\Classes";
        const runVerb = `"${path.join(bin, "apsi-run.cmd")}" "%1"`;   // not named `run`: that is the process helper
        await regAdd(`${cls}\\.apsi`, "", "Apsi.Source");
        await regAdd(`${cls}\\.apsi`, "Content Type", "text/plain");
        await regAdd(`${cls}\\Apsi.Source`, "", "Apsi source file");
        await regAdd(`${cls}\\Apsi.Source\\DefaultIcon`, "", path.join(dir, "brand", "apsi-file.ico"));
        await regAdd(`${cls}\\Apsi.Source\\shell\\open`, "", "Run with Apsi");
        await regAdd(`${cls}\\Apsi.Source\\shell\\open\\command`, "", runVerb);
        // right-click > New > Apsi source file
        await regAdd(`${cls}\\.apsi\\ShellNew`, "NullFile", "");
        await regAdd(`${cls}\\Apsi.Source`, "FriendlyTypeName", "Apsi source file");
        const editor = findVSCode();
        const codeExe = editor && editor.toLowerCase().endsWith("code.cmd") ? path.join(path.dirname(path.dirname(editor)), "Code.exe") : null;
        const editCmd = codeExe && fs.existsSync(codeExe) ? `"${codeExe}" "%1"` : `notepad.exe "%1"`;
        await regAdd(`${cls}\\Apsi.Source\\shell\\edit`, "", "Edit");
        await regAdd(`${cls}\\Apsi.Source\\shell\\edit\\command`, "", editCmd);
        await regAdd(`${cls}\\Apsi.Source\\shell\\test`, "", "Test with Apsi");
        await regAdd(`${cls}\\Apsi.Source\\shell\\test\\command`, "", `"${path.join(bin, "apsi-run.cmd")}" --test "%1"`);
        record.registryKeys.push(`${cls}\\.apsi`, `${cls}\\Apsi.Source`);
        // make Explorer pick up the new icon right away instead of after a restart
        await run("ie4uinit.exe", ["-show"]);
      } else {
        const share = path.join(HOME, ".local", "share");
        const mimeDir = path.join(share, "mime", "packages");
        await fsp.mkdir(mimeDir, { recursive: true });
        const mimeFile = path.join(mimeDir, "apsi.xml");
        await fsp.writeFile(mimeFile, `<?xml version="1.0" encoding="UTF-8"?>\n<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n  <mime-type type="text/x-apsi">\n    <comment>Apsi source file</comment>\n    <sub-class-of type="text/plain"/>\n    <glob pattern="*.apsi"/>\n    <icon name="text-x-apsi"/>\n  </mime-type>\n</mime-info>\n`);
        record.files.push(mimeFile);
        for (const s of [16, 32, 48, 64, 128, 256]) {
          const appIcon = path.join(share, "icons", "hicolor", `${s}x${s}`, "apps", "apsi.png");
          const mimeIcon = path.join(share, "icons", "hicolor", `${s}x${s}`, "mimetypes", "text-x-apsi.png");
          for (const [dst, src] of [[appIcon, `apsi-${s}.png`], [mimeIcon, `apsi-file-${s}.png`]]) {
            await fsp.mkdir(path.dirname(dst), { recursive: true });
            await fsp.copyFile(path.join(dir, "brand", "png", src), dst);
            record.files.push(dst);
          }
        }
        const apps = path.join(share, "applications");
        await fsp.mkdir(apps, { recursive: true });
        const runDesktop = path.join(apps, "apsi-run.desktop");
        await fsp.writeFile(runDesktop, `[Desktop Entry]\nType=Application\nName=Apsi\nComment=Run Apsi programs\nExec="${path.join(bin, "apsi-run")}" %f\nIcon=apsi\nTerminal=true\nMimeType=text/x-apsi;\nNoDisplay=true\n`);
        record.files.push(runDesktop);
        record.mime = true;
        // right-click > New Document > Apsi file, in desktops that use ~/Templates
        const templates = xdgDir("TEMPLATES");
        if (templates && fs.existsSync(templates)) {
          const tpl = path.join(templates, "Apsi file.apsi");
          if (!fs.existsSync(tpl)) {
            await fsp.writeFile(tpl, "# New Apsi program\n\nprint(\"Hello from Apsi!\")\n");
            record.files.push(tpl);
          }
        }
        await run("update-mime-database", [path.join(share, "mime")]);
        await run("xdg-mime", ["default", "apsi-run.desktop", "text/x-apsi"]);
      }
      say("Registered .apsi files");
    }
    finish("assoc");

    // 7 shortcuts
    step("shortcuts");
    if (c.startMenu || c.desktopShortcut) {
      if (IS_WIN) {
        const icon = path.join(dir, "brand", "apsi.ico");
        const startDir = path.join(process.env.APPDATA || "", "Microsoft", "Windows", "Start Menu", "Programs", "Apsi");
        const links = [];
        if (c.startMenu) {
          links.push([path.join(startDir, "Apsi REPL.lnk"), path.join(bin, "apsi-repl.cmd"), "Interactive Apsi prompt"]);
          if (c.examples) links.push([path.join(startDir, "Apsi Examples.lnk"), path.join(dir, "examples"), "Example programs"]);
          if (c.docs) links.push([path.join(startDir, "Apsi Documentation.lnk"), path.join(dir, "docs"), "Documentation"]);
          links.push([path.join(startDir, "Uninstall Apsi.lnk"), path.join(dir, "Uninstall.cmd"), "Remove Apsi"]);
          record.dirs.push(startDir);
        }
        if (c.desktopShortcut) {
          const desk = path.join(HOME, "Desktop", "Apsi REPL.lnk");
          links.push([desk, path.join(bin, "apsi-repl.cmd"), "Interactive Apsi prompt"]);
          record.files.push(desk);
        }
        await fsp.mkdir(startDir, { recursive: true });
        for (const [lnk, target, desc] of links) {
          if (hooks.makeShortcut) hooks.makeShortcut(lnk, { target, icon, iconIndex: 0, description: desc, cwd: dir });
        }
      } else {
        const apps = path.join(HOME, ".local", "share", "applications");
        await fsp.mkdir(apps, { recursive: true });
        const entry = `[Desktop Entry]\nType=Application\nName=Apsi REPL\nComment=Interactive Apsi prompt\nExec="${path.join(bin, "apsi-repl")}"\nIcon=apsi\nTerminal=true\nCategories=Development;Science;\n`;
        if (c.startMenu) {
          const f = path.join(apps, "apsi-repl.desktop");
          await fsp.writeFile(f, entry);
          record.files.push(f);
        }
        if (c.desktopShortcut) {
          const deskDir = path.join(HOME, "Desktop");
          if (fs.existsSync(deskDir)) {
            const f = path.join(deskDir, "apsi-repl.desktop");
            await fsp.writeFile(f, entry);
            await fsp.chmod(f, 0o755);
            record.files.push(f);
          }
        }
      }
      say("Created shortcuts");
    }
    finish("shortcuts");

    // 8 VS Code
    step("editors");
    {
      // the integration files live in the install folder, so `apsi editors install` works later too
      for (const part of ["editors", "vscode"]) {
        const target = path.join(dir, part);
        await fsp.rm(target, { recursive: true, force: true });
        if (fs.existsSync(path.join(payload, part))) await copyTree(path.join(payload, part), target, () => {});
      }
      const action = c.editors ? "install" : "uninstall";
      const r = await run(pythonIn(dir), ["-I", "-B", "-X", "utf8", "-m", "apsi", "editors", action, "--json", "--root", dir],
        { timeoutMs: 600000, env: { ...process.env, PYTHONUTF8: "1", PYTHONNOUSERSITE: "1" } });
      let results = [];
      try { results = JSON.parse(r.out.trim().split(/\r?\n/).pop()).results || []; } catch { results = []; }
      for (const e of results) {
        if (action === "uninstall") continue;
        if (e.ok) {
          record.files.push(...(e.files || []));
          record.dirs.push(...(e.dirs || []));
          if (e.uninstall_cli) record.extensionClis.push(e.uninstall_cli);
          say(`Added Apsi to ${e.name}`);
        } else {
          say(`${e.name}: Apsi support not added (${e.error || "manual step needed"}) — retry later with: apsi editors install ${e.id}`);
        }
      }
      if (c.editors && !results.length) say(r.code === 0 ? "No supported code editor found" : `Editor support skipped: ${(r.err || "").trim().slice(-300)}`);
    }
    finish("editors");

    // 9 optional packages
    step("packages");
    const pkgs = [];
    if (options.packages.gpu) pkgs.push("gpu");
    if (options.packages.onnx) pkgs.push(options.packages.gpu ? "onnx-gpu" : "onnx");
    if (options.packages.torch) pkgs.push("torch-auto");   // CUDA build matched to the NVIDIA driver, ROCm, or CPU
    if (options.packages.transformers) pkgs.push("transformers");
    if (pkgs.length) {
      say(`Downloading ${pkgs.join(", ")} (binary wheels only, over HTTPS)…`);
      const r = await run(pythonIn(dir), ["-I", "-X", "utf8", "-m", "apsi", "pkg", "install", ...pkgs],
        { env: { ...process.env, PYTHONNOUSERSITE: "1", PIP_NO_WARN_SCRIPT_LOCATION: "1" }, onLine: (l) => l.trim() && say(l.trim()) });
      say(r.code === 0 ? "Optional packages installed" : `Some optional packages could not be installed — Apsi works without them; retry later with: apsi pkg install ${pkgs.join(" ")}`);
    }
    finish("packages");

    // 10 security
    step("security");
    const cfgDir = apsiHome();
    await fsp.mkdir(cfgDir, { recursive: true });
    const cfgFile = path.join(cfgDir, "config.json");
    let cfg = {};
    try { cfg = JSON.parse(await fsp.readFile(cfgFile, "utf8")); } catch { cfg = {}; }
    const sec = options.security;
    cfg.profile = sec.profile;
    cfg.scan_on_run = sec.scanOnRun === false ? "off" : "on";
    cfg.audit_log = sec.auditLog !== false;
    cfg.guard_max_overwrites = Math.max(10, Math.min(100000, Number(sec.guardLimit) || 150));
    await fsp.writeFile(cfgFile, JSON.stringify(cfg, null, 2), { mode: 0o600 });
    if (!IS_WIN) { try { await fsp.chmod(cfgDir, 0o700); } catch { /* best effort */ } }
    say(`Security profile: ${sec.profile} · Apsi Guard ${cfg.scan_on_run} · audit log ${cfg.audit_log ? "on" : "off"}`);
    finish("security");

    // 11 uninstaller
    step("uninstaller");
    if (IS_WIN) {
      const ps1 = path.join(dir, "uninstall.ps1");
      await fsp.writeFile(ps1, winUninstallScript(record));
      await fsp.writeFile(path.join(dir, "Uninstall.cmd"),
        `@echo off\r\ncopy /y "%~dp0uninstall.ps1" "%TEMP%\\apsi-uninstall.ps1" >nul\r\npowershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\\apsi-uninstall.ps1"\r\npause\r\n`);
      const key = "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Apsi";
      await regAdd(key, "DisplayName", `Apsi ${info.version}`);
      await regAdd(key, "DisplayVersion", info.version);
      await regAdd(key, "Publisher", "Apsi");
      await regAdd(key, "InstallLocation", dir);
      await regAdd(key, "DisplayIcon", path.join(dir, "brand", "apsi.ico"));
      await regAdd(key, "UninstallString", `"${path.join(dir, "Uninstall.cmd")}"`);
      await regAdd(key, "EstimatedSize", String(Math.round(record.size || 0)), "REG_DWORD");
      await regAdd(key, "NoModify", "1", "REG_DWORD");
      await regAdd(key, "NoRepair", "1", "REG_DWORD");
      await regAdd("HKCU\\Software\\Apsi", "InstallDir", dir);
      await regAdd("HKCU\\Software\\Apsi", "Version", info.version);
      record.registryKeys.push(key, "HKCU\\Software\\Apsi");
      await fsp.writeFile(ps1, winUninstallScript(record));
    } else {
      const sh = path.join(dir, "uninstall.sh");
      await fsp.writeFile(sh, linuxUninstallScript(record));
      await fsp.chmod(sh, 0o755);
    }
    await fsp.writeFile(path.join(dir, "install.json"), JSON.stringify(record, null, 2));
    say("Registered the uninstaller");
    finish("uninstaller");

    // 12 verify
    step("verify");
    const launcher = path.join(bin, IS_WIN ? "apsi.cmd" : "apsi");
    const checks = [["version"]];
    if (options.security.verify) checks.push(["verify"]);
    if (c.examples) checks.push(["test", path.join(dir, "examples", "01_hello.apsi")]);
    for (const args of checks) {
      const r = await runScript(launcher, args, { env: { ...process.env, PYTHONUTF8: "1" } });
      const out = (r.out + r.err).trim().split(/\r?\n/).filter(Boolean);
      if (r.code !== 0) throw new Error(`self-test 'apsi ${args[0]}' failed:\n${out.slice(-5).join("\n")}`);
      say(`✓ apsi ${args[0]}: ${out[out.length - 1] || "ok"}`);
    }
    finish("verify");
    hooks.onProgress && hooks.onProgress({ step: "done", label: "Done", percent: 100 });
    return { ok: true, dir, log, record };
  } catch (e) {
    say(`ERROR: ${e.message}`);
    say("Rolling back the changes made so far…");
    try {
      await undo(record, createdDir);
      say("Rollback complete");
    } catch (e2) {
      say(`Rollback incomplete: ${e2.message}`);
    }
    return { ok: false, dir, log, error: e.message };
  }
}

// ------------------------------------------------------------------ uninstall / rollback
async function undo(record, removeDir = true) {
  if (record.pathEntry && IS_WIN) await winUserPath("remove", record.pathEntry).catch(() => {});
  for (const cli of record.extensionClis || []) await runScript(cli, ["--uninstall-extension", EXT_ID]).catch(() => {});
  for (const k of record.registryKeys || []) if (IS_WIN) await regDelete(k);
  if (record.profileFiles && record.profileFiles.length) await linuxPathBlock("remove", null);
  for (const f of record.files || []) await fsp.rm(f, { force: true });
  for (const d of record.dirs || []) await fsp.rm(d, { recursive: true, force: true });
  if (record.mime) await run("update-mime-database", [path.join(HOME, ".local", "share", "mime")]);
  if (removeDir) await fsp.rm(record.dir, { recursive: true, force: true });
}

async function uninstall(dir, hooks = {}) {
  const f = path.join(dir, "install.json");
  const record = JSON.parse(await fsp.readFile(f, "utf8"));
  hooks.onLog && hooks.onLog(`Removing Apsi ${record.version} from ${record.dir}`);
  await undo(record, true);
  hooks.onLog && hooks.onLog("Apsi has been removed.");
  return { ok: true };
}

module.exports = {
  verifyPayload, validateDir,
  detectSystem, defaultOptions, install, uninstall, payloadDir, defaultInstallDir, readExisting,
  STEPS, _internal: { linuxPathBlock, winUninstallScript, linuxUninstallScript, copyTree },
};
