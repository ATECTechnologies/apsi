// Apsi Setup — Electron main process.
// Security: context isolation, sandboxed renderer, no Node in the UI, strict CSP,
// navigation and new windows blocked, only https links may open externally.
"use strict";

const { app, BrowserWindow, ipcMain, dialog, shell, session } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const core = require("./installer-core.cjs");

const IS_WIN = process.platform === "win32";
const args = process.argv.slice(app.isPackaged ? 1 : 2);
const flag = (name) => args.includes(name);
const value = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};

let win = null;
let splash = null;
let busy = false;
let lastSystem = null;      // what Setup detected: the uninstall target comes from here, never from the UI
let lastInstallDir = null;  // the folder that was actually installed

// ------------------------------------------------------------------ option sanitising
// The renderer is untrusted: everything it sends is rebuilt from a whitelist.
const PROFILES = ["sandbox", "strict", "standard", "developer"];
function sanitizeOptions(o) {
  const sys = lastSystem || {};
  const base = core.defaultOptions(sys);
  if (!o || typeof o !== "object") return base;
  const out = structuredClone(base);
  if (["install", "upgrade", "repair"].includes(o.mode)) out.mode = o.mode;
  if (typeof o.dir === "string") out.dir = o.dir;
  for (const k of Object.keys(base.components)) {
    if (o.components && typeof o.components[k] === "boolean") out.components[k] = o.components[k];
  }
  out.components.runtime = true;
  for (const k of Object.keys(base.packages)) {
    if (o.packages && typeof o.packages[k] === "boolean") out.packages[k] = o.packages[k];
  }
  if (o.security && PROFILES.includes(o.security.profile)) out.security.profile = o.security.profile;
  for (const k of ["verify", "scanOnRun", "auditLog"]) {
    if (o.security && typeof o.security[k] === "boolean") out.security[k] = o.security[k];
  }
  if (o.security && [50, 150, 500, 2000].includes(o.security.guardLimit)) out.security.guardLimit = o.security.guardLimit;
  const v = core.validateDir(out.dir);
  const upgrading = sys.existing && path.resolve(sys.existing.dir) === path.resolve(out.dir);
  if (!v.ok && !upgrading) throw new Error(v.problems.join(" "));
  return out;
}

// Only the bundled UI (file:// inside the app) may call privileged handlers.
function trusted(e) {
  const url = (e.senderFrame && e.senderFrame.url) || "";
  return win && e.sender === win.webContents && url.startsWith("file://");
}
function handle(channel, fn) {
  ipcMain.handle(channel, (e, ...a) => {
    if (!trusted(e)) throw new Error("blocked: untrusted sender");
    return fn(e, ...a);
  });
}

// ------------------------------------------------------------------ silent / unattended mode
async function silent() {
  const log = (m) => process.stdout.write(m + "\n");
  if (flag("--uninstall")) {
    const sys = await core.detectSystem(process.resourcesPath);
    if (!sys.existing) { log("Apsi is not installed."); return 0; }
    await core.uninstall(sys.existing.dir, { onLog: log });
    return 0;
  }
  const sys = await core.detectSystem(process.resourcesPath);
  const o = core.defaultOptions(sys);
  if (value("--dir")) {
    const v = core.validateDir(value("--dir"));
    if (!v.ok) { log("Invalid --dir: " + v.problems.join(" ")); return 2; }
    o.dir = v.dir;
  }
  if (value("--profile")) {
    if (!PROFILES.includes(value("--profile"))) { log("Unknown --profile (sandbox, strict, standard, developer)"); return 2; }
    o.security.profile = value("--profile");
  }
  for (const c of ["examples", "docs", "editors", "fileAssociation", "startMenu"]) {
    if (flag(`--no-${c}`)) o.components[c] = false;
  }
  if (flag("--no-vscode")) o.components.editors = false;          // older name of --no-editors
  if (flag("--no-path")) o.components.addToPath = false;
  if (flag("--desktop-shortcut")) o.components.desktopShortcut = true;
  for (const p of ["gpu", "onnx", "torch", "transformers"]) if (flag(`--with-${p}`)) o.packages[p] = true;
  const res = await core.install(o, {
    resourcesPath: process.resourcesPath, onLog: log,
    makeShortcut: (lnk, opts) => shell.writeShortcutLink(lnk, "create", opts),
  });
  return res.ok ? 0 : 1;
}

// ------------------------------------------------------------------ splash
// Shown the moment Setup starts and kept on top of every other window until the setup window has
// painted its first frame; then it closes and the setup window takes its place. Same size and
// design as the Windows launcher splash (build/splash.bmp), so the hand-over looks seamless.
function createSplash() {
  splash = new BrowserWindow({
    width: 520,
    height: 300,
    frame: false,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    alwaysOnTop: true,
    skipTaskbar: false,
    show: false,
    center: true,
    backgroundColor: "#0b1220",
    title: "Apsi Setup",
    icon: path.join(__dirname, "..", "build", IS_WIN ? "icon.ico" : "icons/256x256.png"),
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, devTools: false, spellcheck: false },
  });
  splash.setAlwaysOnTop(true, "floating");
  splash.removeMenu();
  splash.once("ready-to-show", () => splash && !splash.isDestroyed() && splash.show());
  splash.on("closed", () => { splash = null; });
  splash.loadFile(path.join(__dirname, "splash.html"), { query: { v: app.getVersion() } });
}

function closeSplash() {
  if (splash && !splash.isDestroyed()) splash.destroy();
  splash = null;
}

function revealMain() {
  if (!win || win.isDestroyed() || win.isVisible()) return;
  win.show();
  win.focus();
  closeSplash();
}

// ------------------------------------------------------------------ window
function createWindow() {
  win = new BrowserWindow({
    width: 900,
    height: 610,
    minWidth: 820,
    minHeight: 560,
    resizable: true,
    frame: false,
    show: false,
    center: true,
    backgroundColor: "#ffffff",
    title: "Apsi Setup",
    icon: path.join(__dirname, "..", "build", IS_WIN ? "icon.ico" : "icons/256x256.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      spellcheck: false,
      devTools: !app.isPackaged,
    },
  });
  win.removeMenu();
  win.once("ready-to-show", revealMain);
  // never leave the user looking at a splash: show the window anyway after 20 s
  const fallback = setTimeout(revealMain, 20000);
  win.on("show", () => clearTimeout(fallback));
  win.webContents.on("did-fail-load", (_e, code, desc, _url, isMainFrame) => {
    if (!isMainFrame || code === -3) return; // -3 = navigation aborted (harmless)
    closeSplash();
    dialog.showErrorBox("Apsi Setup", `The setup window could not be loaded (${desc || code}). Please download Apsi Setup again.`);
    app.exit(1);
  });
  win.webContents.on("render-process-gone", (_e, details) => {
    if (details.reason === "clean-exit") return;
    closeSplash();
    const choice = dialog.showMessageBoxSync({
      type: "error", buttons: ["Reload", "Quit"], defaultId: 0, cancelId: 1,
      message: "Apsi Setup stopped unexpectedly.", detail: `Reason: ${details.reason}. Reload to continue.`,
    });
    if (choice === 0 && win && !win.isDestroyed()) {
      win.webContents.once("did-finish-load", revealMain);
      win.reload();
    } else app.exit(1);
  });
  win.on("unresponsive", () => {
    if (busy) return; // copying files can briefly block; the main process keeps working
    const choice = dialog.showMessageBoxSync(win, {
      type: "warning", buttons: ["Wait", "Reload"], defaultId: 0, cancelId: 0,
      message: "Apsi Setup is not responding.",
    });
    if (choice === 1) win.reload();
  });
  win.on("closed", () => { win = null; });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e) => e.preventDefault());
  win.on("close", (e) => {
    if (busy) {
      const choice = dialog.showMessageBoxSync(win, {
        type: "warning", buttons: ["Keep installing", "Quit"], defaultId: 0, cancelId: 0,
        message: "Setup is still running.", detail: "Quitting now may leave Apsi partly installed.",
      });
      if (choice === 0) e.preventDefault();
    }
  });
  win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
}

// ------------------------------------------------------------------ helpers
function onPath(cmd) {
  const exts = IS_WIN ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";") : [""];
  return (process.env.PATH || "").split(path.delimiter).some((d) =>
    d && exts.some((x) => { try { return fs.statSync(path.join(d, cmd + x)).isFile(); } catch { return false; } }));
}

function launch(cmd, args) {
  // a detached helper must never crash Setup: errors (e.g. ENOENT) are emitted asynchronously
  try {
    const child = spawn(cmd, args, { detached: true, stdio: "ignore", windowsHide: false });
    child.on("error", () => {});
    child.unref();
  } catch { /* ignore */ }
}

// ------------------------------------------------------------------ IPC
function register() {
  handle("system:detect", async () => (lastSystem = await core.detectSystem(process.resourcesPath)));
  handle("options:default", async () => core.defaultOptions(lastSystem || await core.detectSystem(process.resourcesPath)));
  handle("dir:validate", (_e, dir) => core.validateDir(typeof dir === "string" ? dir : ""));
  handle("app:startMode", () => (flag("--uninstall") ? "uninstall" : "normal"));
  handle("app:platform", () => process.platform);
  handle("dialog:pickDir", async (_e, current) => {
    const r = await dialog.showOpenDialog(win, {
      title: "Choose the Apsi installation folder", defaultPath: current,
      properties: ["openDirectory", "createDirectory", "promptToCreate"],
    });
    return r.canceled ? null : r.filePaths[0];
  });
  handle("install:start", async (e, options) => {
    if (busy) return { ok: false, error: "already running" };
    let opts;
    try { opts = sanitizeOptions(options); } catch (err) { return { ok: false, error: err.message }; }
    busy = true;
    try {
      const res = await core.install(opts, {
        resourcesPath: process.resourcesPath,
        onProgress: (p) => e.sender.send("install:progress", p),
        onLog: (m) => e.sender.send("install:log", m),
        makeShortcut: (lnk, o) => shell.writeShortcutLink(lnk, "create", o),
      });
      if (res.ok) lastInstallDir = res.dir;
      return res;
    } finally {
      busy = false;
    }
  });
  handle("uninstall:start", async (e) => {
    const sys = lastSystem || await core.detectSystem(process.resourcesPath);
    if (!sys.existing) return { ok: false, error: "Apsi is not installed." };
    busy = true;
    try {
      return await core.uninstall(sys.existing.dir, { onLog: (m) => e.sender.send("install:log", m) });
    } catch (err) {
      return { ok: false, error: err.message };
    } finally {
      busy = false;
    }
  });
  handle("app:open", async (_e, what) => {
    const dir = lastInstallDir || (lastSystem && lastSystem.existing && lastSystem.existing.dir);
    if (!dir) return false;
    const bin = path.join(dir, "bin");
    if (what === "repl") {
      if (IS_WIN) {
        launch(process.env.ComSpec || "cmd.exe", ["/c", "start", "", path.join(bin, "apsi-repl.cmd")]);
        return true;
      }
      const terms = [["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["konsole", "-e"], ["xfce4-terminal", "-x"],
        ["kitty", "--"], ["alacritty", "-e"], ["tilix", "-e"], ["mate-terminal", "-x"], ["xterm", "-e"]];
      const found = terms.find(([t]) => onPath(t));
      if (!found) return false;
      launch(found[0], [found[1], path.join(bin, "apsi-repl")]);
      return true;
    }
    const target = { examples: path.join(dir, "examples"), docs: path.join(dir, "docs"), folder: dir,
      readme: path.join(dir, "README.md") }[what];
    if (target && fs.existsSync(target)) await shell.openPath(target);
    return true;
  });
  handle("link:open", (_e, url) => {
    if (typeof url === "string" && url.startsWith("https://")) shell.openExternal(url);
  });
  ipcMain.on("window:minimize", (e) => trusted(e) && win.minimize());
  ipcMain.on("window:close", (e) => trusted(e) && win.close());
}

// ------------------------------------------------------------------ start
if (flag("--silent")) {
  app.whenReady().then(async () => {
    const code = await silent().catch((e) => { process.stderr.write(String(e) + "\n"); return 1; });
    app.exit(code);
  });
} else if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  // software rendering: no GPU process to start (faster first paint) and no driver-specific
  // blank-window or crash bugs; the UI is simple enough that it makes no visible difference
  app.disableHardwareAcceleration();
  app.on("second-instance", () => {
    if (win && win.isVisible()) { if (win.isMinimized()) win.restore(); win.focus(); }
    else if (splash && !splash.isDestroyed()) splash.focus();
  });
  app.whenReady().then(() => {
    createSplash();
    // deny every browser permission (camera, notifications, clipboard-read, ...) and any remote load
    session.defaultSession.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
    session.defaultSession.setPermissionCheckHandler(() => false);
    session.defaultSession.webRequest.onBeforeRequest((d, cb) => {
      cb({ cancel: !(d.url.startsWith("file://") || d.url.startsWith("devtools://") || d.url.startsWith("data:")) });
    });
    register();
    createWindow();
  });
  app.on("web-contents-created", (_e, wc) => {
    wc.on("will-attach-webview", (ev) => ev.preventDefault());
    wc.on("will-navigate", (ev) => ev.preventDefault());
    wc.setWindowOpenHandler(() => ({ action: "deny" }));
  });
  app.on("window-all-closed", () => app.quit());
}
