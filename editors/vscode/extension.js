// Apsi for VS Code — language server client, run/test/check/watch/REPL commands, apsi.toml tasks.
// Works on Windows 10/11, Linux and macOS. No dependencies besides vscode-languageclient.
"use strict";

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const cp = require("child_process");

const IS_WIN = process.platform === "win32";
let client = null;
let output = null;
let cachedApsi = undefined; // { command, args, version } | null
let statusItem = null;

// ------------------------------------------------------------------ locating Apsi
// Windows refuses to start a .cmd or .bat file without a shell (Node 18.20.2 and later, every current VS Code),
// so a launcher found on PATH runs through cmd.exe; the bundled runtime's python.exe runs directly.
function needsShell(cmd) {
  return IS_WIN && /\.(cmd|bat)$/i.test(cmd);
}

// The command line to start Apsi with these arguments: { command, args, shell }.
function invocation(apsi, extra) {
  const all = [...apsi.args, ...extra];
  if (!needsShell(apsi.command)) return { command: apsi.command, args: all, shell: false };
  return { command: quote(apsi.command), args: all.map(quote), shell: true };
}

function apsiEnv(apsi, more) {
  return { ...process.env, PYTHONUTF8: "1", ...(apsi && apsi.env), ...(more || {}) };
}

// One Apsi command, waited for. Never throws: a failure to start comes back as { error }.
function runApsi(apsi, extra, opts = {}) {
  const inv = invocation(apsi, extra);
  return cp.spawnSync(inv.command, inv.args, {
    encoding: "utf8", windowsHide: true, maxBuffer: 256 * 1024 * 1024, shell: inv.shell,
    cwd: opts.cwd, timeout: opts.timeout, env: apsiEnv(apsi, opts.env),
  });
}

function probe(apsi) {
  try {
    const r = runApsi(apsi, ["version"], { timeout: 15000 });
    const m = r.status === 0 && /Apsi\s+([\w.\-]+)/.exec(r.stdout || "");
    return m ? m[1] : null;
  } catch (e) {
    return null;
  }
}

// The Apsi installs on this computer, newest first: the version in current.txt (`apsi use`), then the others.
// Layout: <root>/<version>/runtime/python/..., and the older single-version layout <root>/runtime/python/...
function installedRuntimes() {
  const roots = [];
  if (IS_WIN) {
    const local = process.env.LOCALAPPDATA || path.join(process.env.USERPROFILE || "", "AppData", "Local");
    roots.push(path.join(local, "Programs", "Apsi"), path.join(process.env.ProgramFiles || "C:\\Program Files", "Apsi"),
      path.join(local, "Apsi"));
  } else {
    roots.push(path.join(process.env.HOME || "", ".local", "share", "apsi"), "/opt/apsi");
  }
  const python = (dir) => (IS_WIN ? path.join(dir, "runtime", "python", "python.exe")
                                  : path.join(dir, "runtime", "python", "bin", "python3"));
  const num = (v) => v.split(/[.\-]/).map((x) => parseInt(x, 10) || 0);
  const newer = (a, b) => { const x = num(a), y = num(b); for (let i = 0; i < Math.max(x.length, y.length); i++) { if ((x[i] || 0) !== (y[i] || 0)) return (y[i] || 0) - (x[i] || 0); } return 0; };
  const out = [];
  for (const root of roots) {
    let versions = [];
    try { versions = fs.readdirSync(root).filter((d) => /^\d+\.\d+/.test(d)).sort(newer); } catch (e) { continue; }
    let current = "";
    try { current = fs.readFileSync(path.join(root, "current.txt"), "utf8").trim(); } catch (e) { /* none */ }
    if (current && versions.includes(current)) versions = [current, ...versions.filter((v) => v !== current)];
    for (const v of versions) {
      const py = python(path.join(root, v));
      if (fs.existsSync(py)) out.push({ command: py, args: ["-I", "-X", "utf8", "-m", "apsi"], env: { APSI_INSTALL_ROOT: root, APSI_INSTALL_VERSION: v, PYTHONNOUSERSITE: "1" } });
    }
    const legacy = python(root);
    if (fs.existsSync(legacy)) out.push({ command: legacy, args: ["-I", "-X", "utf8", "-m", "apsi"], env: { PYTHONNOUSERSITE: "1" } });
  }
  return out;
}

function findApsi(force) {
  if (cachedApsi !== undefined && !force) return cachedApsi;
  const cfg = vscode.workspace.getConfiguration("apsi").get("command") || [];
  const candidates = [];
  if (cfg.length) candidates.push({ command: cfg[0], args: cfg.slice(1) });
  candidates.push(...installedRuntimes());
  if (IS_WIN) {
    const local = process.env.LOCALAPPDATA;
    if (local) candidates.push({ command: path.join(local, "Apsi", "venv", "Scripts", "python.exe"), args: ["-m", "apsi"] });
    candidates.push({ command: "apsi.cmd", args: [] }, { command: "py", args: ["-3", "-m", "apsi"] }, { command: "python", args: ["-m", "apsi"] });
  } else {
    candidates.push({ command: path.join(process.env.HOME || "", ".apsi", "venv", "bin", "python"), args: ["-m", "apsi"] });
    candidates.push({ command: "apsi", args: [] }, { command: "python3", args: ["-m", "apsi"] });
  }
  for (const c of candidates) {
    if (path.isAbsolute(c.command) && !fs.existsSync(c.command)) continue;
    const version = probe(c);
    if (version) return (cachedApsi = { ...c, version });
  }
  return (cachedApsi = null);
}

function requireApsi() {
  const a = findApsi();
  if (!a) {
    vscode.window.showErrorMessage(
      "Apsi was not found. Install it with the Apsi Setup wizard, or set \"apsi.command\" in Settings.",
      "Open Settings").then((c) => c && vscode.commands.executeCommand("workbench.action.openSettings", "apsi.command"));
  }
  return a;
}

// ------------------------------------------------------------------ terminals
function quote(s) {
  if (s === "") return '""';
  if (!/[\s"'&|<>^%$`;()]/.test(s)) return s;
  if (IS_WIN) return `"${s.replace(/"/g, '\\"')}"`;
  return `'${s.replace(/'/g, "'\\''")}'`;
}

function commandLine(apsi, parts) {
  // PowerShell needs `&` to run a quoted program path
  const shell = (vscode.env.shell || "").toLowerCase();
  const prog = quote(apsi.command);
  const lead = IS_WIN && /powershell|pwsh/.test(shell) && prog.startsWith('"') ? "& " : "";
  return lead + [prog, ...apsi.args.map(quote), ...parts.map(quote)].join(" ");
}

function terminal(name, cwd) {
  let t = vscode.window.terminals.find((x) => x.name === name && x.exitStatus === undefined);
  if (!t) t = vscode.window.createTerminal({ name, cwd, iconPath: new vscode.ThemeIcon("play") });
  return t;
}

async function runInTerminal(parts, opts = {}) {
  const apsi = requireApsi();
  if (!apsi) return;
  const cfg = vscode.workspace.getConfiguration("apsi");
  const t = terminal(opts.name || "Apsi", opts.cwd);
  t.show(true);
  if (cfg.get("run.clearTerminal") && !opts.keep) await vscode.commands.executeCommand("workbench.action.terminal.clear");
  t.sendText(commandLine(apsi, parts));
}

async function activeFile(uri) {
  if (uri && uri.fsPath) return uri.fsPath;
  const ed = vscode.window.activeTextEditor;
  if (!ed) { vscode.window.showWarningMessage("Open an .apsi file first."); return null; }
  if (ed.document.isUntitled) {
    const ok = await ed.document.save();
    if (!ok) return null;
  } else if (vscode.workspace.getConfiguration("apsi").get("run.saveBeforeRun") && ed.document.isDirty) {
    await ed.document.save();
  }
  return ed.document.fileName;
}

function folderOf(file) {
  const ws = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(file));
  return ws ? ws.uri.fsPath : path.dirname(file);
}

function extraArgs() {
  return vscode.workspace.getConfiguration("apsi").get("run.arguments") || [];
}

// ------------------------------------------------------------------ REPL
let replTerminal = null;
function repl(show = true) {
  const apsi = requireApsi();
  if (!apsi) return null;
  if (!replTerminal || replTerminal.exitStatus !== undefined) {
    const ws = vscode.workspace.workspaceFolders;
    replTerminal = vscode.window.createTerminal({
      name: "Apsi REPL", cwd: ws && ws.length ? ws[0].uri.fsPath : undefined,
      ...(needsShell(apsi.command)
        ? { shellPath: process.env.ComSpec || "cmd.exe", shellArgs: ["/d", "/c", apsi.command, ...apsi.args, "repl"] }
        : { shellPath: apsi.command, shellArgs: [...apsi.args, "repl"] }),
      env: apsi.env, iconPath: new vscode.ThemeIcon("terminal"),
    });
  }
  if (show) replTerminal.show(true);
  return replTerminal;
}

function sendToRepl() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  let code;
  if (ed.selection.isEmpty) {
    const line = ed.document.lineAt(ed.selection.active.line);
    code = line.text;
    // move to the next non-empty line, like other REPL integrations
    let next = line.lineNumber + 1;
    while (next < ed.document.lineCount && !ed.document.lineAt(next).text.trim()) next++;
    if (next < ed.document.lineCount) {
      const p = new vscode.Position(next, 0);
      ed.selection = new vscode.Selection(p, p);
    }
  } else {
    code = ed.document.getText(ed.selection);
  }
  if (!code.trim()) return;
  const t = repl(true);
  if (!t) return;
  // dedent the block so it runs at top level, and end blocks with an empty line
  const lines = code.replace(/\r\n/g, "\n").split("\n");
  const indent = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^\s*/)[0].length));
  const body = lines.map((l) => l.slice(indent)).join("\n");
  const multi = body.includes("\n") || /:\s*(#.*)?$/.test(body);
  if (multi) {
    t.sendText(":paste", true);
    t.sendText(body, true);
    t.sendText(":end", true);
  } else {
    t.sendText(body, true);
  }
}

// ------------------------------------------------------------------ apsi.toml tasks
function readTasks(root) {
  const file = path.join(root, "apsi.toml");
  if (!fs.existsSync(file)) return {};
  const out = {};
  let inTasks = false;
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].replace(/^﻿/, "").trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("[")) { inTasks = /^\[\s*tasks\s*\]/.test(line); continue; }
    if (!inTasks) continue;
    const m = /^"?([\w.\-]+)"?\s*=\s*(.*)$/.exec(line);
    if (!m) continue;
    let raw = m[2];
    while (raw.startsWith("[") && !raw.includes("]") && i + 1 < lines.length) raw += " " + lines[++i].trim();
    const strings = [...raw.matchAll(/"((?:[^"\\]|\\.)*)"|'([^']*)'/g)].map((x) => x[1] !== undefined ? x[1] : x[2]);
    if (strings.length) out[m[1]] = strings;
  }
  return out;
}

function projectRoots() {
  return (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath)
    .filter((p) => fs.existsSync(path.join(p, "apsi.toml")));
}

async function pickTask() {
  const roots = projectRoots();
  const items = [];
  for (const r of roots) {
    for (const [name, steps] of Object.entries(readTasks(r))) {
      items.push({ label: name, description: steps.join("  →  "), detail: roots.length > 1 ? r : undefined, root: r });
    }
  }
  if (!items.length) {
    vscode.window.showInformationMessage("No apsi.toml [tasks] found in this workspace. Create a project with \"Apsi: New Project…\".");
    return;
  }
  const pick = await vscode.window.showQuickPick(items, { placeHolder: "Apsi task to run" });
  if (pick) runInTerminal(["task", pick.label], { cwd: pick.root });
}

function taskExecution(apsi, parts, cwd) {
  const opts = { cwd, env: apsi.env };
  if (needsShell(apsi.command)) return new vscode.ShellExecution(apsi.command, [...apsi.args, ...parts], opts);
  return new vscode.ProcessExecution(apsi.command, [...apsi.args, ...parts], opts);
}

class TaskProvider {
  provideTasks() {
    const apsi = findApsi();
    if (!apsi) return [];
    const tasks = [];
    const make = (def, label, folder, parts, group) => {
      const exec = taskExecution(apsi, parts, folder.uri.fsPath);
      const t = new vscode.Task(def, folder, label, "apsi", exec, "$apsi");
      if (group) t.group = group;
      return t;
    };
    for (const folder of vscode.workspace.workspaceFolders || []) {
      tasks.push(make({ type: "apsi", command: "test" }, "test", folder, ["test"], vscode.TaskGroup.Test));
      tasks.push(make({ type: "apsi", command: "check" }, "check", folder, ["check", "-q"], vscode.TaskGroup.Build));
      tasks.push(make({ type: "apsi", command: "fmt" }, "format", folder, ["fmt"]));
      if (fs.existsSync(path.join(folder.uri.fsPath, "apsi.toml"))) {
        tasks.push(make({ type: "apsi", command: "run" }, "run project", folder, ["run"], vscode.TaskGroup.Build));
        for (const name of Object.keys(readTasks(folder.uri.fsPath))) {
          tasks.push(make({ type: "apsi", command: "task", task: name }, `task: ${name}`, folder, ["task", name]));
        }
      }
    }
    return tasks;
  }

  resolveTask(task) {
    const d = task.definition;
    const apsi = findApsi();
    if (!apsi || !d.command) return undefined;
    const parts = [d.command];
    if (d.command === "task" && d.task) parts.push(d.task);
    if (d.file) parts.push(d.file);
    if (Array.isArray(d.args)) parts.push(...d.args);
    const folder = task.scope && task.scope.uri ? task.scope.uri.fsPath : undefined;
    return new vscode.Task(d, task.scope || vscode.TaskScope.Workspace, task.name, "apsi",
      taskExecution(apsi, parts, folder), "$apsi");
  }
}

// ------------------------------------------------------------------ other commands
async function newProject() {
  const name = await vscode.window.showInputBox({
    prompt: "Project name (a folder with this name is created)", placeHolder: "my-project",
    validateInput: (v) => (/^[\w.\-]+$/.test(v) ? null : "Use letters, digits, '-', '_' or '.'"),
  });
  if (!name) return;
  const template = await vscode.window.showQuickPick([
    { label: "app", description: "a program with a main file and tests" },
    { label: "lib", description: "a reusable library" },
    { label: "physics", description: "a simulation with units and evolve" },
    { label: "ai", description: "a neural network trained with autodiff" },
    { label: "server", description: "a JSON HTTP API (use web)" },
  ], { placeHolder: "Template" });
  if (!template) return;
  const parent = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, openLabel: "Create here" });
  if (!parent || !parent.length) return;
  const apsi = requireApsi();
  if (!apsi) return;
  const r = runApsi(apsi, ["new", name, "--template", template.label], { cwd: parent[0].fsPath });
  if (r.status !== 0) {
    vscode.window.showErrorMessage(`apsi new failed: ${(r.stderr || r.stdout || (r.error && r.error.message) || "").trim()}`);
    return;
  }
  const dir = vscode.Uri.file(path.join(parent[0].fsPath, name));
  const open = await vscode.window.showInformationMessage(`Created ${name}.`, "Open Folder", "Open in New Window");
  if (open) vscode.commands.executeCommand("vscode.openFolder", dir, open === "Open in New Window");
}

async function showDoc() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  const range = ed.document.getWordRangeAtPosition(ed.selection.active, /[\p{L}_][\p{L}\p{N}_]*(\.[\p{L}_][\p{L}\p{N}_]*)*/u);
  const word = range ? ed.document.getText(range) : "";
  const name = await vscode.window.showInputBox({ prompt: "Apsi built-in or module (e.g. sqrt, nn, nn.adam)", value: word });
  if (!name) return;
  const apsi = requireApsi();
  if (!apsi) return;
  const r = runApsi(apsi, ["doc", name]);
  const ch = getOutput();
  ch.clear();
  ch.appendLine((r.stdout || "") + (r.stderr || "") + (r.error ? `Apsi could not be started: ${r.error.message}` : ""));
  ch.show(true);
}

async function convertPython(uri) {
  let target = uri && uri.fsPath;
  if (!target) {
    const ed = vscode.window.activeTextEditor;
    if (!ed || !/\.py$/i.test(ed.document.fileName)) { vscode.window.showWarningMessage("Open a Python (.py) file first."); return; }
    if (ed.document.isDirty) await ed.document.save();
    target = ed.document.fileName;
  }
  const apsi = requireApsi();
  if (!apsi) return;
  const ch = getOutput();
  // how much to convert: a file and the modules it imports (default), the file alone, or the whole project
  const isFolder = fs.existsSync(target) && fs.statSync(target).isDirectory();
  let extra = [];
  if (!isFolder) {
    const pick = await vscode.window.showQuickPick([
      { label: "$(file-symlink-file) This file and what it imports", description: "recommended", args: [] },
      { label: "$(file) This file only", description: "its own modules stay Python", args: ["--only"] },
      { label: "$(folder-library) The whole project", description: "every .py of the project (pyproject.toml / .git)", args: ["--project"] },
      { label: "$(list-unordered) Show what would be converted", description: "writes nothing", args: ["--plan"] },
    ], { placeHolder: "Convert to Apsi: how much?" });
    if (!pick) return;
    extra = pick.args;
  }
  // the conversion runs in the background (the editor stays responsive); the result message is shown after the
  // progress note has closed, so "Converting to Apsi…" never stays on screen
  const r = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Converting to Apsi…" },
    () => new Promise((resolve) => {
      const inv = invocation(apsi, ["convert", target, ...extra]);
      let stdout = "", stderr = "";
      let child;
      try {
        child = cp.spawn(inv.command, inv.args, { windowsHide: true, shell: inv.shell, env: apsiEnv(apsi, { NO_COLOR: "1" }) });
      } catch (e) {
        resolve({ status: null, stdout, stderr, error: e });
        return;
      }
      child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
      child.stdout.on("data", (d) => { stdout += d; });
      child.stderr.on("data", (d) => { stderr += d; });
      child.on("error", (e) => resolve({ status: null, stdout, stderr, error: e }));
      child.on("close", (code) => resolve({ status: code, stdout, stderr }));
    }));
  ch.clear();
  ch.appendLine(`> ${[apsi.command, ...apsi.args, "convert", target, ...extra].join(" ")}`);
  if (extra.includes("--plan")) {
    if (r.stdout) ch.appendLine(r.stdout.trimEnd());
    if (r.stderr) ch.appendLine(r.stderr.trimEnd());
    ch.show(true);
    return;
  }
  if (r.stdout) ch.appendLine(r.stdout.trimEnd());
  if (r.stderr) ch.appendLine(r.stderr.trimEnd());
  if (r.error) ch.appendLine(`Apsi could not be started: ${r.error.message}`);
  const isDir = isFolder || extra.includes("--project");
  const out = isDir ? null : target.replace(/\.py$/i, ".apsi");
  if (r.status === 0 && out && fs.existsSync(out)) {
    const doc = await vscode.workspace.openTextDocument(out);
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
  }
  const notes = ((r.stdout || "").match(/\((\d+) notes?/) || [])[1];
  if (r.status === 0) {
    vscode.window.showInformationMessage(`Converted to Apsi${notes && notes !== "0" ? ` — ${notes} place(s) to review` : ""}.`, "Show Details")
      .then((pick) => pick && ch.show(true));
  } else {
    const why = r.error ? "Apsi could not be started" : "the converter reported problems";
    vscode.window.showErrorMessage(`Not converted: ${why}. See the Apsi output for the details.`, "Show Details")
      .then((pick) => pick && ch.show(true));
  }
}

async function customCommand() {
  const value = await vscode.window.showInputBox({
    prompt: "Apsi command line (without the word apsi)", placeHolder: "run app.apsi --allow-net   ·   pkg list   ·   doc nn",
  });
  if (!value) return;
  const parts = value.match(/"[^"]*"|'[^']*'|\S+/g).map((s) => s.replace(/^["']|["']$/g, ""));
  const ws = vscode.workspace.workspaceFolders;
  runInTerminal(parts, { cwd: ws && ws.length ? ws[0].uri.fsPath : undefined });
}

function getOutput() {
  if (!output) output = vscode.window.createOutputChannel("Apsi");
  return output;
}

// ------------------------------------------------------------------ language server
async function startServer(context) {
  if (!vscode.workspace.getConfiguration("apsi").get("languageServer.enabled")) return;
  let LanguageClient;
  try {
    ({ LanguageClient } = require("vscode-languageclient/node"));
  } catch (e) {
    getOutput().appendLine("vscode-languageclient is missing; diagnostics are off.");
    return;
  }
  const apsi = findApsi();
  if (!apsi) {
    vscode.window.showWarningMessage("Apsi was not found — syntax colours and snippets only. Install Apsi or set 'apsi.command'.");
    return;
  }
  const inv = invocation(apsi, ["lsp"]);
  const serverOptions = {
    command: inv.command, args: inv.args,
    options: { shell: inv.shell, env: apsiEnv(apsi, { PYTHONDONTWRITEBYTECODE: "1" }) },
  };
  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "apsi" }, { scheme: "untitled", language: "apsi" }],
    outputChannel: getOutput(),
  };
  client = new LanguageClient("apsi", "Apsi Language Server", serverOptions, clientOptions);
  try {
    await client.start();
  } catch (e) {
    const msg = `The Apsi language server did not start (${[apsi.command, ...apsi.args, "lsp"].join(" ")}): ${e && e.message}`;
    getOutput().appendLine(msg);
    vscode.window.showErrorMessage("The Apsi language server did not start. See the Apsi output; \"Apsi: Restart Language Server\" tries again.", "Show Details")
      .then((pick) => pick && getOutput().show(true));
  }
}

async function restartServer(context) {
  cachedApsi = undefined;
  if (client) { try { await client.stop(); } catch (e) { /* already stopped */ } client = null; }
  await startServer(context);
  updateStatus();
  vscode.window.setStatusBarMessage("Apsi language server restarted", 3000);
}

function updateStatus() {
  if (!statusItem) return;
  const ed = vscode.window.activeTextEditor;
  const show = vscode.workspace.getConfiguration("apsi").get("statusBar.enabled") && ed && ed.document.languageId === "apsi";
  if (!show) { statusItem.hide(); return; }
  const apsi = findApsi();
  statusItem.text = apsi ? `$(play) Apsi ${apsi.version}` : "$(warning) Apsi not found";
  statusItem.tooltip = apsi ? `Run this file (${[apsi.command, ...apsi.args].join(" ")})` : "Set apsi.command in Settings";
  statusItem.command = apsi ? "apsi.run" : { command: "workbench.action.openSettings", arguments: ["apsi.command"], title: "Settings" };
  statusItem.show();
}

// ------------------------------------------------------------------ activation
function activate(context) {
  const reg = (id, fn) => context.subscriptions.push(vscode.commands.registerCommand(id, fn));

  // F5: the Apsi debugger speaks the Debug Adapter Protocol (`apsi dap`)
  context.subscriptions.push(vscode.debug.registerDebugAdapterDescriptorFactory("apsi", {
    createDebugAdapterDescriptor() {
      const apsi = requireApsi();
      if (!apsi) return undefined;
      if (needsShell(apsi.command)) {
        return new vscode.DebugAdapterExecutable(process.env.ComSpec || "cmd.exe", ["/d", "/c", apsi.command, ...apsi.args, "dap"], { env: apsiEnv(apsi) });
      }
      return new vscode.DebugAdapterExecutable(apsi.command, [...apsi.args, "dap"], { env: apsiEnv(apsi) });
    },
  }));
  context.subscriptions.push(vscode.debug.registerDebugConfigurationProvider("apsi", {
    resolveDebugConfiguration(folder, config) {
      if (!config.type && !config.request && !config.name) {
        const ed = vscode.window.activeTextEditor;
        if (ed && ed.document.languageId === "apsi") {
          config.type = "apsi"; config.name = "Debug Apsi file"; config.request = "launch"; config.program = "${file}";
        }
      }
      if (!config.program) {
        return vscode.window.showInformationMessage("Open an .apsi file to debug it.").then(() => undefined);
      }
      return config;
    },
  }));

  reg("apsi.run", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["run", f, ...extraArgs()], { cwd: folderOf(f) });
  });
  reg("apsi.runWithArgs", async (uri) => {
    const f = await activeFile(uri);
    if (!f) return;
    const key = `apsi.args:${f}`;
    const value = await vscode.window.showInputBox({
      prompt: "Program arguments and flags (read arguments with args())",
      value: context.workspaceState.get(key, ""), placeHolder: "input.csv --allow-read",
    });
    if (value === undefined) return;
    await context.workspaceState.update(key, value);
    const parts = (value.match(/"[^"]*"|'[^']*'|\S+/g) || []).map((s) => s.replace(/^["']|["']$/g, ""));
    runInTerminal(["run", f, ...parts, ...extraArgs()], { cwd: folderOf(f) });
  });
  reg("apsi.runProject", () => {
    const roots = projectRoots();
    if (!roots.length) { vscode.window.showWarningMessage("No apsi.toml in this workspace."); return; }
    runInTerminal(["run", ...extraArgs()], { cwd: roots[0] });
  });
  reg("apsi.test", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["test", f, ...extraArgs()], { cwd: folderOf(f) });
  });
  reg("apsi.testAll", () => {
    const ws = vscode.workspace.workspaceFolders;
    runInTerminal(["test", ...extraArgs()], { cwd: ws && ws.length ? ws[0].uri.fsPath : undefined });
  });
  reg("apsi.check", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["check", f], { cwd: folderOf(f) });
  });
  reg("apsi.watch", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["watch", f, ...(extraArgs().length ? ["--", ...extraArgs()] : [])], { name: "Apsi Watch", cwd: folderOf(f), keep: true });
  });
  reg("apsi.bench", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["bench", f, ...extraArgs()], { cwd: folderOf(f) });
  });
  reg("apsi.audit", async (uri) => {
    const f = await activeFile(uri);
    if (f) runInTerminal(["audit", f], { cwd: folderOf(f) });
  });
  reg("apsi.repl", () => repl(true));
  reg("apsi.sendToRepl", sendToRepl);
  reg("apsi.loadInRepl", async () => {
    const f = await activeFile();
    const t = f && repl(true);
    if (t) t.sendText(`:load ${quote(f)}`, true);
  });
  reg("apsi.task", pickTask);
  reg("apsi.newProject", newProject);
  reg("apsi.doc", showDoc);
  reg("apsi.custom", customCommand);
  reg("apsi.convertPython", convertPython);
  reg("apsi.restartServer", () => restartServer(context));
  reg("apsi.showOutput", () => getOutput().show(true));

  context.subscriptions.push(vscode.tasks.registerTaskProvider("apsi", new TaskProvider()));

  statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  context.subscriptions.push(statusItem);
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(updateStatus));
  context.subscriptions.push(vscode.window.onDidCloseTerminal((t) => { if (t === replTerminal) replTerminal = null; }));
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
    if (e.affectsConfiguration("apsi.command") || e.affectsConfiguration("apsi.languageServer")) restartServer(context);
    else if (e.affectsConfiguration("apsi.statusBar")) updateStatus();
  }));

  // locating Apsi spawns a process: do it after the editor is up
  setTimeout(() => { updateStatus(); startServer(context); }, 0);
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
