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
function probe(cmd, args) {
  try {
    const r = cp.spawnSync(cmd, [...args, "version"], {
      timeout: 15000, encoding: "utf8", windowsHide: true, shell: IS_WIN && !path.isAbsolute(cmd) && /\.cmd$/i.test(cmd),
    });
    const m = r.status === 0 && /Apsi\s+([\w.\-]+)/.exec(r.stdout || "");
    return m ? m[1] : null;
  } catch (e) {
    return null;
  }
}

function findApsi(force) {
  if (cachedApsi !== undefined && !force) return cachedApsi;
  const cfg = vscode.workspace.getConfiguration("apsi").get("command") || [];
  const candidates = [];
  if (cfg.length) candidates.push([cfg[0], cfg.slice(1)]);
  if (IS_WIN) {
    const local = process.env.LOCALAPPDATA;
    if (local) {
      candidates.push([path.join(local, "Programs", "Apsi", "runtime", "python", "python.exe"), ["-m", "apsi"]]);
      candidates.push([path.join(local, "Apsi", "runtime", "python", "python.exe"), ["-m", "apsi"]]);
      candidates.push([path.join(local, "Apsi", "venv", "Scripts", "python.exe"), ["-m", "apsi"]]);
    }
    candidates.push(["apsi.cmd", []], ["py", ["-3", "-m", "apsi"]], ["python", ["-m", "apsi"]]);
  } else {
    const home = process.env.HOME || "";
    candidates.push([path.join(home, ".local", "share", "apsi", "runtime", "python", "bin", "python3"), ["-m", "apsi"]]);
    candidates.push([path.join(home, ".apsi", "venv", "bin", "python"), ["-m", "apsi"]]);
    candidates.push(["apsi", []], ["python3", ["-m", "apsi"]]);
  }
  for (const [command, args] of candidates) {
    if (path.isAbsolute(command) && !fs.existsSync(command)) continue;
    const version = probe(command, args);
    if (version) return (cachedApsi = { command, args, version });
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
      shellPath: apsi.command, shellArgs: [...apsi.args, "repl"], iconPath: new vscode.ThemeIcon("terminal"),
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

class TaskProvider {
  provideTasks() {
    const apsi = findApsi();
    if (!apsi) return [];
    const tasks = [];
    const make = (def, label, folder, parts, group) => {
      const exec = new vscode.ProcessExecution(apsi.command, [...apsi.args, ...parts], { cwd: folder.uri.fsPath });
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
      new vscode.ProcessExecution(apsi.command, [...apsi.args, ...parts], { cwd: folder }), "$apsi");
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
  const r = cp.spawnSync(apsi.command, [...apsi.args, "new", name, "--template", template.label],
    { cwd: parent[0].fsPath, encoding: "utf8", windowsHide: true });
  if (r.status !== 0) {
    vscode.window.showErrorMessage(`apsi new failed: ${(r.stderr || r.stdout || "").trim()}`);
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
  const r = cp.spawnSync(apsi.command, [...apsi.args, "doc", name], { encoding: "utf8", windowsHide: true });
  const ch = getOutput();
  ch.clear();
  ch.appendLine((r.stdout || "") + (r.stderr || ""));
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
  await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Converting to Apsi…" }, async () => {
    const r = cp.spawnSync(apsi.command, [...apsi.args, "convert", target],
      { encoding: "utf8", windowsHide: true, env: { ...process.env, NO_COLOR: "1", PYTHONUTF8: "1" } });
    ch.clear();
    ch.appendLine((r.stdout || "") + (r.stderr || ""));
    const isDir = fs.existsSync(target) && fs.statSync(target).isDirectory();
    const out = isDir ? null : target.replace(/\.py$/i, ".apsi");
    if (out && fs.existsSync(out)) {
      const doc = await vscode.workspace.openTextDocument(out);
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    }
    const notes = ((r.stdout || "").match(/\((\d+) notes?/) || [])[1];
    const msg = r.status === 0 ? `Converted to Apsi${notes && notes !== "0" ? ` — ${notes} place(s) to review` : ""}.`
                               : "Converted with problems — see the Apsi output.";
    const pick = await vscode.window.showInformationMessage(msg, "Show Details");
    if (pick) ch.show(true);
  });
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
  const serverOptions = {
    command: apsi.command, args: [...apsi.args, "lsp"],
    options: { env: { ...process.env, PYTHONUTF8: "1", PYTHONDONTWRITEBYTECODE: "1" } },
  };
  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "apsi" }, { scheme: "untitled", language: "apsi" }],
    outputChannel: getOutput(),
  };
  client = new LanguageClient("apsi", "Apsi Language Server", serverOptions, clientOptions);
  try {
    await client.start();
  } catch (e) {
    getOutput().appendLine(`language server failed to start: ${e && e.message}`);
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
      return new vscode.DebugAdapterExecutable(apsi.command, [...apsi.args, "dap"]);
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
