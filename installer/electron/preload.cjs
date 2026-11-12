// The only bridge between the UI and the system: a small, explicit API.
"use strict";
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("setup", Object.freeze({
  detect: () => ipcRenderer.invoke("system:detect"),
  defaults: (sys) => ipcRenderer.invoke("options:default", sys),
  startMode: () => ipcRenderer.invoke("app:startMode"),
  platform: () => ipcRenderer.invoke("app:platform"),
  pickDir: (current) => ipcRenderer.invoke("dialog:pickDir", current),
  validateDir: (dir) => ipcRenderer.invoke("dir:validate", dir),
  install: (options) => ipcRenderer.invoke("install:start", options),
  uninstall: () => ipcRenderer.invoke("uninstall:start"),
  open: (what) => ipcRenderer.invoke("app:open", what),
  link: (url) => ipcRenderer.invoke("link:open", url),
  onProgress: (cb) => {
    const h = (_e, p) => cb(p);
    ipcRenderer.on("install:progress", h);
    return () => ipcRenderer.removeListener("install:progress", h);
  },
  onLog: (cb) => {
    const h = (_e, m) => cb(m);
    ipcRenderer.on("install:log", h);
    return () => ipcRenderer.removeListener("install:log", h);
  },
  minimize: () => ipcRenderer.send("window:minimize"),
  close: () => ipcRenderer.send("window:close"),
}));
