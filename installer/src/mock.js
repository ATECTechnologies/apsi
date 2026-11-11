// Fake system API so the UI can be previewed in a browser without Electron.
export function mockSetup() {
  const sys = {
    platform: "win32", arch: "x64", version: "2.0.0",
    payload: { version: "2.0.0", components: { runtime: 97e6, examples: 120e3, docs: 160e3, editors: 520e3 } },
    osName: "Windows 11 (build 22631)", osOk: true, cpu: "Intel(R) Core(TM) i7-12700H", cpuOk: true,
    threads: 20, ramGB: 32, gpus: [{ name: "NVIDIA GeForce RTX 3060 Laptop GPU", compute_capability: 8.6, meets_target: true }],
    gpuOk: true, admin: false, editors: [{ id: "vscode", name: "Visual Studio Code", kind: "extension", found: true }, { id: "cursor", name: "Cursor", kind: "extension", found: true }, { id: "sublime", name: "Sublime Text", kind: "package", found: false }],
    existing: null, installed: { root: null, versions: [] }, defaultDir: "C:\\Users\\you\\AppData\\Local\\Programs\\Apsi", diskFree: 412e9, requiredBytes: 98e6,
    doctor: { simd: ["AVX2", "FMA3", "AVX512F"] },
  };
  let progress = () => {};
  let logger = () => {};
  return {
    detect: async () => { await new Promise((r) => setTimeout(r, 600)); return sys; },
    defaults: async (s) => ({
      mode: "install", dir: s.defaultDir, allUsers: false,
      components: { runtime: true, examples: true, docs: true, editors: true, addToPath: true, fileAssociation: true, startMenu: true, desktopShortcut: false },
      packages: { gpu: false, onnx: false, torch: false, transformers: false },
      security: { profile: "standard", verify: true, scanOnRun: true, auditLog: true, guardLimit: 150 },
    }),
    startMode: async () => "normal",
    platform: async () => "win32",
    pickDir: async (d) => d,
    validateDir: async (d) => {
      const problems = [];
      const warnings = [];
      if (!d.trim()) problems.push("Choose a folder.");
      if (/^[A-Za-z]:\\?$/.test(d)) problems.push("Do not install into the root of a drive.");
      if (/\\Windows(\\|$)/i.test(d)) problems.push("System folders such as C:\\Windows cannot be used.");
      if (/Program Files/i.test(d)) warnings.push("Program Files needs administrator rights; the per-user default is recommended.");
      if (/onedrive/i.test(d)) warnings.push("Synced folders (OneDrive, Dropbox) slow Apsi down and may lock files.");
      return { ok: !problems.length, problems, warnings, dir: d };
    },
    install: async () => {
      const steps = ["Verifying installer integrity", "Preparing", "Copying the Apsi runtime", "Creating the apsi command", "Adding Apsi to PATH", "Registering .apsi files", "Verifying the installation"];
      for (let i = 0; i <= 100; i += 4) {
        await new Promise((r) => setTimeout(r, 60));
        progress({ percent: i, label: steps[Math.min(steps.length - 1, Math.floor(i / 18))] });
        if (i % 20 === 0) logger(`step ${i}%`);
      }
      return { ok: true, dir: sys.defaultDir, log: [] };
    },
    uninstall: async () => ({ ok: true }),
    open: async () => true,
    link: async () => true,
    onProgress: (cb) => { progress = cb; return () => {}; },
    onLog: (cb) => { logger = cb; return () => {}; },
    minimize: () => {},
    close: () => {},
  };
}
