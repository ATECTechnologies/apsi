export function bytes(n) {
  if (n == null) return "unknown";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export const PROFILES = [
  { id: "sandbox", title: "Sandbox", tag: "Most secure",
    text: "Programs cannot read or write files, use the network, environment variables, Python or native code unless you pass --allow-… flags on each run." },
  { id: "strict", title: "Strict", tag: "Recommended for servers",
    text: "Programs may read and write files only inside their own folder. Everything else needs an explicit flag." },
  { id: "standard", title: "Standard", tag: "Recommended",
    text: "Programs may read and write files. Network, environment variables, Python and native libraries still need an explicit flag." },
  { id: "developer", title: "Developer", tag: "Least restrictive",
    text: "Everything is allowed, the same as running every program with --allow-all. Use only on a personal machine." },
];

export const COMPONENTS = [
  { id: "runtime", title: "Apsi runtime", text: "Compiler, standard library, Python 3.12 and numpy (required).", size: "runtime", required: true },
  { id: "examples", title: "Example programs", text: "27 examples: physics, AI, data pipelines, web servers.", size: "examples" },
  { id: "docs", title: "Documentation", text: "Language reference, standard library, guides.", size: "docs" },
  { id: "editors", title: "Code editor support", size: "editors", needsEditor: true,
    text: "Apsi colours, live unit/shape errors, run buttons and snippets in VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim and Notepad++." },
];

export const INTEGRATION = [
  { id: "addToPath", title: "Add apsi to PATH", text: "Use the apsi command from any terminal." },
  { id: "fileAssociation", title: "Associate .apsi files", text: "Double-click to run, right-click to edit or test, with the Apsi file icon." },
  { id: "startMenu", title: "Start menu / app launcher entries", text: "Apsi REPL, examples, documentation and uninstall." },
  { id: "desktopShortcut", title: "Desktop shortcut", text: "A shortcut to the Apsi REPL on your desktop." },
  { id: "gpuPower", title: "Full GPU speed without administrator rights", needsGPU: true,
    text: "Installs Apsi's small GPU helper (Windows asks once) and measures this card for about 2.5 minutes: the power it trains at and the fan range that holds it at 78-79 °C, never 80, at full and steady speed. 'apsi gpu calibrate' measures again; 'apsi gpu remove' takes it all away." },
];

export const PACKAGES = [
  { id: "gpu", title: "NVIDIA GPU acceleration", pkg: "cupy-cuda12x", size: "≈ 400 MB", text: "Run tensors on GPUs with compute capability 7.5+ (GTX 1660 and newer).", needsGPU: true },
  { id: "onnx", title: "ONNX Runtime", pkg: "onnxruntime", size: "≈ 20 MB", text: "Run exported models from PyTorch, TensorFlow and scikit-learn." },
  { id: "torch", title: "PyTorch", pkg: "torch", size: "≈ 200 MB – 2.5 GB", text: "Train use-layers models on the GPU (fit backend=\"torch\"), load checkpoints. The right build is picked for you: CUDA for NVIDIA (by driver version), ROCm for AMD, otherwise CPU." },
  { id: "transformers", title: "Hugging Face Transformers", pkg: "transformers", size: "≈ 15 MB + models", text: "Use pretrained language, vision and audio models with hf_pipeline." },
];
