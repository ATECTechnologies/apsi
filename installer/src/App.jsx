import React, { useEffect, useMemo, useRef, useState } from "react";
import { bytes, PROFILES, COMPONENTS, INTEGRATION, PACKAGES } from "./format.js";

const setup = () => window.setup;

const FLOW = [
  { id: "welcome", title: "Welcome" },
  { id: "system", title: "System check" },
  { id: "location", title: "Location" },
  { id: "components", title: "Components" },
  { id: "packages", title: "Optional packages" },
  { id: "security", title: "Security" },
  { id: "review", title: "Review" },
  { id: "install", title: "Install" },
  { id: "finish", title: "Finish" },
];

// ------------------------------------------------------------------ small UI pieces
function Toggle({ checked, disabled, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled}
      className={`toggle ${checked ? "on" : ""}`} onClick={() => !disabled && onChange(!checked)}>
      <span />
    </button>
  );
}

function OptionRow({ title, text, meta, checked, disabled, onChange, badge }) {
  return (
    <label className={`option ${disabled ? "disabled" : ""}`}>
      <div className="option-text">
        <div className="option-title">{title} {badge && <span className="badge">{badge}</span>}</div>
        <div className="option-desc">{text}</div>
      </div>
      {meta && <div className="option-meta">{meta}</div>}
      <Toggle checked={checked} disabled={disabled} onChange={onChange} label={title} />
    </label>
  );
}

function Check({ ok, warn, label, value }) {
  const cls = ok ? "ok" : warn ? "warn" : "bad";
  const icon = ok ? "✓" : warn ? "!" : "✕";
  return (
    <div className={`check ${cls}`}>
      <span className="check-icon">{icon}</span>
      <div><div className="check-label">{label}</div><div className="check-value">{value}</div></div>
    </div>
  );
}

// ------------------------------------------------------------------ app
export default function App() {
  const [sys, setSys] = useState(null);
  const [error, setError] = useState(null);
  const [opts, setOpts] = useState(null);
  const [step, setStep] = useState("welcome");
  const [maintenance, setMaintenance] = useState(null);   // null | "modify" | "repair" | "uninstall"
  const [installType, setInstallType] = useState("express"); // express | custom
  const [dirCheck, setDirCheck] = useState({ ok: true, problems: [], warnings: [] });
  const [progress, setProgress] = useState({ percent: 0, label: "" });
  const [log, setLog] = useState([]);
  const [result, setResult] = useState(null);
  const [showLog, setShowLog] = useState(false);
  const [finishActions, setFinishActions] = useState({ repl: true, examples: false, docs: false });
  const logRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const s = await setup().detect();
        setSys(s);
        setOpts(await setup().defaults(s));
        if ((await setup().startMode()) === "uninstall" && s.existing) {
          setMaintenance("uninstall");
          setStep("review");
        }
      } catch (e) {
        setError(String(e.message || e));
      }
    })();
    const offP = setup().onProgress((p) => setProgress(p));
    const offL = setup().onLog((m) => setLog((l) => [...l, m]));
    return () => { offP(); offL(); };
  }, []);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log, showLog]);
  const bodyRef = useRef(null);
  useEffect(() => { if (bodyRef.current) bodyRef.current.scrollTop = 0; }, [step]);   // every page starts at its top

  const isWin = sys && sys.platform === "win32";
  const editorNames = ((sys && sys.editors) || []).filter((e) => e.found && e.kind !== "manual").map((e) => e.name);
  const flow = useMemo(() => {
    if (maintenance === "uninstall") return FLOW.filter((s) => ["welcome", "review", "install", "finish"].includes(s.id));
    if (maintenance === "repair") return FLOW.filter((s) => ["welcome", "review", "install", "finish"].includes(s.id));
    if (!maintenance && installType === "express") return FLOW.filter((s) => ["welcome", "system", "review", "install", "finish"].includes(s.id));
    return FLOW;
  }, [maintenance, installType]);

  // validate the folder as the user types (debounced); the main process validates again before installing
  useEffect(() => {
    if (!opts) return undefined;
    const t = setTimeout(async () => {
      const v = setup().validateDir ? await setup().validateDir(opts.dir) : { ok: true, problems: [], warnings: [] };
      const upgrading = sys && sys.existing && (sys.existing.root || sys.existing.dir) === opts.dir;
      setDirCheck(upgrading ? { ok: true, problems: [], warnings: v.warnings || [] } : v);
    }, 250);
    return () => clearTimeout(t);
  }, [opts && opts.dir]);
  const index = flow.findIndex((s) => s.id === step);

  const set = (path, value) => setOpts((o) => {
    const n = structuredClone(o);
    const keys = path.split(".");
    let t = n;
    for (const k of keys.slice(0, -1)) t = t[k];
    t[keys[keys.length - 1]] = value;
    return n;
  });

  const sizeOf = (key) => (sys ? sys.payload.components[key] || 0 : 0);
  const selectedSize = useMemo(() => {
    if (!opts) return 0;
    return COMPONENTS.filter((c) => opts.components[c.id]).reduce((a, c) => a + sizeOf(c.size), 0);
  }, [opts, sys]);

  const blockers = useMemo(() => {
    const b = [];
    if (!sys) return b;
    if (!sys.osOk) b.push(isWin ? "Apsi requires Windows 10 or newer (64-bit)." : "Apsi requires Linux kernel 4.15 or newer.");
    if (sys.diskFree != null && sys.diskFree < sys.requiredBytes * 1.2) b.push("Not enough free disk space.");
    return b;
  }, [sys]);

  async function start() {
    setStep("install");
    setLog([]);
    setProgress({ percent: 0, label: "Starting…" });
    let r;
    if (maintenance === "uninstall") {
      r = await setup().uninstall();
      setProgress({ percent: 100, label: "Done" });
    } else {
      const o = structuredClone(opts);
      if (maintenance === "repair") o.mode = "repair";
      else if (maintenance === "modify") o.mode = "upgrade";
      r = await setup().install(o);
    }
    setResult(r);
    if (!r.ok) setShowLog(true);
    setStep("finish");
  }

  async function finish() {
    if (result && result.ok && maintenance !== "uninstall") {
      for (const [k, v] of Object.entries(finishActions)) if (v) await setup().open(k);
    }
    setup().close();
  }

  function next() {
    if (step === "review") return start();
    if (step === "finish") return finish();
    setStep(flow[index + 1].id);
  }

  // ------------------------------------------------------------------ pages
  function Welcome() {
    if (sys && sys.existing && !maintenance) {
      return (
        <div className="page">
          <h1>Apsi {sys.existing.version} is already installed</h1>
          <p className="lead">Installed in <code>{sys.existing.dir}</code>. Choose what Setup should do.
            {sys.existing.others && sys.existing.others.length > 0 &&
              <> Other versions ({sys.existing.others.join(", ")}) are not changed.</>}</p>
          <div className="cards">
            {[
              ["modify", "Modify or upgrade", `Change components and settings, and update to ${sys.version}.`],
              ["repair", "Repair", "Reinstall the runtime with your current settings and verify it."],
              ["uninstall", "Uninstall", `Remove Apsi ${sys.existing.version} (other installed versions stay).`],
            ].map(([id, t, d]) => (
              <button key={id} className="card" onClick={() => { setMaintenance(id); setStep(id === "modify" ? "system" : "review"); }}>
                <div className="card-title">{t}</div><div className="card-text">{d}</div>
              </button>
            ))}
          </div>
        </div>
      );
    }
    return (
      <div className="page welcome">
        <img className="hero-mark" src="./app-icon.png" alt="" />
        <h1>Welcome to Apsi {sys ? sys.version : ""}</h1>
        <p className="lead">
          A programming language for AI, physics and servers — with physical units, tensor shape checking,
          automatic differentiation, FP64 to FP4 numeric formats and a hardened web server built in.
        </p>
        <ul className="bullets">
          <li>Self-contained: Python 3.12 and numpy are bundled and private to Apsi. No administrator rights needed.</li>
          <li>Every installer file is checked against its SHA-256 list before anything is copied.</li>
          <li>No telemetry. The uninstaller removes exactly what Setup lists on the Review page.</li>
          <li>Versions install side by side: <code>apsi</code> runs the newest, <code>apsi-{sys ? sys.version : "X.Y.Z"}</code> this
            one, and <code>apsi use VERSION</code> switches.{sys && sys.installed && sys.installed.versions.length > 0 &&
              <> Installed now: {sys.installed.versions.join(", ")} (kept).</>}</li>
        </ul>
        {sys && (
          <>
            <div className="group-title">Installation type</div>
            <div className="profiles two">
              {[
                ["express", "Express", "Recommended", "Everything Apsi needs with safe defaults: standard security profile, Apsi Guard on, PATH and .apsi files set up."],
                ["custom", "Custom", "Advanced", "Choose the folder, components, optional GPU / AI packages and the security settings yourself."],
              ].map(([id, t, tag, d]) => (
                <button key={id} className={`profile ${installType === id ? "selected" : ""}`} onClick={() => setInstallType(id)}>
                  <div className="profile-head"><span className="radio" />{t}<span className="badge">{tag}</span></div>
                  <div className="profile-text">{d}</div>
                </button>
              ))}
            </div>
          </>
        )}
        {error && <div className="alert bad">{error}</div>}
        {!sys && !error && <div className="muted">Checking this computer…</div>}
      </div>
    );
  }

  function System() {
    const gpu = sys.gpus && sys.gpus.length ? sys.gpus[0] : null;
    return (
      <div className="page">
        <h1>System check</h1>
        <p className="lead">Apsi targets modern hardware. Setup checked this computer.</p>
        <div className="checks">
          <Check ok={sys.osOk} label="Operating system" value={sys.osName} />
          <Check ok={sys.cpuOk} warn={!sys.cpuOk} label="Processor"
            value={`${sys.cpu} · ${sys.threads} threads${sys.cpuOk ? " · AVX2 + FMA" : " · below the AVX2 target (Apsi still runs)"}`} />
          <Check ok={sys.ramGB >= 4} warn={sys.ramGB < 4} label="Memory" value={`${sys.ramGB.toFixed(1)} GB`} />
          <Check ok={!!gpu && sys.gpuOk} warn label="Graphics"
            value={gpu ? `${gpu.name} · compute capability ${gpu.compute_capability}${sys.gpuOk ? "" : " (below 7.5)"}` : "No NVIDIA GPU detected — the CPU backend will be used"} />
          <Check ok={sys.diskFree == null || sys.diskFree > sys.requiredBytes * 1.2} label="Disk space"
            value={`${bytes(sys.requiredBytes)} required · ${bytes(sys.diskFree)} free`} />
          <Check ok={editorNames.length > 0} warn label="Code editors"
            value={editorNames.length ? `${editorNames.join(", ")} — Apsi support can be added` : "None of the supported editors found (optional)"} />
        </div>
        {blockers.map((b) => <div key={b} className="alert bad">{b}</div>)}
      </div>
    );
  }

  function Location() {
    return (
      <div className="page">
        <h1>Installation folder</h1>
        <p className="lead">The Apsi folder: this version goes into its own <code>{sys ? sys.version : ""}</code> folder inside it, next to any other versions.</p>
        <div className="path-row">
          <input className="path" value={opts.dir} spellCheck={false} onChange={(e) => set("dir", e.target.value)} />
          <button className="btn secondary" onClick={async () => { const d = await setup().pickDir(opts.dir); if (d) set("dir", d); }}>Browse…</button>
        </div>
        {dirCheck.problems && dirCheck.problems.map((p) => <div key={p} className="alert bad">{p}</div>)}
        {dirCheck.warnings && dirCheck.warnings.map((p) => <div key={p} className="alert warn">{p}</div>)}
        <div className="stats">
          <div><span>Space required</span><b>{bytes(selectedSize)}</b></div>
          <div><span>Space available</span><b>{bytes(sys.diskFree)}</b></div>
        </div>
        <div className="note">
          {isWin
            ? "The default location is inside your user profile, so Setup never needs administrator rights and other users are not affected."
            : "On Linux, Apsi installs into your home folder and links the apsi command into ~/.local/bin."}
        </div>
      </div>
    );
  }

  function Components() {
    return (
      <div className="page">
        <h1>Components</h1>
        <p className="lead">Choose what to install. You can change this later by running Setup again.</p>
        <div className="group-title">Apsi</div>
        {COMPONENTS.map((c) => (
          <OptionRow key={c.id} title={c.title}
            text={c.needsEditor ? (editorNames.length ? `Adds Apsi to: ${editorNames.join(", ")}.` : "No supported code editor was found on this computer.") : c.text}
            meta={bytes(sizeOf(c.size))} checked={!!opts.components[c.id]} badge={c.required ? "Required" : null}
            disabled={c.required || (c.needsEditor && !editorNames.length)} onChange={(v) => set(`components.${c.id}`, v)} />
        ))}
        <div className="group-title">Integration</div>
        {INTEGRATION.map((c) => (
          <OptionRow key={c.id} title={c.title} text={c.text} checked={!!opts.components[c.id]}
            onChange={(v) => set(`components.${c.id}`, v)} />
        ))}
      </div>
    );
  }

  function Packages() {
    return (
      <div className="page">
        <h1>Optional packages</h1>
        <p className="lead">These are downloaded from the Python Package Index during installation and need an internet connection. Apsi works fully without them.</p>
        {PACKAGES.map((p) => (
          <OptionRow key={p.id} title={p.title} text={p.needsGPU && !sys.gpuOk ? "Needs an NVIDIA GPU with compute capability 7.5 or newer — not detected." : p.text}
            meta={p.size} checked={!!opts.packages[p.id]} disabled={p.needsGPU && !sys.gpuOk}
            onChange={(v) => set(`packages.${p.id}`, v)} />
        ))}
      </div>
    );
  }

  function Security() {
    return (
      <div className="page">
        <h1>Security</h1>
        <p className="lead">Apsi programs run with explicit permissions. Choose the default for this computer. A single run can always grant more with flags such as <code>--allow-net</code>; change the default later with <code>apsi config --profile …</code>.</p>
        <div className="profiles">
          {PROFILES.map((p) => (
            <button key={p.id} className={`profile ${opts.security.profile === p.id ? "selected" : ""}`}
              onClick={() => set("security.profile", p.id)}>
              <div className="profile-head"><span className="radio" />{p.title}<span className="badge">{p.tag}</span></div>
              <div className="profile-text">{p.text}</div>
            </button>
          ))}
        </div>
        <div className="group-title">Apsi Guard — malware protection</div>
        <OptionRow title="Scan programs before they run" text="Blocks programs that steal credentials, mine crypto, hide code (Trojan Source), escape the sandbox or behave like ransomware."
          checked={opts.security.scanOnRun} onChange={(v) => set("security.scanOnRun", v)} badge="Recommended" />
        <label className="option">
          <div className="option-text">
            <div className="option-title">Ransomware shield</div>
            <div className="option-desc">Stop a program that modifies or deletes more existing files than this within 20 seconds.</div>
          </div>
          <select className="select" value={opts.security.guardLimit} onChange={(e) => set("security.guardLimit", Number(e.target.value))}>
            <option value={50}>50 files — strict</option>
            <option value={150}>150 files — balanced</option>
            <option value={500}>500 files — data work</option>
            <option value={2000}>2000 files — bulk jobs</option>
          </select>
        </label>
        <OptionRow title="Keep a security log" text="Record blocked actions in ~/.apsi/audit.log so you can see what a program tried to do."
          checked={opts.security.auditLog} onChange={(v) => set("security.auditLog", v)} />
        <div className="group-title">Integrity</div>
        <OptionRow title="Verify the installation" text="Check every runtime file against the release manifest and run a self-test after installing."
          checked={opts.security.verify} onChange={(v) => set("security.verify", v)} />
      </div>
    );
  }

  function Review() {
    if (maintenance === "uninstall") {
      return (
        <div className="page">
          <h1>Uninstall Apsi</h1>
          <p className="lead">Setup will remove Apsi {sys.existing.version} from <code>{sys.existing.dir}</code>, together with its PATH entry, file associations, shortcuts and uninstall entry.</p>
          <div className="note">Your own .apsi programs and the settings folder (~/.apsi) are kept.</div>
        </div>
      );
    }
    const items = [
      ["Installation type", maintenance ? "Maintenance" : installType === "express" ? "Express (recommended settings)" : "Custom"],
      ["Action", maintenance === "repair" ? "Repair" : maintenance === "modify" ? `Modify / upgrade to ${sys.version}` : `Install Apsi ${sys.version}`],
      ["Folder", opts.dir],
      ["Components", COMPONENTS.filter((c) => opts.components[c.id]).map((c) => c.title).join(", ")],
      ["Integration", INTEGRATION.filter((c) => opts.components[c.id]).map((c) => c.title).join(", ") || "None"],
      ["Optional packages", PACKAGES.filter((p) => opts.packages[p.id]).map((p) => p.title).join(", ") || "None"],
      ["Security profile", PROFILES.find((p) => p.id === opts.security.profile).title],
      ["Apsi Guard", `${opts.security.scanOnRun ? "Pre-run scan on" : "Pre-run scan off"} · ransomware shield ${opts.security.guardLimit} files / 20 s${opts.security.auditLog ? " · security log" : ""}`],
      ["Disk space", bytes(selectedSize)],
    ];
    return (
      <div className="page">
        <h1>Ready to install</h1>
        <p className="lead">Review your choices. Select Install to begin.</p>
        <dl className="summary">
          {items.map(([k, v]) => (<React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>))}
        </dl>
        {blockers.map((b) => <div key={b} className="alert bad">{b}</div>)}
        {!dirCheck.ok && dirCheck.problems.map((p) => <div key={p} className="alert bad">Folder: {p}</div>)}
        <p className="muted small">Commands after installing: <code>apsi pkg install gpu onnx torch</code> adds libraries any time · <code>apsi scan .</code> checks a folder for malware.</p>
      </div>
    );
  }

  function Install() {
    return (
      <div className="page">
        <h1>{maintenance === "uninstall" ? "Removing Apsi" : "Installing Apsi"}</h1>
        <p className="lead">{progress.label || "Working…"}</p>
        <div className="progress"><div style={{ width: `${progress.percent || 0}%` }} /></div>
        <div className="progress-pct">{Math.round(progress.percent || 0)}%</div>
        <button className="link" onClick={() => setShowLog(!showLog)}>{showLog ? "Hide details" : "Show details"}</button>
        {showLog && <pre className="log" ref={logRef}>{log.join("\n")}</pre>}
      </div>
    );
  }

  function Finish() {
    const ok = result && result.ok;
    if (maintenance === "uninstall") {
      return (
        <div className="page">
          <h1>{ok ? "Apsi has been removed" : "Uninstall did not finish"}</h1>
          <p className="lead">{ok ? "Thank you for using Apsi." : result && result.error}</p>
        </div>
      );
    }
    return (
      <div className="page">
        <div className={`result ${ok ? "ok" : "bad"}`}>{ok ? "✓" : "✕"}</div>
        <h1>{ok ? "Apsi is ready" : "Installation failed"}</h1>
        {ok ? (
          <>
            <p className="lead">Open a new terminal and try the commands below. Terminals and VS Code windows that were
              already open still use the old PATH, so close them first (quit VS Code completely). Still not recognized?
              See docs/TROUBLESHOOTING.md.</p>
            <pre className="code">apsi version{"\n"}apsi run {isWin ? "examples\\01_hello.apsi" : "examples/01_hello.apsi"}{"\n"}apsi doctor</pre>
            <div className="group-title">When I select Finish</div>
            <OptionRow title="Open the Apsi REPL" text="An interactive prompt for trying Apsi." checked={finishActions.repl}
              onChange={(v) => setFinishActions({ ...finishActions, repl: v })} />
            {opts.components.examples && <OptionRow title="Open the examples folder" text="The example programs." checked={finishActions.examples}
              onChange={(v) => setFinishActions({ ...finishActions, examples: v })} />}
            {opts.components.docs && <OptionRow title="Open the documentation" text="Language guide and reference." checked={finishActions.docs}
              onChange={(v) => setFinishActions({ ...finishActions, docs: v })} />}
          </>
        ) : (
          <>
            <p className="lead">{result && result.error}</p>
            <div className="note">All changes were rolled back. The details below show what happened.</div>
            <pre className="log">{log.join("\n")}</pre>
          </>
        )}
      </div>
    );
  }

  const pages = { welcome: Welcome, system: System, location: Location, components: Components, packages: Packages,
    security: Security, review: Review, install: Install, finish: Finish };
    const ready = sys && opts;
  const canNext = ready && !(step === "review" && blockers.length && maintenance !== "uninstall")
    && !(step === "location" && (!opts.dir.trim() || !dirCheck.ok))
    && !(step === "review" && !maintenance && !dirCheck.ok) && !(step === "welcome" && sys.existing && !maintenance);
  const nextLabel = step === "review" ? (maintenance === "uninstall" ? "Uninstall" : maintenance === "repair" ? "Repair" : "Install")
    : step === "finish" ? "Finish" : "Next";

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><img src="./wordmark.png" alt="Apsi" /></div>
        <ol className="steps">
          {flow.map((s, i) => (
            <li key={s.id} className={i < index ? "done" : i === index ? "current" : ""}>
              <span className="dot">{i < index ? "✓" : i + 1}</span>{s.title}
            </li>
          ))}
        </ol>
        <div className="side-foot">Version {sys ? sys.version : "…"}</div>
      </aside>
      <main className="content">
        <header className="titlebar">
          <span className="titlebar-text">Apsi Setup</span>
          <div className="window-buttons">
            <button aria-label="Minimize" onClick={() => setup().minimize()}>&#x2014;</button>
            <button aria-label="Close" className="close" onClick={() => setup().close()}>&#x2715;</button>
          </div>
        </header>
        <section className="body" ref={bodyRef} tabIndex={-1}>{ready || step === "welcome" ? pages[step]() : <div className="page muted">Loading…</div>}</section>
        <footer className="footer">
          <button className="btn ghost" disabled={step === "install"} onClick={() => setup().close()}>
            {step === "finish" ? "Close" : "Cancel"}
          </button>
          <div className="spacer" />
          {index > 0 && !["install", "finish"].includes(step) && (
            <button className="btn secondary" onClick={() => {
              if (index === 1 && maintenance) setMaintenance(null);
              setStep(flow[index - 1].id);
            }}>Back</button>
          )}
          {step !== "install" && (
            <button className={`btn primary ${step === "review" && maintenance === "uninstall" ? "danger" : ""}`}
              disabled={!canNext} onClick={next}>{nextLabel}</button>
          )}
        </footer>
      </main>
    </div>
  );
}
