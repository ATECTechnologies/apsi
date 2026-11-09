# Apsi security model

Apsi assumes any program might be hostile, including one you downloaded, one an AI wrote,
or a dependency of either. The defences are layered. Each layer still holds if another one
is turned off.

| Layer | What it stops | Where |
|---|---|---|
| 1. Capability permissions | file, network, environment, Python and native-code access that was not granted | every built-in function |
| 2. Guarded locations | changes to Apsi's own settings, the runtime, system folders, shell start-up files; reads of credentials | `check_path` |
| 3. Network guard | SSRF to cloud metadata and link-local addresses, DNS tricks, redirect hops to non-allowed hosts, oversized responses | `http_get`, `http_post_json` |
| 4. Apsi Guard static scan | malware patterns, checked before `apsi run` / `apsi serve` and on demand with `apsi scan` | `apsi/guard.py` |
| 5. Behaviour monitor | ransomware and wipers: mass overwrite or delete of existing files, decoy files | runtime |
| 6. Python / native-code control | sandbox escape through `subprocess`, `os`, `ctypes`, ...; native libraries planted in temp or Downloads; library swapping | `pyimport`, `cload` |
| 7. Resource limits | memory exhaustion and runaway programs | `--max-memory`, `--timeout` |
| 8. Integrity | a modified runtime or installer | `apsi verify`, installer SHA-256 check |

## 1. Permissions and profiles

```
apsi run app.apsi                              # profile from `apsi config` (default: standard)
apsi run app.apsi --sandbox                    # nothing allowed: pure computation
apsi run app.apsi --profile strict             # files only inside the program's folder
apsi run app.apsi --allow-read=data --allow-write=out
apsi run app.apsi --allow-net=api.example.com:443,*.example.org
apsi run app.apsi --allow-env=HOME,API_TOKEN
apsi run app.apsi --allow-python               # numpy/torch/... but NOT subprocess, os, ctypes, socket
apsi run app.apsi --allow-python=all           # every Python module (full access: trusted code only)
apsi run app.apsi --allow-ffi=vecmath.so@<sha256>   # native library pinned by hash
apsi config --profile strict                   # change the default for this computer
apsi audit app.apsi                            # which capabilities a program uses + the minimal flags
```

| Profile | read | write | net | env | python | ffi |
|---|---|---|---|---|---|---|
| sandbox | – | – | – | – | – | – |
| strict | program folder | program folder | – (private IPs blocked even with `--allow-net`) | – | – | – |
| standard | anywhere* | anywhere* | – | – | – | – |
| developer | anywhere* | anywhere* | any† | any | all | any |

\* Except the guarded locations below. † Cloud metadata and link-local addresses stay blocked.

## 2. Guarded locations

- **Never writable, under any profile:** `~/.apsi` (so a program cannot raise its own
  permissions), the Apsi runtime and Python installation, system folders (`/etc`, `/usr`,
  `C:\Windows`, `Program Files`), shell start-up files (`.bashrc`, `.profile`, `.zshrc`),
  `~/.local/bin`, the Windows Startup folder and XDG autostart. You can add more folders
  with `APSI_PROTECT=dir1;dir2`.
- **Credentials need an explicit grant:** a blanket `--allow-read` does not cover `~/.ssh`,
  `~/.aws`, `~/.azure`, `~/.kube`, `~/.docker`, `~/.gnupg`, `.netrc`, `.git-credentials`,
  browser profiles (Chrome, Edge, Firefox), Windows Credential Manager folders or
  `/etc/shadow`. Grant one with `--allow-read=~/.ssh`.
- Paths are resolved with `realpath` before every check, so a symlink inside an allowed
  folder cannot point outside it. NUL bytes, `\\?\` device paths, reserved names
  (`CON`, `NUL`, `COM1`, ...) and NTFS alternate data streams (`file.txt:hidden`) are refused.

## 3. Network guard

- Only `http://` and `https://`, and no credentials inside URLs.
- The host is resolved, and every address is checked. Cloud metadata (`169.254.169.254`),
  link-local, multicast and reserved addresses are blocked unless the host is listed exactly
  in `--allow-net`. The strict profile also blocks private and loopback addresses for
  wildcard grants.
- Every redirect hop is checked against the same policy, with at most 5 redirects.
- Responses are capped at 64 MB (`APSI_HTTP_MAX_BYTES`).

## 4. Apsi Guard: static threat scan

`apsi run` and `apsi serve` scan a program before running it:

- **critical** findings always block the run.
- **high** findings block under the sandbox and strict profiles and print a warning
  otherwise.
- `--trust` skips the scan for one run. `APSI_SCAN=off`, or `scan_on_run: "off"` in
  `~/.apsi/config.json`, turns it off.

```
apsi scan downloads/project            # .apsi, .py, .dll/.so/.exe, scripts; exit 1 at "high" or worse
apsi scan . --fail-on critical         # use in CI
apsi scan . --av                       # also Microsoft Defender (MpCmdRun) or ClamAV
apsi scan . --json                     # machine-readable report
```

| Rule | Severity | Detects |
|---|---|---|
| trojan-source | critical | bidirectional override characters (CVE-2021-42574) |
| invisible-char / homoglyph | high | zero-width characters; identifiers using look-alike Cyrillic, Greek or Armenian letters (normal Greek physics names such as `ω`, `π`, `θ` are fine) |
| exfiltration | critical | credential locations read in a program that also uses the network |
| credential-access | high | SSH keys, cloud credentials, browser `Login Data`/`Cookies`, wallets, Telegram/Discord tokens |
| persistence | high | Startup folder, `CurrentVersion\Run`, crontab, systemd, LaunchAgents, `.bashrc`, `LD_PRELOAD` |
| sandbox-escape / dynamic-import | high | `pyimport("subprocess")` and other system modules; computed module names |
| system-library | high | `cload("kernel32")`, `ntdll`, `advapi32`, `libc`, ... |
| download-execute | high | download + write + native or Python execution |
| ransomware-pattern | critical | a loop that walks files, reads them, encrypts them and writes them back |
| suspicious-endpoint | critical–low | mining pools (`stratum+tcp`), xmrig, Discord/Telegram webhooks, paste sites, `.onion`, metadata endpoints, raw IP addresses |
| shell-command | high | `powershell -enc`, `curl … \| sh`, `certutil -urlcache`, `vssadmin delete shadows`, ... |
| obfuscation | medium | long base64/hex blobs, high-entropy strings |
| python-* | critical–medium | in `.py` companions: `exec(base64…)`, injection APIs, `shell=True`, `pickle.loads` |
| disguised-binary, injection-imports | critical / high | executables renamed to documents; PE files importing `WriteProcessMemory` + `CreateRemoteThread` |
| blocklisted | critical | SHA-256 listed in `~/.apsi/blocklist.txt` (one hash per line) |
| yara:* | high | your YARA rules in `~/.apsi/rules/*.yar` (needs `apsi pkg install yara-python`) |
| eicar | critical | the standard antivirus test file, used to check the scanner works |

## 5. Behaviour monitor: ransomware shield

The monitor counts writes that modify existing files. If a program changes more than
**150** (`guard_max_overwrites`, `APSI_GUARD_MAX_OVERWRITES`) within **20 s** (`guard_window_s`),
Apsi stops it. Touching any file whose name starts with `.apsi-decoy` stops a program
immediately: put decoy files in folders you care about. Files a program creates itself do
not count.

## 6. Python and native code

- `--allow-python` allows ordinary libraries. Modules that reach the operating system
  directly (`subprocess`, `os`, `sys`, `ctypes`, `socket`, `shutil`, `importlib`, `pickle`,
  `multiprocessing`, `winreg`, ...) need `--allow-python=all`. Harmless helpers
  (`importlib.util`, `ctypes.util`, `os.path`) are allowed.
- Installed Apsi runs Python isolated (`-I`), so a stray `json.py` in your project folder
  cannot hijack the runtime.
- `cload` refuses libraries in world-writable folders, in the temporary folder, and (unless
  pinned) in Downloads. Pins come from `--allow-ffi=name@sha256` or from `ffi_pins` in the
  config. A pinned library with a different hash is refused.
- **`@native` and `@gpu` code needs no permission and gains none.** The native compiler only accepts computation:
  numbers, tensors passed in, other Apsi functions and `print`. It cannot open files, use the network, run
  programs or call Python or C libraries, so machine code produced from a program can do nothing the program could
  not already do. Index, overflow and division checks are on by default (`@native("fast")` removes them for code
  you trust; out-of-range accesses are then undefined, as in C).

## 7. Packages

`apsi pkg install` installs **binary wheels only** by default, so no package build scripts
run on your machine. Use `--allow-source` to change that. Package indexes must use HTTPS,
and package names containing shell metacharacters are refused. `apsi pkg lock` writes exact
versions with SHA-256 hashes, and `apsi pkg install -r apsi.lock` then requires them.
`apsi pkg audit` runs `pip check` and `pip-audit`.

## 8. Integrity and the installer

- `apsi verify` compares every runtime file with `MANIFEST.json`.
- The setup wizard re-hashes **every payload file** against `payload-files.json`, whose own
  hash is pinned in `payload.json`, before it copies anything. Extra files, such as a
  planted DLL, are refused as well.
- The wizard runs Electron with context isolation, a sandboxed renderer and no Node.js in
  the UI. It uses a strict CSP, denies all browser permissions and blocks every non-local
  request. Each IPC call is checked against its sender, and installer options are rebuilt
  from a whitelist. The uninstall target comes from the detected installation, never from
  the UI.
- The packaged binary has its Electron fuses flipped: no `ELECTRON_RUN_AS_NODE`, no
  `NODE_OPTIONS`, no `--inspect`, and app code loads only from the integrity-checked asar.
- The install folder is validated: system folders, drive roots, symlinks and non-empty
  non-Apsi folders are refused. Synced folders such as OneDrive get a warning.
- `~/.apsi/config.json` is written with mode 600, and blocked actions are logged to
  `~/.apsi/audit.log` (rotated at 5 MB).
- Code signing: set `CSC_LINK` and `CSC_KEY_PASSWORD` before `npm run dist:win`, and
  electron-builder signs with SHA-256 and an RFC 3161 timestamp.

## 9. Executables and caches

- **`apsi build`** fixes a program's permissions when it is built (`apsi build app.apsi --allow-read`); the
  executable runs with exactly those (`apsi build` lists them), and an `apsi.toml` next to where it runs cannot
  add any. A program with check errors is never built. Apsi Guard scans the program every time it starts, as
  with `apsi run`.
- An executable unpacks its program files only below `~/.apsi/apps/NAME-ID`; archive entries that would land
  outside that folder are ignored.
- **The compile cache** (`~/.apsi/cache/compiled`) holds parsed programs as pickles. Every entry is signed with the
  per-user key `~/.apsi/cache.key` (HMAC-SHA256) and verified before it is read; an entry that was modified or copied
  from another computer is deleted without being unpickled. `APSI_NO_CACHE=1` turns the cache off,
  `apsi compile --clear` empties it.
- **The machine-code cache** (`~/.apsi/cache/native`) holds object code produced by the native compiler. Every entry is
  signed with the same per-user key and verified before it is loaded; a modified or foreign entry is deleted and the
  code is compiled again. The key includes the IR, the CPU model and features, and the LLVM and Apsi versions.
  `APSI_NO_CACHE=1` turns it off; `apsi native clear-cache` empties it.
- **The PyTorch backend** (`fit(…, backend="torch")`, `torch_module`, `check_torch`) runs PyTorch, which is Python
  code outside Apsi's sandbox, so it needs the `python` permission like `pyimport`; `backend="auto"` stays on NumPy
  without it.

## Reporting a vulnerability

Do not open a public issue. Send the details and a proof of concept to the maintainers'
security address on the official site. Fixes are released as `2.0.x` patch versions.
