"""Apsi libraries (.apsil): versions, packing, verification, installing, `use NAME`, publishing to a registry."""
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apsi import apsil                                    # noqa: E402
from apsi.apsil import PackageError                       # noqa: E402

LIB = '''# geometry helpers
fn area(w: Length, h: Length) = w * h

fn hyp(a, b) = sqrt(a * a + b * b)

let golden = (1 + sqrt(5)) / 2
let _hidden = 3
'''


def make_lib(root, name="geom-tools", version="1.0.0", src=LIB, deps=None):
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    with open(os.path.join(root, "src", "main.apsi"), "w") as fh:
        fh.write(src)
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write(f"# {name}\n")
    dep = "".join(f'{k} = "{v}"\n' for k, v in (deps or {}).items())
    with open(os.path.join(root, "apsi.toml"), "w") as fh:
        fh.write(f'[project]\nname = "{name}"\nversion = "{version}"\ndescription = "test lib"\n\n'
                 f'[package]\nentry = "src/main.apsi"\nlicense = "ATEC-NC"\n\n[dependencies]\n{dep}')
    return root


class Registry:
    """A tiny in-memory registry speaking the same API as the website."""

    def __init__(self):
        self.pkgs = {}
        self.token = "apsi_" + "t" * 32
        reg = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/v1/token":
                    if self.headers.get("Authorization") != f"Bearer {reg.token}":
                        return self._json(401, {"error": "invalid or expired API token"})
                    return self._json(200, {"name": "tester", "scopes": ["publish:new", "publish:update"]})
                m = re.match(r"^/api/v1/packages/([^/]+)/([^/]+)/download$", self.path)
                if m:
                    data = reg.pkgs.get(m.group(1), {}).get(m.group(2))
                    if not data:
                        return self._json(404, {"error": "not found"})
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                m = re.match(r"^/api/v1/packages/([^/?]+)$", self.path)
                if m:
                    vs = reg.pkgs.get(m.group(1))
                    if not vs:
                        return self._json(404, {"error": "no such library"})
                    out = [{"version": v, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                           for v, d in vs.items()]
                    return self._json(200, {"name": m.group(1), "versions": out,
                                            "latest": apsil.best_version(list(vs))})
                if self.path.startswith("/api/v1/search"):
                    return self._json(200, {"results": [{"name": n, "latest": apsil.best_version(list(v)),
                                                         "summary": ""} for n, v in reg.pkgs.items()]})
                self._json(404, {"error": "?"})

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                body = self.rfile.read(n)          # read the whole request first, like a real server
                if self.headers.get("Authorization") != f"Bearer {reg.token}":
                    return self._json(401, {"error": "invalid token"})
                b = self.headers["Content-Type"].split("boundary=")[1].encode()
                parts = body.split(b"--" + b)
                fields, blob = {}, None
                for p in parts:
                    if b"\r\n\r\n" not in p:
                        continue
                    head, val = p.split(b"\r\n\r\n", 1)
                    val = val[:-2]
                    nm = re.search(rb'name="([^"]+)"', head).group(1).decode()
                    if b"filename=" in head:
                        blob = val
                    else:
                        fields[nm] = val.decode()
                if hashlib.sha256(blob).hexdigest() != fields["sha256"]:
                    return self._json(400, {"error": "sha mismatch"})
                man = apsil.validate_manifest(json.loads(zipfile.ZipFile(io.BytesIO(blob)).read("apsil.json")))
                if man["version"] in reg.pkgs.get(man["name"], {}):
                    return self._json(409, {"error": "version exists"})
                reg.pkgs.setdefault(man["name"], {})[man["version"]] = blob
                self._json(201, {"name": man["name"], "version": man["version"]})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class Versions(unittest.TestCase):
    def test_specs(self):
        S = apsil.satisfies
        self.assertTrue(S("1.4.2", "^1.2"))
        self.assertFalse(S("2.0.0", "^1.2"))
        self.assertTrue(S("0.2.9", "^0.2.1"))
        self.assertFalse(S("0.3.0", "^0.2.1"))
        self.assertTrue(S("1.2.9", "~1.2.3"))
        self.assertFalse(S("1.3.0", "~1.2.3"))
        self.assertTrue(S("1.5.0", ">=1,<2"))
        self.assertFalse(S("2.0.0", ">=1,<2"))
        self.assertTrue(S("1.0.0", "==1.0.0"))
        self.assertTrue(S("1.9.0", "1"))
        self.assertFalse(S("2.0.0-rc.1", "*"))           # pre-releases only when asked for
        self.assertTrue(S("2.0.0-rc.2", ">=2.0.0-rc.1"))
        self.assertEqual(apsil.best_version(["1.0.0", "1.10.0", "1.9.3", "2.0.0-beta"], "*"), "1.10.0")
        self.assertEqual(apsil.best_version(["1.0.0", "1.2.0", "2.1.0"], "^1"), "1.2.0")
        self.assertIsNone(apsil.best_version(["1.0.0"], "^2"))
        self.assertEqual(apsil.split_requirement("geom-tools@^1.2"), ("geom-tools", "^1.2"))
        self.assertEqual(apsil.split_requirement("x>=1,<2"), ("x", ">=1,<2"))
        self.assertEqual(apsil.split_requirement("X"), ("x", "*"))
        for bad in ("1.2", "v1", "1.2.3.4", ""):
            with self.assertRaises(PackageError):
                apsil.parse_version(bad)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = os.path.join(self.tmp, "home")
        self._env = {k: os.environ.get(k) for k in ("APSI_HOME", "APSI_REGISTRY", "APSI_TOKEN")}
        os.environ["APSI_HOME"] = self.home
        for k in ("APSI_REGISTRY", "APSI_TOKEN"):
            os.environ.pop(k, None)
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)


class PackAndInstall(Base):
    def test_pack_is_deterministic_and_verified(self):
        lib = make_lib(os.path.join(self.tmp, "lib"))
        a = apsil.pack(lib, quiet=True)
        h1 = hashlib.sha256(open(a, "rb").read()).hexdigest()
        os.utime(os.path.join(lib, "src", "main.apsi"), (1, 1))
        b = apsil.pack(lib, quiet=True)
        self.assertEqual(h1, hashlib.sha256(open(b, "rb").read()).hexdigest())
        self.assertTrue(a.endswith(os.path.join("dist", "geom-tools-1.0.0.apsil")))
        man = apsil.read_manifest(a)
        self.assertEqual(man["entry"], "src/main.apsi")
        self.assertEqual({f["path"] for f in man["files"]}, {"src/main.apsi", "README.md"})

    def test_pack_rejects_errors(self):
        lib = make_lib(os.path.join(self.tmp, "lib"), src="let x = 1m + 2s\n")
        with self.assertRaises(PackageError) as cm:
            apsil.pack(lib, quiet=True)
        self.assertIn("main.apsi", str(cm.exception))
        lib2 = make_lib(os.path.join(self.tmp, "lib2"), name="Bad Name")
        with self.assertRaises(PackageError):
            apsil.pack(lib2, quiet=True)

    def test_install_and_use(self):
        f = apsil.pack(make_lib(os.path.join(self.tmp, "lib")), quiet=True)
        proj = os.path.join(self.tmp, "proj")
        os.makedirs(proj)
        open(os.path.join(proj, "apsi.toml"), "w").write('[project]\nname = "p"\nversion = "0.1.0"\n')
        apsil.install([f], proj, quiet=True)
        self.assertIn("geom-tools", apsil.installed(proj))
        prog = os.path.join(proj, "main.apsi")
        open(prog, "w").write("use geom_tools\nprint(area(2m, 3m))\nprint(hyp(3, 4))\n"
                              "use geom_tools as g\nprint(g.golden)\n")
        from apsi.interpreter import Interpreter
        from apsi.checker import Checker
        from apsi.parser import parse
        src = open(prog).read()
        self.assertEqual(Checker(Interpreter(), proj).check(parse(src)), [])
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            Interpreter().run(src, prog)
        finally:
            sys.stdout = old
        lines = out.getvalue().split()
        self.assertIn("6", out.getvalue())
        self.assertIn("5.0", lines)
        self.assertIn("1.618", out.getvalue())
        # reference tree-walking engine too
        out = io.StringIO()
        sys.stdout = out
        try:
            Interpreter().run(src, prog, engine="interp")
        finally:
            sys.stdout = old
        self.assertIn("1.618", out.getvalue())
        # hidden names are not exported
        with self.assertRaises(Exception):
            Interpreter().run("use geom_tools\nprint(_hidden)\n", prog)
        apsil.uninstall("geom-tools", proj)
        self.assertNotIn("geom-tools", apsil.installed(proj))

    def test_tampered_files_are_refused(self):
        f = apsil.pack(make_lib(os.path.join(self.tmp, "lib")), quiet=True)
        bad = os.path.join(self.tmp, "bad.apsil")
        with zipfile.ZipFile(f) as zin, zipfile.ZipFile(bad, "w") as zout:
            for i in zin.infolist():
                data = zin.read(i)
                if i.filename == "src/main.apsi":
                    data = data.replace(b"w * h", b"w + h")
                zout.writestr(i, data)
        with self.assertRaises(PackageError):
            apsil.install([bad], None, global_=True, quiet=True)
        # path traversal
        evil = os.path.join(self.tmp, "evil.apsil")
        man = apsil.read_manifest(f)
        man["files"].append({"path": "../../x.apsi", "sha256": "0" * 64, "size": 1})
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("apsil.json", json.dumps(man))
            z.writestr("src/main.apsi", LIB)
            z.writestr("README.md", "# geom-tools\n")
            z.writestr("../../x.apsi", "x")
        with self.assertRaises(PackageError):
            apsil.install([evil], None, global_=True, quiet=True)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "x.apsi")))
        with self.assertRaises(PackageError):
            apsil.read_manifest(os.path.join(self.tmp, "lib", "apsi.toml"))


class RegistryFlow(Base):
    def setUp(self):
        super().setUp()
        self.reg = Registry()

    def tearDown(self):
        self.reg.close()
        super().tearDown()

    def test_publish_install_latest_and_deps(self):
        base = make_lib(os.path.join(self.tmp, "base"), name="base-math", version="1.0.0",
                        src="fn twice(x) = 2 * x\n")
        apsil.publish(apsil.pack(base, quiet=True), self.reg.token, self.reg.url, quiet=True)
        make_lib(base, name="base-math", version="1.2.0", src="fn twice(x) = x + x\n")
        apsil.publish(apsil.pack(base, quiet=True), self.reg.token, self.reg.url, quiet=True)
        make_lib(base, name="base-math", version="2.0.0", src="fn twice(x) = 0\n")
        apsil.publish(apsil.pack(base, quiet=True), self.reg.token, self.reg.url, quiet=True)
        top = make_lib(os.path.join(self.tmp, "top"), name="top-lib", version="0.1.0",
                       src="use base_math\nfn quad(x) = twice(twice(x))\n", deps={"base-math": "^1.0"})
        # pack checks `use base_math`, so the dependency must be installed while building
        apsil.install(["base-math@^1"], None, global_=True, registry=self.reg.url, quiet=True)
        apsil.publish(apsil.pack(top, quiet=True), self.reg.token, self.reg.url, quiet=True)
        with self.assertRaises(PackageError) as cm:
            apsil.publish(apsil.pack(top, quiet=True), self.reg.token, self.reg.url, quiet=True)
        self.assertIn("409", str(cm.exception))
        with self.assertRaises(PackageError) as cm:
            apsil.publish(apsil.pack(top, quiet=True), "apsi_" + "x" * 32, self.reg.url, quiet=True)
        self.assertIn("401", str(cm.exception))

        proj = os.path.join(self.tmp, "proj")
        os.makedirs(proj)
        open(os.path.join(proj, "apsi.toml"), "w").write('[project]\nname = "p"\nversion = "0.1.0"\n')
        done = apsil.install(["top-lib"], proj, registry=self.reg.url, quiet=True)
        self.assertEqual([m["name"] for m in done], ["top-lib", "base-math"])
        inst = apsil.installed(proj)
        self.assertEqual(inst["base-math"][0], "1.2.0")         # ^1.0 picks the newest 1.x, not 2.0.0
        lock = json.load(open(apsil.write_lock(proj)))
        self.assertEqual(lock["packages"]["top-lib"]["version"], "0.1.0")
        from apsi.interpreter import Interpreter
        it = Interpreter()
        env = it.run("use top_lib\nlet r = quad(3)\n", os.path.join(proj, "m.apsi"))
        self.assertEqual(env.lookup("r") if hasattr(env, "lookup") else env.vars["r"], 12)
        # newest when no version is given
        apsil.install(["base-math"], proj, registry=self.reg.url, upgrade=True, quiet=True)
        self.assertEqual(apsil.installed(proj)["base-math"][0], "2.0.0")
        with self.assertRaises(PackageError):
            apsil.install(["nothing-here"], proj, registry=self.reg.url, quiet=True)
        with self.assertRaises(PackageError):
            apsil.install(["base-math@^9"], proj, registry=self.reg.url, quiet=True)

    def test_registry_and_token_storage(self):
        self.assertEqual(apsil.registry_url(), "https://apsi.dev")      # the official registry by default
        with self.assertRaises(PackageError):
            apsil.registry_url("http://example.com")          # http only for localhost
        from apsi.cli import main
        self.assertEqual(main(["lib", "registry", self.reg.url]), 0)
        self.assertEqual(apsil.registry_url(), self.reg.url)
        self.assertEqual(main(["lib", "login", "--token", self.reg.token]), 0)
        self.assertEqual(apsil.stored_token(self.reg.url), self.reg.token)
        if os.name == "posix":
            self.assertEqual(os.stat(apsil._cred_path()).st_mode & 0o777, 0o600)
        lib = make_lib(os.path.join(self.tmp, "lib"))
        os.chdir(lib)
        self.assertEqual(main(["pack"]), 0)
        self.assertEqual(main(["publish"]), 0)                # newest file in dist/, stored token
        self.assertEqual(main(["search", "geom"]), 0)
        os.chdir(self.tmp)
        self.assertEqual(main(["install", "-g", "geom-tools"]), 0)
        self.assertIn("geom-tools", apsil.installed(None))
        self.assertEqual(main(["lib", "list"]), 0)
        self.assertEqual(main(["uninstall", "-g", "geom-tools"]), 0)


if __name__ == "__main__":
    unittest.main()
