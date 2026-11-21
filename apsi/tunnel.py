"""`apsi tunnel`: put a local Apsi server on the internet through Cloudflare, without opening a port.

    apsi tunnel --port 8080                      # a quick tunnel: a random https://….trycloudflare.com address
    apsi tunnel server.apsi --port 8080          # also starts the program (it should listen on 127.0.0.1:8080)
    apsi tunnel server.apsi --name my-site       # a named tunnel set up once with cloudflared (your own domain)

It runs Cloudflare's `cloudflared` program, which connects out to Cloudflare; visitors reach https://your-name and
Cloudflare forwards them to 127.0.0.1:PORT. In the program call app.production(domain="your.domain",
cloudflare=True) (or app.trust_proxy("local")) so the server sees each visitor's address and https."""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

INSTALL = {
    "win32": "winget install --id Cloudflare.cloudflared",
    "darwin": "brew install cloudflared",
    "linux": "see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ "
             "(Debian/Ubuntu: the cloudflared .deb package)",
}
NAMED_SETUP = """Set up a named tunnel once (your domain must be on Cloudflare):
  cloudflared tunnel login
  cloudflared tunnel create {name}
  cloudflared tunnel route dns {name} www.your-domain.com
then write ~/.cloudflared/config.yml:
  tunnel: {name}
  credentials-file: <the .json file `create` printed>
  ingress:
    - hostname: www.your-domain.com
      service: http://127.0.0.1:{port}
    - service: http_status:404"""


def find_cloudflared(explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    found = shutil.which("cloudflared")
    if found:
        return found
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            p = os.path.join(base, "cloudflared", "cloudflared.exe")
            if os.path.isfile(p):
                return p
    return None


def tunnel_command(exe, port, name=None):
    if name:
        return [exe, "tunnel", "--no-autoupdate", "run", name]
    return [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{int(port)}"]


_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _wait_port(port, seconds, proc=None):
    import socket
    end = time.time() + seconds
    while time.time() < end and (proc is None or proc.poll() is None):
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def cmd_tunnel(args):
    exe = find_cloudflared(args.cloudflared)
    if exe is None:
        print("apsi tunnel: cloudflared is not installed. Install it with:", file=sys.stderr)
        print("  " + INSTALL.get(sys.platform, INSTALL["linux"]), file=sys.stderr)
        return 1
    if args.name and args.setup:
        print(NAMED_SETUP.format(name=args.name, port=args.port))
        return 0
    app = None
    if args.file:
        cmd = [sys.executable, "-m", "apsi", "run", "--allow-net"] + list(args.allow or []) + [args.file]
        app = subprocess.Popen(cmd)
        if not _wait_port(args.port, args.wait, app):
            print(f"apsi tunnel: {args.file} " + ("stopped" if app.poll() is not None else
                  f"is not listening on 127.0.0.1:{args.port} after {args.wait:g} s"), file=sys.stderr)
            app.terminate()
            return 1
    elif not _wait_port(args.port, 1.0):
        print(f"apsi tunnel: nothing listens on 127.0.0.1:{args.port} yet (start your server, or pass its file)",
              file=sys.stderr)
    proc = subprocess.Popen(tunnel_command(exe, args.port, args.name), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    shown = [False]

    def pump():
        for line in proc.stdout:
            m = _URL.search(line)
            if m and not shown[0]:
                shown[0] = True
                print(f"\n  public address: {m.group(0)}  ->  http://127.0.0.1:{args.port}\n", flush=True)
            if args.verbose or "ERR" in line or "error" in line.lower():
                sys.stdout.write("  cloudflared: " + line)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    if args.name:
        print(f"  tunnel '{args.name}' -> http://127.0.0.1:{args.port} (the host names of its config.yml)")
    print("  Ctrl+C stops the tunnel" + (" and the program" if app else ""))
    try:
        while proc.poll() is None and (app is None or app.poll() is None):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (proc, app):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
    return proc.returncode or 0


def add_parser(sub):
    p = sub.add_parser("tunnel", help="put a local server on the internet through Cloudflare (cloudflared)")
    p.add_argument("file", nargs="?", help="an .apsi server to start first (it listens on 127.0.0.1:PORT)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--name", help="a named tunnel (your own domain); without it: a quick trycloudflare.com address")
    p.add_argument("--setup", action="store_true", help="with --name: print the one-time setup steps")
    p.add_argument("--cloudflared", help="path of the cloudflared program")
    p.add_argument("--allow", action="append", help="more permission flags for the program, e.g. --allow=--allow-read=data")
    p.add_argument("--wait", type=float, default=20.0, help="seconds to wait for the program to listen")
    p.add_argument("--verbose", action="store_true", help="show all of cloudflared's output")
    p.set_defaults(fn=cmd_tunnel)
