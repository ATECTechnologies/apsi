"""Tiny LSP client used by the test-suite to drive `apsi lsp`."""
import json
import subprocess
import sys


class Client:
    def __init__(self, cwd):
        self.p = subprocess.Popen([sys.executable, "-m", "apsi", "lsp"], cwd=cwd,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def _send(self, msg):
        data = json.dumps(msg).encode()
        self.p.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        self.p.stdin.flush()

    def _read(self):
        headers = {}
        while True:
            line = self.p.stdout.readline().decode().strip()
            if not line:
                break
            k, _, v = line.partition(":")
            headers[k.lower()] = v.strip()
        return json.loads(self.p.stdout.read(int(headers["content-length"])))

    def request(self, method, params):
        self.n += 1
        self._send({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params})
        while True:
            m = self._read()
            if m.get("id") == self.n:
                return m.get("result")

    def notify(self, method, params, expect=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})
        if expect:
            while True:
                m = self._read()
                if m.get("method") == expect:
                    return m["params"]

    def close(self):
        self.request("shutdown", {})
        self._send({"jsonrpc": "2.0", "method": "exit"})
        self.p.wait(timeout=10)
