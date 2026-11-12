"""The debugger through the Debug Adapter Protocol (what VS Code uses)."""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG = """fn area(w, h):
    let a = w * h
    return a

var total = 0
for i in range(5):
    total += area(i, 2)
print("total", total)
"""


class DAPTest(unittest.TestCase):
    def test_breakpoint_stack_variables_step(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "dbg.apsi")
        with open(path, "w") as fh:
            fh.write(PROG)
        p = subprocess.Popen([sys.executable, "-m", "apsi", "dap"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
        q = queue.Queue()

        def reader():
            while True:
                h = {}
                while True:
                    line = p.stdout.readline()
                    if not line:
                        q.put(None)
                        return
                    line = line.strip()
                    if not line:
                        break
                    k, _, v = line.decode().partition(":")
                    h[k.lower()] = v.strip()
                q.put(json.loads(p.stdout.read(int(h["content-length"]))))
        threading.Thread(target=reader, daemon=True).start()
        seq = [0]

        def send(cmd, **args):
            seq[0] += 1
            b = json.dumps({"seq": seq[0], "type": "request", "command": cmd, "arguments": args}).encode()
            p.stdin.write(b"Content-Length: %d\r\n\r\n" % len(b) + b)
            p.stdin.flush()
            return seq[0]

        def wait(pred):
            while True:
                m = q.get(timeout=30)
                self.assertIsNotNone(m, "the adapter exited")
                if pred(m):
                    return m

        def resp(s):
            return wait(lambda m: m.get("type") == "response" and m["request_seq"] == s)

        def ev(name):
            return wait(lambda m: m.get("type") == "event" and m["event"] == name)
        try:
            resp(send("initialize", adapterID="apsi"))
            ev("initialized")
            resp(send("launch", program=path))
            resp(send("setBreakpoints", source={"path": path}, breakpoints=[{"line": 2, "condition": "w == 3"}]))
            resp(send("configurationDone"))
            self.assertEqual(ev("stopped")["body"]["reason"], "breakpoint")
            frames = resp(send("stackTrace", threadId=1))["body"]["stackFrames"]
            self.assertEqual([(f["name"], f["line"]) for f in frames], [("area()", 2), ("<module>", 7)])
            sc = resp(send("scopes", frameId=1))["body"]["scopes"]
            v = {x["name"]: x["value"] for x in
                 resp(send("variables", variablesReference=sc[0]["variablesReference"]))["body"]["variables"]}
            self.assertEqual(v, {"w": "3", "h": "2"})
            self.assertEqual(resp(send("evaluate", expression="w * h + 1", frameId=1))["body"]["result"], "7")
            resp(send("next", threadId=1))
            ev("stopped")
            self.assertEqual(resp(send("stackTrace", threadId=1))["body"]["stackFrames"][0]["line"], 3)
            resp(send("setBreakpoints", source={"path": path}, breakpoints=[]))
            resp(send("continue", threadId=1))
            out = wait(lambda m: m.get("event") == "output" and "total" in m["body"]["output"])
            self.assertIn("total 20", out["body"]["output"])
            self.assertEqual(ev("exited")["body"]["exitCode"], 0)
            send("disconnect")
        finally:
            p.kill()


if __name__ == "__main__":
    unittest.main()
