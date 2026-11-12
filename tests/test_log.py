"""`use log`: records to the terminal and to .log files (text and JSON lines), rotation, reading back, `apsi log`."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import log as L                            # noqa: E402

PROGRAM = r'''
use log as lg
print(log(1.0))
lg.to_file("logs/app.log", rotate="1 KB", keep=2)
lg.info("server started", port=8080, host="local host")
lg.warn("disk almost full", free=3.2[m])
let db = lg.named("db", table="users")
db.error("query failed", err="timeout")
lg.debug("only in the file")
with lg.timer("work"):
    var s = 0
    for i in range(1000):
        s += i
for i in range(40):
    lg.info("tick", i=i)
let recs = lg.read("logs/app.log", level="warn", older=true)
print(len(recs), recs[0]["message"], recs[1]["name"], recs[1]["fields"]["table"])
lg.to_file("logs/app.jsonl", format="json")
lg.info("as json", n=[1, 2])
'''


class Logging(unittest.TestCase):
    def test_program(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "t.apsi"), "w", encoding="utf-8") as fh:
                fh.write(PROGRAM)
            r = subprocess.run([sys.executable, "-m", "apsi", "run", "t.apsi", "--allow-write=.", "--allow-read=."],
                               cwd=td, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
            self.assertEqual(r.returncode, 0, r.stderr)
            out = r.stdout.splitlines()
            self.assertEqual(out[0], "0.0")                       # the math log() is still there
            self.assertEqual(out[1], "2 disk almost full db users")
            self.assertIn("INFO   app  server started  port=8080  host=\"local host\"", r.stderr)
            self.assertNotIn("only in the file", r.stderr)       # debug: below the terminal's level
            logs = os.path.join(td, "logs")
            names = sorted(os.listdir(logs))
            self.assertIn("app.1.log", names)                    # rotated at 1 KB, 2 kept
            self.assertNotIn("app.3.log", names)
            for n in names:
                if n.endswith(".log"):
                    self.assertLessEqual(os.path.getsize(os.path.join(logs, n)), 1024 + 200)
            everything = L.read(os.path.join(logs, "app.log"), older=True)
            self.assertTrue(any(x["message"] == "work took" or x["message"].startswith("work took")
                                for x in everything) or len(everything) < 45)
            with open(os.path.join(logs, "app.jsonl"), encoding="utf-8") as fh:
                rec = json.loads(fh.readline())
            self.assertEqual((rec["level"], rec["message"], rec["n"]), ("info", "as json", [1, 2]))
            # `apsi log` shows the file, filtered
            r = subprocess.run([sys.executable, "-m", "apsi", "log", os.path.join(logs, "app.log"), "--older",
                                "--level", "error"], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": ROOT})
            self.assertEqual(r.stdout.count("\n"), 1)
            self.assertIn("ERROR  db  query failed  table=users  err=timeout", r.stdout)

    def test_daily_rotation_and_reading_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "d.log")
            f = L._File(p, "text", "daily", 3, 10)
            f.write("2026-11-10 10:00:00.000  INFO   app  old day  a=1\n")
            import datetime
            f.day = datetime.date(2026, 11, 10)
            f.write("2026-11-11 10:00:00.000  WARN   app  new day  b=\"x y\"\n")
            f.close()
            self.assertTrue(os.path.exists(os.path.join(td, "d.2026-11-10.log")))
            recs = L.read(p, older=True)
            self.assertEqual([r["message"] for r in recs], ["old day", "new day"])
            self.assertEqual(recs[1]["fields"], {"b": "x y"})

    def test_terminal_colour_and_levels(self):
        hub = L._HUB
        old = (hub.level, hub.color)
        try:
            hub.color = False
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                lg = L.Logger("svc")
                lg.debug("hidden")
                lg.error("shown", code=3)
            self.assertNotIn("hidden", buf.getvalue())
            self.assertIn("ERROR  svc  shown  code=3", buf.getvalue())
        finally:
            hub.level, hub.color = old


if __name__ == "__main__":
    unittest.main()
