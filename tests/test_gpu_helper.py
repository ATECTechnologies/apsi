"""The GPU helper: the key is required, limits never go above the card's default, the protocol over 127.0.0.1."""
import ctypes
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from apsi import gpu_helper


class FakeNvml:
    def __init__(self):
        self.limit = 170000

    def nvmlDeviceGetHandleByIndex_v2(self, i, h):
        return 0 if i.value == 0 else 1

    def nvmlDeviceGetPowerManagementLimit(self, h, v):
        v._obj.value = self.limit
        return 0

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h, v):
        v._obj.value = 170000
        return 0

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h, lo, hi):
        lo._obj.value, hi._obj.value = 100000, 212000
        return 0

    def nvmlDeviceSetPowerManagementLimit(self, h, mw):
        self.limit = mw.value
        return 0

    def nvmlDeviceGetCount_v2(self, n):
        n._obj.value = 1
        return 0


class Helper(unittest.TestCase):
    def test_requests(self):
        lib = FakeNvml()
        self.assertEqual(gpu_helper.handle(lib, {"key": "x", "op": "status"}, "k")["error"], "wrong key")
        r = gpu_helper.handle(lib, {"key": "k", "op": "set", "mw": 250000}, "k")
        self.assertEqual((r["ok"], r["mw"], lib.limit), (True, 170000, 170000))     # never above the default
        r = gpu_helper.handle(lib, {"key": "k", "op": "set", "mw": 50000}, "k")
        self.assertEqual(r["mw"], 100000)                                            # never below the minimum
        gpu_helper.handle(lib, {"key": "k", "op": "set", "mw": 130000}, "k")
        self.assertEqual(gpu_helper.handle(lib, {"key": "k", "op": "status"}, "k")["current_mw"], 130000)
        gpu_helper.handle(lib, {"key": "k", "op": "default"}, "k")
        self.assertEqual(lib.limit, 170000)
        self.assertFalse(gpu_helper.handle(lib, {"key": "k", "op": "status", "gpu": 3}, "k")["ok"])

    def test_over_the_socket(self):
        lib = FakeNvml()
        with tempfile.TemporaryDirectory() as home:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            os.makedirs(os.path.join(home, ".apsi"))
            with open(os.path.join(home, ".apsi", "gpu-helper.json"), "w") as fh:
                json.dump({"key": "secret", "port": port, "start_limit": "80%"}, fh)
            with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}), \
                    mock.patch.object(gpu_helper, "_nvml", lambda: lib):
                threading.Thread(target=gpu_helper.serve, daemon=True).start()
                for _ in range(50):
                    r = gpu_helper.call("status")
                    if r:
                        break
                    time.sleep(0.05)
                self.assertEqual(r["current_mw"], 136000)                           # the start limit: 80 %
                self.assertTrue(gpu_helper.call("set", mw=120000)["ok"])
                self.assertEqual(lib.limit, 120000)


if __name__ == "__main__":
    unittest.main()
