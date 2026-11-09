"""WebSocket server and client (RFC 6455)."""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apsi.web import App  # noqa: E402
from apsi.websocket import Client, frame, _xor  # noqa: E402


class WebSocketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = cls.app = App(log=False)
        app.websocket("/echo", lambda ws, msg: ws.send(msg))

        def chat(ws):
            ws.join("room")
            ws.send("joined")
            for m in ws:
                app.room("room").send(m)
        app.websocket("/chat", chat)
        app.websocket("/limited", lambda ws, msg: ws.send(msg), max_message=1000)
        cls.srv = app.run_background()
        cls.url = f"ws://127.0.0.1:{cls.srv.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_echo_text_binary_large_ordered(self):
        c = Client(self.url + "/echo")
        c.send("héllo")
        self.assertEqual(c.receive(timeout=5), "héllo")
        c.send(b"\x00\xff")
        self.assertEqual(c.receive(timeout=5), b"\x00\xff")
        big = "z" * (3 << 20)
        c.send(big)
        self.assertEqual(c.receive(timeout=10), big)
        for i in range(500):
            c.send(str(i))
        self.assertEqual([c.receive(timeout=5) for _ in range(500)], [str(i) for i in range(500)])
        c.send({"k": [1, 2]})
        self.assertEqual(c.receive(timeout=5), '{"k": [1, 2]}')
        c.close()

    def test_rooms(self):
        a, b = Client(self.url + "/chat"), Client(self.url + "/chat")
        self.assertEqual(a.receive(timeout=5), "joined")
        self.assertEqual(b.receive(timeout=5), "joined")
        a.send("hi all")
        self.assertEqual(b.receive(timeout=5), "hi all")
        self.assertEqual(a.receive(timeout=5), "hi all")
        a.close()
        b.close()

    def test_limits_and_protocol_errors(self):
        c = Client(self.url + "/limited")
        c.send("x" * 5000)
        self.assertIsNone(c.receive(timeout=5))          # closed with 1009
        c.close()
        # an unmasked client frame is a protocol error
        c = Client(self.url + "/echo")
        c.sock.sendall(frame(1, b"unmasked"))
        self.assertIsNone(c.receive(timeout=5))
        c.close()

    def test_masking(self):
        self.assertEqual(_xor(_xor(b"abcdefg", b"\x01\x02\x03\x04"), b"\x01\x02\x03\x04"), b"abcdefg")


if __name__ == "__main__":
    unittest.main()
