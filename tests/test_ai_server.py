"""The OpenAI-style model server: answers, streaming, stop strings, keys, the queue, clients that leave, metrics."""
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from test_hub_models import tiny_llama          # noqa: E402
from apsi import hub as H, ai_server             # noqa: E402


def call(srv, method, path, body=None, key=None, raw=False):
    c = http.client.HTTPConnection(srv.host, srv.port, timeout=60)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (data if raw else json.loads(data or b"null")), dict(r.getheaders())


class ModelServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        tiny_llama(cls.td.name, "llama", ctx=256)
        cls.lm = H.Model(cls.td.name, device="cpu")
        cls.srv = ai_server.Server(cls.lm, port=0, name="tiny", log=False).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()
        cls.td.cleanup()

    def test_models_health_and_errors(self):
        s, j, _ = call(self.srv, "GET", "/health")
        self.assertEqual((s, j["status"], j["model"]), (200, "ok", "tiny"))
        s, j, _ = call(self.srv, "GET", "/v1/models")
        self.assertEqual(j["data"][0]["id"], "tiny")
        s, j, _ = call(self.srv, "POST", "/v1/chat/completions", {"messages": "hi"})
        self.assertEqual((s, j["error"]["code"]), (400, 400))
        s, j, _ = call(self.srv, "POST", "/v1/completions", {"prompt": "x", "temperature": 99})
        self.assertEqual(s, 400)
        s, j, _ = call(self.srv, "POST", "/v1/nothing", {})
        self.assertEqual(s, 404)

    def test_completion_matches_generate(self):
        want = self.lm.generate("hello", max_new_tokens=8)
        s, j, _ = call(self.srv, "POST", "/v1/completions", {"prompt": "hello", "max_tokens": 8, "temperature": 0})
        self.assertEqual(s, 200)
        self.assertEqual(j["choices"][0]["text"], want)
        self.assertEqual(j["usage"]["completion_tokens"], self.lm.last_tokens)
        self.assertGreater(j["usage"]["prompt_tokens"], 0)

    def test_chat_and_stream_agree(self):
        body = {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 12, "temperature": 0.8,
                "seed": 7}
        s, j, _ = call(self.srv, "POST", "/v1/chat/completions", body)
        self.assertEqual(s, 200)
        whole = j["choices"][0]["message"]["content"]
        s, raw, hd = call(self.srv, "POST", "/v1/chat/completions", dict(body, stream=True), raw=True)
        self.assertEqual(s, 200)
        self.assertIn("text/event-stream", hd["Content-Type"])
        events = [ln[6:] for ln in raw.decode().split("\n") if ln.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(text.strip(), whole)
        self.assertIn(chunks[-1]["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertIn("usage", chunks[-1])

    def test_stop_strings(self):
        full = self.lm.generate("hello", max_new_tokens=16)
        if len(full) < 4:
            self.skipTest("the tiny model wrote too little")
        stop = full[2:4]
        s, j, _ = call(self.srv, "POST", "/v1/completions", {"prompt": "hello", "max_tokens": 16, "stop": stop})
        self.assertEqual(j["choices"][0]["text"], full[:full.find(stop)])
        s, raw, _ = call(self.srv, "POST", "/v1/completions",
                         {"prompt": "hello", "max_tokens": 16, "stop": [stop], "stream": True}, raw=True)
        evs = [json.loads(ln[6:]) for ln in raw.decode().split("\n") if ln.startswith("data: {")]
        self.assertEqual("".join(e["choices"][0]["text"] for e in evs), full[:full.find(stop)])

    def test_many_clients_at_once(self):
        res, errs = [], []

        def one(i):
            try:
                s, j, _ = call(self.srv, "POST", "/v1/completions",
                               {"prompt": "hello", "max_tokens": 6, "temperature": 0, "stream": i % 2 == 1},
                               raw=i % 2 == 1)
                res.append(s)
            except Exception as e:                       # noqa: BLE001
                errs.append(e)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        self.assertEqual(errs, [])
        self.assertEqual(res, [200] * 16)
        s, m, _ = call(self.srv, "GET", "/metrics")
        self.assertGreaterEqual(m["ok"], 16)
        self.assertIsNotNone(m["latency_s"]["p95"])

    def test_client_that_leaves_frees_the_model(self):
        sock = socket.create_connection((self.srv.host, self.srv.port))
        body = json.dumps({"prompt": "hello", "max_tokens": 200, "stream": True}).encode()
        sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        sock.recv(200)
        sock.close()
        s, j, _ = call(self.srv, "POST", "/v1/completions", {"prompt": "hello", "max_tokens": 4})
        self.assertEqual(s, 200)

    def test_api_key_and_full_queue(self):
        srv = ai_server.Server(self.lm, port=0, api_key="k1", max_queue=0, log=False).start()
        try:
            self.assertEqual(call(srv, "GET", "/v1/models")[0], 401)
            self.assertEqual(call(srv, "GET", "/v1/models", key="k1")[0], 200)
            self.assertEqual(call(srv, "GET", "/health")[0], 200)          # health needs no key
            srv.gate.next_ticket += 1                                      # someone is being served
            s, j, hd = call(srv, "POST", "/v1/completions", {"prompt": "hi"}, key="k1")
            self.assertEqual((s, hd.get("Retry-After")), (429, "2"))
            srv.gate.serving += 1
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
