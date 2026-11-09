"""use hub: tokenizer.json engines (WordPiece, byte-level BPE, SentencePiece-style BPE with byte fallback) and the
download path (redirect to storage, token only on the Hub host, SHA-256 check, resume, cache, offline) against a
local server that answers like the Hugging Face Hub."""
import hashlib
import http.server
import json
import os
import tempfile
import threading
import unittest

from apsi import hub as H


def wordpiece():
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "hello", "!",
                                         "world", "cafe"])}
    return H.Tokenizer({
        "model": {"type": "WordPiece", "vocab": vocab, "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100},
        "normalizer": {"type": "BertNormalizer", "lowercase": True, "clean_text": True, "handle_chinese_chars": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "[CLS]"}}, {"Sequence": {"id": "A"}},
                                      {"SpecialToken": {"id": "[SEP]"}}],
                           "pair": []},
        "added_tokens": [{"id": 0, "content": "[PAD]", "special": True}, {"id": 1, "content": "[UNK]", "special": True},
                         {"id": 2, "content": "[CLS]", "special": True}, {"id": 3, "content": "[SEP]", "special": True}],
    })


def byte_bpe():
    G = "Ġ"                                           # "Ġ": a space in GPT-2's byte alphabet
    vocab = {t: i for i, t in enumerate(["h", "e", "l", "o", G, "w", "r", "d", "he", "ll", "hell", "hello", G + "w",
                                         G + "wo", "or", G + "wor", "ld", G + "world", "!", "<|endoftext|>"])}
    merges = ["h e", "l l", "he ll", "hell o", f"{G} w", "o r", f"{G}w or", "l d", f"{G}wor ld"]
    return H.Tokenizer({"model": {"type": "BPE", "vocab": vocab, "merges": merges},
                        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
                        "added_tokens": [{"id": 19, "content": "<|endoftext|>", "special": True}]})


def sp_bpe():
    vocab = {"<unk>": 0, "<s>": 1, "▁": 2, "▁h": 3, "i": 4, "▁hi": 5, "<0xE2>": 6, "<0x9C>": 7, "<0x93>": 8}
    return H.Tokenizer({"model": {"type": "BPE", "vocab": vocab, "merges": ["▁ h", "▁h i"], "byte_fallback": True,
                                  "unk_token": "<unk>"},
                        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                       {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
                        "post_processor": {"type": "TemplateProcessing", "single": [
                            {"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}], "pair": []},
                        "added_tokens": [{"id": 1, "content": "<s>", "special": True}]})


class Tokenizers(unittest.TestCase):
    def test_wordpiece(self):
        t = wordpiece()
        e = t.encode("UnAffable, hello world!")
        self.assertEqual(e.tokens, ["[CLS]", "un", "##aff", "##able", "[UNK]", "hello", "world", "!", "[SEP]"])
        self.assertEqual(e.ids[:4], [2, 4, 5, 6])
        self.assertEqual(t.encode("Café").tokens[1], "cafe")            # accents stripped with lowercase
        self.assertEqual(t.decode(t.encode("hello world!").ids), "hello world!")

    def test_byte_level_bpe(self):
        t = byte_bpe()
        e = t.encode("hello world!<|endoftext|>")
        self.assertEqual(e.tokens, ["hello", "Ġworld", "!", "<|endoftext|>"])
        self.assertEqual(t.decode(e.ids, skip_special_tokens=True), "hello world!")

    def test_sentencepiece_bpe_with_byte_fallback(self):
        t = sp_bpe()
        e = t.encode("hi ✓")
        self.assertEqual(e.tokens, ["<s>", "▁hi", "▁", "<0xE2>", "<0x9C>", "<0x93>"])
        self.assertEqual(t.decode(e.ids), "hi ✓")


class FakeHub(http.server.BaseHTTPRequestHandler):
    files = {}
    seen_auth = []

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.route(head=True)

    def do_GET(self):
        self.route(head=False)

    def route(self, head):
        self.seen_auth.append((self.path, self.headers.get("Authorization")))
        if self.path.startswith("/api/models/me/tiny"):
            body = json.dumps({"id": "me/tiny", "siblings": [{"rfilename": n} for n in self.files]}).encode()
            return self.reply(200, body, {"Content-Type": "application/json"}, head)
        if self.path.startswith("/me/tiny/resolve/main/"):
            name = self.path.split("/resolve/main/", 1)[1]
            data = self.files.get(name)
            if data is None:
                return self.reply(404, b"", {}, head)
            sha = hashlib.sha256(data).hexdigest()
            if name.endswith(".bin"):                       # large file: redirect to storage
                port = self.server.server_address[1]          # storage lives on another host name, as on the Hub
                return self.reply(302, b"", {"Location": f"http://localhost:{port}/storage/{name}", "X-Repo-Commit": "abc123",
                                             "X-Linked-Etag": f'"{sha}"', "X-Linked-Size": str(len(data))}, head)
            return self.reply(200, data, {"X-Repo-Commit": "abc123", "ETag": f'"{sha}"'}, head)
        if self.path.startswith("/storage/"):
            data = self.files[self.path[len("/storage/"):]]
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                return self.reply(206, data[start:], {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"},
                                  head)
            return self.reply(200, data, {}, head)
        self.reply(404, b"", {}, head)

    def reply(self, code, body, headers, head):
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)


class Downloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.old = H.ENDPOINT
        H.ENDPOINT = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.td = tempfile.TemporaryDirectory()
        os.environ["APSI_HUB_CACHE"] = cls.td.name
        FakeHub.files = {"config.json": b'{"hidden_size": 8}', "weights.bin": os.urandom(300_000)}

    @classmethod
    def tearDownClass(cls):
        H.ENDPOINT = cls.old
        cls.srv.shutdown()
        os.environ.pop("APSI_HUB_CACHE", None)
        cls.td.cleanup()

    def test_download_cache_resume_and_token(self):
        m = H.module()
        m["login"]("hf_secret")
        self.assertEqual(sorted(m["files"]("me/tiny")), ["config.json", "weights.bin"])
        self.assertEqual(m["config"]("me/tiny"), {"hidden_size": 8})
        part = os.path.join(self.td.name, "models", "me--tiny", "abc123", "weights.bin.part")
        os.makedirs(os.path.dirname(part), exist_ok=True)
        with open(part, "wb") as fh:                        # an interrupted earlier download
            fh.write(FakeHub.files["weights.bin"][:100_000])
        p = m["download"]("me/tiny", "weights.bin", progress=False)
        with open(p, "rb") as fh:
            self.assertEqual(fh.read(), FakeHub.files["weights.bin"])
        storage = [a for path, a in FakeHub.seen_auth if path.startswith("/storage/")]
        self.assertTrue(storage and all(a is None for a in storage), "the token never leaves the Hub host")
        self.assertTrue(any(a == "Bearer hf_secret" for path, a in FakeHub.seen_auth if path.startswith("/api/")))
        m["offline"](True)
        try:
            self.assertEqual(m["download"]("me/tiny", "config.json"), os.path.join(os.path.dirname(p), "config.json"))
        finally:
            m["offline"](False)
            m["logout"]()

    def test_corrupt_download_is_rejected(self):
        good = FakeHub.files["weights.bin"]
        try:
            FakeHub.files["bad.bin"] = good
            orig = FakeHub.route

            def tamper(self, head):
                if self.path == "/storage/bad.bin" and not head:
                    return self.reply(200, b"x" * len(good), {}, head)
                return orig(self, head)
            FakeHub.route = tamper
            with self.assertRaises(H.HubError):
                H.module()["download"]("me/tiny", "bad.bin", progress=False)
        finally:
            FakeHub.route = orig
            FakeHub.files.pop("bad.bin", None)

    def test_names_are_checked(self):
        with self.assertRaises(H.HubError):
            H.module()["download"]("me/tiny", "../../etc/passwd")
        with self.assertRaises(H.HubError):
            H.module()["files"]("../x")


if __name__ == "__main__":
    unittest.main()
