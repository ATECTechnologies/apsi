"""db.mongo: BSON round trips, SCRAM against the RFC 7677 test vector, and the OP_MSG protocol against a small
in-process server (hello, SCRAM-SHA-256 login checked on the server side, insert, find with getMore, update, delete,
errors). Set APSI_TEST_MONGO=mongodb://… to also run against a real server."""
import base64
import datetime
import decimal
import hashlib
import hmac
import os
import re
import socket
import struct
import threading
import unittest

from apsi import db_mongo as G


class FakeMongo(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.docs = []
        self.cursors = {}

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self.session, args=(c,), daemon=True).start()

    def session(self, c):
        st = {"authed": False}
        while True:
            try:
                head = b""
                while len(head) < 16:
                    part = c.recv(16 - len(head))
                    if not part:
                        return
                    head += part
                ln, rid, _, op = struct.unpack("<iiii", head)
                body = b""
                while len(body) < ln - 16:
                    body += c.recv(ln - 16 - len(body))
            except OSError:
                return
            cmd = G.decode(body, 5)
            reply = self.handle(cmd, st)
            out = struct.pack("<I", 0) + b"\x00" + G.encode(reply)
            c.sendall(struct.pack("<iiii", 16 + len(out), 1, rid, 2013) + out)

    def handle(self, cmd, st):
        name = next(iter(cmd))
        if name == "hello":
            return {"ok": 1.0, "maxWireVersion": 21, "saslSupportedMechs": ["SCRAM-SHA-256"]}
        if name == "saslStart":
            first = bytes(cmd["payload"]).decode()
            bare = first[3:]
            cn = dict(p.split("=", 1) for p in bare.split(","))["r"]
            st["salt"], st["nonce"], st["bare"] = b"0123456789abcdef", cn + "SRV", bare
            st["sfirst"] = f"r={st['nonce']},s={base64.b64encode(st['salt']).decode()},i=4096"
            return {"ok": 1.0, "conversationId": 1, "done": False, "payload": G.Binary(st["sfirst"].encode())}
        if name == "saslContinue":
            final = bytes(cmd["payload"]).decode()
            f = dict(p.split("=", 1) for p in final.split(","))
            salted = hashlib.pbkdf2_hmac("sha256", b"secret", st["salt"], 4096)
            ck = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
            auth = f"{st['bare']},{st['sfirst']},c=biws,r={f['r']}".encode()
            sig = hmac.new(hashlib.sha256(ck).digest(), auth, hashlib.sha256).digest()
            if base64.b64decode(f["p"]) != bytes(a ^ b for a, b in zip(ck, sig)):
                return {"ok": 0.0, "errmsg": "Authentication failed.", "code": 18, "codeName": "AuthenticationFailed"}
            sk = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
            st["authed"] = True
            v = base64.b64encode(hmac.new(sk, auth, hashlib.sha256).digest()).decode()
            return {"ok": 1.0, "conversationId": 1, "done": True, "payload": G.Binary(f"v={v}".encode())}
        if not st["authed"]:
            return {"ok": 0.0, "errmsg": "command requires authentication", "code": 13, "codeName": "Unauthorized"}
        if name == "insert":
            self.docs += cmd["documents"]
            return {"ok": 1.0, "n": len(cmd["documents"])}
        if name == "find":
            hits = [d for d in self.docs if all(d.get(k) == v for k, v in cmd["filter"].items())]
            self.cursors[77] = hits[2:]
            return {"ok": 1.0, "cursor": {"id": 77 if len(hits) > 2 else 0, "ns": f"{cmd['$db']}.{cmd['find']}",
                                          "firstBatch": hits[:2]}}
        if name == "getMore":
            rest = self.cursors.pop(cmd["getMore"], [])
            return {"ok": 1.0, "cursor": {"id": 0, "ns": "shop.items", "nextBatch": rest}}
        if name == "update":
            return {"ok": 1.0, "n": 1, "nModified": 1}
        if name == "delete":
            n = len(self.docs)
            self.docs = []
            return {"ok": 1.0, "n": n}
        if name == "ping":
            return {"ok": 1.0}
        return {"ok": 0.0, "errmsg": f"no such command: '{name}'", "code": 59, "codeName": "CommandNotFound"}


class MongoTests(unittest.TestCase):
    def test_bson_round_trip(self):
        oid = G.ObjectId()
        doc = {"_id": oid, "s": "héllo ✓", "i": 7, "big": 1 << 40, "f": 1.5, "t": True, "n": None,
               "l": [1, "two", {"three": 3.0}], "b": b"\x00\x01", "d": decimal.Decimal("12.3450"),
               "neg": decimal.Decimal("-0.001"), "when": datetime.datetime(2026, 11, 10, 4, 5, 6, 789000,
                                                                          tzinfo=datetime.timezone.utc),
               "re": re.compile("^a.*z$", re.I), "nested": {"z": 1, "a": 2}}
        back = G.decode(G.encode(doc))
        self.assertEqual(list(back), list(doc))                     # key order is kept
        for k in doc:
            if k == "re":
                self.assertEqual((back[k].pattern, bool(back[k].flags & re.I)), ("^a.*z$", True))
            else:
                self.assertEqual(back[k], doc[k], k)
        self.assertEqual(str(back["d"]), "12.3450")
        self.assertEqual(G.encode({"hello": "world"}).hex(), "160000000268656c6c6f0006000000776f726c640000")
        self.assertEqual(len(str(oid)), 24)
        self.assertEqual(G.ObjectId(str(oid)), oid)

    def test_scram_sha256_rfc7677(self):
        sc = G.Scram("SCRAM-SHA-256", "user", "pencil", nonce="rOprNGfwEbeRWgbNEkqO")
        self.assertEqual(sc.first(), b"n,,n=user,r=rOprNGfwEbeRWgbNEkqO")
        final = sc.final(b"r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096")
        self.assertEqual(final, b"c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,"
                                b"p=dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=")
        sc.verify(b"v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=")
        with self.assertRaises(G.MongoError):
            sc.verify(b"v=AAAA")

    def test_protocol(self):
        srv = FakeMongo()
        srv.start()
        m = G.Mongo(f"mongodb://app:secret@127.0.0.1:{srv.port}/shop?authSource=admin")
        self.assertTrue(m.ping())
        items = m.collection("items")
        oid = items.insert_one({"name": "pen", "kind": "office", "price": 1.25})
        self.assertIsInstance(oid, G.ObjectId)
        items.insert_many([{"name": f"n{i}", "kind": "office"} for i in range(4)])
        found = items.find({"kind": "office"})
        self.assertEqual([d["name"] for d in found], ["pen", "n0", "n1", "n2", "n3"])     # two batches (getMore)
        self.assertEqual(len(items.find({"kind": "office"}, limit=3)), 3)
        self.assertEqual(items.find_one({"name": "pen"})["price"], 1.25)
        self.assertEqual(m["shop"]["items"].update_one({"name": "pen"}, {"$set": {"price": 2}})["modified"], 1)
        self.assertEqual(items.delete_many({}), 5)
        with self.assertRaises(G.MongoError) as e:
            m.command({"nonsense": 1})
        self.assertEqual(e.exception.code, 59)
        m.close()
        with self.assertRaises(G.MongoError):
            G.Mongo(f"mongodb://app:wrong@127.0.0.1:{srv.port}/shop")

    @unittest.skipUnless(os.environ.get("APSI_TEST_MONGO"), "no real MongoDB server")
    def test_real_server(self):
        m = G.Mongo(os.environ["APSI_TEST_MONGO"])
        c = m.collection("apsi_test")
        c.drop()
        c.insert_many([{"k": i, "v": decimal.Decimal("1.5")} for i in range(250)])
        self.assertEqual(c.count_documents({"k": {"$lt": 100}}), 100)
        self.assertEqual(len(c.find({}, sort={"k": -1}, batch_size=50)), 250)
        c.drop()


if __name__ == "__main__":
    unittest.main()
