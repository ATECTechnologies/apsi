"""db.mysql against a small in-process MySQL server that speaks the real wire protocol (handshake v10,
caching_sha2_password fast path, an authentication switch to mysql_native_password, text result sets with and
without CLIENT_DEPRECATE_EOF, OK/ERR packets, several result sets). Set APSI_TEST_MYSQL=mysql://… to also run the
same checks against a real server."""
import decimal
import hashlib
import os
import socket
import struct
import threading
import unittest

from apsi import db_mysql as M


def pkt(seq, payload):
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def lenenc_str(b):
    return M._lenenc_int(len(b)) + b


class FakeMySQL(threading.Thread):
    """Accepts connections; users: app/secret (caching_sha2), legacy/old (switched to native password)."""

    def __init__(self, deprecate_eof=True):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.deprecate = deprecate_eof
        self.queries = []

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self.session, args=(c,), daemon=True).start()

    def recv(self, c):
        h = b""
        while len(h) < 4:
            h += c.recv(4 - len(h))
        n = h[0] | h[1] << 8 | h[2] << 16
        d = b""
        while len(d) < n:
            d += c.recv(n - len(d))
        return h[3], d

    def session(self, c):
        nonce = bytes(range(1, 21))
        caps = (M.CLIENT_PROTOCOL_41 | M.CLIENT_SECURE_CONNECTION | M.CLIENT_PLUGIN_AUTH | M.CLIENT_CONNECT_WITH_DB
                | M.CLIENT_PLUGIN_AUTH_LENENC | M.CLIENT_MULTI_RESULTS | M.CLIENT_MULTI_STATEMENTS | M.CLIENT_TRANSACTIONS
                | (M.CLIENT_DEPRECATE_EOF if self.deprecate else 0))
        hs = (b"\x0a8.0.36-fake\0" + struct.pack("<I", 7) + nonce[:8] + b"\0" + struct.pack("<H", caps & 0xFFFF)
              + bytes([45]) + struct.pack("<H", 2) + struct.pack("<H", caps >> 16) + bytes([21]) + b"\0" * 10
              + nonce[8:] + b"\0" + b"caching_sha2_password\0")
        c.sendall(pkt(0, hs))
        seq, resp = self.recv(c)
        r = M._Reader(resp)
        client_caps = r.u32()
        r.take(4 + 1 + 23)
        user = r.nul().decode()
        auth = r.lenenc_bytes()
        db = r.nul().decode() if client_caps & M.CLIENT_CONNECT_WITH_DB else ""
        seq += 1
        if user == "app" and auth == M.sha2_scramble("secret", nonce):
            c.sendall(pkt(seq, b"\x01\x03"))
            c.sendall(pkt(seq + 1, b"\x00\x00\x00\x02\x00\x00\x00"))
        elif user == "legacy":
            n2 = b"abcdefghijklmnopqrst"
            c.sendall(pkt(seq, b"\xfemysql_native_password\0" + n2 + b"\0"))
            s2, a2 = self.recv(c)
            if a2 != M.native_scramble("old", n2):
                c.sendall(pkt(s2 + 1, b"\xff\x15\x04#28000Access denied"))
                return c.close()
            c.sendall(pkt(s2 + 1, b"\x00\x00\x00\x02\x00\x00\x00"))
        else:
            c.sendall(pkt(seq, b"\xff\x15\x04#28000Access denied for user"))
            return c.close()
        self.db = db
        while True:
            try:
                seq, p = self.recv(c)
            except (OSError, IndexError):
                return
            if not p or p[0] == 0x01:
                return c.close()
            if p[0] == 0x0E:
                c.sendall(pkt(1, b"\x00\x00\x00\x02\x00\x00\x00"))
                continue
            sql = p[1:].decode()
            self.queries.append(sql)
            if sql.startswith("select"):
                self.resultset(c, more=False)
            elif sql.startswith("multi"):
                self.resultset(c, more=True)
                self.resultset(c, more=False, seq0=10)
            elif sql.startswith("bad"):
                c.sendall(pkt(1, b"\xff\x28\x04#42000You have an error in your SQL syntax"))
            else:
                c.sendall(pkt(1, b"\x00" + M._lenenc_int(3) + M._lenenc_int(42) + b"\x02\x00\x00\x00"))

    def resultset(self, c, more, seq0=1):
        cols = [("id", 3, 63), ("name", 253, 45), ("price", 246, 63), ("blob", 252, 63), ("nothing", 253, 45)]
        s = seq0
        c.sendall(pkt(s, M._lenenc_int(len(cols))))
        for name, t, cs in cols:
            s += 1
            body = b"".join(lenenc_str(x) for x in (b"def", b"shop", b"t", b"t", name.encode(), name.encode()))
            body += b"\x0c" + struct.pack("<HIBHB", cs, 255, t, 0, 0) + b"\0\0"
            c.sendall(pkt(s, body))
        if not self.deprecate:
            s += 1
            c.sendall(pkt(s, b"\xfe\x00\x00\x02\x00"))
        for row in [(b"1", "pen ✓".encode(), b"1.25", b"\x00\xff", None), (b"2", b"ink", b"10.00", b"", None)]:
            s += 1
            c.sendall(pkt(s, b"".join(b"\xfb" if v is None else lenenc_str(v) for v in row)))
        s += 1
        status = 0x0002 | (M.SERVER_MORE_RESULTS if more else 0)
        if self.deprecate:
            c.sendall(pkt(s, b"\xfe\x00\x00" + struct.pack("<HH", status, 0)))
        else:
            c.sendall(pkt(s, b"\xfe\x00\x00" + struct.pack("<H", status)))


class MySQLProtocol(unittest.TestCase):
    def check(self, deprecate):
        srv = FakeMySQL(deprecate)
        srv.start()
        my = M.MySQL(f"mysql://app:secret@127.0.0.1:{srv.port}/shop")
        self.assertEqual(my.server_version, "8.0.36-fake")
        rows = my.query("select id, name, price, blob, nothing from item where price < ?", 10)
        self.assertEqual(rows.columns, ["id", "name", "price", "blob", "nothing"])
        self.assertEqual(rows[0].id, 1)
        self.assertEqual(rows[0].name, "pen ✓")
        self.assertEqual(rows[0].price, decimal.Decimal("1.25"))
        self.assertEqual(rows[0].blob, b"\x00\xff")
        self.assertIsNone(rows[1].nothing)
        self.assertIn("price < 10", srv.queries[-1])
        r = my.query("insert into item (name) values (?)", "it's \"quoted\"\\; drop table item; --")
        self.assertEqual((r.rowcount, r.insert_id), (3, 42))
        self.assertEqual(srv.queries[-1], "insert into item (name) values ('it\\'s \\\"quoted\\\"\\\\; drop table item; --')")
        self.assertEqual(len(my.query("multi; select 2")), 2)
        with self.assertRaises(M.MySQLError) as e:
            my.query("bad sql")
        self.assertIn("syntax", str(e.exception))
        self.assertEqual(e.exception.code, "42000")
        self.assertTrue(my.ping())
        my.insert_many("item", [["a", 1], ["b", 2]], columns=["name", "price"])
        self.assertIn("insert into `item` (`name`, `price`) values ('a', 1), ('b', 2)", srv.queries)
        my.close()
        legacy = M.MySQL(host="127.0.0.1", port=srv.port, user="legacy", password="old")
        self.assertEqual(legacy.query("select 1")[0].id, 1)
        with self.assertRaises(M.MySQLError):
            M.MySQL(host="127.0.0.1", port=srv.port, user="app", password="wrong")

    def test_with_deprecate_eof(self):
        self.check(True)

    def test_classic_eof(self):
        self.check(False)

    def test_binding_never_touches_literals(self):
        self.assertEqual(M.bind("select '?' , ?, `a?b` -- ?\n, $1", ["x"]), "select '?' , 'x', `a?b` -- ?\n, 'x'")
        self.assertEqual(M.escape(None), "NULL")
        self.assertEqual(M.escape(b"\x01"), "X'01'")
        self.assertEqual(M.escape([1, 2]), "(1, 2)")
        with self.assertRaises(M.MySQLError):
            M.bind("select ?, ?", [1])

    def test_scrambles(self):
        nonce = bytes(range(20))
        p1 = hashlib.sha1(b"pw").digest()
        want = bytes(a ^ b for a, b in zip(p1, hashlib.sha1(nonce + hashlib.sha1(p1).digest()).digest()))
        self.assertEqual(M.native_scramble("pw", nonce), want)
        self.assertEqual(M.native_scramble("", nonce), b"")
        self.assertEqual(len(M.sha2_scramble("pw", nonce)), 32)

    @unittest.skipUnless(os.environ.get("APSI_TEST_MYSQL"), "no real MySQL server")
    def test_real_server(self):
        my = M.MySQL(os.environ["APSI_TEST_MYSQL"])
        my.execute("create temporary table t (id int primary key, s text, d decimal(9,2))")
        my.insert_many("t", [[1, "a'b", decimal.Decimal("1.50")]])
        r = my.query("select * from t where id = ?", 1).one()
        self.assertEqual((r.s, r.d), ("a'b", decimal.Decimal("1.50")))


if __name__ == "__main__":
    unittest.main()
