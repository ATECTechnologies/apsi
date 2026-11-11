"""MySQL and MariaDB for `use db` — the client/server protocol written in Apsi's runtime, no driver to install.

    use db
    let my = db.mysql("mysql://shop:secret@localhost:3306/shop")       # needs --allow-net=localhost:3306
    my.execute("create table if not exists item (id int auto_increment primary key, name text, price decimal(9,2))")
    my.query("insert into item (name, price) values (?, ?)", "pen", 1.25)
    for row in my.query("select id, name, price from item where price < ?", 10):
        print(row.id, row.name, row.price)

Authentication: caching_sha2_password (MySQL 8 default; fast path, full path over TLS or with the server's RSA key)
and mysql_native_password (MariaDB, MySQL 5.7), with authentication switches. TLS with ?ssl=true (or
ssl-mode=REQUIRED). Values are sent with the connection in utf8mb4 and escaped by the client, so a value can never
end a string literal (no SQL injection); `?`, `%s` and `$1` placeholders are accepted.
"""
import datetime as _dt
import decimal
import hashlib
import json
import socket
import ssl
import struct
import threading
from contextlib import contextmanager
from urllib.parse import urlparse, unquote, parse_qs

from .db import DatabaseError, Result, _row_type, _check, _plain, _ident

CLIENT_LONG_PASSWORD = 1
CLIENT_FOUND_ROWS = 2
CLIENT_LONG_FLAG = 4
CLIENT_CONNECT_WITH_DB = 8
CLIENT_PROTOCOL_41 = 0x200
CLIENT_SSL = 0x800
CLIENT_TRANSACTIONS = 0x2000
CLIENT_SECURE_CONNECTION = 0x8000
CLIENT_MULTI_STATEMENTS = 0x10000
CLIENT_MULTI_RESULTS = 0x20000
CLIENT_PLUGIN_AUTH = 0x80000
CLIENT_PLUGIN_AUTH_LENENC = 0x200000
CLIENT_DEPRECATE_EOF = 0x1000000
SERVER_MORE_RESULTS = 0x0008
UTF8MB4 = 45


class MySQLError(DatabaseError):
    kind = "DatabaseError"


# ---------------------------------------------------------------- authentication scrambles
def native_scramble(password, nonce):
    """mysql_native_password: SHA1(pw) XOR SHA1(nonce + SHA1(SHA1(pw)))."""
    if not password:
        return b""
    p1 = hashlib.sha1(password.encode()).digest()
    p2 = hashlib.sha1(p1).digest()
    p3 = hashlib.sha1(nonce + p2).digest()
    return bytes(a ^ b for a, b in zip(p1, p3))


def sha2_scramble(password, nonce):
    """caching_sha2_password: SHA256(pw) XOR SHA256(SHA256(SHA256(pw)) + nonce)."""
    if not password:
        return b""
    p1 = hashlib.sha256(password.encode()).digest()
    p2 = hashlib.sha256(p1).digest()
    p3 = hashlib.sha256(p2 + nonce).digest()
    return bytes(a ^ b for a, b in zip(p1, p3))


# ---------------------------------------------------------------- packet helpers
def _lenenc_int(n):
    if n < 251:
        return bytes([n])
    if n < 1 << 16:
        return b"\xfc" + struct.pack("<H", n)
    if n < 1 << 24:
        return b"\xfd" + struct.pack("<I", n)[:3]
    return b"\xfe" + struct.pack("<Q", n)


class _Reader:
    def __init__(self, data):
        self.d, self.i = data, 0

    def byte(self):
        b = self.d[self.i]
        self.i += 1
        return b

    def take(self, n):
        v = self.d[self.i:self.i + n]
        self.i += n
        return v

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def lenenc(self):
        b = self.byte()
        if b < 251:
            return b
        if b == 0xFB:
            return None
        if b == 0xFC:
            return self.u16()
        if b == 0xFD:
            return struct.unpack("<I", self.take(3) + b"\0")[0]
        return struct.unpack("<Q", self.take(8))[0]

    def lenenc_bytes(self):
        n = self.lenenc()
        return None if n is None else self.take(n)

    def nul(self):
        j = self.d.index(b"\0", self.i)
        v = self.d[self.i:j]
        self.i = j + 1
        return v

    def rest(self):
        v = self.d[self.i:]
        self.i = len(self.d)
        return v


# ---------------------------------------------------------------- values
def escape(v):
    """A value as an SQL literal for the text protocol (utf8mb4 connection)."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise MySQLError("MySQL cannot store NaN or infinity")
        return repr(v)
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "X'" + bytes(v).hex() + "'"
    if isinstance(v, _dt.datetime):
        return "'" + v.isoformat(" ", timespec="microseconds") + "'"
    if isinstance(v, _dt.date):
        return "'" + v.isoformat() + "'"
    if isinstance(v, _dt.timedelta):
        s = int(v.total_seconds())
        sign = "-" if s < 0 else ""
        s = abs(s)
        return f"'{sign}{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{abs(v.microseconds):06d}'"
    if isinstance(v, (dict, list, tuple)) and not isinstance(v, str):
        if isinstance(v, (list, tuple)):
            return "(" + ", ".join(escape(x) for x in v) + ")" if v else "(NULL)"
        v = json.dumps(v, ensure_ascii=False)
    s = str(v)
    out = []
    for ch in s:
        out.append({"\0": "\\0", "\n": "\\n", "\r": "\\r", "\\": "\\\\", "'": "\\'", '"': '\\"', "\x1a": "\\Z"}.get(ch, ch))
    return "'" + "".join(out) + "'"


def bind(sql, params):
    """Put escaped values into ?, %s or $N placeholders (never inside string literals or comments)."""
    if not params:
        return sql
    out, i, n, k = [], 0, len(sql), 0
    while i < n:
        c = sql[i]
        if c in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == "\\":
                    j += 2
                    continue
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i:j + 1])
            i = j + 1
            continue
        if sql.startswith("--", i) or c == "#":
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out.append(sql[i:j])
            i = j
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(sql[i:j])
            i = j
            continue
        if c == "?" or sql.startswith("%s", i):
            if k >= len(params):
                raise MySQLError(f"the query has more placeholders than the {len(params)} value(s) given")
            out.append(escape(params[k]))
            k += 1
            i += 1 if c == "?" else 2
            continue
        if c == "$" and i + 1 < n and sql[i + 1].isdigit():
            j = i + 1
            while j < n and sql[j].isdigit():
                j += 1
            idx = int(sql[i + 1:j]) - 1
            if not 0 <= idx < len(params):
                raise MySQLError(f"${idx + 1} has no value ({len(params)} given)")
            out.append(escape(params[idx]))
            k = max(k, idx + 1)
            i = j
            continue
        out.append(c)
        i += 1
    if k < len(params):
        raise MySQLError(f"{len(params)} values given but the query uses {k}")
    return "".join(out)


_INT_T = {1, 2, 3, 8, 9, 13}
_FLOAT_T = {4, 5}
_BLOB_T = {249, 250, 251, 252, 253, 254, 15}


def _decode(t, charset, flags, raw):
    if raw is None:
        return None
    if t in _INT_T:
        return int(raw)
    if t in _FLOAT_T:
        return float(raw)
    if t in (0, 246):
        return decimal.Decimal(raw.decode())
    if t in (7, 12):
        s = raw.decode()
        if s.startswith("0000-00-00"):
            return None
        return _dt.datetime.fromisoformat(s)
    if t == 10:
        s = raw.decode()
        return None if s.startswith("0000") else _dt.date.fromisoformat(s)
    if t == 11:
        s = raw.decode()
        neg = s.startswith("-")
        h, m, rest = s.lstrip("-").split(":")
        sec = float(rest)
        td = _dt.timedelta(hours=int(h), minutes=int(m), seconds=sec)
        return -td if neg else td
    if t == 245:
        return json.loads(raw.decode())
    if t == 16:
        return int.from_bytes(raw, "big")
    if t in _BLOB_T and charset == 63:
        return bytes(raw)
    return raw.decode("utf-8", "replace")


class MySQL:
    """One connection. query(sql, *params) → Result of records; execute(sql) for statements without values."""

    def __init__(self, url=None, host="localhost", port=3306, user=None, password=None, database=None,
                 timeout=10.0, ssl_mode=None, interp=None, charset="utf8mb4"):
        url, password = _plain(url), _plain(password)
        if url:
            u = urlparse(url)
            if u.scheme not in ("mysql", "mariadb"):
                raise MySQLError("db.mysql: the URL must start with mysql:// or mariadb://")
            host = u.hostname or host
            port = u.port or port
            user = unquote(u.username) if u.username else user
            password = unquote(u.password) if u.password is not None else password
            database = u.path.lstrip("/") or database
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            mode = q.get("ssl-mode") or q.get("sslmode") or ("REQUIRED" if q.get("ssl") in ("true", "1") else None)
            ssl_mode = ssl_mode or mode
        self.host, self.port, self.user, self.database = host, int(port), user or "root", database
        _check(interp, self.host, self.port, "db.mysql()")
        self.lock = threading.RLock()
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 0
        self.tls = False
        self.server_version = ""
        try:
            self._handshake(password or "", (ssl_mode or "").upper())
        except BaseException:
            self.sock.close()
            raise

    # ---- transport
    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise MySQLError("the MySQL server closed the connection")
            buf += chunk
        return bytes(buf)

    def _read(self):
        data = b""
        while True:
            head = self._recv_exact(4)
            n = head[0] | head[1] << 8 | head[2] << 16
            self.seq = (head[3] + 1) & 0xFF
            data += self._recv_exact(n)
            if n < 0xFFFFFF:
                return data

    def _write(self, payload):
        while True:
            part, payload = payload[:0xFFFFFF], payload[0xFFFFFF:]
            self.sock.sendall(struct.pack("<I", len(part))[:3] + bytes([self.seq]) + part)
            self.seq = (self.seq + 1) & 0xFF
            if len(part) < 0xFFFFFF:
                return

    def _error(self, p):
        r = _Reader(p)
        r.byte()
        code = r.u16()
        state = None
        if r.d[r.i:r.i + 1] == b"#":
            r.byte()
            state = r.take(5).decode()
        msg = r.rest().decode("utf-8", "replace")
        return MySQLError(f"MySQL error {code}: {msg}", code=state or str(code), detail=code)

    # ---- connecting
    def _handshake(self, password, ssl_mode):
        p = self._read()
        if p[0] == 0xFF:
            raise self._error(p)
        r = _Reader(p)
        if r.byte() != 10:
            raise MySQLError("unsupported MySQL protocol version")
        self.server_version = r.nul().decode()
        self.thread_id = r.u32()
        nonce = r.take(8)
        r.byte()
        caps = r.u16()
        r.byte()                                          # character set
        r.u16()                                           # status
        caps |= r.u16() << 16
        alen = r.byte()
        r.take(10)
        if caps & CLIENT_SECURE_CONNECTION:
            nonce += r.take(max(13, alen - 8))[:-1] if alen else r.take(13)[:-1]
        plugin = r.nul().decode() if caps & CLIENT_PLUGIN_AUTH and r.i < len(r.d) else "mysql_native_password"
        self.server_caps = caps
        want = (CLIENT_LONG_PASSWORD | CLIENT_FOUND_ROWS | CLIENT_LONG_FLAG | CLIENT_PROTOCOL_41 | CLIENT_TRANSACTIONS
                | CLIENT_SECURE_CONNECTION | CLIENT_MULTI_STATEMENTS | CLIENT_MULTI_RESULTS | CLIENT_PLUGIN_AUTH
                | CLIENT_PLUGIN_AUTH_LENENC | CLIENT_DEPRECATE_EOF)
        if self.database:
            want |= CLIENT_CONNECT_WITH_DB
        self.caps = want & caps | CLIENT_PROTOCOL_41
        if ssl_mode in ("REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY", "PREFERRED"):
            if caps & CLIENT_SSL:
                self.caps |= CLIENT_SSL
                self._write(struct.pack("<IIB23x", self.caps, 1 << 24, UTF8MB4))
                ctx = ssl.create_default_context()
                if ssl_mode in ("REQUIRED", "PREFERRED"):
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                elif ssl_mode == "VERIFY_CA":
                    ctx.check_hostname = False
                self.sock = ctx.wrap_socket(self.sock, server_hostname=self.host)
                self.tls = True
            elif ssl_mode != "PREFERRED":
                raise MySQLError("the MySQL server does not offer TLS but ssl-mode=REQUIRED was asked for")
        auth = self._scramble(plugin, password, nonce)
        body = struct.pack("<IIB23x", self.caps, 1 << 24, UTF8MB4) + self.user.encode() + b"\0"
        body += (_lenenc_int(len(auth)) + auth) if self.caps & CLIENT_PLUGIN_AUTH_LENENC else bytes([len(auth)]) + auth
        if self.database:
            body += self.database.encode() + b"\0"
        body += plugin.encode() + b"\0"
        self._write(body)
        self._auth_result(plugin, password, nonce)

    def _scramble(self, plugin, password, nonce):
        if plugin == "mysql_native_password":
            return native_scramble(password, nonce)
        if plugin == "caching_sha2_password":
            return sha2_scramble(password, nonce)
        if plugin == "mysql_clear_password":
            if not self.tls:
                raise MySQLError("the server asks for the password in clear text; connect with ?ssl=true")
            return password.encode() + b"\0"
        raise MySQLError(f"unsupported MySQL authentication plugin '{plugin}'")

    def _auth_result(self, plugin, password, nonce):
        while True:
            p = self._read()
            k = p[0]
            if k == 0x00:
                return
            if k == 0xFF:
                raise self._error(p)
            if k == 0xFE:                                  # authentication switch
                r = _Reader(p)
                r.byte()
                plugin = r.nul().decode()
                nonce = r.rest().rstrip(b"\0")
                self._write(self._scramble(plugin, password, nonce))
                continue
            if k == 0x01 and plugin == "caching_sha2_password":
                if p[1:2] == b"\x03":                     # fast authentication worked; OK follows
                    continue
                if p[1:2] == b"\x04":                     # full authentication
                    if self.tls:
                        self._write(password.encode() + b"\0")
                        continue
                    self._write(b"\x02")                  # ask for the server's RSA public key
                    key = self._read()[1:]
                    self._write(self._rsa(password, nonce, key))
                    continue
            raise MySQLError(f"unexpected reply during MySQL authentication (0x{k:02x})")

    @staticmethod
    def _rsa(password, nonce, pem):
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError:
            raise MySQLError("this MySQL account needs TLS (?ssl=true) or the 'cryptography' package for its password")
        pw = password.encode() + b"\0"
        xored = bytes(b ^ nonce[i % len(nonce)] for i, b in enumerate(pw))
        key = serialization.load_pem_public_key(pem)
        return key.encrypt(xored, padding.OAEP(mgf=padding.MGF1(hashes.SHA1()), algorithm=hashes.SHA1(), label=None))

    # ---- queries
    def _command(self, code, payload=b""):
        self.seq = 0
        self._write(bytes([code]) + payload)

    def _read_results(self):
        """All result sets of the last command; the last one is returned (statements before it ran too)."""
        last = Result()
        while True:
            p = self._read()
            if p[0] == 0xFF:
                raise self._error(p)
            if p[0] == 0x00:
                r = _Reader(p)
                r.byte()
                affected = r.lenenc()
                insert_id = r.lenenc()
                status = r.u16()
                last = Result([], [], f"OK {affected}", affected or 0)
                last.insert_id = insert_id
                if not status & SERVER_MORE_RESULTS:
                    return last
                continue
            r = _Reader(p)
            ncols = r.lenenc()
            cols = []
            for _ in range(ncols):
                c = _Reader(self._read())
                for _ in range(4):
                    c.lenenc_bytes()                      # catalog, schema, table, original table
                name = c.lenenc_bytes().decode()
                c.lenenc_bytes()                          # original name
                c.lenenc()
                charset = c.u16()
                c.u32()
                typ = c.byte()
                flags = c.u16()
                cols.append((name, typ, charset, flags))
            if not self.caps & CLIENT_DEPRECATE_EOF:
                self._read()                              # EOF after the column definitions
            rt = _row_type([c[0] for c in cols])
            rows = []
            status = 0
            while True:
                p = self._read()
                deprecate = self.caps & CLIENT_DEPRECATE_EOF
                if p[0] == 0xFE and (len(p) < 0xFFFFFF if deprecate else len(p) < 9):
                    rr = _Reader(p)
                    rr.byte()
                    if self.caps & CLIENT_DEPRECATE_EOF:
                        rr.lenenc()
                        rr.lenenc()
                        status = rr.u16()
                    else:
                        rr.u16()
                        status = rr.u16()
                    break
                if p[0] == 0xFF:
                    raise self._error(p)
                rr = _Reader(p)
                rows.append(rt(_decode(t, cs, fl, rr.lenenc_bytes()) for _, t, cs, fl in cols))
            last = Result(rows, [c[0] for c in cols], f"SELECT {len(rows)}", len(rows))
            if not status & SERVER_MORE_RESULTS:
                return last

    def query(self, sql, *params):
        with self.lock:
            self._command(0x03, bind(sql, params).encode("utf-8"))
            return self._read_results()

    def execute(self, sql):
        return self.query(sql)

    def insert_many(self, table, rows, columns=None, chunk=1000):
        rows = [list(r.values()) if isinstance(r, dict) else list(r) for r in rows]
        if not rows:
            return 0
        cols = f" ({', '.join(_mident(c) for c in columns)})" if columns else ""
        n = 0
        with self.transaction():
            for s in range(0, len(rows), chunk):
                part = rows[s:s + chunk]
                values = ", ".join("(" + ", ".join(escape(v) for v in r) + ")" for r in part)
                self.query(f"insert into {_mident(table)}{cols} values {values}")
                n += len(part)
        return n

    @contextmanager
    def transaction(self):
        self.query("start transaction")
        try:
            yield self
        except BaseException:
            self.query("rollback")
            raise
        else:
            self.query("commit")

    def ping(self):
        with self.lock:
            self._command(0x0E)
            p = self._read()
            if p[0] == 0xFF:
                raise self._error(p)
            return True

    def close(self):
        try:
            with self.lock:
                self._command(0x01)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    def __repr__(self):
        return f"<mysql {self.user}@{self.host}:{self.port}/{self.database or ''} server {self.server_version}>"


def _mident(name):
    s = str(name)
    if not s or "\0" in s:
        raise MySQLError(f"invalid name {name!r}")
    return ".".join("`" + part.replace("`", "``") + "`" for part in s.split("."))


_ = _ident
