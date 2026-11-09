"""`use db` — PostgreSQL, Redis and SQLite, written in Apsi's runtime (no driver to install).

    use db
    let pg = db.postgres("postgres://user:pass@localhost:5432/shop")      # needs --allow-net=localhost:5432
    pg.execute("create table if not exists item (id serial primary key, name text, price numeric)")
    pg.query("insert into item (name, price) values ($1, $2)", "pen", 1.25)
    for row in pg.query("select id, name, price from item where price < $1", 10):
        print(row.id, row.name, row.price)                 # rows are records; row["name"] works too
    with pg.transaction():
        pg.query("update item set price = price * 1.1")

    let r = db.redis("redis://:secret@localhost:6379/0")
    r.set("hits", 0)
    r.incr("hits")
    let p = r.pipeline()                                   # many commands, one round trip
    for i in range(1000):
        p.rpush("queue", i)
    p.execute()

    let lite = db.sqlite("shop.db")                         # needs --allow-read/--allow-write for the file

PostgreSQL: the v3 wire protocol with SCRAM-SHA-256 / md5 / password authentication, TLS (sslmode=require), the
extended query protocol with $1 parameters (never string-pasted, so no SQL injection), prepared statements reused
per connection, results decoded to Apsi values (int, float, Decimal, bool, text, bytea, json/jsonb, dates, uuid,
1-D arrays), transactions, COPY-free bulk insert with `insert_many`, LISTEN/NOTIFY and a thread-safe pool.
Redis: RESP with pipelining, pub/sub, every command by name (`r.hset(…)`, `r.zadd(…)`, `r.command("XADD", …)`).
"""
import base64
import datetime as _dt
import decimal
import hashlib
import hmac
import json
import os
import queue
import socket
import ssl
import struct
import threading
import uuid
from contextlib import contextmanager
from urllib.parse import urlparse, unquote, parse_qs

from .errors import RuntimeErr


class DatabaseError(RuntimeErr):
    """An error reported by the database server (has .code: the SQLSTATE for PostgreSQL)."""
    kind = "DatabaseError"

    def __init__(self, msg, code=None, detail=None):
        super().__init__(msg)
        self.code = code
        self.detail = detail


def _check(interp, host, port, what):
    if interp is None:
        return
    from .security import check_net, resolve_checked
    check_net(interp, host, port, what)
    resolve_checked(interp, host, port, what)


class Row(tuple):
    """A result row: a tuple that also answers row.column and row["column"]."""
    __slots__ = ()
    _cols = ()

    def __getattr__(self, k):
        try:
            return tuple.__getitem__(self, self._cols[k])
        except KeyError:
            raise AttributeError(f"no column '{k}' (columns: {', '.join(self._cols)})")

    def __getitem__(self, k):
        if isinstance(k, str):
            return tuple.__getitem__(self, self._cols[k])
        return tuple.__getitem__(self, k)

    def as_dict(self):
        return dict(zip(self._cols, self))

    def __repr__(self):
        return "row(" + ", ".join(f"{c}={v!r}" for c, v in zip(self._cols, self)) + ")"


def _row_type(cols):
    return type("Row", (Row,), {"__slots__": (), "_cols": {c: i for i, c in enumerate(cols)}})


class Result(list):
    """Rows of a query (a list), with .columns, .rowcount (rows affected) and .status ("INSERT 0 3" …)."""

    def __init__(self, rows=(), columns=(), status="", rowcount=0):
        super().__init__(rows)
        self.columns = list(columns)
        self.status = status
        self.rowcount = rowcount

    def one(self):
        if len(self) != 1:
            raise DatabaseError(f"expected exactly one row, got {len(self)}")
        return self[0]

    def scalar(self):
        return self[0][0] if self else None

    def dicts(self):
        return [r.as_dict() for r in self]

    def column(self, name):
        i = self.columns.index(name)
        return [r[i] for r in self]

    def to_dataframe(self):
        from .dataframe import DataFrame
        return DataFrame({c: [r[i] for r in self] for i, c in enumerate(self.columns)})


# ======================================================================================== PostgreSQL
_PG_INT = {20, 21, 23, 26, 28}
_PG_FLOAT = {700, 701}


def _pg_decode(oid, raw):
    s = raw.decode("utf-8")
    if oid in _PG_INT:
        return int(s)
    if oid in _PG_FLOAT:
        return float(s)
    if oid == 1700:
        return decimal.Decimal(s)
    if oid == 16:
        return s == "t"
    if oid in (114, 3802):
        return json.loads(s)
    if oid == 17:
        return bytes.fromhex(s[2:]) if s.startswith("\\x") else s.encode()
    if oid == 1082:
        return _dt.date.fromisoformat(s)
    if oid == 1114:
        return _dt.datetime.fromisoformat(s)
    if oid == 1184:
        return _dt.datetime.fromisoformat(s if not s[-3] in "+-" else s + ":00")
    if oid == 1083:
        return _dt.time.fromisoformat(s)
    if oid == 2950:
        return uuid.UUID(s)
    if oid in (1007, 1016, 1005, 1021, 1022, 1009, 1015, 1000):     # 1-D arrays
        inner = {1007: 23, 1016: 20, 1005: 21, 1021: 700, 1022: 701, 1009: 25, 1015: 25, 1000: 16}[oid]
        return _pg_array(s, inner)
    return s


def _pg_array(s, inner):
    body = s[1:-1]
    if not body:
        return []
    out, cur, q, esc, quoted = [], [], False, False, False
    for ch in body:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            q = not q
            quoted = True
        elif ch == "," and not q:
            out.append(("".join(cur), quoted))
            cur, quoted = [], False
        else:
            cur.append(ch)
    out.append(("".join(cur), quoted))
    return [None if (v == "NULL" and not qd) else _pg_decode(inner, v.encode()) for v, qd in out]


def _pg_encode(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return b"t" if v else b"f"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return b"\\x" + bytes(v).hex().encode()
    if isinstance(v, (dict, list)) and not (isinstance(v, list) and v and all(isinstance(x, (int, float, str))
                                                                             for x in v) and False):
        if isinstance(v, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return ("{" + ",".join(repr(x) for x in v) + "}").encode()
        return json.dumps(v, ensure_ascii=False).encode()
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat().encode()
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:                                      # noqa: BLE001
        pass
    return str(v).encode()


_PG_ARRAY_OIDS = {1000, 1005, 1007, 1016, 1021, 1022, 1009, 1015, 1014, 1231, 2951, 1115, 1182, 1001}


def _pg_array_literal(v):
    parts = []
    for x in v:
        if x is None:
            parts.append("NULL")
        elif isinstance(x, (list, tuple)):
            parts.append(_pg_array_literal(x))
        else:
            s = _pg_encode(x).decode()
            parts.append('"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"')
    return "{" + ",".join(parts) + "}"


def _pg_param(v, oid):
    if isinstance(v, (list, tuple)) and (oid in _PG_ARRAY_OIDS or oid == 0 and not isinstance(v, list)):
        return _pg_array_literal(v).encode()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, default=str).encode()
    return _pg_encode(v)


class _Buf:
    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()

    def read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buf)))
            if not chunk:
                raise DatabaseError("the database closed the connection")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def _plain(v):
    """The text of an env.secret(...) value (or the value itself)."""
    return v.reveal() if hasattr(v, "reveal") and callable(v.reveal) else v


class Postgres:
    """One PostgreSQL connection. Not shared between threads — use db.pool() for that."""

    def __init__(self, url=None, host="localhost", port=5432, user=None, password=None, database=None,
                 timeout=10.0, sslmode=None, interp=None, application_name="apsi"):
        url, password = _plain(url), _plain(password)   # env.secret(...) values work directly
        if url:
            u = urlparse(url)
            host = u.hostname or host
            port = u.port or port
            user = unquote(u.username) if u.username else user
            password = unquote(u.password) if u.password else password
            database = (u.path or "/")[1:] or database
            q = parse_qs(u.query)
            sslmode = q.get("sslmode", [sslmode])[0]
        user = user or os.environ.get("PGUSER") or "postgres"
        if password is None and interp is not None:
            password = _env(interp, "PGPASSWORD")
        database = database or user
        _check(interp, host, port, "db.postgres()")
        self.host, self.port, self.user, self.database = host, port, user, database
        self._interp = interp
        self._stmts = {}
        self.params = {}
        self.notices = []
        self.notifications = []
        self._lock = threading.Lock()
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if sslmode in ("require", "verify-ca", "verify-full"):
            self.sock.sendall(struct.pack("!ii", 8, 80877103))
            if self.sock.recv(1) != b"S":
                raise DatabaseError("the server does not support TLS")
            ctx = ssl.create_default_context()
            if sslmode == "require":
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            self.sock = ctx.wrap_socket(self.sock, server_hostname=host)
        self.sock.settimeout(None)
        self.r = _Buf(self.sock)
        body = b""
        for k, v in (("user", user), ("database", database), ("application_name", application_name),
                     ("client_encoding", "UTF8")):
            body += k.encode() + b"\0" + str(v).encode() + b"\0"
        body += b"\0"
        self.sock.sendall(struct.pack("!ii", len(body) + 8, 196608) + body)
        self._auth(password)

    # ---------------------------------------------------------------- protocol
    def _send(self, kind, payload=b""):
        return kind + struct.pack("!i", len(payload) + 4) + payload

    def _msg(self):
        head = self.r.read(5)
        kind, n = head[:1], struct.unpack("!i", head[1:])[0]
        return kind, self.r.read(n - 4)

    def _error(self, payload):
        fields = {}
        for part in payload.split(b"\0"):
            if part:
                fields[chr(part[0])] = part[1:].decode("utf-8", "replace")
        return DatabaseError(f"{fields.get('S', 'ERROR')}: {fields.get('M', '')}" +
                             (f" ({fields['D']})" if "D" in fields else ""), fields.get("C"), fields.get("D"))

    def _auth(self, password):
        while True:
            kind, p = self._msg()
            if kind == b"R":
                code = struct.unpack("!i", p[:4])[0]
                if code == 0:
                    continue
                if code == 3:
                    self._need_pw(password)
                    self.sock.sendall(self._send(b"p", password.encode() + b"\0"))
                elif code == 5:
                    self._need_pw(password)
                    inner = hashlib.md5(password.encode() + self.user.encode()).hexdigest()
                    outer = "md5" + hashlib.md5(inner.encode() + p[4:8]).hexdigest()
                    self.sock.sendall(self._send(b"p", outer.encode() + b"\0"))
                elif code == 10:
                    self._need_pw(password)
                    self._scram(password, p[4:])
                else:
                    raise DatabaseError(f"unsupported authentication method {code}")
            elif kind == b"S":
                k, v = p.split(b"\0")[:2]
                self.params[k.decode()] = v.decode()
            elif kind == b"K":
                self.pid, self.secret = struct.unpack("!ii", p)
            elif kind == b"E":
                raise self._error(p)
            elif kind == b"Z":
                return

    def _need_pw(self, password):
        if password is None:
            raise DatabaseError("the server asks for a password (put it in the URL, password=…, or PGPASSWORD)")

    def _scram(self, password, mechs):
        if b"SCRAM-SHA-256" not in mechs:
            raise DatabaseError("the server offers no supported SASL mechanism")
        nonce = base64.b64encode(os.urandom(18)).decode()
        first_bare = f"n=,r={nonce}"
        msg = b"SCRAM-SHA-256\0" + struct.pack("!i", len(first_bare) + 3) + b"n,," + first_bare.encode()
        self.sock.sendall(self._send(b"p", msg))
        kind, p = self._msg()
        if kind == b"E":
            raise self._error(p)
        server_first = p[4:].decode()
        attrs = dict(x.split("=", 1) for x in server_first.split(","))
        if not attrs["r"].startswith(nonce):
            raise DatabaseError("SCRAM: the server nonce does not extend ours")
        salt = base64.b64decode(attrs["s"])
        salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(attrs["i"]))
        client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
        stored = hashlib.sha256(client_key).digest()
        final_wo = f"c=biws,r={attrs['r']}"
        auth_msg = f"{first_bare},{server_first},{final_wo}".encode()
        sig = hmac.new(stored, auth_msg, hashlib.sha256).digest()
        proof = base64.b64encode(bytes(a ^ b for a, b in zip(client_key, sig))).decode()
        self.sock.sendall(self._send(b"p", f"{final_wo},p={proof}".encode()))
        kind, p = self._msg()
        if kind == b"E":
            raise self._error(p)
        server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
        want = base64.b64encode(hmac.new(server_key, auth_msg, hashlib.sha256).digest()).decode()
        got = dict(x.split("=", 1) for x in p[4:].decode().split(","))
        if not hmac.compare_digest(got.get("v", ""), want):
            raise DatabaseError("SCRAM: the server's signature is wrong (not the server you think)")

    def _read_until_ready(self):
        cols = oids = None
        rows, status, err = [], "", None
        rtype = None
        while True:
            kind, p = self._msg()
            if kind == b"D":
                n = struct.unpack("!h", p[:2])[0]
                off, vals = 2, []
                for i in range(n):
                    ln = struct.unpack("!i", p[off:off + 4])[0]
                    off += 4
                    if ln < 0:
                        vals.append(None)
                    else:
                        vals.append(_pg_decode(oids[i], p[off:off + ln]))
                        off += ln
                rows.append(rtype(vals))
            elif kind == b"T":
                n = struct.unpack("!h", p[:2])[0]
                off, cols, oids = 2, [], []
                for _ in range(n):
                    end = p.index(b"\0", off)
                    cols.append(p[off:end].decode())
                    off = end + 1
                    oids.append(struct.unpack("!i", p[off + 6:off + 10])[0])
                    off += 18
                rtype = _row_type(cols)
            elif kind == b"C":
                status = p[:-1].decode()
            elif kind == b"E":
                err = self._error(p)
            elif kind == b"N":
                self.notices.append(self._error(p).msg)
            elif kind == b"A":
                pid = struct.unpack("!i", p[:4])[0]
                ch, payload = p[4:].split(b"\0")[:2]
                self.notifications.append((ch.decode(), payload.decode(), pid))
            elif kind == b"Z":
                if err is not None:
                    raise err
                parts = status.split()
                count = int(parts[-1]) if parts and parts[-1].isdigit() else len(rows)
                return Result(rows, cols or [], status, count)
            # 1 ParseComplete, 2 BindComplete, n NoData, t ParameterDescription, s PortalSuspended: nothing to do

    # ---------------------------------------------------------------- API
    def _prepare(self, sql):
        """Parse once per connection and learn the parameter types (lists become arrays or json as the column
        needs)."""
        hit = self._stmts.get(sql)
        if hit is not None:
            return hit
        self._n = getattr(self, "_n", 0) + 1
        name = f"apsi_{self._n}".encode()
        self.sock.sendall(self._send(b"P", name + b"\0" + sql.encode() + b"\0" + struct.pack("!h", 0)) +
                          self._send(b"D", b"S" + name + b"\0") + self._send(b"S"))
        oids, err = [], None
        while True:
            kind, p = self._msg()
            if kind == b"t":
                n = struct.unpack("!h", p[:2])[0]
                oids = list(struct.unpack(f"!{n}i", p[2:2 + 4 * n]))
            elif kind == b"E":
                err = self._error(p)
            elif kind == b"Z":
                break
        if err is not None:
            raise err
        if len(self._stmts) >= 256:                         # keep the server-side cache bounded
            old_sql, (old, _) = next(iter(self._stmts.items()))
            del self._stmts[old_sql]
            self.sock.sendall(self._send(b"C", b"S" + old + b"\0") + self._send(b"S"))
            self._read_until_ready()
        self._stmts[sql] = (name, oids)
        return name, oids

    def query(self, sql, *params):
        """Run one statement with $1, $2 … parameters (sent separately — no SQL injection). Returns a Result."""
        with self._lock:
            name, oids = self._prepare(sql)
            vals = [_pg_param(v, oids[i] if i < len(oids) else 0) for i, v in enumerate(params)]
            bind = b"\0" + name + b"\0" + struct.pack("!h", 0) + struct.pack("!h", len(vals))
            for v in vals:
                bind += struct.pack("!i", -1) if v is None else struct.pack("!i", len(v)) + v
            bind += struct.pack("!h", 0)
            self.sock.sendall(self._send(b"B", bind) + self._send(b"D", b"P\0") +
                              self._send(b"E", b"\0" + struct.pack("!i", 0)) + self._send(b"S"))
            return self._read_until_ready()

    def execute(self, sql):
        """Run one or more statements without parameters (the simple protocol). Returns the last Result."""
        with self._lock:
            self.sock.sendall(self._send(b"Q", sql.encode() + b"\0"))
            return self._read_until_ready()

    def insert_many(self, table, rows, columns=None, chunk=1000):
        """Insert many rows in few round trips (multi-row VALUES, parameters, transactional)."""
        rows = [list(r.values()) if isinstance(r, dict) else list(r) for r in rows]
        if not rows:
            return 0
        if columns is None and isinstance(rows[0], dict):
            columns = list(rows[0])
        ncol = len(rows[0])
        cols = f" ({', '.join(_ident(c) for c in columns)})" if columns else ""
        total = 0
        with self.transaction():
            for i in range(0, len(rows), chunk):
                part = rows[i:i + chunk]
                ph = ", ".join("(" + ", ".join(f"${k * ncol + j + 1}" for j in range(ncol)) + ")"
                               for k in range(len(part)))
                flat = [v for r in part for v in r]
                total += self.query(f"insert into {_ident(table)}{cols} values {ph}", *flat).rowcount
        return total

    @contextmanager
    def transaction(self):
        self.execute("begin")
        try:
            yield self
        except BaseException:
            self.execute("rollback")
            raise
        else:
            self.execute("commit")

    def listen(self, channel):
        self.execute(f"listen {_ident(channel)}")

    def notify(self, channel, payload=""):
        self.query("select pg_notify($1, $2)", channel, payload)

    def poll(self, timeout=0.0):
        """Notifications received so far (after listen()); waits up to `timeout` seconds for one."""
        if not self.notifications and timeout:
            self.sock.settimeout(timeout)
            try:
                kind, p = self._msg()
                if kind == b"A":
                    pid = struct.unpack("!i", p[:4])[0]
                    ch, payload = p[4:].split(b"\0")[:2]
                    self.notifications.append((ch.decode(), payload.decode(), pid))
            except (socket.timeout, TimeoutError):
                pass
            finally:
                self.sock.settimeout(None)
        out, self.notifications = self.notifications, []
        return out

    def close(self):
        try:
            self.sock.sendall(self._send(b"X"))
        except OSError:
            pass
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    def __repr__(self):
        return f"<postgres {self.user}@{self.host}:{self.port}/{self.database}>"


def _ident(name):
    name = str(name)
    if not name.replace("_", "").replace(".", "").isalnum():
        return '"' + name.replace('"', '""') + '"'
    return ".".join('"' + p + '"' if p.lower() != p else p for p in name.split("."))


class Pool:
    """A fixed number of connections shared by threads: `with pool.connection() as c: c.query(…)`, or pool.query(…)."""

    def __init__(self, factory, size=8):
        self.factory = factory
        self.q = queue.LifoQueue()
        self.size = size
        self.made = 0
        self.lock = threading.Lock()

    @contextmanager
    def connection(self):
        try:
            c = self.q.get_nowait()
        except queue.Empty:
            with self.lock:
                make = self.made < self.size
                if make:
                    self.made += 1
            c = self.factory() if make else self.q.get()
        try:
            yield c
        except Exception:
            try:
                c.execute("rollback") if isinstance(c, Postgres) else None
            except Exception:                                  # noqa: BLE001 - broken: drop it
                with self.lock:
                    self.made -= 1
                c = None
            raise
        finally:
            if c is not None:
                self.q.put(c)

    def query(self, sql, *params):
        with self.connection() as c:
            return c.query(sql, *params)

    def execute(self, sql):
        with self.connection() as c:
            return c.execute(sql)


# ======================================================================================== Redis
class RedisError(DatabaseError):
    kind = "RedisError"


class Redis:
    def __init__(self, url=None, host="localhost", port=6379, password=None, db=0, username=None, timeout=10.0,
                 decode=True, tls=False, interp=None):
        url, password = _plain(url), _plain(password)   # env.secret(...) values work directly
        if url:
            u = urlparse(url)
            host = u.hostname or host
            port = u.port or port
            password = unquote(u.password) if u.password else password
            username = unquote(u.username) if u.username else username
            if u.path and u.path != "/":
                db = int(u.path[1:])
            tls = tls or u.scheme == "rediss"
        _check(interp, host, port, "db.redis()")
        self.host, self.port, self.decode = host, port, decode
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if tls:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=host)
        self.sock.settimeout(None)
        self.r = _Buf(self.sock)
        self._lock = threading.Lock()
        if password:
            self.command(*(("AUTH", username, password) if username else ("AUTH", password)))
        if db:
            self.command("SELECT", db)

    @staticmethod
    def _pack(args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            if isinstance(a, bytes):
                b = a
            elif isinstance(a, bool):
                b = b"1" if a else b"0"
            elif isinstance(a, (dict, list)):
                b = json.dumps(a).encode()
            else:
                b = str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(b), b))
        return b"".join(out)

    def _line(self):
        buf = self.r.buf
        while True:
            i = buf.find(b"\r\n")
            if i >= 0:
                line = bytes(buf[:i])
                del buf[:i + 2]
                return line
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RedisError("Redis closed the connection")
            buf += chunk

    def _reply(self):
        line = self._line()
        t, rest = line[:1], line[1:]
        if t == b"+":
            return rest.decode() if self.decode else rest
        if t == b"-":
            return RedisError(rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n < 0:
                return None
            data = self.r.read(n + 2)[:-2]
            return data.decode("utf-8", "replace") if self.decode else data
        if t == b"*":
            n = int(rest)
            if n < 0:
                return None
            return [self._reply() for _ in range(n)]
        raise RedisError(f"unexpected reply {line[:40]!r}")

    def command(self, *args):
        with self._lock:
            self.sock.sendall(self._pack(args))
            r = self._reply()
        if isinstance(r, RedisError):
            raise r
        return _shape(args, r)

    def pipeline(self):
        return _Pipeline(self)

    def subscribe(self, *channels):
        """Pub/sub: returns a generator of (channel, message); this connection is then only for listening."""
        self.sock.sendall(self._pack(("SUBSCRIBE",) + channels))
        for _ in channels:
            self._reply()

        def listen():
            while True:
                m = self._reply()
                if isinstance(m, list) and m and m[0] in ("message", b"message"):
                    yield m[1], m[2]
        return listen()

    def publish(self, channel, message):
        return self.command("PUBLISH", channel, message)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cmd = _ALIAS.get(name, name.upper().replace("_", " ").split())

        def call(*args):
            return self.command(*cmd, *args)
        call.__name__ = name
        return call

    def close(self):
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
        return f"<redis {self.host}:{self.port}>"


_ALIAS = {"delete": ["DEL"], "type_": ["TYPE"], "exec_": ["EXEC"]}
_PAIRS = {"HGETALL", "CONFIG GET", "XINFO", "HRANDFIELD"}


def _shape(args, r):
    """Replies that are pairs become dicts (HGETALL); everything else as Redis sent it."""
    if isinstance(r, list) and args:
        name = str(args[0]).upper()
        if name == "CONFIG" and len(args) > 1:
            name = "CONFIG " + str(args[1]).upper()
        if name in _PAIRS and len(r) % 2 == 0:
            return dict(zip(r[0::2], r[1::2]))
    return r


class _Pipeline:
    def __init__(self, conn):
        self.conn = conn
        self.cmds = []

    def command(self, *args):
        self.cmds.append(args)
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cmd = _ALIAS.get(name, name.upper().replace("_", " ").split())
        return lambda *a: self.command(*cmd, *a)

    def execute(self):
        c = self.conn
        with c._lock:
            c.sock.sendall(b"".join(c._pack(a) for a in self.cmds))
            out = [c._reply() for _ in self.cmds]
        cmds, self.cmds = self.cmds, []
        return [r if isinstance(r, RedisError) else _shape(a, r) for a, r in zip(cmds, out)]


# ======================================================================================== SQLite
class SQLite:
    """SQLite through Python's built-in sqlite3, with Apsi's file permissions. Same API as Postgres (? or $1)."""

    def __init__(self, path=":memory:", interp=None):
        import sqlite3
        if path != ":memory:" and interp is not None:
            from .security import check_path
            path = check_path(interp, path, "rw", "db.sqlite()")
        self.c = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()

    def _run(self, sql, params):
        import re
        sql = re.sub(r"\$(\d+)", r"?\1", sql)
        with self._lock:
            cur = self.c.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rt = _row_type(cols)
            rows = [rt(r) for r in cur.fetchall()] if cols else []
            return Result(rows, cols, "", cur.rowcount if cur.rowcount >= 0 else len(rows))

    def query(self, sql, *params):
        return self._run(sql, params)

    def execute(self, sql):
        with self._lock:
            self.c.executescript(sql)
        return Result()

    def insert_many(self, table, rows, columns=None, chunk=1000):
        rows = [list(r.values()) if isinstance(r, dict) else list(r) for r in rows]
        if not rows:
            return 0
        cols = f" ({', '.join(_ident(c) for c in columns)})" if columns else ""
        with self._lock:
            self.c.execute("begin")
            self.c.executemany(f"insert into {_ident(table)}{cols} values ({', '.join('?' * len(rows[0]))})", rows)
            self.c.execute("commit")
        return len(rows)

    @contextmanager
    def transaction(self):
        self.execute("begin")
        try:
            yield self
        except BaseException:
            self.execute("rollback")
            raise
        else:
            self.execute("commit")

    def close(self):
        self.c.close()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False


def _env(interp, name):
    try:
        from .security import check_env
        check_env(interp, name)
        return os.environ.get(name)
    except Exception:                                      # noqa: BLE001 - no env permission: no default
        return None


def module(interp=None):
    def postgres(url=None, **kw):
        return Postgres(url, interp=interp, **kw)

    def redis(url=None, **kw):
        return Redis(url, interp=interp, **kw)

    def sqlite(path=":memory:"):
        return SQLite(path, interp=interp)

    def pool(url, size=8, **kw):
        if str(url).startswith(("redis://", "rediss://")):
            return Pool(lambda: Redis(url, interp=interp, **kw), size)
        return Pool(lambda: Postgres(url, interp=interp, **kw), size)
    return {"postgres": postgres, "redis": redis, "sqlite": sqlite, "pool": pool, "DatabaseError": DatabaseError,
            "RedisError": RedisError, "Result": Result}
