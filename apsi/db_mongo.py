"""MongoDB for `use db` — BSON and the OP_MSG wire protocol written in Apsi's runtime, no driver to install.

    use db
    let mongo = db.mongo("mongodb://app:secret@localhost:27017/shop?authSource=admin")   # --allow-net=localhost:27017
    let items = mongo.collection("items")                      # or mongo["shop"]["items"]
    items.insert_one({"name": "pen", "price": 1.25, "tags": ["office"]})
    for doc in items.find({"price": {"$lt": 10}}, sort={"price": 1}, limit=20):
        print(doc["name"], doc["price"])
    items.update_one({"name": "pen"}, {"$set": {"price": 1.5}})
    print(items.count_documents({}))
    print(items.aggregate([{"$group": {"_id": none, "total": {"$sum": "$price"}}}]))

Authentication: SCRAM-SHA-256 and SCRAM-SHA-1 (authSource=…), TLS with ?tls=true. MongoDB 3.6 and newer.
Values: dict ↔ document (key order kept), list ↔ array, int (32/64-bit), float, Decimal (decimal128), str, bytes,
bool, none, datetime (UTC), ObjectId, regular expressions.
"""
import base64
import datetime as _dt
import decimal
import hashlib
import hmac
import os
import re
import socket
import ssl
import struct
import threading
import unicodedata
from urllib.parse import urlparse, unquote, parse_qs

from .db import DatabaseError, _check, _plain


class MongoError(DatabaseError):
    kind = "DatabaseError"


# ---------------------------------------------------------------- BSON
class ObjectId:
    """A 12-byte MongoDB id: ObjectId() makes a new one, ObjectId("65f…") reads one."""
    _counter = int.from_bytes(os.urandom(3), "big")
    _rand = os.urandom(5)
    _lock = threading.Lock()
    __slots__ = ("binary",)

    def __init__(self, value=None):
        if value is None:
            with ObjectId._lock:
                ObjectId._counter = (ObjectId._counter + 1) & 0xFFFFFF
                c = ObjectId._counter
            import time
            self.binary = struct.pack(">I", int(time.time())) + ObjectId._rand + c.to_bytes(3, "big")
        elif isinstance(value, ObjectId):
            self.binary = value.binary
        elif isinstance(value, (bytes, bytearray)) and len(value) == 12:
            self.binary = bytes(value)
        elif isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{24}", value):
            self.binary = bytes.fromhex(value)
        else:
            raise MongoError(f"not an ObjectId: {value!r}")

    @property
    def time(self):
        return _dt.datetime.fromtimestamp(struct.unpack(">I", self.binary[:4])[0], _dt.timezone.utc)

    def __str__(self):
        return self.binary.hex()

    def __repr__(self):
        return f"ObjectId('{self.binary.hex()}')"

    def __eq__(self, o):
        return isinstance(o, ObjectId) and o.binary == self.binary

    def __hash__(self):
        return hash(self.binary)


class Binary(bytes):
    """Bytes with a BSON binary subtype (0 generic, 4 UUID …)."""

    def __new__(cls, data, subtype=0):
        b = super().__new__(cls, data)
        b.subtype = subtype
        return b


def _cstring(s):
    b = s.encode("utf-8")
    if b"\0" in b:
        raise MongoError("BSON keys cannot contain a NUL character")
    return b + b"\0"


def _dec128_encode(d):
    d = decimal.Decimal(d)
    sign, digits, exp = d.as_tuple()
    if not isinstance(exp, int):
        raise MongoError("decimal128 cannot hold NaN or infinity here")
    coeff = int("".join(map(str, digits)) or "0")
    if coeff >= 1 << 113 or not -6176 <= exp <= 6111:
        raise MongoError(f"{d} does not fit in decimal128")
    hi = (sign << 63) | ((exp + 6176) << 49) | (coeff >> 64)
    lo = coeff & ((1 << 64) - 1)
    return struct.pack("<QQ", lo, hi)


def _dec128_decode(b):
    lo, hi = struct.unpack("<QQ", b)
    sign = hi >> 63
    if (hi >> 61) & 3 == 3:
        comb = (hi >> 58) & 0x1F
        if comb == 0x1E:
            return decimal.Decimal("-Infinity" if sign else "Infinity")
        if comb == 0x1F:
            return decimal.Decimal("NaN")
        exp = ((hi >> 47) & 0x3FFF) - 6176
        coeff = 0                                        # non-canonical: treated as zero by the specification
    else:
        exp = ((hi >> 49) & 0x3FFF) - 6176
        coeff = ((hi & ((1 << 49) - 1)) << 64) | lo
    return decimal.Decimal((sign, tuple(int(c) for c in str(coeff)), exp))


def encode(doc):
    """A dict (or list of pairs) → BSON bytes."""
    body = bytearray()
    items = doc.items() if hasattr(doc, "items") else doc
    for k, v in items:
        _element(body, str(k), v)
    return struct.pack("<i", len(body) + 5) + bytes(body) + b"\0"


def _element(out, key, v):
    k = _cstring(key)
    if v is None:
        out += b"\x0a" + k
    elif v is True or v is False:
        out += b"\x08" + k + (b"\x01" if v else b"\x00")
    elif isinstance(v, int):
        if -(1 << 31) <= v < 1 << 31:
            out += b"\x10" + k + struct.pack("<i", v)
        elif -(1 << 63) <= v < 1 << 63:
            out += b"\x12" + k + struct.pack("<q", v)
        else:
            raise MongoError(f"{v} does not fit in a 64-bit BSON integer")
    elif isinstance(v, float):
        out += b"\x01" + k + struct.pack("<d", v)
    elif isinstance(v, decimal.Decimal):
        out += b"\x13" + k + _dec128_encode(v)
    elif isinstance(v, str):
        b = v.encode("utf-8")
        out += b"\x02" + k + struct.pack("<i", len(b) + 1) + b + b"\0"
    elif isinstance(v, ObjectId):
        out += b"\x07" + k + v.binary
    elif isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        out += b"\x05" + k + struct.pack("<iB", len(b), getattr(v, "subtype", 0)) + b
    elif isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
        ms = int((v - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)) // _dt.timedelta(milliseconds=1))
        out += b"\x09" + k + struct.pack("<q", ms)
    elif isinstance(v, _dt.date):
        _element(out, key, _dt.datetime(v.year, v.month, v.day, tzinfo=_dt.timezone.utc))
    elif isinstance(v, re.Pattern):
        flags = "".join(c for f, c in ((re.I, "i"), (re.M, "m"), (re.S, "s"), (re.X, "x")) if v.flags & f)
        out += b"\x0b" + k + _cstring(v.pattern) + _cstring(flags)
    elif hasattr(v, "items"):
        out += b"\x03" + k + encode(v)
    elif isinstance(v, (list, tuple)):
        out += b"\x04" + k + encode((str(i), x) for i, x in enumerate(v))
    elif hasattr(v, "tolist"):
        _element(out, key, v.tolist())
    else:
        raise MongoError(f"cannot store a {type(v).__name__} in MongoDB")


def decode(b, offset=0, as_list=False):
    """BSON bytes → dict (or list for arrays)."""
    n = struct.unpack_from("<i", b, offset)[0]
    end = offset + n - 1
    i = offset + 4
    out = [] if as_list else {}
    while i < end:
        t = b[i]
        j = b.index(b"\0", i + 1)
        key = b[i + 1:j].decode("utf-8")
        i = j + 1
        if t == 0x01:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        elif t in (0x02, 0x0D, 0x0E):
            ln = struct.unpack_from("<i", b, i)[0]
            v = b[i + 4:i + 3 + ln].decode("utf-8")
            i += 4 + ln
        elif t in (0x03, 0x04):
            ln = struct.unpack_from("<i", b, i)[0]
            v = decode(b, i, as_list=t == 0x04)
            i += ln
        elif t == 0x05:
            ln, sub = struct.unpack_from("<iB", b, i)
            data = b[i + 5:i + 5 + ln]
            if sub == 4 and ln == 16:
                import uuid
                v = uuid.UUID(bytes=bytes(data))
            else:
                v = Binary(data, sub)
            i += 5 + ln
        elif t == 0x07:
            v = ObjectId(bytes(b[i:i + 12]))
            i += 12
        elif t == 0x08:
            v = b[i] == 1
            i += 1
        elif t == 0x09:
            ms = struct.unpack_from("<q", b, i)[0]
            v = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(milliseconds=ms)
            i += 8
        elif t in (0x0A, 0x06, 0xFF, 0x7F):
            v = None
        elif t == 0x0B:
            j = b.index(b"\0", i)
            pat = b[i:j].decode()
            k2 = b.index(b"\0", j + 1)
            fl = b[j + 1:k2].decode()
            i = k2 + 1
            flags = 0
            for c, f in (("i", re.I), ("m", re.M), ("s", re.S), ("x", re.X)):
                if c in fl:
                    flags |= f
            try:
                v = re.compile(pat, flags)
            except re.error:
                v = pat
        elif t == 0x10:
            v = struct.unpack_from("<i", b, i)[0]
            i += 4
        elif t == 0x11:
            inc, ts = struct.unpack_from("<II", b, i)
            v = (ts, inc)
            i += 8
        elif t == 0x12:
            v = struct.unpack_from("<q", b, i)[0]
            i += 8
        elif t == 0x13:
            v = _dec128_decode(bytes(b[i:i + 16]))
            i += 16
        elif t == 0x0F:                                    # code with scope (deprecated)
            ln = struct.unpack_from("<i", b, i)[0]
            v = None
            i += ln
        else:
            raise MongoError(f"unknown BSON type 0x{t:02x}")
        if as_list:
            out.append(v)
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------- SCRAM (RFC 5802 / 7677)
def _saslprep(s):
    return unicodedata.normalize("NFKC", s)


class Scram:
    def __init__(self, mech, user, password, nonce=None):
        self.mech = mech
        self.h = hashlib.sha256 if mech == "SCRAM-SHA-256" else hashlib.sha1
        self.user = user.replace("=", "=3D").replace(",", "=2C")
        if mech == "SCRAM-SHA-1":                         # MongoDB hashes the password first for SHA-1
            password = hashlib.md5(f"{user}:mongo:{password}".encode()).hexdigest()
        else:
            password = _saslprep(password)
        self.password = password.encode()
        self.nonce = nonce or base64.b64encode(os.urandom(24)).decode()
        self.first_bare = f"n={self.user},r={self.nonce}"

    def first(self):
        return ("n,," + self.first_bare).encode()

    def final(self, server_first):
        s = server_first.decode()
        f = dict(p.split("=", 1) for p in s.split(","))
        if not f["r"].startswith(self.nonce):
            raise MongoError("authentication failed: the server nonce does not extend ours")
        salted = hashlib.pbkdf2_hmac(self.h().name, self.password, base64.b64decode(f["s"]), int(f["i"]))
        client_key = hmac.new(salted, b"Client Key", self.h).digest()
        stored = self.h(client_key).digest()
        without_proof = f"c=biws,r={f['r']}"
        auth_msg = f"{self.first_bare},{s},{without_proof}".encode()
        sig = hmac.new(stored, auth_msg, self.h).digest()
        proof = bytes(a ^ b for a, b in zip(client_key, sig))
        server_key = hmac.new(salted, b"Server Key", self.h).digest()
        self.server_sig = base64.b64encode(hmac.new(server_key, auth_msg, self.h).digest()).decode()
        return f"{without_proof},p={base64.b64encode(proof).decode()}".encode()

    def verify(self, server_final):
        f = dict(p.split("=", 1) for p in server_final.decode().split(","))
        if "e" in f:
            raise MongoError(f"authentication failed: {f['e']}")
        if not hmac.compare_digest(f.get("v", ""), self.server_sig):
            raise MongoError("authentication failed: the server could not prove it knows the password")


# ---------------------------------------------------------------- connection
OP_MSG = 2013


class Mongo:
    def __init__(self, url=None, host="localhost", port=27017, user=None, password=None, database=None,
                 auth_source=None, tls=False, timeout=10.0, interp=None, mechanism=None):
        url, password = _plain(url), _plain(password)
        if url:
            u = urlparse(url)
            if u.scheme != "mongodb":
                raise MongoError("db.mongo: the URL must start with mongodb:// (mongodb+srv:// needs DNS SRV lookups: "
                                 "give the host instead)")
            host = u.hostname or host
            port = u.port or port
            user = unquote(u.username) if u.username else user
            password = unquote(u.password) if u.password is not None else password
            database = u.path.lstrip("/") or database
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            auth_source = q.get("authSource", auth_source)
            tls = tls or q.get("tls") in ("true", "1") or q.get("ssl") in ("true", "1")
            mechanism = q.get("authMechanism", mechanism)
        self.host, self.port = host, int(port)
        self.database = database or "test"
        _check(interp, self.host, self.port, "db.mongo()")
        self.lock = threading.RLock()
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if tls:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)
        self.req = 0
        try:
            hello = self.command({"hello": 1, "client": {"application": {"name": "apsi"},
                                                         "driver": {"name": "apsi", "version": "2.4.4"}}}, db="admin")
            self.server = hello
            if user:
                mechs = hello.get("saslSupportedMechs") or []
                mech = mechanism or ("SCRAM-SHA-256" if "SCRAM-SHA-256" in mechs or not mechs else "SCRAM-SHA-1")
                self._auth(mech, user, password or "", auth_source or database or "admin")
        except BaseException:
            self.sock.close()
            raise

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise MongoError("the MongoDB server closed the connection")
            buf += c
        return bytes(buf)

    def command(self, cmd, db=None):
        """Run a database command ({"ping": 1}, {"listCollections": 1} …) and return its reply."""
        doc = dict(cmd)
        doc["$db"] = db or self.database
        with self.lock:
            self.req = (self.req + 1) & 0x7FFFFFFF
            body = struct.pack("<I", 0) + b"\x00" + encode(doc)
            self.sock.sendall(struct.pack("<iiii", 16 + len(body), self.req, 0, OP_MSG) + body)
            ln, _, _, op = struct.unpack("<iiii", self._recv(16))
            data = self._recv(ln - 16)
        if op != OP_MSG:
            raise MongoError(f"unexpected MongoDB reply (opcode {op})")
        i = 4
        reply = None
        while i < len(data):
            kind = data[i]
            i += 1
            if kind == 0:
                n = struct.unpack_from("<i", data, i)[0]
                reply = decode(data, i)
                i += n
            else:
                n = struct.unpack_from("<i", data, i)[0]
                i += n
        if reply is None:
            raise MongoError("empty MongoDB reply")
        if reply.get("ok") != 1 and reply.get("ok") != 1.0:
            raise MongoError(f"MongoDB: {reply.get('errmsg', 'command failed')}", code=reply.get("code"),
                             detail=reply.get("codeName"))
        if reply.get("writeErrors"):
            e = reply["writeErrors"][0]
            raise MongoError(f"MongoDB: {e.get('errmsg')}", code=e.get("code"))
        return reply

    def _auth(self, mech, user, password, source):
        sc = Scram(mech, user, password)
        r = self.command({"saslStart": 1, "mechanism": mech, "payload": Binary(sc.first()),
                          "options": {"skipEmptyExchange": True}}, db=source)
        proof = sc.final(bytes(r["payload"]))
        r = self.command({"saslContinue": 1, "conversationId": r["conversationId"], "payload": Binary(proof)},
                         db=source)
        sc.verify(bytes(r["payload"]))
        while not r.get("done"):
            r = self.command({"saslContinue": 1, "conversationId": r["conversationId"], "payload": Binary(b"")},
                             db=source)

    def __getitem__(self, name):
        return Database(self, name)

    def collection(self, name, database=None):
        return Collection(self, database or self.database, name)

    def ping(self):
        return self.command({"ping": 1}, db="admin").get("ok") == 1

    def list_databases(self):
        return [d["name"] for d in self.command({"listDatabases": 1, "nameOnly": True}, db="admin")["databases"]]

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
        return f"<mongo {self.host}:{self.port}/{self.database}>"


class Database:
    def __init__(self, conn, name):
        self.conn, self.name = conn, name

    def __getitem__(self, coll):
        return Collection(self.conn, self.name, coll)

    def collection(self, name):
        return Collection(self.conn, self.name, name)

    def command(self, cmd):
        return self.conn.command(cmd, db=self.name)

    def list_collections(self):
        r = self.command({"listCollections": 1, "nameOnly": True})
        return [c["name"] for c in Cursor(self.conn, self.name, r["cursor"]).all()]

    def drop(self):
        return self.command({"dropDatabase": 1})


class Cursor:
    """Iterates a server cursor, fetching further batches (getMore) as needed."""

    def __init__(self, conn, db, cur, limit=0):
        self.conn, self.db = conn, db
        self.batch = list(cur.get("firstBatch", []))
        self.id = cur.get("id", 0)
        self.ns = cur.get("ns", f"{db}.")
        self.limit = limit
        self.seen = 0

    def __iter__(self):
        while True:
            while self.batch:
                if self.limit and self.seen >= self.limit:
                    return
                self.seen += 1
                yield self.batch.pop(0)
            if not self.id or self.limit and self.seen >= self.limit:
                return
            coll = self.ns.split(".", 1)[1]
            r = self.conn.command({"getMore": self.id, "collection": coll}, db=self.db)
            self.batch = list(r["cursor"].get("nextBatch", []))
            self.id = r["cursor"].get("id", 0)

    def all(self):
        return list(self)


class Collection:
    def __init__(self, conn, db, name):
        self.conn, self.db, self.name = conn, db, name

    def _cmd(self, cmd):
        return self.conn.command(cmd, db=self.db)

    def insert_one(self, doc):
        doc = dict(doc)
        doc.setdefault("_id", ObjectId())
        self._cmd({"insert": self.name, "documents": [doc]})
        return doc["_id"]

    def insert_many(self, docs, ordered=True, chunk=10000):
        docs = [dict(d) for d in docs]
        for d in docs:
            d.setdefault("_id", ObjectId())
        for s in range(0, len(docs), chunk):
            self._cmd({"insert": self.name, "documents": docs[s:s + chunk], "ordered": ordered})
        return [d["_id"] for d in docs]

    def find(self, filter=None, projection=None, sort=None, limit=0, skip=0, batch_size=0):
        cmd = {"find": self.name, "filter": filter or {}}
        if projection:
            cmd["projection"] = projection
        if sort:
            cmd["sort"] = sort
        if limit:
            cmd["limit"] = int(limit)
        if skip:
            cmd["skip"] = int(skip)
        if batch_size:
            cmd["batchSize"] = int(batch_size)
        return Cursor(self.conn, self.db, self._cmd(cmd)["cursor"], int(limit or 0)).all()

    def find_one(self, filter=None, projection=None, sort=None):
        r = self.find(filter, projection, sort, limit=1)
        return r[0] if r else None

    def _update(self, filter, update, multi, upsert):
        r = self._cmd({"update": self.name, "updates": [{"q": filter, "u": update, "multi": multi, "upsert": upsert}]})
        return {"matched": r.get("n", 0), "modified": r.get("nModified", 0),
                "upserted": (r.get("upserted") or [{}])[0].get("_id")}

    def update_one(self, filter, update, upsert=False):
        return self._update(filter, update, False, upsert)

    def update_many(self, filter, update, upsert=False):
        return self._update(filter, update, True, upsert)

    def replace_one(self, filter, doc, upsert=False):
        return self._update(filter, dict(doc), False, upsert)

    def delete_one(self, filter):
        return self._cmd({"delete": self.name, "deletes": [{"q": filter, "limit": 1}]}).get("n", 0)

    def delete_many(self, filter):
        return self._cmd({"delete": self.name, "deletes": [{"q": filter, "limit": 0}]}).get("n", 0)

    def count_documents(self, filter=None):
        r = self.aggregate([{"$match": filter or {}}, {"$count": "n"}])
        return r[0]["n"] if r else 0

    def aggregate(self, pipeline):
        r = self._cmd({"aggregate": self.name, "pipeline": list(pipeline), "cursor": {}})
        return Cursor(self.conn, self.db, r["cursor"]).all()

    def distinct(self, key, filter=None):
        return self._cmd({"distinct": self.name, "key": key, "query": filter or {}})["values"]

    def create_index(self, keys, unique=False, name=None):
        keys = {keys: 1} if isinstance(keys, str) else dict(keys)
        name = name or "_".join(f"{k}_{v}" for k, v in keys.items())
        self._cmd({"createIndexes": self.name, "indexes": [{"key": keys, "name": name, "unique": bool(unique)}]})
        return name

    def drop(self):
        try:
            self._cmd({"drop": self.name})
        except MongoError as e:
            if "ns not found" not in str(e):
                raise

    def __repr__(self):
        return f"<collection {self.db}.{self.name}>"
