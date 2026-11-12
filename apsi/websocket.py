"""WebSocket (RFC 6455) for `use web`: server routes, rooms/broadcast and a client.

    use web
    let app = web.server()

    @app.websocket("/echo")
    fn echo(ws, msg):                      # message mode: called for each message (scales to many thousands of
        ws.send(msg)                       # connections — no thread is parked per connection)

    @app.websocket("/chat")
    fn chat(ws):                           # session mode: a conversation, one thread per connection
        ws.join("lobby")
        for msg in ws:                     # ends when the client closes
            app.room("lobby").send({"from": ws.id, "text": msg})

    let c = web.websocket_connect("ws://localhost:8080/echo")   # needs --allow-net
    c.send("hi"); print(c.receive())

Text and binary messages, dicts/lists sent as JSON, fragmentation, ping/pong keep-alive (dead peers dropped),
the closing handshake, a size limit per message (max_message, 16 MB by default), UTF-8 checking of text, masking
enforced from clients, back-pressure on slow receivers, and Origin checking (`origins=[…]`).
"""
import asyncio
import base64
import collections
import hashlib
import json
import os
import queue
import socket
import ssl
import struct
import threading
import uuid
from urllib.parse import urlparse

from .errors import RuntimeErr

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 8, 9, 10
_CLOSED = object()


class WebSocketClosed(RuntimeErr):
    kind = "WebSocketClosed"


def accept_key(key):
    return base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode()


def frame(op, payload, mask=False):
    n = len(payload)
    head = bytearray([0x80 | op])
    mbit = 0x80 if mask else 0
    if n < 126:
        head.append(mbit | n)
    elif n < 65536:
        head.append(mbit | 126)
        head += struct.pack("!H", n)
    else:
        head.append(mbit | 127)
        head += struct.pack("!Q", n)
    if mask:
        key = os.urandom(4)
        head += key
        payload = _xor(payload, key)
    return bytes(head) + payload


def _xor(data, key):
    if not data:
        return b""
    n = len(data)
    k = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(k, "little")).to_bytes(n, "little")


def _encode(msg):
    if isinstance(msg, (bytes, bytearray, memoryview)):
        return OP_BIN, bytes(msg)
    if isinstance(msg, str):
        return OP_TEXT, msg.encode("utf-8")
    return OP_TEXT, json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")


class Room:
    def __init__(self, name):
        self.name = name
        self.members = set()
        self.lock = threading.Lock()

    def send(self, msg, exclude=None):
        """Send to every member (one encoding, many writes). Returns how many received it."""
        op, data = _encode(msg)
        packet = frame(op, data)
        with self.lock:
            members = list(self.members)
        n = 0
        for ws in members:
            if ws is exclude:
                continue
            if ws._send_raw(packet):
                n += 1
        return n

    def __len__(self):
        return len(self.members)


class ServerSocket:
    """One accepted WebSocket connection (used from handler threads)."""

    def __init__(self, app, req, writer, loop, max_message):
        self.app, self.request, self.writer, self.loop = app, req, writer, loop
        self.id = uuid.uuid4().hex[:12]
        self.path = req.path
        self.params = getattr(req, "params", {})
        self.inbox = queue.Queue()
        self.closed = False
        self.max_message = max_message
        self.rooms = set()
        self.close_code = None
        self._pending = 0
        self._mq = collections.deque()
        self._draining = False
        self._mlock = threading.Lock()
        self.state = {}                       # free storage for the application (user name, …)

    # ---- sending (thread-safe)
    def _send_raw(self, packet):
        if self.closed:
            return False
        if self._pending > (8 << 20):         # a receiver that does not read: stop queueing (back-pressure)
            return False
        self._pending += len(packet)

        def write():
            self._pending -= len(packet)
            if not self.writer.is_closing():
                self.writer.write(packet)
        try:
            self.loop.call_soon_threadsafe(write)
        except RuntimeError:
            return False
        return True

    def send(self, msg):
        op, data = _encode(msg)
        if not self._send_raw(frame(op, data)):
            if self.closed:
                raise WebSocketClosed("the connection is closed")
        return self

    def send_json(self, obj):
        return self.send(json.dumps(obj, default=str))

    def receive(self, timeout=None):
        """The next message (str for text, bytes for binary), or none when the connection closed."""
        if self.closed and self.inbox.empty():
            return None
        try:
            m = self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if m is _CLOSED else m

    def receive_json(self, timeout=None):
        m = self.receive(timeout)
        return None if m is None else json.loads(m)

    def __iter__(self):
        while True:
            m = self.receive()
            if m is None:
                return
            yield m

    def join(self, room):
        r = self.app.room(room)
        with r.lock:
            r.members.add(self)
        self.rooms.add(r)
        return r

    def leave(self, room):
        r = self.app.room(room)
        with r.lock:
            r.members.discard(self)
        self.rooms.discard(r)

    def close(self, code=1000, reason=""):
        if not self.closed:
            self._send_raw(frame(OP_CLOSE, struct.pack("!H", code) + reason.encode()[:120]))
            self.closed = True
            self.loop.call_soon_threadsafe(self._shutdown)

    def _shutdown(self):
        try:
            self.writer.close()
        except Exception:                                  # noqa: BLE001
            pass

    def _gone(self):
        self.closed = True
        for r in list(self.rooms):
            with r.lock:
                r.members.discard(self)
        self.inbox.put(_CLOSED)

    def __repr__(self):
        return f"<websocket {self.id} {self.path}>"


async def serve(app, route, req, reader, writer, headers):
    """Upgrade an HTTP connection and run the WebSocket until it closes (called from the event loop)."""
    loop = asyncio.get_running_loop()
    key = headers.get("sec-websocket-key", "")
    if headers.get("sec-websocket-version") != "13" or len(base64.b64decode(key + "==")[:16]) != 16:
        writer.write(b"HTTP/1.1 400 Bad Request\r\nSec-WebSocket-Version: 13\r\nContent-Length: 0\r\n\r\n")
        return
    origins = route.options.get("origins")
    if origins and headers.get("origin") not in origins:
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        return
    proto = ""
    want = [p.strip() for p in headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    if want and route.options.get("subprotocols"):
        pick = next((p for p in want if p in route.options["subprotocols"]), None)
        if pick:
            proto = f"Sec-WebSocket-Protocol: {pick}\r\n"
    writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept_key(key)}\r\n{proto}\r\n").encode())
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    ws = ServerSocket(app, req, writer, loop, route.options.get("max_message", 16 << 20))
    handler = route.handler
    message_mode = route.options.get("message_mode")

    def run_session():
        try:
            handler(ws)
        except Exception as e:                             # noqa: BLE001 - a failing handler closes its socket
            app._log_ws_error(ws, e)
        finally:
            if not ws.closed:
                ws.close(1000)

    if not message_mode:
        threading.Thread(target=run_session, name=f"ws-{ws.id}", daemon=True).start()
    on_open = route.options.get("on_open")
    if message_mode and on_open:
        loop.run_in_executor(None, on_open, ws)
    ping_every = route.options.get("ping", 20.0)
    parts, part_op, size = [], 0, 0
    try:
        while not ws.closed:
            try:
                h = await asyncio.wait_for(reader.readexactly(2), ping_every * 2 if ping_every else None)
            except asyncio.TimeoutError:
                if ping_every:
                    writer.write(frame(OP_PING, b"apsi"))
                    try:
                        h = await asyncio.wait_for(reader.readexactly(2), ping_every)
                    except asyncio.TimeoutError:
                        break                               # no answer to a ping: the peer is gone
                else:
                    continue
            fin, op = h[0] & 0x80, h[0] & 0x0F
            if h[0] & 0x70:
                await _fail(writer, 1002, "reserved bits set")
                break
            if not h[1] & 0x80:
                await _fail(writer, 1002, "client frames must be masked")
                break
            n = h[1] & 0x7F
            if n == 126:
                n = struct.unpack("!H", await reader.readexactly(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", await reader.readexactly(8))[0]
            if op >= 8 and (n > 125 or not fin):
                await _fail(writer, 1002, "bad control frame")
                break
            if size + n > ws.max_message:
                await _fail(writer, 1009, "message too big")
                break
            mkey = await reader.readexactly(4)
            data = _xor(await reader.readexactly(n), mkey)
            if op == OP_PING:
                writer.write(frame(OP_PONG, data))
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                code = struct.unpack("!H", data[:2])[0] if len(data) >= 2 else 1005
                ws.close_code = code
                if not ws.closed:
                    writer.write(frame(OP_CLOSE, data[:2]))
                break
            if op in (OP_TEXT, OP_BIN):
                if parts:
                    await _fail(writer, 1002, "new message inside a fragmented one")
                    break
                part_op = op
            elif op != OP_CONT or not parts:
                await _fail(writer, 1002, "unexpected continuation or unknown opcode")
                break
            parts.append(data)
            size += n
            if not fin:
                continue
            payload = b"".join(parts)
            parts, size = [], 0
            if part_op == OP_TEXT:
                try:
                    msg = payload.decode("utf-8")
                except UnicodeDecodeError:
                    await _fail(writer, 1007, "invalid UTF-8")
                    break
            else:
                msg = payload
            part_op = 0
            if message_mode:
                # one drain task per connection at a time: messages are handled in order, in batches
                with ws._mlock:
                    ws._mq.append(msg)
                    start = not ws._draining
                    ws._draining = True
                if start:
                    loop.run_in_executor(None, _drain, app, handler, ws)
            else:
                ws.inbox.put(msg)
            if writer.transport.get_write_buffer_size() > (4 << 20):
                await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass
    finally:
        ws._gone()
        on_close = route.options.get("on_close")
        if on_close:
            loop.run_in_executor(None, on_close, ws)
        try:
            await writer.drain()
        except Exception:                                  # noqa: BLE001
            pass
        writer.close()


def _drain(app, handler, ws):
    q = ws._mq
    while True:
        with ws._mlock:
            if not q:
                ws._draining = False
                return
            msg = q.popleft()
        try:
            handler(ws, msg)
        except Exception as e:                             # noqa: BLE001
            app._log_ws_error(ws, e)


async def _fail(writer, code, reason):
    writer.write(frame(OP_CLOSE, struct.pack("!H", code) + reason.encode()))
    try:
        await writer.drain()
    except Exception:                                      # noqa: BLE001
        pass


# ======================================================================================== client
class Client:
    """A blocking WebSocket client: send(), receive(), iteration, close(); ping answered automatically."""

    def __init__(self, url, headers=None, timeout=10.0, subprotocols=None, max_message=16 << 20):
        u = urlparse(url)
        if u.scheme not in ("ws", "wss"):
            raise RuntimeErr("websocket_connect() needs a ws:// or wss:// URL")
        host, port = u.hostname, u.port or (443 if u.scheme == "wss" else 80)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if u.scheme == "wss":
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        lines = [f"GET {path} HTTP/1.1", f"Host: {host}:{port}", "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        if subprotocols:
            lines.append("Sec-WebSocket-Protocol: " + ", ".join(subprotocols))
        for k, v in (headers or {}).items():
            lines.append(f"{k}: {v}")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        self.buf = bytearray()
        head = self._read_until(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise RuntimeErr(f"WebSocket handshake failed: {status.decode(errors='replace')}")
        if accept_key(key).encode() not in head:
            raise RuntimeErr("WebSocket handshake failed: wrong Sec-WebSocket-Accept")
        self.sock.settimeout(None)
        self.closed = False
        self.max_message = max_message
        self.lock = threading.Lock()

    def _read_until(self, sep):
        while sep not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeErr("the server closed the connection")
            self.buf += chunk
        i = self.buf.index(sep) + len(sep)
        out = bytes(self.buf[:i])
        del self.buf[:i]
        return out

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buf)))
            if not chunk:
                raise WebSocketClosed("the server closed the connection")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def send(self, msg):
        op, data = _encode(msg)
        with self.lock:
            self.sock.sendall(frame(op, data, mask=True))
        return self

    def receive(self, timeout=None):
        self.sock.settimeout(timeout)
        parts, op0 = [], None
        try:
            while True:
                h = self._read(2)
                fin, op = h[0] & 0x80, h[0] & 0x0F
                n = h[1] & 0x7F
                if n == 126:
                    n = struct.unpack("!H", self._read(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", self._read(8))[0]
                if n > self.max_message:
                    raise RuntimeErr("WebSocket message too big")
                data = self._read(n)
                if op == OP_PING:
                    with self.lock:
                        self.sock.sendall(frame(OP_PONG, data, mask=True))
                    continue
                if op == OP_PONG:
                    continue
                if op == OP_CLOSE:
                    self.closed = True
                    try:
                        self.sock.sendall(frame(OP_CLOSE, data[:2], mask=True))
                    except OSError:
                        pass
                    return None
                if op0 is None:
                    op0 = op
                parts.append(data)
                if fin:
                    payload = b"".join(parts)
                    return payload.decode("utf-8") if op0 == OP_TEXT else payload
        except (socket.timeout, TimeoutError):
            return None
        finally:
            if not self.closed:
                self.sock.settimeout(None)

    def __iter__(self):
        while True:
            m = self.receive()
            if m is None:
                return
            yield m

    def close(self, code=1000):
        if not self.closed:
            try:
                self.sock.sendall(frame(OP_CLOSE, struct.pack("!H", code), mask=True))
            except OSError:
                pass
            self.closed = True
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False
