"""`use layout` — exact binary layouts: every byte of a C struct, a packet, a file header or a device register, the
same in Apsi and in C.

  c_struct(name, fields, pack=0, endian="little")   a layout: fields [(name, type), …] in order
      types  u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool8 char, array(type, n), another c_struct,
             bits(u8|u16|u32|u64, [(name, width), …])   (bit fields, the lowest bits first)
      pack   0: the C compiler's natural alignment (the default of GCC, Clang and MSVC for these types);
             1, 2, 4, 8: #pragma pack(n) / __attribute__((packed)) for 1
      endian "little", "big" or "native"
  L.size, L.align, L.offset("field"), L.describe()
  L.pack(record or dict) -> bytes        L.unpack(bytes, offset=0) -> record (missing fields: zero)
  L.view(buffer, offset=0)               read and write fields in place (a bytearray, a NumPy array, memory)
  L.numpy_dtype()                        a NumPy dtype with the same offsets: arrays of records in one read
  L.c_header(guard=True)                 the C definition with static assertions of every offset and the size,
                                         bit fields as the base integer plus _SHIFT / _MASK macros (no compiler
                                         decides their layout)
  register_map(name, base, registers)    device registers [(name, offset, type, bits?)] -> c_header() with the
                                         addresses as volatile pointers, read(buffer) / write over a memory image
  sizeof(t), alignof(t), offsetof(L, "field")
"""
import struct as _st

import numpy as np

from .errors import RuntimeErr
from .values import Record


class LayoutError(RuntimeErr):
    pass


class Scalar:
    def __init__(self, name, code, size, ctype, np_code):
        self.name, self.code, self.size, self.align, self.ctype, self.np = name, code, size, size, ctype, np_code

    def describe(self):
        return self.name

    __repr__ = describe


u8, u16, u32, u64 = (Scalar("u8", "B", 1, "uint8_t", "u1"), Scalar("u16", "H", 2, "uint16_t", "u2"),
                     Scalar("u32", "I", 4, "uint32_t", "u4"), Scalar("u64", "Q", 8, "uint64_t", "u8"))
i8, i16, i32, i64 = (Scalar("i8", "b", 1, "int8_t", "i1"), Scalar("i16", "h", 2, "int16_t", "i2"),
                     Scalar("i32", "i", 4, "int32_t", "i4"), Scalar("i64", "q", 8, "int64_t", "i8"))
f32, f64 = Scalar("f32", "f", 4, "float", "f4"), Scalar("f64", "d", 8, "double", "f8")
bool8 = Scalar("bool8", "?", 1, "bool", "?")
char = Scalar("char", "c", 1, "char", "S1")
SCALARS = {s.name: s for s in (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool8, char)}


class Array:
    def __init__(self, item, n):
        self.item, self.n = _type(item), int(n)
        if self.n < 0:
            raise LayoutError("array(type, n): n is 0 or more")
        self.size, self.align = self.item.size * self.n, self.item.align

    def describe(self):
        return f"array({self.item.describe()}, {self.n})"

    __repr__ = describe


class Bits:
    """Bit fields packed into one unsigned integer, the lowest bits first."""

    def __init__(self, base, fields):
        self.base = _type(base)
        if self.base not in (u8, u16, u32, u64):
            raise LayoutError("bits(base, …): the base is u8, u16, u32 or u64")
        self.fields, pos = [], 0
        for name, width in _pairs(fields):
            width = int(width)
            if width < 1:
                raise LayoutError(f"bit field {name}: a width of 1 or more")
            self.fields.append((str(name), pos, width))
            pos += width
        if pos > 8 * self.base.size:
            raise LayoutError(f"bit fields of {pos} bits do not fit in {self.base.name}")
        self.size, self.align = self.base.size, self.base.align

    def describe(self):
        return f"bits({self.base.name}, {', '.join(f'{n}:{w}' for n, _, w in self.fields)})"

    __repr__ = describe


def _pairs(fields):
    if isinstance(fields, dict):
        return list(fields.items())
    if isinstance(fields, Record):
        return list(fields.fields.items())
    out = []
    for f in fields:
        if not isinstance(f, (list, tuple)) or len(f) != 2:
            raise LayoutError("fields are [(name, type), …]")
        out.append((f[0], f[1]))
    return out


def _type(t):
    if isinstance(t, (Scalar, Array, Bits, CStruct)):
        return t
    if isinstance(t, str) and t in SCALARS:
        return SCALARS[t]
    raise LayoutError(f"not a layout type: {t!r} (u8 … i64, f32, f64, bool8, char, array(), bits(), c_struct())")


class CStruct:
    def __init__(self, name, fields, pack=0, endian="little"):
        self.name, self.pack_n = str(name), int(pack or 0)
        if self.pack_n not in (0, 1, 2, 4, 8, 16):
            raise LayoutError("pack: 0 (natural), 1, 2, 4, 8 or 16")
        if endian not in ("little", "big", "native"):
            raise LayoutError('endian: "little", "big" or "native"')
        self.endian = endian
        self.prefix = {"little": "<", "big": ">", "native": "="}[endian]
        self.fields, off, align = [], 0, 1
        seen = set()
        for fname, ftype in _pairs(fields):
            fname, t = str(fname), _type(ftype)
            if fname in seen:
                raise LayoutError(f"{self.name}: the field {fname} twice")
            seen.add(fname)
            a = min(t.align, self.pack_n) if self.pack_n else t.align
            off = -(-off // a) * a
            self.fields.append((fname, t, off))
            off += t.size
            align = max(align, a)
        self.align = align
        self.size = -(-off // align) * align if off else 0
        self._by = {n: (t, o) for n, t, o in self.fields}

    # ---------------------------------------------------------------- inspection
    def offset(self, field):
        if field not in self._by:
            raise LayoutError(f"{self.name} has no field {field}")
        return self._by[field][1]

    def describe(self):
        rows = [f"{self.name}: {self.size} bytes, align {self.align}, {self.endian}-endian"
                + (f", pack {self.pack_n}" if self.pack_n else "")]
        for n, t, o in self.fields:
            rows.append(f"  {o:>5}  {t.size:>4}  {n}: {t.describe()}")
        return "\n".join(rows)

    __repr__ = describe

    # ---------------------------------------------------------------- bytes
    def pack(self, values=None):
        buf = bytearray(self.size)
        self._write(buf, 0, values or {})
        return bytes(buf)

    def unpack(self, data, offset=0):
        data = bytes(data) if not isinstance(data, (bytes, bytearray, memoryview)) else data
        if len(data) - offset < self.size:
            raise LayoutError(f"{self.name}: {self.size} bytes needed, {len(data) - offset} given")
        return self._read(memoryview(data), offset)

    def _get(self, values, name):
        if isinstance(values, Record):
            return values.fields.get(name)
        if isinstance(values, dict):
            return values.get(name)
        return getattr(values, name, None)

    def _write(self, buf, base, values):
        for n, t, o in self.fields:
            _put(buf, base + o, t, self._get(values, n), self.prefix)

    def _read(self, mv, base):
        return Record(**{n: _get(mv, base + o, t, self.prefix) for n, t, o in self.fields})

    def view(self, buffer, offset=0):
        return View(self, buffer, offset)

    # ---------------------------------------------------------------- NumPy and C
    def numpy_dtype(self):
        names, formats, offsets = [], [], []
        for n, t, o in self.fields:
            names.append(n)
            formats.append(_np_type(t, self.prefix))
            offsets.append(o)
        return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": self.size})

    def c_header(self, guard=True):
        parts, done = [], set()
        _c_defs(self, parts, done)
        body = "\n\n".join(parts)
        if not guard:
            return body
        g = f"APSI_LAYOUT_{self.name.upper()}_H"
        return (f"/* {self.name}: made by Apsi (use layout). {self.endian}-endian, {self.size} bytes. */\n"
                f"#ifndef {g}\n#define {g}\n\n#include <stdint.h>\n#include <stdbool.h>\n#include <stddef.h>\n\n"
                f"{body}\n\n#endif\n")


def _c_defs(s, parts, done):
    if s.name in done:
        return
    for _, t, _ in s.fields:
        inner = t.item if isinstance(t, Array) else t
        if isinstance(inner, CStruct):
            _c_defs(inner, parts, done)
    # each struct once, whichever header defines it first; the assertions below stay outside, so a struct of the
    # same name with another layout in another header stops the build instead of being taken silently
    lines = [f"#ifndef APSI_LAYOUT_DEF_{s.name.upper()}", f"#define APSI_LAYOUT_DEF_{s.name.upper()}"]
    if s.pack_n:
        lines.append(f"#pragma pack(push, {s.pack_n})")
    lines.append(f"typedef struct {s.name} {{")
    for n, t, o in s.fields:
        lines.append(f"    {_c_decl(t, n)};")
    lines.append(f"}} {s.name};")
    if s.pack_n:
        lines.append("#pragma pack(pop)")
    lines.append("#endif")
    lines.append(f"_Static_assert(sizeof({s.name}) == {s.size}, \"{s.name}: size\");")
    for n, t, o in s.fields:
        lines.append(f"_Static_assert(offsetof({s.name}, {n}) == {o}, \"{s.name}.{n}: offset\");")
    for n, t, o in s.fields:
        if isinstance(t, Bits):
            for bn, sh, w in t.fields:
                mask = ((1 << w) - 1) << sh
                pre = f"{s.name.upper()}_{n.upper()}_{bn.upper()}"
                lines.append(f"#ifndef {pre}_SHIFT\n#define {pre}_SHIFT {sh}u\n#define {pre}_MASK 0x{mask:X}u\n#endif")
    parts.append("\n".join(lines))
    done.add(s.name)


def _c_decl(t, n):
    if isinstance(t, Scalar):
        return f"{t.ctype} {n}"
    if isinstance(t, Bits):
        return f"{t.base.ctype} {n}"
    if isinstance(t, CStruct):
        return f"{t.name} {n}"
    if isinstance(t, Array):
        inner = _c_decl(t.item, n)
        return f"{inner}[{t.n}]"
    raise LayoutError(f"no C form for {t!r}")


def _np_type(t, prefix):
    if isinstance(t, Scalar):
        return np.dtype(t.np).newbyteorder(prefix) if t.np not in ("u1", "i1", "?", "S1") else np.dtype(t.np)
    if isinstance(t, Bits):
        return _np_type(t.base, prefix)
    if isinstance(t, CStruct):
        return t.numpy_dtype()
    if isinstance(t, Array):
        return (_np_type(t.item, prefix), (t.n,))
    raise LayoutError(f"no NumPy form for {t!r}")


def _put(buf, off, t, v, prefix):
    if isinstance(t, Scalar):
        if v is None:
            return
        if t is char:
            v = v.encode() if isinstance(v, str) else bytes(v)
            if len(v) != 1:
                raise LayoutError("char: one byte")
        elif t.code in "fd":
            v = float(v)
        elif t is bool8:
            v = bool(v)
        else:
            v = int(v)
            lo, hi = _range(t)
            if not lo <= v <= hi:
                raise LayoutError(f"{v} does not fit in {t.name} ({lo} … {hi})")
        _st.pack_into(prefix + t.code, buf, off, v)
    elif isinstance(t, Bits):
        if v is None:
            return
        word = 0
        for bn, sh, w in t.fields:
            x = v.get(bn) if isinstance(v, dict) else (v.fields.get(bn) if isinstance(v, Record) else
                                                         getattr(v, bn, None))
            x = int(x or 0)
            if not 0 <= x < (1 << w):
                raise LayoutError(f"bit field {bn}: {x} does not fit in {w} bits")
            word |= x << sh
        _st.pack_into(prefix + t.base.code, buf, off, word)
    elif isinstance(t, CStruct):
        if v is not None:
            t._write(buf, off, v)
    elif isinstance(t, Array):
        if v is None:
            return
        if t.item is char and isinstance(v, (str, bytes)):
            raw = v.encode() if isinstance(v, str) else v
            if len(raw) > t.n:
                raise LayoutError(f"text of {len(raw)} bytes in array(char, {t.n})")
            buf[off:off + len(raw)] = raw
            return
        vals = list(np.asarray(v).reshape(-1)) if not isinstance(v, (list, tuple)) else list(v)
        if len(vals) > t.n:
            raise LayoutError(f"{len(vals)} values in array of {t.n}")
        for i, x in enumerate(vals):
            _put(buf, off + i * t.item.size, t.item, x, prefix)
    else:
        raise LayoutError(f"not a layout type: {t!r}")


def _range(t):
    bits = 8 * t.size
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if t.code.islower() else (0, (1 << bits) - 1)


def _get(mv, off, t, prefix):
    if isinstance(t, Scalar):
        v = _st.unpack_from(prefix + t.code, mv, off)[0]
        return v.decode("latin-1") if t is char else v
    if isinstance(t, Bits):
        word = _st.unpack_from(prefix + t.base.code, mv, off)[0]
        return Record(**{bn: (word >> sh) & ((1 << w) - 1) for bn, sh, w in t.fields})
    if isinstance(t, CStruct):
        return t._read(mv, off)
    if isinstance(t, Array):
        if t.item is char:
            return bytes(mv[off:off + t.n]).split(b"\0", 1)[0].decode("utf-8", "replace")
        if isinstance(t.item, Scalar) and t.item is not bool8:
            return np.frombuffer(mv, np.dtype(t.item.np).newbyteorder(prefix), t.n, off).copy()
        return [_get(mv, off + i * t.item.size, t.item, prefix) for i in range(t.n)]
    raise LayoutError(f"not a layout type: {t!r}")


class View:
    """A layout over memory you own: reading a field decodes its bytes there, writing encodes them there."""

    def __init__(self, layout, buffer, offset=0):
        object.__setattr__(self, "_l", layout)
        mv = memoryview(buffer)
        if mv.readonly:
            raise LayoutError("view(): the buffer is read-only (use a bytearray or a NumPy array)")
        mv = mv.cast("B") if mv.format != "B" or mv.ndim != 1 else mv
        if len(mv) - offset < layout.size:
            raise LayoutError(f"{layout.name}: {layout.size} bytes needed at {offset}, {len(mv) - offset} there")
        object.__setattr__(self, "_m", mv)
        object.__setattr__(self, "_o", int(offset))

    def __getattr__(self, n):
        l_ = object.__getattribute__(self, "_l")
        if n not in l_._by:
            raise AttributeError(n)
        t, o = l_._by[n]
        if isinstance(t, CStruct):
            return View(t, self._m, self._o + o)
        return _get(self._m, self._o + o, t, l_.prefix)

    def __setattr__(self, n, v):
        l_ = self._l
        if n not in l_._by:
            raise LayoutError(f"{l_.name} has no field {n}")
        t, o = l_._by[n]
        _put(self._m, self._o + o, t, v, l_.prefix)

    def to_record(self):
        return self._l._read(self._m, self._o)

    def __repr__(self):
        return f"view({self._l.name} at {self._o})"


class RegisterMap:
    """Device registers at fixed addresses: [(name, offset, type, bits?)]."""

    def __init__(self, name, base, registers):
        self.name, self.base, self.regs = str(name), int(base), []
        used = []
        for r in registers:
            r = list(r)
            if len(r) not in (3, 4):
                raise LayoutError("registers are [(name, offset, type), …] or with bit fields [(…, bits(…))]")
            rname, off, t = str(r[0]), int(r[1]), _type(r[2])
            if t not in (u8, u16, u32, u64):
                raise LayoutError(f"register {rname}: u8, u16, u32 or u64")
            if off % t.size:
                raise LayoutError(f"register {rname}: offset {off} is not aligned to {t.size}")
            for n2, o2, s2 in used:
                if off < o2 + s2 and o2 < off + t.size:
                    raise LayoutError(f"register {rname} overlaps {n2}")
            fields = Bits(t, r[3]) if len(r) == 4 else None
            self.regs.append((rname, off, t, fields))
            used.append((rname, off, t.size))
        self.size = max((o + t.size for _, o, t, _ in self.regs), default=0)

    def c_header(self, guard=True):
        up = self.name.upper()
        lines = [f"#define {up}_BASE 0x{self.base:08X}u"]
        for rn, off, t, bits in self.regs:
            lines.append(f"#define {up}_{rn.upper()}_OFFSET 0x{off:X}u")
            lines.append(f"#define {up}_{rn.upper()} (*(volatile {t.ctype} *)({up}_BASE + 0x{off:X}u))")
            for bn, sh, w in (bits.fields if bits else []):
                lines.append(f"#define {up}_{rn.upper()}_{bn.upper()}_SHIFT {sh}u")
                lines.append(f"#define {up}_{rn.upper()}_{bn.upper()}_MASK 0x{(((1 << w) - 1) << sh):X}u")
        body = "\n".join(lines)
        if not guard:
            return body
        g = f"APSI_REGS_{up}_H"
        return f"/* {self.name} registers: made by Apsi (use layout). */\n#ifndef {g}\n#define {g}\n\n" \
               f"#include <stdint.h>\n\n{body}\n\n#endif\n"

    def read(self, image, endian="little"):
        """Every register (and its bit fields) from a memory image of the block (bytes from base on)."""
        pre = {"little": "<", "big": ">", "native": "="}[endian]
        mv = memoryview(bytes(image))
        out = {}
        for rn, off, t, bits in self.regs:
            v = _st.unpack_from(pre + t.code, mv, off)[0]
            out[rn] = Record(value=v, **{bn: (v >> sh) & ((1 << w) - 1) for bn, sh, w in bits.fields}) if bits \
                else v
        return Record(**out)

    def write(self, image, register, value, endian="little"):
        """Write one register (a number, or its bit fields as a record) into a bytearray image."""
        pre = {"little": "<", "big": ">", "native": "="}[endian]
        for rn, off, t, bits in self.regs:
            if rn == register:
                if bits is not None and not isinstance(value, (int, np.integer)):
                    buf = bytearray(t.size)
                    _put(buf, 0, bits, value, pre)
                    value = _st.unpack_from(pre + t.code, buf, 0)[0]
                _st.pack_into(pre + t.code, image, off, int(value))
                return image
        raise LayoutError(f"{self.name} has no register {register}")


def c_struct(name, fields, pack=0, endian="little"):
    return CStruct(name, fields, pack, endian)


def array(item, n):
    return Array(item, n)


def bits(base, fields):
    return Bits(base, fields)


def register_map(name, base, registers):
    return RegisterMap(name, base, registers)


def sizeof(t):
    return _type(t).size


def alignof(t):
    return _type(t).align


def offsetof(layout, field):
    return layout.offset(field)


def module():
    m = dict(c_struct=c_struct, array=array, bits=bits, register_map=register_map, sizeof=sizeof, alignof=alignof,
             offsetof=offsetof)
    m.update(SCALARS)
    return m
