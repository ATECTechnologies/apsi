# Embedded and binary layouts

`use layout` describes data the way a C compiler lays it out, byte for byte: firmware structs, packets on a wire,
file headers, device registers. One description gives the bytes in Apsi, a NumPy dtype for arrays of records, and
a C header with static assertions of every offset — so Apsi and the C side can never disagree silently.

## Structs

```apsi
use layout

let Sample = layout.c_struct("Sample", [
    ("id", layout.u16),
    ("flags", layout.bits(layout.u8, [("ok", 1), ("mode", 3)])),    # bit fields, the lowest bits first
    ("value", layout.f32),
    ("name", layout.array(layout.char, 8)),
    ("count", layout.u64)])

print(Sample.describe())        # every field's offset and size
let raw = Sample.pack(record(id=0xBEEF, flags=record(ok=1, mode=5), value=2.5, name="probe", count=42))
let s = Sample.unpack(raw)      # a record: s.flags.mode, s.name, …
write_text("sample.h", Sample.c_header())
```

| | |
|---|---|
| types | `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool8 char`, `array(type, n)`, another `c_struct`, `bits(base, [(name, width), …])` |
| `pack=` | `0` the natural alignment (GCC, Clang and MSVC on x86-64, ARM, AArch64, RISC-V), `1` packed, `2`, `4`, `8`, `16` like `#pragma pack(n)` |
| `endian=` | `"little"` (default), `"big"` (network order, many file formats), `"native"` |
| `L.size`, `L.align`, `L.offset("f")` | the layout's numbers (also `sizeof`, `alignof`, `offsetof`) |
| `L.pack(values)` / `L.unpack(bytes, offset=0)` | values ↔ bytes; a number that does not fit its type is refused, missing fields are zero |
| `L.view(buffer, offset=0)` | read and write the fields of memory you own (a `bytearray`, a tensor) in place |
| `L.numpy_dtype()` | arrays of records: `np.frombuffer(data, L.numpy_dtype())` reads millions at once |
| `L.c_header()` | the C definition, `_Static_assert` of the size and of every offset, `_SHIFT` / `_MASK` macros for bit fields |

Bit fields are stored as their base integer with shift and mask macros in C, not as C bit fields: the C standard
lets each compiler choose how bit fields are laid out, so Apsi never depends on that choice. 32-bit x86 Linux
aligns 8-byte members to 4: the static assertions stop such a build instead of letting it read wrong data.

## Device registers

```apsi
let UART = layout.register_map("UART0", 0x40001000, [
    ("DR", 0x00, layout.u32),
    ("SR", 0x04, layout.u32, [("txe", 1), ("rxne", 1)]),
    ("BRR", 0x08, layout.u16)])
write_text("uart0.h", UART.c_header())       # UART0_SR, UART0_SR_RXNE_MASK, … as volatile accesses
let image = bytearray(UART.size)               # a memory image (a dump, a simulator, a test)
UART.write(image, "SR", record(txe=1, rxne=1))
print(UART.read(image).SR.rxne)
```

Registers must be aligned to their size and may not overlap; the header uses `volatile` pointers at
`BASE + offset`, the form every vendor's device headers use.

## Calling C with these buffers

```apsi
use ffi
let lib = ffi.cload("libdevice.so")                          # --allow-ffi
let fill = lib.fn("fill", [ffi.ptr_void, ffi.u16], ffi.void)  # void fill(Sample *s, uint16_t id);
let mem = bytearray(Sample.size)
fill(Sample.view(mem), 7)                                     # C writes into Apsi's memory, no copy
```

`ptr_void` and `ptr_u8` take a `bytearray`, a contiguous tensor or a layout view (its address from the view's
offset); `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64` are the integer and float arguments. Numbers can be written in base
16, 2 and 8: `0xFF`, `0b1010_0101`, `0o755`.
