"""Enumerations: `enum Color:` declarations and enum_type(name, members)."""
import enum as _enum

from .errors import RuntimeErr


def enum_type(name, members):
    """An enumeration: members [(name, value or none), …]; a member without a value is the one before it plus one
    (the first: 0, as in C). Members: Color.RED, .name, .value; Color(1) and Color["RED"] look them up; for c in
    Color iterates in order; members compare equal only to themselves."""
    items, nxt, seen = [], 0, set()
    for m in members:
        if not isinstance(m, (list, tuple)) or len(m) != 2:
            raise RuntimeErr("enum members are (name, value) pairs")
        n, v = str(m[0]), m[1]
        if n in seen:
            raise RuntimeErr(f"enum {name}: the member {n} twice")
        seen.add(n)
        if v is None:
            v = nxt
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise RuntimeErr(f"enum {name}.{n}: a value is a number or text")
        if isinstance(v, int):
            nxt = v + 1
        items.append((n, v))
    try:
        return _enum.Enum(str(name), items)
    except (TypeError, ValueError) as e:
        raise RuntimeErr(f"enum {name}: {e}") from None
