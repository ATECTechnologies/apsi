def kind(x):
    match x:
        case 0:
            return "zero"
        case [a, b]:
            return f"pair {a} {b}"
        case {"k": v}:
            return f"dict {v}"
        case str() as s:
            return "str " + s
        case _:
            return "other"
print([kind(v) for v in (0, [1, 2], {"k": 3}, "s", 4.5)])
