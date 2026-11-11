class MyErr(ValueError):
    pass
def risky(n):
    if n < 0:
        raise MyErr(f"neg {n}")
    return 10 / n
for n in (2, 0, -1):
    try:
        print(risky(n))
    except ZeroDivisionError as e:
        print("zero", type(e).__name__)
    except ValueError as e:
        print("val", e, isinstance(e, MyErr))
    else:
        print("ok")
    finally:
        print("done", n)
try:
    {}["k"]
except KeyError as e:
    print("key", repr(e))
try:
    try:
        raise RuntimeError("a")
    except RuntimeError as e:
        raise TypeError("b") from e
except TypeError as e:
    print(e, repr(e.__cause__))
