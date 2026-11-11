class E(Exception):
    pass
try:
    raise E("x")
except Exception as e:
    print("caught", e, type(e).__name__)
