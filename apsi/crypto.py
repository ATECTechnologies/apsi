"""`use crypto` — security primitives built on vetted implementations (hashlib/hmac/secrets).

Password hashes use scrypt where available and PBKDF2-HMAC-SHA256 otherwise (e.g. in
the WebAssembly playground); verification accepts both formats.

Apsi never rolls its own ciphers: everything here wraps audited standard algorithms.
"""
import base64
import hashlib
import hmac
import secrets

from .errors import RuntimeErr

_HASHES = {"sha256", "sha512", "sha3_256", "sha3_512", "blake2b", "blake2s"}


def _bytes(x):
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    raise RuntimeErr("crypto functions take Str values")


def hash(data, algo="sha256"):
    if algo not in _HASHES:
        raise RuntimeErr(f"unknown hash '{algo}'; use one of {', '.join(sorted(_HASHES))}")
    return hashlib.new(algo, _bytes(data)).hexdigest()


def hmac_sign(key, message, algo="sha256"):
    return hmac.new(_bytes(key), _bytes(message), algo).hexdigest()


def hmac_verify(key, message, signature, algo="sha256"):
    return hmac.compare_digest(hmac_sign(key, message, algo), str(signature))


def secure_equal(a, b):
    """Constant-time comparison (prevents timing attacks)."""
    return hmac.compare_digest(_bytes(a), _bytes(b))


def random_token(nbytes=32):
    return secrets.token_urlsafe(int(nbytes))


def random_bytes_hex(nbytes=32):
    return secrets.token_hex(int(nbytes))


def secure_randint(lo, hi):
    """Uniform integer in [lo, hi) from the OS CSPRNG."""
    return int(lo) + secrets.randbelow(int(hi) - int(lo))


PBKDF2_ITERATIONS = 600_000      # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256


def _pbkdf2_sha256(password, salt, iterations, dklen=32):
    """hashlib.pbkdf2_hmac when available; otherwise a pure-Python RFC 8018 implementation
    (used in WebAssembly builds without OpenSSL)."""
    if hasattr(hashlib, "pbkdf2_hmac"):
        return hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=dklen)
    inner = hmac.new(password, digestmod=hashlib.sha256)
    out = b""
    block = 1
    while len(out) < dklen:
        h = inner.copy()
        h.update(salt + block.to_bytes(4, "big"))
        u = h.digest()
        acc = int.from_bytes(u, "big")
        for _ in range(iterations - 1):
            h = inner.copy()
            h.update(u)
            u = h.digest()
            acc ^= int.from_bytes(u, "big")
        out += acc.to_bytes(32, "big")
        block += 1
    return out[:dklen]


def _b64(v):
    return base64.b64encode(v).decode()


def password_hash(password, n=2 ** 14, r=8, p=1):
    """Salted password hash to store in a database.

    scrypt (memory-hard) when the platform provides it, otherwise PBKDF2-HMAC-SHA256
    with 600k iterations. Format: 'scrypt$n$r$p$salt$hash' or 'pbkdf2_sha256$iter$salt$hash'.
    """
    salt = secrets.token_bytes(16)
    if hasattr(hashlib, "scrypt"):
        dk = hashlib.scrypt(_bytes(password), salt=salt, n=n, r=r, p=p, dklen=32)
        return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(dk)}"
    iterations = PBKDF2_ITERATIONS if hasattr(hashlib, "pbkdf2_hmac") else 100_000
    dk = _pbkdf2_sha256(_bytes(password), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(dk)}"


def password_verify(password, stored):
    try:
        parts = str(stored).split("$")
        if parts[0] == "scrypt" and len(parts) == 6:
            _, n, r, p, salt, dk = parts
            salt, dk = base64.b64decode(salt), base64.b64decode(dk)
            got = hashlib.scrypt(_bytes(password), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(dk))
        elif parts[0] == "pbkdf2_sha256" and len(parts) == 4:
            _, it, salt, dk = parts
            salt, dk = base64.b64decode(salt), base64.b64decode(dk)
            got = _pbkdf2_sha256(_bytes(password), salt, int(it), len(dk))
        else:
            return False
        return hmac.compare_digest(got, dk)
    except (ValueError, TypeError, AttributeError):
        return False


def b64encode(text):
    return base64.b64encode(_bytes(text)).decode()


def b64decode(text):
    return base64.b64decode(_bytes(text)).decode("utf-8", "replace")


def module():
    return {"hash": hash, "hmac_sign": hmac_sign, "hmac_verify": hmac_verify,
            "secure_equal": secure_equal, "random_token": random_token,
            "random_hex": random_bytes_hex, "secure_randint": secure_randint,
            "password_hash": password_hash, "password_verify": password_verify,
            "b64encode": b64encode, "b64decode": b64decode}
