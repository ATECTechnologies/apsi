"""Entry point used by the browser playground (Pyodide) and `apsi` embedding.

run(code) executes Apsi source in a sandbox (no files, Python, FFI or network)
and returns {"output": str, "error": str | None, "ok": bool, "tests": str}.
"""
import contextlib
import io

from .errors import ApsiError, CheckFailed


def run(code, filename="playground.apsi", run_tests=True):
    from .interpreter import Interpreter
    from .security import Policy
    it = Interpreter(policy=Policy.profile("sandbox"))
    out = io.StringIO()
    error = None
    tests = ""
    with contextlib.redirect_stdout(out):
        try:
            it.run(code, f"<{filename}>")
        except CheckFailed as e:
            error = "\n".join(x.format(code, filename) for x in e.errors)
            error += f"\n\n{len(e.errors)} error(s) found by the checker; nothing was run."
        except ApsiError as e:
            error = e.format(code, filename)
        except Exception as e:                          # never crash the page
            error = f"internal error: {type(e).__name__}: {e}"
    if error is None and run_tests and it.tests:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passed, failed = it.run_tests(report=lambda s: print(s))
        tests = buf.getvalue() + f"\n{passed} passed, {len(failed)} failed"
    return {"output": out.getvalue(), "error": error, "ok": error is None, "tests": tests}
