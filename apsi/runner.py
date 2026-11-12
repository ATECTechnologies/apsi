"""Run Apsi programs on a thread with a large stack.

Deeply recursive Apsi programs need many interpreter frames. On the default stack (1–8 MB) CPython can
overflow the C stack and the process dies with a segmentation fault before Python's recursion limit is
reached. Running the program on a thread with a big stack lets the recursion limit trigger first, so deep
recursion either works or ends with a clean `RuntimeError: maximum recursion depth exceeded`.

Ctrl+C (and SIGTERM) still work: the main thread keeps the signal handlers and forwards them — to a
registered graceful-stop hook (the web server registers one) or as a KeyboardInterrupt into the program
thread.
"""
import ctypes
import signal
import sys
import threading

# Windows accepts thread stacks strictly below 256 MiB, so the first size is 255 MiB; smaller sizes are tried when
# the system refuses (the recursion limit is scaled with the stack, so the limit always fits the stack).
STACK_SIZES_MB = (255, 128, 64, 32)
FRAMES_PER_MB = 235                       # ≈ 60,000 frames on 255 MiB, measured with the deepest engine

_stop_hooks = []          # callables that stop the running program gracefully (e.g. a web server)
_lock = threading.Lock()


def add_stop_hook(fn):
    with _lock:
        _stop_hooks.append(fn)


def remove_stop_hook(fn):
    with _lock:
        if fn in _stop_hooks:
            _stop_hooks.remove(fn)


def _inject(thread, exc_type):
    """Raise exc_type inside `thread` at its next Python instruction."""
    try:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(exc_type))
    except Exception:
        pass


def _interrupt(thread):
    with _lock:
        hooks = list(_stop_hooks)
    if hooks:
        for h in hooks:
            try:
                h()
            except Exception:
                pass
    else:
        _inject(thread, KeyboardInterrupt)


def call(fn, *args, **kwargs):
    """fn(*args, **kwargs) on a big-stack thread; returns its result or re-raises its exception."""
    if threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)
    box = {}
    done = threading.Event()

    def target():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:          # SystemExit and KeyboardInterrupt travel back too
            box["error"] = e
        finally:
            done.set()

    old_size, mb = None, 0
    for mb in STACK_SIZES_MB:
        try:
            old_size = threading.stack_size(mb * 1024 * 1024)
            break
        except (ValueError, RuntimeError, OverflowError):
            continue
    if old_size is None:
        return fn(*args, **kwargs)          # platform without configurable stacks
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, mb * FRAMES_PER_MB))
    worker = threading.Thread(target=target, name="apsi-program", daemon=True)
    try:
        worker.start()
    except (RuntimeError, MemoryError):
        threading.stack_size(old_size)
        sys.setrecursionlimit(old_limit)
        return fn(*args, **kwargs)
    finally:
        try:
            threading.stack_size(old_size)
        except (ValueError, RuntimeError):
            pass

    old_term = None
    if hasattr(signal, "SIGTERM"):
        try:
            old_term = signal.signal(signal.SIGTERM, lambda *_: _interrupt(worker))
        except (ValueError, OSError):
            old_term = None
    interrupts = 0
    try:
        # an Event, not join()/is_alive(): an interrupted join() can wrongly report the thread as finished
        while not done.is_set():
            try:
                done.wait(0.2)
            except KeyboardInterrupt:
                interrupts += 1
                if interrupts >= 3:          # the program ignores Ctrl+C (e.g. stuck in native code)
                    raise
                _interrupt(worker)
    finally:
        if old_term is not None:
            try:
                signal.signal(signal.SIGTERM, old_term)
            except (ValueError, OSError):
                pass
    if "error" in box:
        raise box["error"]
    return box.get("value")
