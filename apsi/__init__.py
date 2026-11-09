"""Apsi — a language for AI and physics with units, shapes and autodiff built in."""
__version__ = "2.0.0"

import sys as _sys

if _sys.version_info < (3, 12):                   # pragma: no cover - depends on the interpreter
    raise ImportError(f"Apsi {__version__} needs Python 3.12 or newer (this is "
                      f"{_sys.version_info.major}.{_sys.version_info.minor}); install a newer Python or use the "
                      f"Windows installer / portable Linux build, which bring their own")

from .interpreter import Interpreter, run_source, run_file  # noqa: E402,F401
