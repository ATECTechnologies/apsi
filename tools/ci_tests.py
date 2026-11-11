"""Runs the toolchain tests in CI with a watchdog.

Every test is listed by name (-v). If the whole run takes longer than the limit (a test that hangs, for example a
deadlock on one runner), the stack of every thread is printed and the run stops with an error, so the log shows
exactly which test and which line were stuck instead of waiting for the six-hour job limit.

    python tools/ci_tests.py [minutes]        (default 20)
"""
import faulthandler
import os
import sys
import unittest

# the same import path as `python -m unittest discover tests` run from the repository: the current folder first,
# not this tools folder
sys.path[0] = os.getcwd()

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
faulthandler.enable()
faulthandler.dump_traceback_later(minutes * 60, exit=True)
result = unittest.main(module=None, argv=["ci_tests", "discover", "tests", "-v"], exit=False).result
faulthandler.cancel_dump_traceback_later()
sys.exit(0 if result.wasSuccessful() else 1)
