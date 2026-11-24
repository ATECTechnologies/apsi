"""The native backend (apsi build --native): each program prints exactly what `apsi run` prints."""
import glob
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
PROGS = sorted(glob.glob(os.path.join(HERE, "native_aot", "*.apsi")))
CC = shutil.which("clang") or shutil.which("gcc")


@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize("prog", PROGS, ids=[os.path.basename(p) for p in PROGS])
def test_native_matches_run(prog, tmp_path):
    from apsi.aot.build import build
    exe = build(prog, str(tmp_path / "prog"))
    want = subprocess.run([sys.executable, "-m", "apsi", "run", prog], capture_output=True, text=True,
                          cwd=os.path.dirname(HERE))
    got = subprocess.run([exe], capture_output=True, text=True)
    assert got.stdout == want.stdout
    assert got.returncode == want.returncode
