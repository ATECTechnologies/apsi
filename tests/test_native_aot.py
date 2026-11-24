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


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_gpu_kernels_emulated(tmp_path, monkeypatch):
    """The card's kernels and the runtime's GPU path, run on host threads (APSI_GPU_EMU)."""
    from apsi.aot.build import build
    monkeypatch.setenv("APSI_GPU_EMU_BUILD", "1")
    exe = build(os.path.join(HERE, "native_aot_gpu.apsi"), str(tmp_path / "gpu"))
    got = subprocess.run([exe], capture_output=True, text=True, env=dict(os.environ, APSI_GPU_EMU="1"))
    assert got.returncode == 0, got.stderr
    lines = got.stdout.splitlines()
    assert lines[0] == "true" and lines[1].startswith("gpu [[6., 6.],")
    assert lines[-1] == "true"
    assert lines[-2].startswith("0.0002282") and lines[-2].endswith("gpu")
