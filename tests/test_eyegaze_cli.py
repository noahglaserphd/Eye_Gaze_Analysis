from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eyegaze_cli_help_exits_zero() -> None:
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "eyegaze", "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "install" in r.stdout
    assert "run" in r.stdout
