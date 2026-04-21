"""
Console entry: eyegaze install | eyegaze run

Project root is the directory containing core/app.py (next to this package when installed editable).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    root = here.parent
    if (root / "core" / "app.py").is_file():
        return root
    cwd = Path.cwd()
    if (cwd / "core" / "app.py").is_file():
        return cwd
    raise FileNotFoundError(
        "Could not locate core/app.py. Run this command from the repository root "
        "or install with: pip install -e ."
    )


def cmd_install() -> int:
    root = _project_root()
    req = root / "requirements.txt"
    if not req.is_file():
        print(f"Missing {req}", file=sys.stderr)
        return 1
    print(f"Installing from {req} …")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        cwd=str(root),
    )
    return int(r.returncode)


def cmd_run(extra: list[str] | None = None) -> int:
    root = _project_root()
    app = root / "core" / "app.py"
    if not app.is_file():
        print(f"Missing {app}", file=sys.stderr)
        return 1
    argv = [sys.executable, "-m", "streamlit", "run", str(app)]
    if extra:
        argv.extend(extra)
    r = subprocess.run(argv, cwd=str(root))
    return int(r.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eyegaze",
        description="Eye Gaze Analysis: install dependencies and run the Streamlit dashboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("install", help="Install Python dependencies from requirements.txt")

    p_run = sub.add_parser("run", help="Start the interactive dashboard (streamlit run core/app.py)")
    p_run.add_argument(
        "streamlit_args",
        nargs="*",
        help="Extra arguments passed through to streamlit (e.g. --server.port 8502)",
    )

    args = parser.parse_args(argv)
    if args.command == "install":
        return cmd_install()
    if args.command == "run":
        return cmd_run(args.streamlit_args if args.streamlit_args else None)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
