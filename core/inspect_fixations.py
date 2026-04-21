"""
Print top fixations by duration for a fixation CSV under fixations/.

Usage:
  python inspect_fixations.py
  python inspect_fixations.py --session MyRecording
  python inspect_fixations.py --file fixations/MyRecording_fixations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FIX_DIR = ROOT / "fixations"


def resolve_path(session: str | None, file: str | None) -> Path:
    if file:
        p = Path(file)
        if not p.is_absolute():
            p = ROOT / p
        return p
    if session:
        return FIX_DIR / f"{session}_fixations.csv"
    files = sorted(FIX_DIR.glob("*_fixations.csv"))
    if not files:
        raise SystemExit(f"No *_fixations.csv files in {FIX_DIR}")
    if len(files) > 1:
        names = ", ".join(f.stem.replace("_fixations", "") for f in files[:10])
        more = f" (+{len(files) - 10} more)" if len(files) > 10 else ""
        raise SystemExit(
            "Multiple sessions found; use --session NAME or --file PATH.\n"
            f"Examples: {names}{more}"
        )
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect fixation durations for a session.")
    ap.add_argument("--session", "-s", help="Session stem (e.g. Rec1 for Rec1_fixations.csv)")
    ap.add_argument("--file", "-f", help="Path to a fixations CSV")
    ap.add_argument("--top", "-n", type=int, default=5, help="Number of rows to show")
    args = ap.parse_args()

    path = resolve_path(args.session, args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    fix = pd.read_csv(path)
    if len(fix) == 0:
        print(f"{path.name}: empty")
        return
    if "duration_s" not in fix.columns:
        raise SystemExit(f"No duration_s column in {path}")

    print(fix.sort_values("duration_s", ascending=False).head(args.top))


if __name__ == "__main__":
    main()
