import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable

from pipeline_config import WINDOW_SIZE
from window_utils import compute_window_metrics, compute_windows_from_fix

ROOT = Path(__file__).resolve().parent
fix_dir = ROOT / "fixations"
sac_dir = ROOT / "saccades"
out_dir = ROOT / "time_windows"


def main(log: Callable[[str], None] | None = None) -> None:
    emit = log or print
    out_dir.mkdir(parents=True, exist_ok=True)

    if not fix_dir.exists():
        raise SystemExit(f"Missing folder: {fix_dir}")

    fix_files = sorted(fix_dir.glob("*_fixations.csv"))

    win_label = f"{WINDOW_SIZE:g}s"

    for fix_file in fix_files:

        session = fix_file.stem.replace("_fixations", "")
        sac_file = sac_dir / f"{session}_saccades.csv"

        fix = pd.read_csv(fix_file)

        if len(fix) == 0:
            emit(f"Skipping {session}: empty fixations")
            continue

        if "end_s" not in fix.columns or "start_s" not in fix.columns:
            emit(f"Skipping {session}: missing start_s/end_s columns")
            continue

        max_time = float(fix["end_s"].max())
        if not np.isfinite(max_time) or max_time <= 0:
            emit(f"Skipping {session}: invalid end_s (max={max_time})")
            continue

        if sac_file.exists():
            sac = pd.read_csv(sac_file)
        else:
            emit(f"Warning: missing {sac_file.name} — using empty saccades")
            sac = pd.DataFrame(columns=["start_s", "amplitude_norm"])

        win_df = compute_windows_from_fix(fix, WINDOW_SIZE)
        if len(win_df) == 0:
            emit(f"Skipping {session}: no full {WINDOW_SIZE}s window (max_time={max_time:.3f}s)")
            continue

        df = compute_window_metrics(fix, sac, win_df)
        df.insert(0, "session", session)

        csv_out = out_dir / f"{session}_windows.csv"
        df.to_csv(csv_out, index=False)

        # timeline figure
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        t = df["window_start"]

        axs[0].plot(t, df["fixation_count"], marker="o")
        axs[0].set_title(f"Fixation count per {win_label} window")
        axs[0].set_ylabel("fixations")

        axs[1].plot(t, df["mean_fix_duration"], marker="o")
        axs[1].set_title(f"Mean fixation duration per {win_label} window")
        axs[1].set_ylabel("seconds")

        axs[2].plot(t, df["mean_saccade_amp"], marker="o")
        axs[2].set_title(f"Mean saccade amplitude per {win_label} window")
        axs[2].set_ylabel("amplitude")

        axs[2].set_xlabel("time (s)")

        fig.tight_layout()

        fig_out = out_dir / f"{session}_timeline.png"
        plt.savefig(fig_out)
        plt.close()

        emit(f"Processed {session}")


if __name__ == "__main__":
    main()
