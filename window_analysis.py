import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline_config import WINDOW_SIZE

ROOT = Path(__file__).resolve().parent
fix_dir = ROOT / "fixations"
sac_dir = ROOT / "saccades"
out_dir = ROOT / "time_windows"


def main() -> None:
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
            print(f"Skipping {session}: empty fixations")
            continue

        if "end_s" not in fix.columns or "start_s" not in fix.columns:
            print(f"Skipping {session}: missing start_s/end_s columns")
            continue

        max_time = float(fix["end_s"].max())
        if not np.isfinite(max_time) or max_time <= 0:
            print(f"Skipping {session}: invalid end_s (max={max_time})")
            continue

        if sac_file.exists():
            sac = pd.read_csv(sac_file)
        else:
            print(f"Warning: missing {sac_file.name} — using empty saccades")
            sac = pd.DataFrame(columns=["start_s", "amplitude_norm"])

        windows = np.arange(0, max_time + WINDOW_SIZE, WINDOW_SIZE)
        if len(windows) < 2:
            print(f"Skipping {session}: no full {WINDOW_SIZE}s window (max_time={max_time:.3f}s)")
            continue

        rows = []

        for start in windows[:-1]:

            end = start + WINDOW_SIZE

            f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)]
            s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)]

            mean_amp = 0.0
            if len(s) and "amplitude_norm" in s.columns:
                mean_amp = float(s["amplitude_norm"].mean())

            if len(f) and "duration_s" in f.columns:
                mean_dur = float(f["duration_s"].mean())
                total_dur = float(f["duration_s"].sum())
            elif len(f):
                mean_dur = 0.0
                total_dur = 0.0
            else:
                mean_dur = 0.0
                total_dur = 0.0

            row = {
                "session": session,
                "window_start": start,
                "window_end": end,
                "fixation_count": len(f),
                "mean_fix_duration": mean_dur,
                "total_fix_time": total_dur,
                "mean_saccade_amp": mean_amp,
            }

            rows.append(row)

        df = pd.DataFrame(rows)

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

        print(f"Processed {session}")


if __name__ == "__main__":
    main()
