import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WINDOW_SIZE = 30.0  # seconds

root = Path(".")
fix_dir = root / "fixations"
sac_dir = root / "saccades"
out_dir = root / "time_windows"

out_dir.mkdir(exist_ok=True)

fix_files = sorted(fix_dir.glob("*_fixations.csv"))

for fix_file in fix_files:

    session = fix_file.stem.replace("_fixations","")
    sac_file = sac_dir / f"{session}_saccades.csv"

    fix = pd.read_csv(fix_file)
    sac = pd.read_csv(sac_file)

    max_time = fix["end_s"].max()

    windows = np.arange(0, max_time + WINDOW_SIZE, WINDOW_SIZE)

    rows = []

    for start in windows[:-1]:

        end = start + WINDOW_SIZE

        f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)]
        s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)]

        row = {
            "session": session,
            "window_start": start,
            "window_end": end,
            "fixation_count": len(f),
            "mean_fix_duration": f["duration_s"].mean() if len(f) else 0,
            "total_fix_time": f["duration_s"].sum() if len(f) else 0,
            "mean_saccade_amp": s["amplitude_norm"].mean() if len(s) else 0
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    csv_out = out_dir / f"{session}_windows.csv"
    df.to_csv(csv_out, index=False)

    # timeline figure
    fig, axs = plt.subplots(3,1, figsize=(10,8))

    t = df["window_start"]

    axs[0].plot(t, df["fixation_count"], marker="o")
    axs[0].set_title("Fixation count per 30s window")
    axs[0].set_ylabel("fixations")

    axs[1].plot(t, df["mean_fix_duration"], marker="o")
    axs[1].set_title("Mean fixation duration per window")
    axs[1].set_ylabel("seconds")

    axs[2].plot(t, df["mean_saccade_amp"], marker="o")
    axs[2].set_title("Mean saccade amplitude per window")
    axs[2].set_ylabel("amplitude")

    axs[2].set_xlabel("time (s)")

    fig.tight_layout()

    fig_out = out_dir / f"{session}_timeline.png"
    plt.savefig(fig_out)
    plt.close()

    print(f"Processed {session}")