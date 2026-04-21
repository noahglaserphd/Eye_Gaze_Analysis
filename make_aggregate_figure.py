# make_aggregate_figure.py
# Creates ONE dataset-level summary figure from ALL sessions:
#   - global fixation heatmap (duration-weighted)
#   - fixation duration histogram (all sessions)
#   - saccade amplitude histogram (all sessions)
#   - per-session metrics scatter (fixations vs mean duration)
#
# Reads:
#   fixations/*_fixations.csv
#   saccades/*_saccades.csv
#
# Writes:
#   figures/ALL_sessions_summary.png
#
# Requirements:
#   pip install pandas numpy matplotlib

from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_config import (
    GAUSSIAN_HEATMAP_AGGREGATE_RADIUS,
    GAUSSIAN_HEATMAP_AGGREGATE_SIGMA,
    HEATMAP_BINS_AGGREGATE_PNG,
)
from viz_utils import duration_weighted_fixation_heatmap

ROOT = Path(__file__).resolve().parent
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main(log: Callable[[str], None] | None = None) -> None:
    emit = log or print
    fix_files = sorted(FIX_DIR.glob("*_fixations.csv"))
    sac_files = sorted(SAC_DIR.glob("*_saccades.csv"))

    if not fix_files:
        raise SystemExit(f"No fixation files found in {FIX_DIR}")
    if not sac_files:
        raise SystemExit(f"No saccade files found in {SAC_DIR}")

    # Load and concatenate all fixations
    fix_list = []
    session_metrics = []

    for f in fix_files:
        df = pd.read_csv(f)
        session = f.stem.replace("_fixations", "")
        df["session"] = session
        fix_list.append(df)

        if len(df):
            session_metrics.append(
                {
                    "session": session,
                    "num_fixations": len(df),
                    "mean_fix_duration": float(df["duration_s"].mean()),
                    "median_fix_duration": float(df["duration_s"].median()),
                    "max_fix_duration": float(df["duration_s"].max()),
                }
            )
        else:
            session_metrics.append(
                {
                    "session": session,
                    "num_fixations": 0,
                    "mean_fix_duration": np.nan,
                    "median_fix_duration": np.nan,
                    "max_fix_duration": np.nan,
                }
            )

    fix_all = pd.concat(fix_list, ignore_index=True)

    # Load and concatenate all saccades
    sac_list = []
    for s in sac_files:
        df = pd.read_csv(s)
        df["session"] = s.stem.replace("_saccades", "")
        sac_list.append(df)
    sac_all = pd.concat(sac_list, ignore_index=True)

    # Compute global heatmap
    heat = duration_weighted_fixation_heatmap(
        fix_all,
        bins=HEATMAP_BINS_AGGREGATE_PNG,
        gaussian_radius=GAUSSIAN_HEATMAP_AGGREGATE_RADIUS,
        gaussian_sigma=GAUSSIAN_HEATMAP_AGGREGATE_SIGMA,
    )

    # Build figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    (ax1, ax2), (ax3, ax4) = axs
    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.35)

    # Title / dataset summary
    total_fix = len(fix_all)
    total_sac = len(sac_all)
    if total_fix and "duration_s" in fix_all.columns:
        mean_fix = float(fix_all["duration_s"].mean())
    else:
        mean_fix = np.nan
    if total_sac and "amplitude_norm" in sac_all.columns:
        mean_amp = float(sac_all["amplitude_norm"].mean())
    else:
        mean_amp = np.nan

    fig.text(
        0.5,
        0.95,
        f"ALL Sessions Summary | Fixations: {total_fix} (mean {mean_fix:.3f}s) | "
        f"Saccades: {total_sac} (mean amp {mean_amp:.3f}) | Sessions: {len(fix_files)}",
        ha="center",
        va="top",
        fontsize=12,
    )

    # Panel 1: global heatmap
    ax1.imshow(heat, origin="upper", extent=[0, 1, 1, 0], aspect="auto")
    ax1.set_title("Global fixation heatmap (duration-weighted)")
    ax1.set_xlabel("x (norm)")
    ax1.set_ylabel("y (norm)")

    # Panel 2: fixation duration histogram (all)
    if total_fix and "duration_s" in fix_all.columns:
        ax2.hist(fix_all["duration_s"].dropna(), bins=40)
    ax2.set_title("Fixation duration distribution (all sessions)")
    ax2.set_xlabel("duration (s)")
    ax2.set_ylabel("count")

    # Panel 3: saccade amplitude histogram (all)
    if total_sac and "amplitude_norm" in sac_all.columns:
        ax3.hist(sac_all["amplitude_norm"].dropna(), bins=40)
    ax3.set_title("Saccade amplitude distribution (all sessions)")
    ax3.set_xlabel("amplitude (norm)")
    ax3.set_ylabel("count")

    # Panel 4: per-session scatter
    m = pd.DataFrame(session_metrics)
    ax4.scatter(m["num_fixations"], m["mean_fix_duration"])
    ax4.set_title("Per-session: fixation count vs mean duration")
    ax4.set_xlabel("num fixations")
    ax4.set_ylabel("mean fixation duration (s)")

    out = FIG_DIR / "ALL_sessions_summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    emit(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
