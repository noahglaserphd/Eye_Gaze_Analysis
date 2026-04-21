# make_figures.py
# Reads:
#   gaze_samples/<stem>.csv
#   fixations/<stem>_fixations.csv
#   saccades/<stem>_saccades.csv
# Writes:
#   figures/<stem>_summary.png
#
# Install:
#   pip install matplotlib pandas numpy

from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.pipeline_config import (
    GAUSSIAN_HEATMAP_SESSION_RADIUS,
    GAUSSIAN_HEATMAP_SESSION_SIGMA,
    HEATMAP_BINS_SESSION_PNG,
)
from core.viz_utils import (
    duration_weighted_fixation_heatmap,
    safe_max,
    safe_mean,
    safe_median,
)

ROOT = Path(__file__).resolve().parent
GAZE_DIR = ROOT / "gaze_samples"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_summary_for_stem(stem: str, emit: Callable[[str], None]) -> None:
    gaze_file = GAZE_DIR / f"{stem}.csv"
    fix_file = FIX_DIR / f"{stem}_fixations.csv"
    sac_file = SAC_DIR / f"{stem}_saccades.csv"

    if not (gaze_file.exists() and fix_file.exists() and sac_file.exists()):
        emit(f"Skipping {stem}: missing one of gaze/fix/sac CSVs")
        return

    fix = pd.read_csv(fix_file)
    sac = pd.read_csv(sac_file)

    n_fix = len(fix)
    n_sac = len(sac)

    mean_fix = safe_mean(fix["duration_s"]) if n_fix else float("nan")
    med_fix = safe_median(fix["duration_s"]) if n_fix else float("nan")
    max_fix = safe_max(fix["duration_s"]) if n_fix else float("nan")

    mean_amp = safe_mean(sac["amplitude_norm"]) if n_sac else float("nan")
    mean_sac_dur = safe_mean(sac["duration_s"]) if n_sac else float("nan")

    heat = duration_weighted_fixation_heatmap(
        fix,
        bins=HEATMAP_BINS_SESSION_PNG,
        gaussian_radius=GAUSSIAN_HEATMAP_SESSION_RADIUS,
        gaussian_sigma=GAUSSIAN_HEATMAP_SESSION_SIGMA,
    )

    # ---- Build figure using subplots (clean layout) ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    (ax1, ax2), (ax3, ax4) = axs

    # Reserve space for a clean header (prevents overlap)
    fig.subplots_adjust(top=0.86, wspace=0.25, hspace=0.35)

    header = (
        f"{stem}    "
        f"Fixations: {n_fix} (mean {mean_fix:.3f}s, median {med_fix:.3f}s, max {max_fix:.3f}s)    "
        f"Saccades: {n_sac} (mean amp {mean_amp:.3f}, mean dur {mean_sac_dur:.3f}s)"
    )
    fig.text(0.5, 0.94, header, ha="center", va="top", fontsize=12)

    # 1) Heatmap
    ax1.imshow(heat, origin="upper", extent=[0, 1, 1, 0], aspect="auto")
    ax1.set_title("Fixation heatmap (duration-weighted)")
    ax1.set_xlabel("x (norm)")
    ax1.set_ylabel("y (norm)")

    # 2) Scanpath
    ax2.set_title("Scanpath (circle size ∝ fixation duration)")
    ax2.set_xlabel("x (norm)")
    ax2.set_ylabel("y (norm)")
    ax2.set_ylim(1, 0)

    if n_fix:
        # sizes relative to max duration; avoid division by zero
        denom = max_fix if np.isfinite(max_fix) and max_fix > 0 else 1.0
        sizes = 30 + 250 * (fix["duration_s"] / denom)
        ax2.plot(fix["x_norm"], fix["y_norm"], "-", linewidth=1.2)
        ax2.scatter(fix["x_norm"], fix["y_norm"], s=sizes, alpha=0.85)

    # 3) Fixation duration histogram
    ax3.set_title("Fixation duration distribution")
    ax3.set_xlabel("duration (s)")
    ax3.set_ylabel("count")
    if n_fix:
        ax3.hist(fix["duration_s"], bins=25)

    # 4) Saccade amplitude histogram
    ax4.set_title("Saccade amplitude distribution")
    ax4.set_xlabel("amplitude (norm)")
    ax4.set_ylabel("count")
    if n_sac:
        ax4.hist(sac["amplitude_norm"], bins=25)

    out = FIG_DIR / f"{stem}_summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    emit(f"Wrote {out.relative_to(ROOT)}")


def main(log: Callable[[str], None] | None = None) -> None:
    emit = log or print
    if not GAZE_DIR.exists():
        raise SystemExit(f"Missing folder: {GAZE_DIR}")

    stems = sorted([p.stem for p in GAZE_DIR.glob("*.csv")])
    if not stems:
        raise SystemExit(f"No gaze sample CSVs found in {GAZE_DIR}")

    for stem in stems:
        plot_summary_for_stem(stem, emit)


if __name__ == "__main__":
    main()
