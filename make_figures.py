# make_figures.py
# Reads:
#   gaze_samples/<stem>.csv
#   fixations/<stem>_fixations.csv
#   saccades/<stem>_saccades.csv
# Writes:
#   figures/<stem>_summary.png
#
# Install (in video_cv):
#   pip install matplotlib pandas numpy

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
GAZE_DIR = ROOT / "gaze_samples"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

HEATMAP_BINS = 140

def gaussian_kernel(radius: int = 6, sigma: float = 2.2) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= k.sum()
    return k

def convolve2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=float)

    # dependency-free convolution (fast enough for ~140x140)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(window * kernel)
    return out

def fixation_heatmap(fix: pd.DataFrame, bins: int = HEATMAP_BINS) -> np.ndarray:
    if len(fix) == 0:
        return np.zeros((bins, bins), dtype=float)

    x = fix["x_norm"].to_numpy(dtype=float)
    y = fix["y_norm"].to_numpy(dtype=float)
    w = fix["duration_s"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[valid], y[valid], w[valid]

    if len(x) == 0:
        return np.zeros((bins, bins), dtype=float)

    H, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[[0, 1], [0, 1]],
        weights=w
    )

    # histogram2d returns (xbins, ybins); transpose for image coords
    H = H.T

    k = gaussian_kernel()
    Hs = convolve2d_same(H, k)
    return Hs

def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")

def safe_median(series: pd.Series) -> float:
    return float(series.median()) if len(series) else float("nan")

def safe_max(series: pd.Series) -> float:
    return float(series.max()) if len(series) else float("nan")

def plot_summary_for_stem(stem: str) -> None:
    gaze_file = GAZE_DIR / f"{stem}.csv"
    fix_file = FIX_DIR / f"{stem}_fixations.csv"
    sac_file = SAC_DIR / f"{stem}_saccades.csv"

    if not (gaze_file.exists() and fix_file.exists() and sac_file.exists()):
        print(f"Skipping {stem}: missing one of gaze/fix/sac CSVs")
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

    heat = fixation_heatmap(fix)

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
    print(f"Wrote {out.relative_to(ROOT)}")

def main() -> None:
    if not GAZE_DIR.exists():
        raise SystemExit(f"Missing folder: {GAZE_DIR}")

    stems = sorted([p.stem for p in GAZE_DIR.glob("*.csv")])
    if not stems:
        raise SystemExit(f"No gaze sample CSVs found in {GAZE_DIR}")

    for stem in stems:
        plot_summary_for_stem(stem)

if __name__ == "__main__":
    main()