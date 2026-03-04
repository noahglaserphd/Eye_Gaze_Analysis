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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

HEATMAP_BINS = 180

def gaussian_kernel(radius: int = 7, sigma: float = 2.6) -> np.ndarray:
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
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

def fixation_heatmap_all(fix_all: pd.DataFrame, bins: int = HEATMAP_BINS) -> np.ndarray:
    if len(fix_all) == 0:
        return np.zeros((bins, bins), dtype=float)

    x = fix_all["x_norm"].to_numpy(dtype=float)
    y = fix_all["y_norm"].to_numpy(dtype=float)
    w = fix_all["duration_s"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[valid], y[valid], w[valid]

    H, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[[0, 1], [0, 1]],
        weights=w
    )
    H = H.T
    return convolve2d_same(H, gaussian_kernel())

def main():
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
        df["session"] = f.stem.replace("_fixations", "")
        fix_list.append(df)

        if len(df):
            session_metrics.append({
                "session": df["session"].iloc[0],
                "num_fixations": len(df),
                "mean_fix_duration": float(df["duration_s"].mean()),
                "median_fix_duration": float(df["duration_s"].median()),
                "max_fix_duration": float(df["duration_s"].max())
            })
        else:
            session_metrics.append({
                "session": df["session"].iloc[0],
                "num_fixations": 0,
                "mean_fix_duration": np.nan,
                "median_fix_duration": np.nan,
                "max_fix_duration": np.nan
            })

    fix_all = pd.concat(fix_list, ignore_index=True)

    # Load and concatenate all saccades
    sac_list = []
    for s in sac_files:
        df = pd.read_csv(s)
        df["session"] = s.stem.replace("_saccades", "")
        sac_list.append(df)
    sac_all = pd.concat(sac_list, ignore_index=True)

    # Compute global heatmap
    heat = fixation_heatmap_all(fix_all)

    # Build figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    (ax1, ax2), (ax3, ax4) = axs
    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.35)

    # Title / dataset summary
    total_fix = len(fix_all)
    total_sac = len(sac_all)
    mean_fix = float(fix_all["duration_s"].mean()) if total_fix else np.nan
    mean_amp = float(sac_all["amplitude_norm"].mean()) if total_sac else np.nan

    fig.text(
        0.5, 0.95,
        f"ALL Sessions Summary | Fixations: {total_fix} (mean {mean_fix:.3f}s) | "
        f"Saccades: {total_sac} (mean amp {mean_amp:.3f}) | Sessions: {len(fix_files)}",
        ha="center", va="top", fontsize=12
    )

    # Panel 1: global heatmap
    ax1.imshow(heat, origin="upper", extent=[0, 1, 1, 0], aspect="auto")
    ax1.set_title("Global fixation heatmap (duration-weighted)")
    ax1.set_xlabel("x (norm)")
    ax1.set_ylabel("y (norm)")

    # Panel 2: fixation duration histogram (all)
    ax2.hist(fix_all["duration_s"], bins=40)
    ax2.set_title("Fixation duration distribution (all sessions)")
    ax2.set_xlabel("duration (s)")
    ax2.set_ylabel("count")

    # Panel 3: saccade amplitude histogram (all)
    ax3.hist(sac_all["amplitude_norm"], bins=40)
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
    print(f"Wrote {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()