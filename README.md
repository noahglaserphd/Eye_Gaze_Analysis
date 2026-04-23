# Video-Based Gaze Analysis from Eye-Tracking Overlays

An open-source research pipeline for extracting and analyzing gaze behavior from video recordings that contain a visible gaze overlay.

This system detects gaze markers embedded in recorded videos and converts them into structured gaze data such as gaze coordinates, fixations, and saccades. The data can then be explored through visualizations, temporal analysis, and an interactive dashboard.

Unlike traditional eye-tracking analysis tools that depend on proprietary ecosystems, this pipeline works directly with screen recordings containing a gaze cursor or overlay. This allows researchers to analyze gaze behavior without relying on vendor APIs or proprietary analysis software.

---

# What This System Does

This pipeline converts eye-tracking overlays from recorded videos into analyzable gaze data. It automatically detects the gaze marker in each video frame, reconstructs gaze coordinates, identifies fixations and saccades, and generates metrics describing visual attention. Researchers can then explore these data through an interactive dashboard that shows heatmaps, scanpaths, temporal attention patterns, AOI analysis, and video playback synchronized with gaze behavior.

---

# Project Workflow

## Recommended — unified dashboard

1. Install dependencies and FFmpeg (see **Installation**).
2. Put recordings in `videos/` **or** pre-built gaze CSVs in `gaze_samples/`.
3. Start the app: `python -m eyegaze run` (after `pip install -e .`) or `streamlit run core/app.py`
4. In the sidebar, open **Data pipeline** and click **Run full pipeline**. This runs extraction (when videos are present), fixation summaries, time-window metrics, and matplotlib figures on disk — the same steps the CLI scripts perform, without a separate terminal.

You can also run individual pipeline steps from the sidebar (e.g. only **Videos → events** or only **Time windows**).

## Alternative — command-line scripts

The same logic lives in `core/pipeline_runner.py` and is exposed by the existing scripts (`core/extract_dot.py`, `core/analyze_gaze_csv.py`, etc.) for batch or headless use.

Compatibility note: root-level shim scripts are also provided (`app.py`, `extract_dot.py`, etc.) so older command examples continue to work after the `core/` reorganization.

---

# Repository Structure

    videos/                      recorded videos containing gaze overlays
    
    gaze_samples/                extracted gaze coordinate samples
    fixations/                   fixation event data
    saccades/                    saccade event data
    metrics/                     fixation summary metrics
    time_windows/                temporal gaze metrics
    
    core/                        all Python source modules and scripts
      app.py                     interactive dashboard (pipeline + exploration)
      extract_dot.py             extracts gaze marker positions from video
      analyze_gaze_csv.py        processes gaze coordinates
      summarize_fixations.py     computes fixation statistics
      window_analysis.py         computes time-window gaze metrics
      make_figures.py            generates per-session visualization figures
      make_aggregate_figure.py   generates multi-session summary figures
      pipeline_runner.py         shared pipeline implementation (dashboard + CLI)
      pipeline_config.py         shared defaults (window size, heatmap bins)
      gaze_detection.py          shared fixation/saccade detection logic
      viz_utils.py               shared offline heatmap helpers
      window_utils.py            shared window slicing + metrics logic
      dashboard_aoi.py           shared AOI normalization/assignment helpers
      dashboard_charts.py        shared Plotly chart builders used by app.py
    
    eyegaze/                     console CLI (`python -m eyegaze …`)
    eyegaze.cmd                  Windows launcher when `eyegaze.exe` is not on PATH
    tests/                       pytest suite for core processing helpers

---

# Installation

Python **3.9 or newer** is recommended.

## One-command setup (recommended)

Create everything (Python deps + FFmpeg) in one command:

    conda env create -f environment.yml

Then activate:

    conda activate gaze_dash

Or use the PowerShell all-in-one script (create/update env, install editable package, run app):

    .\setup_run.ps1

Register the **`eyegaze` command** (one-time, from the repository root):

    pip install -e .

Start the dashboard:

**On Windows (Microsoft Store Python and many default installs), `pip` puts `eyegaze.exe` in a `Scripts` folder that is not on your `PATH`, so `eyegaze` may not run from PowerShell.** Use any of these instead:

    python -m eyegaze run

Or, from the repository folder only, use the bundled launcher:

    .\eyegaze.cmd run

If `eyegaze` works in your terminal (Scripts is on `PATH`), you can use:

    eyegaze run

Note: the Conda environment already installs project dependencies and FFmpeg. `eyegaze install` is optional and mainly useful if you skip Conda and want to install from `requirements.txt`.

Extra Streamlit options are passed through, for example:

    python -m eyegaze run -- --server.port 8502

## Alternative: manual setup

Create a Conda environment:

    conda create -n gaze_dash-try python=3.10
    conda activate gaze_dash-try

Install required libraries (see `requirements.txt` for version ranges):

    pip install -r requirements.txt

Install **FFmpeg** (required for video previews and exports):

    conda install -c conda-forge ffmpeg

Optional: export Plotly charts as PNG from the dashboard:

    pip install "kaleido>=0.2.1,<2"

Shared defaults for window length and heatmap resolution live in `core/pipeline_config.py`. The repository includes a `.gitignore` that excludes typical generated folders (`gaze_samples/`, `fixations/`, `videos/`, etc.); adjust it if you need to version data in git.

Other shared modules: `core/gaze_detection.py` (I-DT fixations and saccades), `core/viz_utils.py` (offline heatmaps for PNG figures).

---

# Running Tests

Install dependencies (includes pytest):

    pip install -r requirements.txt

Run the test suite from the repository root:

    python -m pytest -q

Current test coverage focuses on core, deterministic logic:

- fixation and saccade derivation (`core/gaze_detection.py`)
- time-window generation and metrics (`core/window_utils.py`)
- AOI normalization and assignment (`core/dashboard_aoi.py`)

---

# Step-by-step reference (CLI)

The steps below match what **Data pipeline** in the app runs on disk. If you already used **Run full pipeline** in `core/app.py`, you can go straight to exploration (see **Step 7**) or use these commands for scripts and reproducibility.

---

# Step 1 — Prepare Videos

Place recorded gameplay or task recordings inside the `videos/` folder (optional if you only use pre-built gaze CSVs in `gaze_samples/`).

Example:

    videos/
       Rec1.mkv
       Rec2.mp4
       Rec3.mov

Each filename becomes the **session identifier** used throughout the analysis.

---

# Step 2 — Extract gaze from video (and derive events)

Run:

    python core/extract_dot.py

Or in the app: **Data pipeline** → **Videos → events** or **Run full pipeline**.

This scans each video for the gaze marker, writes timestamped gaze coordinates, then runs fixation and saccade detection and updates `metrics/fixation_summary.csv`.

Outputs (per session stem):

    gaze_samples/<session>.csv
    fixations/<session>_fixations.csv
    saccades/<session>_saccades.csv

---

# Step 3 — Process gaze CSVs only (no video decode)

Use this when you already have `gaze_samples/<session>.csv` and want to recompute fixations/saccades without re-running video extraction:

    python core/analyze_gaze_csv.py

Or in the app: **Gaze CSVs → events** (or **Run full pipeline** when there are no videos in `videos/` but gaze CSVs exist).

Outputs:

    fixations/<session>_fixations.csv
    saccades/<session>_saccades.csv

---

# Step 4 — Generate fixation statistics

    python core/summarize_fixations.py

Or in the app: **Summarize fixations** (included in **Run full pipeline**).

Produces summary statistics describing fixation behavior across sessions.

Output:

    metrics/fixation_summary.csv

---

# Step 5 — Generate session visualizations (PNG)

    python core/make_figures.py

Or in the app: **Session PNG figures** (included in **Run full pipeline**).

Creates figures including:

- fixation heatmaps  
- scanpath diagrams  
- fixation duration distributions  
- saccade amplitude distributions  

Dataset-level summary (all sessions in one figure):

    python core/make_aggregate_figure.py

Or in the app: **Aggregate figure** (included in **Run full pipeline**). Output: `figures/ALL_sessions_summary.png`.

---

# Step 6 — Generate time window metrics

    python core/window_analysis.py

Or in the app: **Time windows** (included in **Run full pipeline**).

This divides each session into time windows (default **30 seconds**, set in `core/pipeline_config.py` as `WINDOW_SIZE`) and calculates gaze metrics for each window.

Example segmentation:

    0–30 seconds
    30–60 seconds
    60–90 seconds

Metrics calculated per window include:

- fixation count  
- mean fixation duration  
- total fixation time  
- mean saccade amplitude  

These metrics allow researchers to analyze **temporal changes in attention**.

---

# Step 7 — Launch the interactive dashboard

Start the dashboard:

    streamlit run core/app.py

Or, if you ran `pip install -e .`:

    eyegaze run

The dashboard opens in your browser. Use **Data pipeline** in the sidebar to run the same processing as Steps 2–6 on disk, then use **Explore** to inspect sessions. You can also run the CLI steps first and use the app for exploration only.

---

# Dashboard Capabilities

The dashboard allows researchers to interactively explore gaze behavior and to **run the full processing pipeline** from the sidebar.

Features include:

- **Data pipeline** controls (full run or individual steps: video extraction, gaze CSV processing, summaries, time windows, PNG figures, aggregate figure)  
- selecting recorded sessions  
- inspecting time windows (default length in `pipeline_config.py`)  
- viewing fixation heatmaps  
- visualizing scanpaths  
- examining fixation duration distributions  
- analyzing saccade amplitude distributions  
- viewing synchronized video clips  
- exporting charts and video segments  
- exporting full-session outputs  

The dashboard is designed primarily for **exploratory analysis and qualitative inspection** of visual attention patterns.

---

# AOI (Area of Interest) Analysis

The system supports **AOI-based gaze analysis**.

AOIs define specific regions of the screen such as interface elements, menus, or HUD components.

Examples:

- bottom HUD  
- minimap  
- score display  
- inventory menu  
- navigation interface  

For each AOI the dashboard can compute:

- fixation count  
- total dwell time  
- mean fixation duration  

This enables researchers to measure **how much attention users allocate to different interface components**.

---

# AOI Editor

The dashboard includes an **interactive AOI editor**.

Researchers can:

1. Open the AOI editor in the dashboard  
2. Draw rectangular regions directly on a reference frame from the recorded video  
3. Save the AOIs  

Saved AOIs are written to:

    aoi/aoi_definitions.json

Once saved, the AOIs are automatically applied to gaze data for analysis.

---

# Example Research Applications

This pipeline can support research involving:

- player attention during gameplay  
- usability evaluation of interface layouts  
- visual search behavior  
- cognitive load during task performance  
- interaction strategies in complex systems  

---

# Citation (APA 7)

Glaser, N. (2026).  
*Video-based gaze analysis from eye-tracking overlays* [Computer software].  
School of Information Science & Learning Technologies, University of Missouri.

https://github.com/noahglaserphd/Eye_Gaze_Analysis

---

# Institutional Attribution

Developed by **Noah Glaser** and **Sean Gallagher**
School of Information Science & Learning Technologies  
University of Missouri  

Copyright © 2026 University of Missouri.

---

# Disclaimer

This project is an independent research tool and is **not affiliated with or endorsed by any eye-tracking hardware manufacturer**.

The software operates exclusively on **video recordings containing visible gaze overlays** and does not interact with proprietary device APIs.

Users are responsible for ensuring their use of eye-tracking hardware and recorded data complies with applicable licensing agreements.

---

# Research Ethics

Researchers using this tool with human participants should ensure compliance with:

- institutional review board (IRB) requirements  
- participant consent procedures  
- privacy and data protection regulations  

Recorded sessions may contain identifiable information and should be handled appropriately.
