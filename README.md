# Video-Based Gaze Analysis from Eye-Tracking Overlays

An open-source research pipeline for extracting and analyzing gaze behavior from video recordings that contain a visible gaze overlay.

This system detects gaze markers embedded in recorded videos and converts them into structured gaze data such as gaze coordinates, fixations, and saccades. The data can then be explored through visualizations, temporal analysis, and an interactive dashboard.

Unlike traditional eye-tracking analysis tools that depend on proprietary ecosystems, this pipeline works directly with screen recordings containing a gaze cursor or overlay. This allows researchers to analyze gaze behavior without relying on vendor APIs or proprietary analysis software.

---

# What This System Does

This pipeline converts eye-tracking overlays from recorded videos into analyzable gaze data. It automatically detects the gaze marker in each video frame, reconstructs gaze coordinates, identifies fixations and saccades, and generates metrics describing visual attention. Researchers can then explore these data through an interactive dashboard that shows heatmaps, scanpaths, temporal attention patterns, AOI analysis, and video playback synchronized with gaze behavior.

---

# Project Workflow

The system runs in two stages.

## Stage 1 — Offline Processing

Scripts extract gaze coordinates from video files and compute gaze metrics.

## Stage 2 — Interactive Dashboard

A Streamlit dashboard allows exploration of sessions, time windows, gaze visualizations, AOI analysis, and synchronized video playback.

The dashboard **does not process raw video**. It only visualizes results produced by the processing scripts.

---

# Repository Structure

    videos/                      recorded videos containing gaze overlays
    
    gaze_samples/                extracted gaze coordinate samples
    fixations/                   fixation event data
    saccades/                    saccade event data
    metrics/                     fixation summary metrics
    time_windows/                temporal gaze metrics
    
    extract_dot.py               extracts gaze marker positions from video
    analyze_gaze_csv.py          processes gaze coordinates
    summarize_fixations.py       computes fixation statistics
    window_analysis.py           computes time-window gaze metrics
    
    make_figures.py              generates per-session visualization figures
    make_aggregate_figure.py     generates multi-session summary figures
    
    app.py                       interactive dashboard

---

# Installation

Python **3.9 or newer** is recommended.

Create a Conda environment:

    conda create -n gaze_dash-try python=3.10
    conda activate gaze_dash-try

Install required libraries:

    pip install streamlit plotly pandas numpy opencv-python matplotlib pillow
    pip install streamlit-drawable-canvas-fix

Install **FFmpeg** (required for video previews and exports):

    conda install -c conda-forge ffmpeg

Optional (for exporting PNG figures):

    pip install kaleido

---

# Step 1 — Prepare Videos

Place recorded gameplay or task recordings inside the `videos/` folder.

Example:

    videos/
       Rec1.mkv
       Rec2.mp4
       Rec3.mov

Each filename becomes the **session identifier** used throughout the analysis.

---

# Step 2 — Extract Gaze Coordinates

Run the gaze extraction script:

    python extract_dot.py

This script scans videos and detects the gaze marker in each frame.

Outputs:

    gaze_samples/<session>.csv

These files contain timestamped gaze coordinates for each frame.

---

# Step 3 — Process Gaze Data

Convert gaze samples into structured eye-movement events:

    python analyze_gaze_csv.py

Outputs:

    fixations/<session>_fixations.csv
    saccades/<session>_saccades.csv

These datasets contain fixation events and saccade events derived from gaze coordinates.

---

# Step 4 — Generate Fixation Statistics

    python summarize_fixations.py

Produces summary statistics describing fixation behavior across sessions.

Output:

    metrics/fixation_summary.csv

---

# Step 5 — Generate Session Visualizations

    python make_figures.py

Creates figures including:

- fixation heatmaps  
- scanpath diagrams  
- fixation duration distributions  
- saccade amplitude distributions  

---

# Step 6 — Generate Time Window Metrics

    python window_analysis.py

This divides each session into **30-second windows** and calculates gaze metrics for each window.

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

# Step 7 — Launch the Interactive Dashboard

Start the dashboard:

    streamlit run app.py

The dashboard will open automatically in a browser.

---

# Dashboard Capabilities

The dashboard allows researchers to interactively explore gaze behavior.

Features include:

- selecting recorded sessions  
- inspecting **30-second time windows**  
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

Developed by **Noah Glaser**  
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
