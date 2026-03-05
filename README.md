# Video-Based Gaze Analysis from Eye-Tracking Overlays

An open-source research pipeline for extracting and analyzing gaze behavior from video recordings that contain a visible gaze overlay.

This system detects gaze markers embedded in recorded videos and converts them into structured gaze data such as gaze coordinates, fixations, and saccades. The data can then be explored through visualizations, temporal analysis, and an interactive dashboard.

Unlike traditional eye-tracking analysis tools that depend on proprietary ecosystems, this pipeline works directly with screen recordings containing a gaze cursor or overlay. This allows researchers to analyze gaze behavior without relying on vendor APIs or proprietary analysis software.

---

# What This System Does

This pipeline converts eye-tracking overlays from recorded videos into analyzable gaze data.

It automatically detects the gaze marker in each video frame, reconstructs gaze coordinates, identifies fixations and saccades, and generates metrics describing visual attention.

Researchers can then explore these data through an interactive dashboard that shows heatmaps, scanpaths, temporal attention patterns, and video playback synchronized with gaze behavior.

---

# Project Workflow

The system runs in two stages.

## Stage 1 — Offline Processing

Scripts extract gaze coordinates from video files and compute gaze metrics.

## Stage 2 — Interactive Dashboard

A Streamlit dashboard allows exploration of sessions, time windows, gaze visualizations, AOI analysis, and synchronized video playback.

The dashboard **does not process raw video**.  
It only visualizes results produced by the processing scripts.

---

# Repository Structure
