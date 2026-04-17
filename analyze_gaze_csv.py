"""
Process gaze sample CSVs into fixations/ and saccades/ using the same I-DT settings as extract_dot.py.

Reads:  gaze_samples/<session>.csv
Writes: fixations/<session>_fixations.csv, saccades/<session>_saccades.csv

CLI entrypoint; the Streamlit app calls the same logic via pipeline_runner.run_analyze_gaze_csvs.
"""

from __future__ import annotations


def main() -> None:
    from pipeline_runner import run_analyze_gaze_csvs

    try:
        run_analyze_gaze_csvs(log=print)
    except ValueError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
