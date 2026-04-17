"""
Aggregate fixation statistics from fixations/*_fixations.csv into metrics/fixation_summary.csv.

CLI entrypoint; the Streamlit app uses pipeline_runner.run_summarize_fixations.
"""

from __future__ import annotations


def main() -> None:
    from pipeline_runner import run_summarize_fixations

    try:
        run_summarize_fixations(log=print)
    except ValueError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
