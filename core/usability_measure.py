"""
Compute NASA-TLX usability/workload scores from questionnaire responses.

Input CSV expects one row per response and six rating columns (0-100):
  - mental_demand
  - physical_demand
  - temporal_demand
  - performance
  - effort
  - frustration

Optional weighted TLX columns (0-5 each; usually from pairwise comparisons):
  - mental_demand_weight
  - physical_demand_weight
  - temporal_demand_weight
  - performance_weight
  - effort_weight
  - frustration_weight
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NASA_TLX_SCALES = (
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
)
WEIGHT_SUFFIX = "_weight"


def _weight_col(scale: str) -> str:
    return f"{scale}{WEIGHT_SUFFIX}"


def make_template(path: Path) -> None:
    cols = ["participant_id", "session_id", *NASA_TLX_SCALES, *[_weight_col(s) for s in NASA_TLX_SCALES]]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def _validate_ratings(df: pd.DataFrame) -> None:
    missing = [c for c in NASA_TLX_SCALES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required NASA-TLX columns: {missing}")
    for col in NASA_TLX_SCALES:
        bad = (~df[col].isna()) & ((df[col] < 0) | (df[col] > 100))
        if bad.any():
            raise ValueError(f"Column '{col}' has values outside 0-100.")


def _compute_weighted_tlx(df: pd.DataFrame, strict_weighting: bool) -> pd.Series:
    weight_cols = [_weight_col(s) for s in NASA_TLX_SCALES]
    has_any_weight = any(c in df.columns for c in weight_cols)
    if not has_any_weight:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    missing = [c for c in weight_cols if c not in df.columns]
    if missing:
        if strict_weighting:
            raise ValueError(f"Missing weight columns for weighted TLX: {missing}")
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    for col in weight_cols:
        bad = (~df[col].isna()) & ((df[col] < 0) | (df[col] > 5))
        if bad.any():
            raise ValueError(f"Column '{col}' has values outside 0-5.")

    weights = df[weight_cols]
    sums = weights.sum(axis=1)
    if strict_weighting and not (sums == 15).all():
        raise ValueError("Weighted TLX requires row-wise weight sum of 15.")

    weighted_sum = sum(df[s] * df[_weight_col(s)] for s in NASA_TLX_SCALES)
    out = weighted_sum / sums.where(sums != 0, pd.NA)
    return out.astype("Float64")


def _interpret_raw_tlx(score: float) -> str:
    # NASA-TLX has no universal cutoffs; these labels are heuristic and should
    # be interpreted relative to your own conditions/tasks.
    if pd.isna(score):
        return "unknown"
    if score < 30:
        return "low workload"
    if score < 55:
        return "moderate workload"
    if score < 75:
        return "high workload"
    return "very high workload"


def score_nasa_tlx(input_csv: Path, output_csv: Path, strict_weighting: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    _validate_ratings(df)

    out = df.copy()
    out["nasa_tlx_raw"] = out.loc[:, NASA_TLX_SCALES].mean(axis=1)
    out["nasa_tlx_weighted"] = _compute_weighted_tlx(out, strict_weighting=strict_weighting)
    out["workload_interpretation"] = out["nasa_tlx_raw"].map(_interpret_raw_tlx)
    out.to_csv(output_csv, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute NASA-TLX scores from questionnaire responses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("metrics") / "nasa_tlx_responses.csv",
        help="Input responses CSV (default: metrics/nasa_tlx_responses.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics") / "nasa_tlx_scores.csv",
        help="Output scored CSV (default: metrics/nasa_tlx_scores.csv).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Write a blank template CSV to this path and exit.",
    )
    parser.add_argument(
        "--strict-weighting",
        action="store_true",
        help="Require all weight columns and enforce row-wise weight sum of 15.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.template:
        args.template.parent.mkdir(parents=True, exist_ok=True)
        make_template(args.template)
        print(f"Wrote template: {args.template}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored = score_nasa_tlx(args.input, args.output, strict_weighting=args.strict_weighting)
    print(f"Scored {len(scored)} response(s) -> {args.output}")
    print(f"Mean raw TLX: {scored['nasa_tlx_raw'].mean():.2f}")
    if scored["nasa_tlx_weighted"].notna().any():
        print(f"Mean weighted TLX: {scored['nasa_tlx_weighted'].mean(skipna=True):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
