import pandas as pd
from pathlib import Path

rows = []

for fix_file in sorted(Path(".").glob("*_fixations.csv")):
    fix = pd.read_csv(fix_file)

    rows.append({
        "file": fix_file.name,
        "num_fixations": len(fix),
        "mean_fixation_duration": fix["duration_s"].mean(),
        "median_fixation_duration": fix["duration_s"].median(),
        "longest_fixation": fix["duration_s"].max()
    })

summary = pd.DataFrame(rows)

summary.to_csv("fixation_summary.csv", index=False)

print("Saved fixation_summary.csv")