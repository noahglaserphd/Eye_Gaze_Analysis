import pandas as pd

fix = pd.read_csv("2026-03-03 14-56-00_fixations.csv")

print(fix.sort_values("duration_s", ascending=False).head())