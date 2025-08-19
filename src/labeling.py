W = 15
FEATURES_IN = "./model/features_W15.csv"
LABELS_OUT = "./model/labels_by_time_W15.csv"
MERGED_OUT = "./model/features_labeled_W15.csv"

# MM:SS, label
DURATIONS = [
    ("3:27", "streaming"),
    ("2:05", "speedtest"),
    ("1:36", "idle"),
    ("2:17", "web"),
    ("1:46", "streaming"),
    ("0:51", "speedtest"),
    ("1:13", "idle"),
]

import pandas as pd
import numpy as np

def mmss_to_seconds(s: str) -> int:
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)

feat = pd.read_csv(FEATURES_IN)

intervals = []
t = 0
for d, lab in DURATIONS:
    dur = mmss_to_seconds(d)
    intervals.append((t, t + dur, lab))
    t += dur

lbl_df = pd.DataFrame(intervals, columns=["t_start", "t_end", "label"])

labels = feat[["win_id", "t_rel_start"]].copy()

def assign_label(t_rel_start: float) -> str | float:
    for _, r in lbl_df.iterrows():
        if r["t_start"] <= t_rel_start < r["t_end"]:
            return r["label"]
    return np.nan  

labels["label"] = labels["t_rel_start"].apply(assign_label)

labels.to_csv(LABELS_OUT, index=False)

merged = feat.merge(labels[["win_id", "label"]], on="win_id", how="left")
merged.to_csv(MERGED_OUT, index=False)

counts = merged["label"].value_counts(dropna=False)
print(f"Saved:\n- {LABELS_OUT}\n- {MERGED_OUT}\n")
print("Label counts (incl. NaN = fuori intervalli):")
print(counts)