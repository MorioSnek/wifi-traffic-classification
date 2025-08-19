FEAT_IN = "./model/features_labeled_W15.csv"
CM_OUT = "./img/confusion_matrix_W15.png"

TEST_SIZE = 0.40
RANDOM_STATE = 42
USE_TIME_SPLIT = True

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load
df = pd.read_csv(FEAT_IN)
df = df.dropna(subset=["label"]).copy()

# Feature set
FEATS = [
    "mbps_Uplink","mbps_Downlink",
    "pps_Uplink","pps_Downlink",
    "len_mean_Uplink","len_mean_Downlink",
    "iat_mean_Uplink","iat_mean_Downlink",
    "iat_var_Uplink","iat_var_Downlink",
    "rssi_mean_Uplink","rssi_mean_Downlink",
    "snr_mean_Uplink","snr_mean_Downlink",
    "sleep_fraction",
    "qos_frac_BE","qos_frac_BK","qos_frac_VI","qos_frac_VO",
]

X_all = df[FEATS].fillna(0.0)
y_all = df["label"].astype(str)


if "t_rel_start" not in df.columns:
    raise ValueError("t_rel_start non presente nel dataset: necessario per lo split temporale.")
df_sorted = df.sort_values("t_rel_start").reset_index(drop=True)
cut = int(len(df_sorted) * (1.0 - TEST_SIZE))
train_df = df_sorted.iloc[:cut]
test_df  = df_sorted.iloc[cut:]
Xtr = train_df[FEATS].fillna(0.0); ytr = train_df["label"].astype(str)
Xte = test_df[FEATS].fillna(0.0);  yte = test_df["label"].astype(str)

# Model
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
clf.fit(Xtr, ytr)

print("Train acc:", clf.score(Xtr, ytr))
print("Test  acc:", clf.score(Xte, yte))
y_pred = clf.predict(Xte)

preferred = ["idle","web","streaming","speedtest","shopping","social"]
labels = [c for c in preferred if c in y_all.unique().tolist()]
if not labels:
    labels = sorted(y_all.unique().tolist())

print("\nClassification report:\n",
      classification_report(yte, y_pred, labels=labels))

cm = confusion_matrix(yte, y_pred, labels=labels)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(CM_OUT, dpi=150)
plt.close(fig)