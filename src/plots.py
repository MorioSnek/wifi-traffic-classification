# Params
W = 15
MY_MAC = "b2:09:6d:5b:24:7d"
CSV_PATH = "./private/traffic.csv"

SLEEP_FRACTION_THRESHOLD = 0.6
SHADE_COLOR = "#e57373"
SHADE_ALPHA = 0.18

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load
df_raw = pd.read_csv(CSV_PATH)
num = lambda s: pd.to_numeric(s, errors="coerce")

df_raw["frame.time_epoch"] = num(df_raw["frame.time_epoch"])
df_raw["frame.len"] = num(df_raw["frame.len"])
df_raw["wlan.fc.type"] = num(df_raw["wlan.fc.type"])
df_raw["wlan.fc.pwrmgt"] = num(df_raw.get("wlan.fc.pwrmgt", 0)).fillna(0).astype(int)
df_raw["wlan_radio.snr"] = num(df_raw.get("wlan_radio.snr", np.nan))

for c in ["wlan.sa","wlan.da","wlan.ta","wlan.ra"]:
    df_raw[c] = df_raw[c].astype(str).str.lower()

t_min = float(df_raw["frame.time_epoch"].min())
t_max = float(df_raw["frame.time_epoch"].max())

# Sleep
upl = df_raw[(df_raw["wlan.sa"] == MY_MAC) | (df_raw["wlan.ta"] == MY_MAC)].copy()
upl = upl.sort_values("frame.time_epoch")
upl["state"] = upl["wlan.fc.pwrmgt"].clip(0,1).astype(int)      # 0=awake, 1=sleep
upl["grp"] = upl["state"].ne(upl["state"].shift()).cumsum()
runs = (upl.groupby("grp")
          .agg(state=("state","last"), start=("frame.time_epoch","min"), end=("frame.time_epoch","max"))
          .reset_index(drop=True))
if not runs.empty:
    runs.loc[runs.index[0], "start"] = t_min
    runs.loc[runs.index[-1], "end"] = t_max
    nxt = runs["start"].shift(-1)
    runs.loc[runs.index[:-1], "end"] = nxt.iloc[:-1].values

# Window grid
grid = np.arange(np.floor(t_min/W)*W, np.ceil(t_max/W)*W, W)
win = pd.DataFrame({"win_id": np.arange(len(grid)),
                    "t_abs_start": grid,
                    "t_abs_end": grid + W})
t0 = float(win["t_abs_start"].min())
win["t_rel_start"] = win["t_abs_start"] - t0
win["t_rel_end"]   = win["t_abs_end"]   - t0

# sleep_time per window
sleep_time = np.zeros(len(win))
for _, r in runs[runs["state"]==1].iterrows():
    s, e = r["start"] - t0, r["end"] - t0
    i0 = max(0, int(np.floor(s / W)))
    i1 = min(len(win)-1, int(np.floor((e - 1e-9) / W)))
    for i in range(i0, i1+1):
        ws, we = win.loc[i, ["t_rel_start","t_rel_end"]]
        sleep_time[i] += max(0.0, min(e, we) - max(s, ws))
sleep = win[["win_id","t_rel_start","t_rel_end"]].copy()
sleep["sleep_fraction"] = sleep_time / W

# Select data frames & direction
df = df_raw[df_raw["wlan.fc.type"] == 2].copy()
is_up   = (df["wlan.sa"] == MY_MAC) | (df["wlan.ta"] == MY_MAC)
is_down = (df["wlan.da"] == MY_MAC) | (df["wlan.ra"] == MY_MAC)
df = df[is_up | is_down].copy()
df["direction"] = np.where(is_up, "Uplink", "Downlink")

df["t_rel"] = df["frame.time_epoch"] - t0
df["win_id"] = pd.cut(
    df["t_rel"], bins=np.r_[win["t_rel_start"].values, win["t_rel_end"].values[-1]],
    right=False, labels=win["win_id"].values
).astype(int)

# Per-window features
for c in ["radiotap.dbm_antsignal", "wlan_radio.snr"]:
    if c not in df.columns: df[c] = np.nan

agg = (df.groupby(["win_id","direction"])
         .agg(pkts=("frame.len","count"),
              bytes=("frame.len","sum"),
              len_mean=("frame.len","mean"),
              len_var=("frame.len","var"),
              rssi_mean=("radiotap.dbm_antsignal","mean"),
              rssi_std=("radiotap.dbm_antsignal","std"),
              snr_mean=("wlan_radio.snr","mean"),
              snr_std=("wlan_radio.snr","std"))
         .reset_index())

dfs = df.sort_values(["win_id","direction","frame.time_epoch"]).copy()
dfs["iat"] = dfs.groupby(["win_id","direction"])["frame.time_epoch"].diff()
iat = (dfs.loc[dfs["iat"]>0]
         .groupby(["win_id","direction"])["iat"]
         .agg(iat_mean="mean", iat_var="var")
         .reset_index())

feat = agg.merge(iat, on=["win_id","direction"], how="left")

wide = feat.pivot(index="win_id", columns="direction")
wide.columns = ["_".join(c).strip() for c in wide.columns.to_flat_index()]
wide = wide.reset_index()

for base in ["pkts","bytes","len_mean","len_var","rssi_mean","rssi_std","snr_mean","snr_std","iat_mean","iat_var"]:
    for d in ["Uplink","Downlink"]:
        col = f"{base}_{d}"
        if col not in wide.columns: wide[col] = np.nan

wide["pps_Uplink"]   = wide["pkts_Uplink"]   / W
wide["pps_Downlink"] = wide["pkts_Downlink"] / W
wide["mbps_Uplink"]  = (8 * wide["bytes_Uplink"])  / (W * 1e6)
wide["mbps_Downlink"]= (8 * wide["bytes_Downlink"])/ (W * 1e6)

wide = wide.merge(win[["win_id","t_rel_start"]], on="win_id", how="left").sort_values("t_rel_start")

# QoS fractions
def tid2ac(tid: int) -> str:
    if tid in (6,7):  return "AC_VO"
    if tid in (4,5):  return "AC_VI"
    if tid in (1,2):  return "AC_BK"
    return "AC_BE"

qos = df.dropna(subset=["wlan.qos.priority"]).copy()
qos["tid"] = qos["wlan.qos.priority"].astype(int)
qos["ac"]  = qos["tid"].map(tid2ac)
qcnt = (qos.groupby(["win_id","ac"])["frame.len"].count().rename("pkts").reset_index())
qwide = qcnt.pivot(index="win_id", columns="ac", values="pkts").fillna(0.0)
for ac in ["AC_BE","AC_BK","AC_VI","AC_VO"]:
    if ac not in qwide.columns: qwide[ac] = 0.0
qwide = qwide.reset_index()
tot = qwide[["AC_BE","AC_BK","AC_VI","AC_VO"]].sum(axis=1).replace(0, np.nan)
qfrac = qwide.copy()
for ac in ["AC_BE","AC_BK","AC_VI","AC_VO"]:
    qfrac[ac] = qwide[ac] / tot
qfrac = qfrac.merge(win[["win_id","t_rel_start"]], on="win_id", how="left").sort_values("t_rel_start")

# Tools
def shade(ax):
    for _, r in sleep[sleep["sleep_fraction"]>=SLEEP_FRACTION_THRESHOLD].iterrows():
        ax.axvspan(r["t_rel_start"], r["t_rel_end"], color=SHADE_COLOR, alpha=SHADE_ALPHA, linewidth=0)

def lineplot(x, ys, labels, title, ylab, path, ylim0=True, x_end=None):
    fig, ax = plt.subplots()
    for y, lab in zip(ys, labels): ax.plot(x, y, label=lab)
    shade(ax)
    ax.set_title(title)
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel(ylab)
    ax.set_xlim(left=0, right=x_end)
    if ylim0: ax.set_ylim(bottom=0)
    ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(path, dpi=150)

# Export features for ML
qos_feats = qfrac.rename(columns={
    "AC_BE":"qos_frac_BE","AC_BK":"qos_frac_BK","AC_VI":"qos_frac_VI","AC_VO":"qos_frac_VO"
})[["win_id","qos_frac_BE","qos_frac_BK","qos_frac_VI","qos_frac_VO"]]

features = wide[[
    "win_id","t_rel_start",
    "mbps_Uplink","mbps_Downlink",
    "pps_Uplink","pps_Downlink",
    "len_mean_Uplink","len_mean_Downlink",
    "iat_mean_Uplink","iat_mean_Downlink",
    "iat_var_Uplink","iat_var_Downlink",
    "rssi_mean_Uplink","rssi_mean_Downlink",
    "snr_mean_Uplink","snr_mean_Downlink"
]].merge(sleep[["win_id","sleep_fraction"]], on="win_id", how="left") \
 .merge(qos_feats, on="win_id", how="left") \
 .fillna(0.0)

features.to_csv("./model/features_W15.csv", index=False)

# ---------- Plots ----------
x = wide["t_rel_start"].values
x_end = t_max - t0

lineplot(x,
         [wide["mbps_Uplink"], wide["mbps_Downlink"]],
         ["Uplink","Downlink"],
         f"Throughput per window (W = {W} s)",
         "Throughput (Mb/s)",
         f"./img/plot_throughput_W{W}.png",
         ylim0=True, x_end=x_end)

lineplot(x,
         [wide["pps_Uplink"], wide["pps_Downlink"]],
         ["Uplink","Downlink"],
         f"Frames per second per window (W = {W} s)",
         "Frames per second (fps)",
         f"./img/plot_fps_W{W}.png",
         ylim0=True, x_end=x_end)

lineplot(x,
         [wide["len_mean_Uplink"], wide["len_mean_Downlink"]],
         ["Uplink","Downlink"],
         f"Average frame size per window (W = {W} s)",
         "Average frame size (bytes)",
         f"./img/plot_frame_size_avg_W{W}.png",
         ylim0=True, x_end=x_end)

lineplot(x,
         [wide["iat_mean_Uplink"], wide["iat_mean_Downlink"]],
         ["Uplink","Downlink"],
         f"Average inter-arrival time per window (W = {W} s)",
         "Average inter-arrival time (s)",
         f"./img/plot_iat_avg_W{W}.png",
         ylim0=True, x_end=x_end)

lineplot(x,
         [wide["iat_var_Uplink"], wide["iat_var_Downlink"]],
         ["Uplink","Downlink"],
         f"Variance of inter-arrival time per window (W = {W} s)",
         "Variance (s^2)",
         f"./img/plot_iat_var_W{W}.png",
         ylim0=True, x_end=x_end)

# RSSI
rssi = wide["rssi_mean_Uplink"].where(~wide["rssi_mean_Uplink"].isna(), wide["rssi_mean_Downlink"])
fig, ax = plt.subplots()
ax.plot(x, rssi, label="RSSI"); ax.set_title("Received Signal Strength Indicator (RSSI)")
ax.set_xlabel("Relative time (s)"); ax.set_ylabel("RSSI (dBm)")
ax.set_xlim(left=0, right=x_end); ax.grid(True); fig.tight_layout()
fig.savefig(f"./img/plot_rssi_W{W}.png", dpi=150)

# QoS stack
order = ["AC_VO","AC_VI","AC_BK","AC_BE"]
colors = {"AC_VO":"#e57373","AC_VI":"#ffb74d","AC_BK":"#b2dfdb","AC_BE":"#90caf9"}
labels = {"AC_VO":"Voice","AC_VI":"Video","AC_BK":"Background","AC_BE":"Best Effort"}
fig, ax = plt.subplots()
ax.stackplot(qfrac["t_rel_start"].values,
             *[qfrac[ac].values for ac in order],
             labels=[labels[a] for a in order],
             colors=[colors[a] for a in order], alpha=0.95)
shade(ax)
ax.set_title(f"QoS Access Categories (W = {W} s)")
ax.set_xlabel("Relative time (s)"); ax.set_ylabel("Fraction of frames")
ax.set_xlim(left=0, right=x_end); ax.set_ylim(0,1)
ax.legend(loc="upper right", ncol=2, fontsize=9); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"./img/plot_qos_composition_W{W}.png", dpi=150)

# SNR
snr = wide["snr_mean_Uplink"].where(~wide["snr_mean_Uplink"].isna(), wide["snr_mean_Downlink"])
fig, ax = plt.subplots()
ax.plot(x, snr, label="SNR"); ax.set_title("Signal-to-Noise Ratio")
ax.set_xlabel("Relative time (s)"); ax.set_ylabel("SNR (dB)")
ax.set_xlim(left=0, right=x_end); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(True); fig.tight_layout()
fig.savefig(f"./img/plot_snr_W{W}.png", dpi=150)

print("Plots saved in ./img")