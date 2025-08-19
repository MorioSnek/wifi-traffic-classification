# === Parameters ===
W = 15    
MY_MAC = "b4:96:a5:4c:b0:70"
CSV_PATH = "./private/traffic.csv"


SLEEP_FRACTION_THRESHOLD = 0.6
SHADE_COLOR = "#e57373"
SHADE_ALPHA = 0.18


# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df_raw = pd.read_csv(CSV_PATH)

# Types & normalization
df_raw["frame.time_epoch"] = pd.to_numeric(df_raw["frame.time_epoch"], errors="coerce")
df_raw["frame.len"] = pd.to_numeric(df_raw["frame.len"], errors="coerce")
df_raw["wlan.fc.type"] = pd.to_numeric(df_raw["wlan.fc.type"], errors="coerce")
if "wlan.fc.pwrmgt" in df_raw.columns:
    df_raw["wlan.fc.pwrmgt"] = pd.to_numeric(df_raw["wlan.fc.pwrmgt"], errors="coerce").fillna(0).astype(int)
else:
    df_raw["wlan.fc.pwrmgt"] = 0

for c in ["wlan.sa","wlan.da","wlan.ta","wlan.ra"]:
    df_raw[c] = df_raw[c].astype(str).str.lower()

t_min = float(df_raw["frame.time_epoch"].min())
t_max = float(df_raw["frame.time_epoch"].max())

# Build non-windowed awake/sleep intervals
uplink_all = df_raw[(df_raw["wlan.sa"] == MY_MAC) | (df_raw["wlan.ta"] == MY_MAC)].copy()
uplink_all = uplink_all.sort_values("frame.time_epoch")
uplink_all["state"] = uplink_all["wlan.fc.pwrmgt"].clip(0,1).astype(int)  # 0=awake, 1=sleep

# Compress into runs of constant state
uplink_all["chg"] = uplink_all["state"].ne(uplink_all["state"].shift(1)).cumsum()
runs = (
    uplink_all.groupby("chg")
              .agg(state=("state","last"),
                   start=("frame.time_epoch","min"),
                   end=("frame.time_epoch","max"))
              .reset_index(drop=True)
)

if not runs.empty:
    runs["next_start"] = runs["start"].shift(-1)
    runs.loc[runs.index[:-1], "end"] = runs.loc[runs.index[:-1], "next_start"]
    runs = runs.drop(columns=["next_start"])

    # Make sure the very first run starts at t_min and the last ends at t_max
    runs.loc[runs.index[0], "start"] = t_min
    runs.loc[runs.index[-1], "end"] = t_max

# Sleep fraction per window
win_start_abs = np.arange(np.floor(t_min / W)*W, np.ceil(t_max / W)*W, W)
win_id = np.arange(len(win_start_abs))
win_map = pd.DataFrame({
    "win_id": win_id,
    "t_abs_start": win_start_abs,
    "t_abs_end": win_start_abs + W
})
t0 = float(win_map["t_abs_start"].min())
win_map["t_rel_start"] = win_map["t_abs_start"] - t0
win_map["t_rel_end"]   = win_map["t_abs_end"]   - t0

sleep_time = np.zeros(len(win_map), dtype=float)
if not runs.empty:
    # Iterate only over sleep intervals
    for _, r in runs[runs["state"] == 1].iterrows():
        s = r["start"] - t0
        e = r["end"]   - t0
        if e <= win_map["t_rel_start"].iloc[0] or s >= win_map["t_rel_end"].iloc[-1]:
            continue
        # Map to window indices
        i0 = max(0, int(np.floor(s / W)))
        i1 = min(len(win_map) - 1, int(np.floor((e - 1e-9) / W)))
        for i in range(i0, i1 + 1):
            ws, we = win_map.loc[i, ["t_rel_start","t_rel_end"]]
            overlap = max(0.0, min(e, we) - max(s, ws))
            sleep_time[i] += overlap

sleep_windows = win_map.copy()
sleep_windows["sleep_time"] = sleep_time
sleep_windows["sleep_fraction"] = sleep_windows["sleep_time"] / W

# Features per window
df = df_raw[df_raw["wlan.fc.type"] == 2].copy()

is_up = (df["wlan.sa"] == MY_MAC) | (df["wlan.ta"] == MY_MAC)
is_down = (df["wlan.da"] == MY_MAC) | (df["wlan.ra"] == MY_MAC)
df = df[is_up | is_down].copy()
df["direction"] = np.where(is_up, "Uplink", "Downlink")

# Assign win_id using absolute start grid above
df["win_id"] = pd.cut(
    df["frame.time_epoch"] - t0,
    bins=np.r_[win_map["t_rel_start"].values, win_map["t_rel_end"].values[-1]],
    right=False, labels=win_id
).astype(int)

# Aggregate
agg = df.groupby(["win_id","direction"]).agg(
    pkts=("frame.len","count"),
    bytes=("frame.len","sum"),
    len_mean=("frame.len","mean"),
    len_var=("frame.len","var"),
    rssi_mean=("radiotap.dbm_antsignal","mean"),
    rssi_std=("radiotap.dbm_antsignal","std"),
).reset_index()

# Inter-arrival times inside each window/direction
dfs = df.sort_values(["win_id","direction","frame.time_epoch"]).copy()
dfs["iat"] = dfs.groupby(["win_id","direction"])["frame.time_epoch"].diff()
iat = (dfs.loc[dfs["iat"] > 0]
         .groupby(["win_id","direction"])["iat"]
         .agg(iat_mean="mean", iat_var="var")
         .reset_index())

feat = agg.merge(iat, on=["win_id","direction"], how="left")

# Wide
wide = feat.pivot(index="win_id", columns="direction")
wide.columns = ["_".join(c).strip() for c in wide.columns.to_flat_index()]
wide = wide.reset_index()

# Derived (normalized by W)
for base in ["pkts","bytes","len_mean","len_var","rssi_mean","rssi_std","iat_mean","iat_var"]:
    for d in ["Uplink","Downlink"]:
        col = f"{base}_{d}"
        if col not in wide.columns: wide[col] = np.nan

wide["pps_Uplink"] = wide["pkts_Uplink"] / W
wide["pps_Downlink"] = wide["pkts_Downlink"] / W
wide["mbps_Uplink"] = (8 * wide["bytes_Uplink"]) / (W * 1e6)
wide["mbps_Downlink"] = (8 * wide["bytes_Downlink"]) / (W * 1e6)

# Join time axis
wide = wide.merge(win_map[["win_id","t_rel_start"]], on="win_id", how="left").sort_values("t_rel_start")

# Shading
def shade_sleep_fraction(ax):
    sw = sleep_windows[sleep_windows["sleep_fraction"] >= SLEEP_FRACTION_THRESHOLD]
    for _, row in sw.iterrows():
        ax.axvspan(row["t_rel_start"], row["t_rel_end"], color=SHADE_COLOR, alpha=SHADE_ALPHA, linewidth=0)

def _tid_to_ac(tid: int) -> str:
    if tid in (6, 7):   return "AC_VO"
    if tid in (4, 5):   return "AC_VI"
    if tid in (0, 3):   return "AC_BE"
    if tid in (1, 2):   return "AC_BK"
    return "AC_BE"

def compute_qos_fractions(df: pd.DataFrame, win_map: pd.DataFrame) -> pd.DataFrame:
    qos = df.copy()
    qos["wlan.qos.priority"] = pd.to_numeric(qos["wlan.qos.priority"], errors="coerce")
    qos = qos.dropna(subset=["wlan.qos.priority"])
    qos["tid"] = qos["wlan.qos.priority"].astype(int)
    qos["ac"] = qos["tid"].map(_tid_to_ac)

    counts = (
        qos.groupby(["win_id", "ac"])["frame.len"]
           .count()
           .rename("pkts")
           .reset_index()
    )

    wide_qos = counts.pivot(index="win_id", columns="ac", values="pkts").fillna(0.0)
    wide_qos = wide_qos.merge(win_map[["win_id","t_rel_start"]], on="win_id", how="left")
    wide_qos = wide_qos.sort_values("t_rel_start")

    order = ["AC_BE", "AC_BK", "AC_VI", "AC_VO"]
    for ac in order:
        if ac not in wide_qos.columns:
            wide_qos[ac] = 0.0

    total = wide_qos[order].sum(axis=1).replace(0, np.nan)
    frac = wide_qos.copy()
    for ac in order:
        frac[ac] = wide_qos[ac] / total

    return frac[["win_id", "t_rel_start"] + order]

# Plots
# Throughput
fig1, ax1 = plt.subplots()
ax1.plot(wide["t_rel_start"], wide["mbps_Uplink"]*W, label="Uplink")
ax1.plot(wide["t_rel_start"], wide["mbps_Downlink"]*W, label="Downlink")
shade_sleep_fraction(ax1)
ax1.set_title(f"Throughput per window (W = {W} s)")
ax1.set_xlabel("Relative time (s)")
ax1.set_ylabel("Throughput (Mb/s)")
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.legend(); ax1.grid(True); fig1.tight_layout()
fig1.savefig(f"./data/plot_throughput_W{W}.png", dpi=150)

# Frames per second
fig2, ax2 = plt.subplots()
ax2.plot(wide["t_rel_start"], wide["pps_Uplink"]*W, label="Uplink")
ax2.plot(wide["t_rel_start"], wide["pps_Downlink"]*W, label="Downlink")
shade_sleep_fraction(ax2)
ax2.set_title(f"Frames per second per window (W = {W} s)")
ax2.set_xlabel("Relative time (s)")
ax2.set_ylabel("Frames per second (fps)")
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.legend(); ax2.grid(True); fig2.tight_layout()
fig2.savefig(f"./data/plot_fps_W{W}.png", dpi=150)

# Average frame size
fig3, ax3 = plt.subplots()
ax3.plot(wide["t_rel_start"], wide["len_mean_Uplink"], label="Uplink")
ax3.plot(wide["t_rel_start"], wide["len_mean_Downlink"], label="Downlink")
shade_sleep_fraction(ax3)
ax3.set_title(f"Average frame size per window (W = {W} s)")
ax3.set_xlabel("Relative time (s)")
ax3.set_ylabel("Average frame size (bytes)")
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)
ax3.legend(); ax3.grid(True); fig3.tight_layout()
fig3.savefig(f"./data/plot_frame_size_avg_W{W}.png", dpi=150)

# Inter-arrival times
fig4, ax4 = plt.subplots()
ax4.plot(wide["t_rel_start"], wide["iat_mean_Uplink"], label="Uplink")
ax4.plot(wide["t_rel_start"], wide["iat_mean_Downlink"], label="Downlink")
shade_sleep_fraction(ax4)
ax4.set_title(f"Average inter-arrival time per window (W = {W} s)")
ax4.set_xlabel("Relative time (s)")
ax4.set_ylabel("Average inter-arrival time (s)")
ax4.set_xlim(left=0)
ax4.set_ylim(bottom=0)
ax4.legend(); ax4.grid(True); fig4.tight_layout()
fig4.savefig(f"./data/plot_iat_avg_W{W}.png", dpi=150)

fig5, ax5 = plt.subplots()
ax5.plot(wide["t_rel_start"], wide["iat_var_Uplink"], label="Uplink")
ax5.plot(wide["t_rel_start"], wide["iat_var_Downlink"], label="Downlink")
shade_sleep_fraction(ax5)
ax5.set_title(f"Variance of inter-arrival time per window (W = {W} s)")
ax5.set_xlabel("Relative time (s)")
ax5.set_ylabel("Variance (s^2)")
ax5.set_xlim(left=0)
ax5.set_ylim(bottom=0)
ax5.legend(); ax5.grid(True); fig5.tight_layout()
fig5.savefig(f"./data/plot_iat_var_W{W}.png", dpi=150)

# RSSI
rssi = wide["rssi_mean_Uplink"].where(~wide["rssi_mean_Uplink"].isna(), wide["rssi_mean_Downlink"])
fig6, ax6 = plt.subplots()
ax6.plot(wide["t_rel_start"], rssi, label="RSSI")
ax6.set_title("Received Signal Strength Indicator (RSSI)")
ax6.set_xlabel("Relative time (s)")
ax6.set_ylabel("RSSI (dBm)")
ax6.grid(True); fig6.tight_layout()
fig6.savefig(f"./data/plot_rssi_W{W}.png", dpi=150)

# QoS
qos_frac = compute_qos_fractions(df, win_map)
fig7, ax7 = plt.subplots()
x = qos_frac["t_rel_start"].values
order = ["AC_VO", "AC_VI", "AC_BK", "AC_BE"]
color_map = {
    "AC_VO": "#e57373",
    "AC_VI": "#ffb74d",
    "AC_BK": "#b2dfdb",
    "AC_BE": "#90caf9",
}
label_map = {
    "AC_VO": "Voice",
    "AC_VI": "Video",
    "AC_BK": "Background",
    "AC_BE": "Best Effort",
}
ys      = [qos_frac[ac].values for ac in order]
colors  = [color_map[ac]       for ac in order]
labels  = [label_map[ac]       for ac in order]
ax7.stackplot(x, *ys, labels=labels, colors=colors, alpha=0.95)
ax7.set_title(f"QoS Access Categories (W = {W} s)")
ax7.set_xlabel("Relative time (s)")
ax7.set_ylabel("Fraction of frames")
ax7.set_xlim(left=0, right=t_max - t0)   # finisce dove finisce la cattura
ax7.set_ylim(bottom=0, top=1)
ax7.legend(loc="upper right", ncol=2, fontsize=9)
ax7.grid(True, axis="y", alpha=0.3)
fig7.tight_layout()
fig7.savefig(f"./data/plot_qos_composition_W{W}.png", dpi=150)

print("Plots saved in ./data")

