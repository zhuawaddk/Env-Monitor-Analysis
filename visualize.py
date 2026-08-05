"""
环境监测数据可视化
生成6张分析图表
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"  -> {name}")


df = pd.read_csv(f"{DATA_DIR}/env_monitor_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["month"] = df["timestamp"].dt.month
df["hour"] = df["timestamp"].dt.hour

for col in ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for station in df["station_id"].unique():
    mask = df["station_id"] == station
    numeric_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]
    df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].interpolate(method="linear", limit_direction="both")

def aqi_level(aqi):
    if pd.isna(aqi): return "未知"
    if aqi <= 50: return "优"
    elif aqi <= 100: return "良"
    elif aqi <= 150: return "轻度污染"
    elif aqi <= 200: return "中度污染"
    elif aqi <= 300: return "重度污染"
    return "严重污染"

df["aqi_level"] = df["AQI"].apply(aqi_level)

# ============================================================
# 图1: 各站点 PM2.5 月度均值对比
# ============================================================
pivot = df.pivot_table(values="PM25", index="month", columns="station_id", aggfunc="mean")
station_colors = {"S01_城区站": "#C0504D", "S02_工业园站": "#8C6E4A",
                   "S03_郊区站": "#4D8066", "S04_滨海站": "#2B5B84", "S05_山脚站": "#7EA8C4"}

fig, ax = plt.subplots(figsize=(12, 6))
for st in pivot.columns:
    ax.plot(pivot.index, pivot[st], marker="o", linewidth=2, markersize=5,
            label=st, color=station_colors.get(st, None))
ax.set_title("Monthly Average PM2.5 by Station", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("PM2.5")
ax.set_xticks(range(1, 13))
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)
save(fig, "01_pm25_monthly_by_station.png")
plt.close()

# ============================================================
# 图2: 各站点空气质量等级分布 (堆叠柱状)
# ============================================================
aqi_pivot = df.pivot_table(index="station_id", columns="aqi_level",
                            values="PM25", aggfunc="count", fill_value=0)
aqi_pivot_pct = aqi_pivot.div(aqi_pivot.sum(axis=1), axis=0) * 100
level_colors = {"优": "#4D8066", "良": "#7EA8C4", "轻度污染": "#F4A300",
                "中度污染": "#C0504D", "重度污染": "#8C6E4A", "严重污染": "#660000"}
level_order = [l for l in ["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"] if l in aqi_pivot_pct.columns]
aqi_pivot_pct = aqi_pivot_pct[level_order]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(aqi_pivot_pct))
for level in level_order:
    ax.barh(aqi_pivot_pct.index, aqi_pivot_pct[level], left=bottom,
            color=level_colors.get(level, "#999"), label=level, height=0.6)
    bottom += aqi_pivot_pct[level].values
ax.set_title("AQI Level Distribution by Station", fontsize=13, fontweight="bold")
ax.set_xlabel("Percentage (%)")
ax.legend(fontsize=8)
for i, (idx, row) in enumerate(aqi_pivot_pct.iterrows()):
    cumsum = 0
    for level in level_order:
        v = row[level]
        if v > 5:
            ax.text(cumsum + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if level in ("中度污染", "重度污染", "严重污染") else "black")
        cumsum += v
save(fig, "02_aqi_level_by_station.png")
plt.close()

# ============================================================
# 图3: 相关性热力图
# ============================================================
corr_df = pd.read_csv(f"{OUTPUT_DIR}/correlation_matrix.csv", index_col=0)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_title("Environmental Indicator Correlation Matrix", fontsize=14, fontweight="bold")
labels = corr_df.columns
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Correlation", fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr_df.values[i][j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr_df.values[i][j]) > 0.6 else "black")
save(fig, "03_correlation_heatmap.png")
plt.close()

# ============================================================
# 图4: 季节 PM2.5 箱线图
# ============================================================
df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                 3: "Spring", 4: "Spring", 5: "Spring",
                                 6: "Summer", 7: "Summer", 8: "Summer",
                                 9: "Autumn", 10: "Autumn", 11: "Autumn"})
season_order = ["Spring", "Summer", "Autumn", "Winter"]
season_data = [df[df["season"] == s]["PM25"].dropna().values for s in season_order]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(season_data, labels=season_order, patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5})
colors = ["#4D8066", "#C0504D", "#F4A300", "#2B5B84"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("PM2.5 Distribution by Season", fontsize=14, fontweight="bold")
ax.set_ylabel("PM2.5")
ax.grid(axis="y", alpha=0.3)
save(fig, "04_pm25_season_boxplot.png")
plt.close()

# ============================================================
# 图5: PM2.5 日变化模式 (各站点)
# ============================================================
hourly = df.groupby(["station_id", "hour"])["PM25"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
for st in hourly["station_id"].unique():
    data = hourly[hourly["station_id"] == st]
    ax.plot(data["hour"], data["PM25"], marker="o", linewidth=2, markersize=4,
            label=st, color=station_colors.get(st, None))
ax.set_title("Diurnal PM2.5 Pattern by Station", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average PM2.5")
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.axvspan(7, 9, alpha=0.08, color="red", label="Morning Rush")
ax.axvspan(17, 19, alpha=0.08, color="red", label="Evening Rush")
save(fig, "05_diurnal_pattern.png")
plt.close()

# ============================================================
# 图6: 温度-PM2.5 散点 (按季节着色)
# ============================================================
season_color = {"Spring": "#4D8066", "Summer": "#C0504D", "Autumn": "#F4A300", "Winter": "#2B5B84"}
sample = df.sample(2000, random_state=42)

fig, ax = plt.subplots(figsize=(8, 6))
for s in season_order:
    mask = sample["season"] == s
    ax.scatter(sample.loc[mask, "temperature"], sample.loc[mask, "PM25"],
               c=season_color[s], label=s, alpha=0.5, s=15)
ax.set_title("Temperature vs PM2.5 by Season", fontsize=14, fontweight="bold")
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("PM2.5")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
save(fig, "06_temp_vs_pm25_scatter.png")
plt.close()

print("\n可视化完成! 图表已保存至 output/ 目录:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print(f"  - {f}")
