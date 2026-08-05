"""
环境监测数据分析与建模
技术栈: Python, Pandas, NumPy, Scikit-learn, Matplotlib
流程: 数据清洗 → 异常检测 → 特征工程 → 相关性分析 → PM2.5预测建模
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 数据加载与概览
# ============================================================
print("=" * 60)
print("[1/6] 数据加载...")
df = pd.read_csv(f"{DATA_DIR}/env_monitor_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month

print(f"  总记录数: {len(df):,}")
print(f"  监测站点: {df['station_id'].nunique()}")
print(f"  时间跨度: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"  数据维度: {df.shape[1]} 列 x {df.shape[0]:,} 行")

# ============================================================
# 2. 缺失值检测与插值
# ============================================================
print("\n" + "=" * 60)
print("[2/6] 缺失值处理...")

numeric_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]
missing_before = df[numeric_cols].isnull().sum()
print(f"  处理前缺失值:")
for col in numeric_cols:
    cnt = missing_before[col]
    if cnt > 0:
        print(f"    {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 按站点分组做时间序列线性插值
df = df.copy()
for station in df["station_id"].unique():
    mask = df["station_id"] == station
    df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )

missing_after = df[numeric_cols].isnull().sum().sum()
print(f"  处理后剩余缺失值: {missing_after}")

# ============================================================
# 3. 异常值检测 (IQR)
# ============================================================
print("\n" + "=" * 60)
print("[3/6] 异常值检测 (IQR方法)...")

outlier_cols = ["PM25", "PM10", "SO2", "NO2"]
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct = len(outliers) / len(df) * 100
    if pct > 0.1:
        # Cap outliers to bounds
        df.loc[df[col] > upper, col] = upper
        df.loc[df[col] < lower, col] = lower
    print(f"  {col}: 异常 {len(outliers)} 条 ({pct:.2f}%), 已截尾")

# ============================================================
# 4. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("[4/6] 特征工程...")

# AQI 等级
def aqi_level(aqi):
    if aqi <= 50: return "优"
    elif aqi <= 100: return "良"
    elif aqi <= 150: return "轻度污染"
    elif aqi <= 200: return "中度污染"
    elif aqi <= 300: return "重度污染"
    return "严重污染"

df["aqi_level"] = df["AQI"].apply(aqi_level)

# 温度区间
df["temp_range"] = pd.cut(df["temperature"], bins=[-20, 0, 10, 20, 30, 50],
                           labels=["严寒", "寒冷", "凉爽", "温暖", "炎热"])

# 滞后特征 (每个站点独立计算)
df["PM25_lag1h"] = np.nan
df["PM25_lag24h"] = np.nan
for st in df["station_id"].unique():
    mask = df["station_id"] == st
    df.loc[mask, "PM25_lag1h"] = df.loc[mask, "PM25"].shift(1)
    df.loc[mask, "PM25_lag24h"] = df.loc[mask, "PM25"].shift(24)

# 滚动统计
for st in df["station_id"].unique():
    mask = df["station_id"] == st
    df.loc[mask, "PM25_roll6h"] = df.loc[mask, "PM25"].rolling(6, min_periods=1).mean()

df = df.dropna(subset=["PM25_lag1h", "PM25_lag24h"])
print(f"  新增特征: aqi_level, temp_range, PM25_lag1h, PM25_lag24h, PM25_roll6h")
print(f"  特征工程后记录数: {len(df):,}")

# ============================================================
# 5. 探索性分析
# ============================================================
print("\n" + "=" * 60)
print("[5/6] 探索性分析...")

# 各站点统计
station_stats = df.groupby("station_id").agg(
    avg_pm25=("PM25", "mean"),
    max_pm25=("PM25", "max"),
    avg_aqi=("AQI", "mean"),
    avg_temp=("temperature", "mean"),
    avg_humidity=("humidity", "mean"),
    good_days=("aqi_level", lambda x: (x.isin(["优", "良"])).mean() * 100)
).round(1)
station_stats["good_days"] = station_stats["good_days"].round(1)
station_stats.to_csv(f"{OUTPUT_DIR}/station_comparison.csv", encoding="utf-8-sig")
print("\n  各站点空气质量对比:")
print(station_stats.to_string())

# 季节分析
df["season"] = df["month"].map({12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
                                  6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"})
season_aqi = df.groupby("season").agg(
    avg_pm25=("PM25", "mean"),
    avg_aqi=("AQI", "mean"),
    avg_temp=("temperature", "mean")
).round(1).reindex(["春", "夏", "秋", "冬"])
season_aqi.to_csv(f"{OUTPUT_DIR}/season_analysis.csv", encoding="utf-8-sig")
print(f"\n  季节分析:")
print(season_aqi.to_string())

# 相关性矩阵
corr_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed"]
corr_matrix = df[corr_cols].corr().round(3)
corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv", encoding="utf-8-sig")
print(f"\n  PM2.5与各指标相关系数:")
pm25_corr = corr_matrix["PM25"].drop("PM25").sort_values(ascending=False)
print(pm25_corr.to_string())

# AQI等级分布
aqi_dist = df["aqi_level"].value_counts(normalize=True) * 100
aqi_dist.to_csv(f"{OUTPUT_DIR}/aqi_distribution.csv", encoding="utf-8-sig")
print(f"\n  AQI等级分布:")
print(aqi_dist.to_string())

# ============================================================
# 6. PM2.5 预测建模
# ============================================================
print("\n" + "=" * 60)
print("[6/6] PM2.5 预测建模 (Random Forest)...")

features = ["PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity",
            "wind_speed", "PM25_lag1h", "PM25_lag24h", "PM25_roll6h"]
X = df[features].values
y = df["PM25"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  训练集: {len(X_train):,} | 测试集: {len(X_test):,}")

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R2: {r2:.4f}")

# 特征重要性
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False, encoding="utf-8-sig")
print(f"\n  特征重要性 Top 5:")
for _, row in importance.head(5).iterrows():
    print(f"    {row['feature']:20s} {row['importance']:.4f}")

# 预测 vs 实际 (样本)
print(f"\n  预测 vs 实际 (前10条):")
for i in range(10):
    print(f"    实际: {y_test[i]:6.1f}  |  预测: {y_pred[i]:6.1f}  |  误差: {abs(y_test[i]-y_pred[i]):5.1f}")

print("\n" + "=" * 60)
print("分析完成!")
print(f"  - output/station_comparison.csv   各站点空气质量对比")
print(f"  - output/season_analysis.csv      季节分析")
print(f"  - output/correlation_matrix.csv   污染物相关性矩阵")
print(f"  - output/aqi_distribution.csv      AQI等级分布")
print(f"  - output/feature_importance.csv    PM2.5预测特征重要性")
print(f"  PM2.5预测模型: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
