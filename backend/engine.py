import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "pm25_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity",
            "wind_speed", "PM25_lag1h", "PM25_lag24h", "PM25_roll6h"]

AQI_LEVELS = ["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"]
STATION_NAMES_CN = {
    "S01_城区站": "城区站", "S02_工业园站": "工业园站",
    "S03_郊区站": "郊区站", "S04_滨海站": "滨海站", "S05_山脚站": "山脚站"
}
STATION_TYPES_CN = {"urban": "城区", "industrial": "工业园", "suburban": "郊区", "coastal": "滨海", "mountain": "山区"}


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "env_monitor_data.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    for col in ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df):
    numeric_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]
    for station in df["station_id"].unique():
        mask = df["station_id"] == station
        df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].interpolate(method="linear", limit_direction="both")
    return df


def cap_outliers(df):
    for col in ["PM25", "PM10", "SO2", "NO2"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df.loc[df[col] > upper, col] = upper
        df.loc[df[col] < lower, col] = lower
    return df


def aqi_level_name(aqi):
    if aqi <= 50: return "优"
    elif aqi <= 100: return "良"
    elif aqi <= 150: return "轻度污染"
    elif aqi <= 200: return "中度污染"
    elif aqi <= 300: return "重度污染"
    return "严重污染"


def add_features(df):
    df["aqi_level"] = df["AQI"].apply(aqi_level_name)
    df["season"] = df["month"].map({12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
                                     6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"})
    for st in df["station_id"].unique():
        mask = df["station_id"] == st
        df.loc[mask, "PM25_lag1h"] = df.loc[mask, "PM25"].shift(1)
        df.loc[mask, "PM25_lag24h"] = df.loc[mask, "PM25"].shift(24)
        df.loc[mask, "PM25_roll6h"] = df.loc[mask, "PM25"].rolling(6, min_periods=1).mean()
    df = df.dropna(subset=["PM25_lag1h", "PM25_lag24h"])
    return df


def preprocess(df):
    df = clean_data(df)
    df = cap_outliers(df)
    df = add_features(df)
    return df


def get_station_overview(df):
    stats = df.groupby("station_id").agg(
        avg_pm25=("PM25", "mean"),
        max_pm25=("PM25", "max"),
        avg_aqi=("AQI", "mean"),
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        good_days=("aqi_level", lambda x: (x.isin(["优", "良"])).mean() * 100)
    ).round(1)
    stats["good_days"] = stats["good_days"].round(1)
    stats = stats.reset_index()
    stats["station_id"] = stats["station_id"].map(STATION_NAMES_CN)
    stats = stats.rename(columns={"station_id": "name"})
    return stats.to_dict(orient="records")
    stats = df.groupby("station_id").agg(
        avg_pm25=("PM25", "mean"),
        max_pm25=("PM25", "max"),
        avg_aqi=("AQI", "mean"),
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        good_days=("aqi_level", lambda x: (x.isin(["优", "良"])).mean() * 100)
    ).round(1)
    stats["good_days"] = stats["good_days"].round(1)
    stats.index = [STATION_NAMES_CN.get(s, s) for s in stats.index]
    return stats.reset_index().rename(columns={"station_id": "name"}).to_dict(orient="records")


def get_season_analysis(df):
    season = df.groupby("season").agg(
        avg_pm25=("PM25", "mean"),
        avg_aqi=("AQI", "mean"),
        avg_temp=("temperature", "mean")
    ).round(1).reindex(["春", "夏", "秋", "冬"])
    return season.reset_index().to_dict(orient="records")


def get_correlation(df):
    cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed"]
    corr = df[cols].corr().round(3)
    result = []
    for i in cols:
        row = {"indicator": i}
        for j in cols:
            row[j] = float(corr.loc[i, j])
        result.append(row)
    return result


def get_aqi_distribution(df):
    dist = df["aqi_level"].value_counts(normalize=True) * 100
    return [{"level": lv, "percentage": round(float(dist.get(lv, 0)), 1)} for lv in AQI_LEVELS]


def get_latest_data(df):
    latest_ts = df["timestamp"].max()
    latest = df[df["timestamp"] == latest_ts]
    result = []
    for _, row in latest.iterrows():
        result.append({
            "station": STATION_NAMES_CN.get(row["station_id"], row["station_id"]),
            "station_type": STATION_TYPES_CN.get(row["station_type"], row["station_type"]),
            "timestamp": str(row["timestamp"]),
            "PM25": round(float(row["PM25"]), 1),
            "PM10": round(float(row["PM10"]), 1),
            "SO2": round(float(row["SO2"]), 1),
            "NO2": round(float(row["NO2"]), 1),
            "CO": round(float(row["CO"]), 2),
            "O3": round(float(row["O3"]), 1),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "wind_speed": float(row["wind_speed"]),
            "AQI": int(row["AQI"]),
            "aqi_level": aqi_level_name(row["AQI"])
        })
    return result


def get_alerts(df):
    latest = get_latest_data(df)
    alerts = []
    thresholds = {"PM25": 75, "PM10": 150, "AQI": 100}
    for station in latest:
        for key, threshold in thresholds.items():
            if station[key] > threshold:
                alerts.append({
                    "station": station["station"],
                    "metric": key,
                    "value": station[key],
                    "threshold": threshold,
                    "level": "warning" if station[key] < threshold * 1.5 else "critical"
                })
    return alerts


def train_model(df):
    X = df[FEATURES].values
    y = df["PM25"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4)
    }
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).to_dict(orient="records")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)
    return {"metrics": metrics, "importance": importance}


def load_model():
    if not os.path.exists(MODEL_PATH):
        df = preprocess(load_data())
        train_model(df)
    return joblib.load(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        df = preprocess(load_data())
        train_model(df)
    return joblib.load(MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = preprocess(load_data())
    return train_model(df)


def predict_single(data: dict):
    model = load_model()
    features = FEATURES
    X = np.array([[data.get(f, 0) for f in features]])
    pred = float(model.predict(X)[0])
    return {"pm25_prediction": round(pred, 1)}


def predict_batch(items: list):
    model = load_model()
    X = np.array([[d.get(f, 0) for f in FEATURES] for d in items])
    preds = [round(float(p), 1) for p in model.predict(X)]
    return [{"predicted_pm25": p} for p in preds]


def get_chart_pm25_monthly(df):
    pivot = df.pivot_table(values="PM25", index="month", columns="station_id", aggfunc="mean")
    result = []
    for month in range(1, 13):
        entry = {"month": month}
        for st in pivot.columns:
            entry[STATION_NAMES_CN.get(st, st)] = round(float(pivot.loc[month, st]), 1) if month in pivot.index else None
        result.append(entry)
    return result


def get_chart_diurnal_pattern(df):
    hourly = df.groupby(["station_id", "hour"])["PM25"].mean().reset_index()
    result = []
    for st in hourly["station_id"].unique():
        series = []
        for h in range(24):
            val = hourly[(hourly["station_id"] == st) & (hourly["hour"] == h)]["PM25"]
            series.append(round(float(val.iloc[0]), 1) if len(val) > 0 else None)
        result.append({"station": STATION_NAMES_CN.get(st, st), "hourly_pm25": series})
    return result


def get_chart_season_boxplot(df):
    season_order = ["春", "夏", "秋", "冬"]
    result = []
    for s in season_order:
        vals = df[df["season"] == s]["PM25"].dropna()
        result.append({
            "season": s,
            "min": round(float(vals.min()), 1),
            "q1": round(float(vals.quantile(0.25)), 1),
            "median": round(float(vals.median()), 1),
            "q3": round(float(vals.quantile(0.75)), 1),
            "max": round(float(vals.max()), 1),
            "mean": round(float(vals.mean()), 1)
        })
    return result


def get_feature_importance():
    df = preprocess(load_data())
    if not os.path.exists(MODEL_PATH):
        train_model(df)
    model = load_model()
    return [{"feature": f, "importance": round(float(i), 4)}
            for f, i in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)]


def answer_question(question: str) -> str:
    df = preprocess(load_data())
    q = question.lower()

    if any(w in q for w in ["pm2.5", "pm25", "pm", "细颗粒物"]):
        avg = df["PM25"].mean()
        max_v = df["PM25"].max()
        worst = df.loc[df["PM25"].idxmax()]
        return (
            f"根据2024年全年监测数据:\n"
            f"  - 所有站点PM2.5均值: {avg:.1f} μg/m³\n"
            f"  - 最高PM2.5: {max_v:.1f} μg/m³, 发生在{worst['station_id']} ({str(worst['timestamp'])})\n"
            f"  - 工业园站均值最高, 山脚站均值最低"
        )

    if any(w in q for w in ["aqi", "空气质量", "指数"]):
        dist = df["aqi_level"].value_counts(normalize=True) * 100
        good_pct = dist.get("优", 0) + dist.get("良", 0)
        return (
            f"空气质量分析:\n"
            f"  - 优良率: {good_pct:.1f}%\n"
            f"  - 优: {dist.get('优', 0):.1f}% | 良: {dist.get('良', 0):.1f}%\n"
            f"  - 轻度污染: {dist.get('轻度污染', 0):.1f}% | 中度+: {dist.get('中度污染', 0)+dist.get('重度污染', 0)+dist.get('严重污染', 0):.1f}%"
        )

    if any(w in q for w in ["季节", "冬季", "夏季", "春季", "秋季", "冬天", "夏天"]):
        season = df.groupby("season")["PM25"].mean().round(1)
        spring = season.get("春", 0)
        summer = season.get("夏", 0)
        autumn = season.get("秋", 0)
        winter = season.get("冬", 0)
        return (
            f"各季节PM2.5均值 (μg/m³):\n"
            f"  春季: {spring:.1f} | 夏季: {summer:.1f}\n"
            f"  秋季: {autumn:.1f} | 冬季: {winter:.1f}\n"
            f"  冬季污染最重, 夏季最轻, 呈现出明显的季节性特征"
        )

    if any(w in q for w in ["站点", "监测站", "站"]):
        stations = get_station_overview(df)
        lines = ["各监测站点空气质量对比:"]
        for s in stations:
            lines.append(f"  {s['name']}: PM2.5均值 {s['avg_pm25']} μg/m³, AQI均值 {s['avg_aqi']}, 优良率 {s['good_days']}%")
        return "\n".join(lines)

    if any(w in q for w in ["预测", "predict", "forecast"]):
        return (
            "本系统使用随机森林模型预测PM2.5浓度。预测需要输入以下指标:\n"
            "  PM10, SO2, NO2, CO, O3, 温度, 湿度, 风速, PM2.5滞后值(1h/24h), PM2.5滚动均值(6h)\n"
            "  请通过 /api/predict 接口提交数据进行预测"
        )

    if any(w in q for w in ["警报", "告警", "超标", "预警", "alarm"]):
        alerts = get_alerts(df)
        if alerts:
            lines = [f"当前共有 {len(alerts)} 项超标告警:"]
            for a in alerts:
                lines.append(f"  {a['station']}: {a['metric']}={a['value']} (阈值{a['threshold']}, 级别: {a['level']})")
            return "\n".join(lines)
        return "当前无超标告警, 所有监测指标正常"

    return (
        "我是环境监测AI助手, 可以回答以下问题:\n"
        "  - PM2.5 / AQI 整体情况\n"
        "  - 各站点空气质量对比\n"
        "  - 季节污染特征分析\n"
        "  - 当前超标告警\n"
        "  - PM2.5预测说明\n"
        "请告诉我你想了解什么?"
    )
