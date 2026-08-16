import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from backend import engine
from backend.llm_chat import answer_question_smart, llm_configured, get_last_error, LLM_PROVIDER
from backend.realtime import fetch_realtime, realtime_configured
from backend.weather_alerts import compute_weather_alerts, fetch_official_warnings, qweather_configured
from backend.engine import (
    load_data, preprocess, get_city_overview, get_city_stats, get_season_analysis,
    get_correlation, get_aqi_distribution, get_latest_data, get_alerts,
    train_model, train_all, predict_single, predict_batch, get_feature_importance,
    get_chart_pm25_monthly, get_chart_diurnal_pattern, get_chart_season_boxplot,
    get_forecast_alerts, answer_question, POLLUTANTS, THRESHOLDS, CITIES, DEFAULT_CITY
)

app = FastAPI(title="多城市空气质量监测AI系统", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files & Root Route ─────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


_df_cache = None


def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = preprocess(load_data())
    return _df_cache


# ─── Models ───────────────────────────────────────────────

class PredictInput(BaseModel):
    PM25: float = 0
    PM10: float = 0
    SO2: float = 0
    NO2: float = 0
    CO: float = 0
    O3: float = 0
    temperature: float = 0
    humidity: float = 0
    wind_speed: float = 0
    PM25_lag1h: float = 0
    PM25_lag24h: float = 0
    PM25_roll6h: float = 0
    PM10_lag1h: float = 0
    PM10_lag24h: float = 0
    PM10_roll6h: float = 0
    O3_lag1h: float = 0
    O3_lag24h: float = 0
    O3_roll6h: float = 0
    NO2_lag1h: float = 0
    NO2_lag24h: float = 0
    NO2_roll6h: float = 0


class ChatRequest(BaseModel):
    question: str
    city: str = DEFAULT_CITY


# ─── 校验 ─────────────────────────────────────────────────

def _check_city(city: str):
    try:
        return engine.check_city(city)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _check_pollutant(pollutant: str):
    p = pollutant.upper().replace(".", "")
    if p == "PM2_5":
        p = "PM25"
    if p not in POLLUTANTS:
        raise HTTPException(400, f"暂不支持 {pollutant}，可选: {POLLUTANTS}")
    return p


# ─── Data Endpoints ───────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "多城市空气质量监测AI系统",
        "chat_mode": LLM_PROVIDER if llm_configured() else "rule-fallback",
        "chat_error": get_last_error() if llm_configured() else "",
        "realtime": "waqi" if realtime_configured() else "offline",
        "cities": [{"key": k, "name": v} for k, v in CITIES.items()],
    }


@app.get("/api/overview")
def overview(city: str = DEFAULT_CITY):
    df = get_df()
    stats = get_city_stats(df, _check_city(city))
    stats["total_records"] = len(df)
    stats["cities"] = df["city"].nunique()
    return stats


@app.get("/api/cities")
def cities():
    return get_city_overview(get_df())


@app.get("/api/cities/latest")
def latest():
    return get_latest_data(get_df())


@app.get("/api/cities/realtime")
def cities_realtime():
    """
    城市对比表数据: 配置 WAQI_TOKEN 时返回 6 城市实时数据（与仪表盘实时板块同源）,
    否则降级为历史基座（CAMS 再分析）最新时点。
    """
    if realtime_configured():
        rows = []
        for key, name in CITIES.items():
            live = fetch_realtime(key)
            if not live:
                continue
            pol = live["pollutants"]
            rows.append({
                "city": key, "station": name, "time": live["time"],
                "AQI": live["aqi"],
                "PM25": pol.get("PM25"), "PM10": pol.get("PM10"),
                "SO2": pol.get("SO2"), "NO2": pol.get("NO2"),
                "CO": pol.get("CO"), "O3": pol.get("O3"),
                "temperature": live["weather"].get("temperature"),
                "humidity": live["weather"].get("humidity"),
                "wind_speed": live["weather"].get("wind_speed"),
                "aqi_level": engine.aqi_level_name(live["aqi"]) if isinstance(live["aqi"], (int, float)) else "-",
            })
        if rows:
            return {"live": True, "source": "WAQI 实时（地面监测站）", "rows": rows}

    rows = get_latest_data(get_df())
    return {
        "live": False,
        "source": f"历史基座最新时点（CAMS 再分析，截至 {rows[0]['timestamp'][:10]}；未配置 WAQI_TOKEN）",
        "rows": rows,
    }


@app.get("/api/season-analysis")
def season_analysis(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_season_analysis(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/correlation")
def correlation(city: str = DEFAULT_CITY):
    return get_correlation(get_df(), _check_city(city))


@app.get("/api/aqi-distribution")
def aqi_distribution(city: str = DEFAULT_CITY):
    return get_aqi_distribution(get_df(), _check_city(city))


@app.get("/api/alerts")
def alerts(city: str = DEFAULT_CITY):
    """当前告警: 有实时数据时基于实时值, 否则基于历史基座最新时点"""
    city = _check_city(city)
    live = fetch_realtime(city)
    if live:
        out = []
        for key, threshold in THRESHOLDS.items():
            v = live["pollutants"].get(key)
            if isinstance(v, (int, float)) and v > threshold:
                out.append({
                    "station": CITIES[city], "metric": key, "value": v,
                    "threshold": threshold,
                    "level": "warning" if v < threshold * 1.5 else "critical",
                    "time": live["time"], "source": "live",
                })
        return out
    return get_alerts(get_df(), city)


# ─── Chart Data Endpoints ─────────────────────────────────

@app.get("/api/charts/pm25-monthly")
def chart_pm25_monthly(city: str = None, pollutant: str = "PM25"):
    """不带 city 时返回各城市对比, 带 city 时返回单城市; pollutant 指定分析污染物"""
    return get_chart_pm25_monthly(get_df(), _check_city(city) if city else None, _check_pollutant(pollutant))


@app.get("/api/charts/diurnal-pattern")
def chart_diurnal(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_chart_diurnal_pattern(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/charts/season-boxplot")
def chart_season(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_chart_season_boxplot(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/charts/pollutant-means")
def chart_pollutant_means(city: str = DEFAULT_CITY):
    """单城市各污染物年均值"""
    return engine.get_chart_pollutant_means(get_df(), _check_city(city))


@app.get("/api/charts/primary-pollutant")
def chart_primary_pollutant(city: str = DEFAULT_CITY):
    """首要污染物（IAQI 最高者）占比"""
    return engine.get_chart_primary_pollutant(get_df(), _check_city(city))


# ─── ML Endpoints ─────────────────────────────────────────

@app.post("/api/model/train")
def api_train(city: str = DEFAULT_CITY):
    """训练指定城市全部污染物模型（时间序列 80/20 划分）"""
    return {"status": "ok", "models": train_all(get_df(), _check_city(city))}


@app.get("/api/model/feature-importance")
def feature_importance(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    p = _check_pollutant(pollutant)
    items = get_feature_importance(get_df(), p, _check_city(city))
    total = sum(i["importance"] for i in items)
    for i in items:
        i["percentage"] = round(i["importance"] / total * 100, 1)
    return items


@app.post("/api/predict")
def api_predict(data: PredictInput, pollutant: str = "PM25", city: str = DEFAULT_CITY):
    return predict_single(data.model_dump(), _check_pollutant(pollutant), _check_city(city))


@app.get("/api/predict/prefill")
def api_predict_prefill(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    """用该城市最新观测值预填预测表单（含真实滞后/滚动特征）"""
    return engine.build_prefill(get_df(), _check_pollutant(pollutant), _check_city(city))


@app.post("/api/predict/batch")
def api_predict_batch(data: List[PredictInput], pollutant: str = "PM25", city: str = DEFAULT_CITY):
    return predict_batch([d.model_dump() for d in data], _check_pollutant(pollutant), _check_city(city))


@app.get("/api/alerts/forecast")
def alerts_forecast(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    """预测性预警: 指定城市未来 24 小时超标风险（有实时数据时以实时观测为起点）"""
    city = _check_city(city)
    p = _check_pollutant(pollutant)
    live = fetch_realtime(city)
    return get_forecast_alerts(get_df(), p, city, live=live)


@app.get("/api/alerts/comprehensive")
def alerts_comprehensive(city: str = DEFAULT_CITY):
    """
    综合预警中心:
    - pollution: 六项污染物多级预警（IAQI>100/150/200 → 黄/橙/红）
    - forecast:  四种污染物未来 24h 预测超标
    - weather:   高温/大风/暴雨多级预警（Open-Meteo 未来 24h 预报）
    - official:  气象部门官方预警（台风等, 需配置 QWEATHER_KEY）
    """
    city = _check_city(city)
    df = get_df()

    live = fetch_realtime(city)
    if live:
        vals = live["pollutants"]
        basis = f"实时数据（{live['time']}，WAQI）"
    else:
        row = engine.get_city_df(df, city).iloc[-1]
        vals = {k: float(row[k]) for k in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"]}
        basis = f"历史基座最新时点（{row['timestamp']}）"
    pollution = engine.get_pollutant_alerts_multilevel(vals, basis)

    forecast = []
    for p in POLLUTANTS:
        f = get_forecast_alerts(df, p, city, live=live)[0]
        if f["will_exceed"]:
            forecast.append({
                "category": "空气污染·预测",
                "metric": engine.P_LABELS[p],
                "value": f["forecast_24h_max"],
                "unit": "μg/m³",
                "threshold": f["threshold"],
                "level": "orange" if f["level"] == "critical" else "yellow",
                "level_name": "橙色预警" if f["level"] == "critical" else "黄色预警",
                "basis": f"未来24h预测，预计{f['first_exceed_in_hours']}h后超标",
            })

    weather = compute_weather_alerts(city)
    official = fetch_official_warnings(city) or []

    return {
        "city": CITIES[city],
        "basis": basis,
        "pollution": pollution,
        "forecast": forecast,
        "weather": weather,
        "official": official,
        "official_enabled": qweather_configured(),
    }


@app.get("/api/realtime")
def realtime(city: str = DEFAULT_CITY):
    """
    实时空气质量: 配置了 WAQI_TOKEN 时返回实时数据（含阈值告警）,
    否则降级为数据集最新一条记录（live=False）。
    """
    city = _check_city(city)
    live = fetch_realtime(city)
    if live:
        alerts = []
        for key, threshold in THRESHOLDS.items():
            v = live["pollutants"].get(key)
            if v and v > threshold:
                alerts.append({
                    "metric": key, "value": v, "threshold": threshold,
                    "level": "warning" if v < threshold * 1.5 else "critical"
                })
        live["alerts"] = alerts
        return live

    # 降级: 历史数据最新时点
    df = engine.get_city_df(get_df(), city)
    row = df.iloc[-1]
    hist_alerts = get_alerts(get_df(), city)
    return {
        "live": False,
        "city": city,
        "source": f"历史数据最新时点（未配置实时源 WAQI_TOKEN）",
        "time": str(row["timestamp"]),
        "stations": [{
            "station": CITIES[city],
            "PM25": round(float(row["PM25"]), 1),
            "PM10": round(float(row["PM10"]), 1),
            "SO2": round(float(row["SO2"]), 1),
            "NO2": round(float(row["NO2"]), 1),
            "O3": round(float(row["O3"]), 1),
            "AQI": int(row["AQI"]),
            "aqi_level": row["aqi_level"],
        }],
        "alerts": hist_alerts,
    }


# ─── AI Chat ──────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    result = answer_question_smart(req.question, engine, _check_city(req.city))
    return {"question": req.question, **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
