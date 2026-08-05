import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from backend.engine import (
    load_data, preprocess, get_station_overview, get_season_analysis,
    get_correlation, get_aqi_distribution, get_latest_data, get_alerts,
    train_model, predict_single, predict_batch, get_feature_importance,
    get_chart_pm25_monthly, get_chart_diurnal_pattern, get_chart_season_boxplot,
    answer_question
)

app = FastAPI(title="环境监测AI系统", version="1.0.0")

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


class ChatRequest(BaseModel):
    question: str


# ─── Data Endpoints ───────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "环境监测AI系统"}


@app.get("/api/overview")
def overview():
    df = get_df()
    return {
        "total_records": len(df),
        "stations": df["station_id"].nunique(),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "dimensions": df.shape[1],
        "columns": list(df.columns)
    }


@app.get("/api/stations")
def stations():
    return get_station_overview(get_df())


@app.get("/api/stations/latest")
def latest():
    return get_latest_data(get_df())


@app.get("/api/stations/{station_id}/latest")
def station_latest(station_id: str):
    df = get_df()
    if station_id not in df["station_id"].unique():
        raise HTTPException(404, f"站点 {station_id} 不存在")
    sdf = df[df["station_id"] == station_id].sort_values("timestamp")
    row = sdf.iloc[-1]
    return {
        "station": station_id,
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
        "aqi_level": row["aqi_level"]
    }


@app.get("/api/season-analysis")
def season_analysis():
    return get_season_analysis(get_df())


@app.get("/api/correlation")
def correlation():
    return get_correlation(get_df())


@app.get("/api/aqi-distribution")
def aqi_distribution():
    return get_aqi_distribution(get_df())


@app.get("/api/alerts")
def alerts():
    return get_alerts(get_df())


# ─── Chart Data Endpoints ─────────────────────────────────

@app.get("/api/charts/pm25-monthly")
def chart_pm25_monthly():
    return get_chart_pm25_monthly(get_df())


@app.get("/api/charts/diurnal-pattern")
def chart_diurnal():
    return get_chart_diurnal_pattern(get_df())


@app.get("/api/charts/season-boxplot")
def chart_season():
    return get_chart_season_boxplot(get_df())


# ─── ML Endpoints ─────────────────────────────────────────

@app.post("/api/model/train")
def api_train():
    result = train_model(get_df())
    return {"status": "ok", **result}


@app.get("/api/model/feature-importance")
def feature_importance():
    items = get_feature_importance()
    total = sum(i["importance"] for i in items)
    for i in items:
        i["percentage"] = round(i["importance"] / total * 100, 1)
    return items


@app.post("/api/predict")
def api_predict(data: PredictInput):
    return predict_single(data.model_dump())


@app.post("/api/predict/batch")
def api_predict_batch(data: List[PredictInput]):
    return predict_batch([d.model_dump() for d in data])


# ─── AI Chat ──────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    answer = answer_question(req.question)
    return {"question": req.question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
