"""
环境监测数据生成脚本
输出: 5个监测站点，每小时一条记录，跨度12个月，约21万条
指标: PM2.5、PM10、SO2、NO2、CO、O3、温度、湿度、风速、AQI
"""
import csv
import math
import os
import random
from datetime import datetime, timedelta

random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

stations = {
    "S01_城区站": {"lat": 23.13, "lon": 113.26, "type": "urban", "base_pm25": 35, "base_no2": 40},
    "S02_工业园站": {"lat": 23.08, "lon": 113.45, "type": "industrial", "base_pm25": 50, "base_so2": 25},
    "S03_郊区站": {"lat": 23.30, "lon": 113.15, "type": "suburban", "base_pm25": 20, "base_o3": 55},
    "S04_滨海站": {"lat": 22.80, "lon": 113.60, "type": "coastal", "base_pm25": 15, "base_o3": 60},
    "S05_山脚站": {"lat": 23.50, "lon": 113.80, "type": "mountain", "base_pm25": 12, "base_temp": 2},
}

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31, 23)
total_hours = int((end - start).total_seconds() / 3600) + 1

with open(f"{DATA_DIR}/env_monitor_data.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "station_id", "station_type",
                 "PM25", "PM10", "SO2", "NO2", "CO", "O3",
                 "temperature", "humidity", "wind_speed", "AQI"])

    for h in range(total_hours):
        ts = start + timedelta(hours=h)
        day_of_year = ts.timetuple().tm_yday
        hour_of_day = ts.hour

        # 季节因子 (冬季高, 夏季低)
        season = math.sin((day_of_year - 80) / 365 * 2 * math.pi)

        # 日循环因子 (早晚高峰)
        rush_hour = math.exp(-((hour_of_day - 8) ** 2) / 8) + math.exp(-((hour_of_day - 18) ** 2) / 8)

        # 天气基线
        base_temp = 15 + 15 * math.sin((day_of_year - 100) / 365 * 2 * math.pi)
        base_temp += -5 + hour_of_day * 1.2 if hour_of_day < 14 else 5 - (hour_of_day - 14) * 1.2

        base_humidity = 60 + 20 * math.sin((day_of_year + 30) / 365 * 2 * math.pi)
        base_wind = 3 + 2 * math.sin((day_of_year + 180) / 365 * 2 * math.pi)

        for sid, cfg in stations.items():
            # 污染物浓度 (受季节 + 早晚高峰 + 站点特征影响)
            pm25 = max(1, cfg["base_pm25"] + 20 * season + 15 * rush_hour + random.gauss(0, 8))
            pm10 = max(2, pm25 * 1.8 + random.gauss(0, 12))
            so2 = max(0.5, cfg.get("base_so2", 5) + 8 * season + random.gauss(0, 3))
            no2 = max(1, cfg.get("base_no2", 15) + 10 * rush_hour + random.gauss(0, 8))
            co = max(0.2, 0.8 + 0.5 * rush_hour + random.gauss(0, 0.15))
            o3 = max(2, 40 + 20 * math.sin((day_of_year + 60) / 365 * 2 * math.pi) + random.gauss(0, 10))

            # 气象
            t_offset = cfg.get("base_temp", 0)
            temp = round(base_temp + t_offset + random.gauss(0, 1.5), 1)
            humidity = max(15, min(100, round(base_humidity + random.gauss(0, 5), 1)))
            wind = max(0, round(base_wind + random.gauss(0, 1), 1))

            # 简化 AQI 计算 (取各分指数最大值)
            aqi_pm25 = min(500, pm25 / 35 * 50 if pm25 <= 35 else 50 + (pm25 - 35) / 40 * 50 if pm25 <= 75 else
                           100 + (pm25 - 75) / 40 * 50 if pm25 <= 115 else 150 + (pm25 - 115) / 35 * 50 if pm25 <= 150 else
                           200 + (pm25 - 150) / 50 * 100)
            aqi = int(max(aqi_pm25,
                          min(500, no2 / 40 * 50) if no2 <= 40 else min(500, 50 + (no2 - 40) / 40 * 50),
                          min(500, o3 / 100 * 50)))

            # 模拟缺失值 (3%)
            if random.random() < 0.03:
                missing_field = random.choice(["PM25", "PM10", "SO2", "NO2", "CO", "O3", "wind_speed"])
                vals = [ts.strftime("%Y-%m-%d %H:%M:%S"), sid, cfg["type"],
                        round(pm25, 1), round(pm10, 1), round(so2, 1), round(no2, 1),
                        round(co, 1), round(o3, 1), temp, humidity, wind, aqi]
                idx_map = {h: i for i, h in enumerate(["timestamp", "station_id", "station_type",
                         "PM25", "PM10", "SO2", "NO2", "CO", "O3",
                         "temperature", "humidity", "wind_speed", "AQI"])}
                vals[idx_map[missing_field]] = ""
                w.writerow(vals)
            else:
                w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), sid, cfg["type"],
                            round(pm25, 1), round(pm10, 1), round(so2, 1), round(no2, 1),
                            round(co, 1), round(o3, 1), temp, humidity, wind, aqi])

print(f"数据生成完成: {total_hours}x{len(stations)}={total_hours*len(stations):,} 条记录")
print(f"站点: {list(stations.keys())}")
print(f"文件: data/env_monitor_data.csv")
