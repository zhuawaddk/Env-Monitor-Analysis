# 多城市空气质量监测 AI 分析系统

基于机器学习的多城市空气质量监测、预测与预警系统。以 Open-Meteo（CAMS 空气质量再分析 + ERA5 气象再分析）的 6 城市近一年小时级真实数据为分析基座，按城市分别训练随机森林模型实现 PM2.5 / PM10 / O3 / NO2 浓度预测与未来 24 小时超标预警，通过 FastAPI 提供 RESTful 接口，前端支持城市一键切换，并接入 WAQI 实时数据与 FastGPT 大模型问答。

## 功能特性

- **多城市支持**：北京、上海、广州、深圳、成都、西安 6 城市，前端顶部一键切换，全部图表、预测、问答联动
- **数据分析**：AQI 等级分布、季节特征、日变化模式、污染物与气象相关性矩阵、城市间对比
- **浓度预测**：按城市分别训练随机森林模型，支持 PM2.5 / PM10 / O3 / NO2 四种污染物
- **预测性预警**：未来 24 小时递归预测，输出超标风险等级、预计峰值与首次超标时间，附 24h 预测曲线
- **综合预警中心**：六项污染物按 IAQI 分级（黄/橙/红），未来 24h 预测超标预警，以及基于 Open-Meteo 预报的高温（黄/橙/红）、大风（蓝/黄/橙/红）、暴雨（蓝/黄/橙/红）多级预警；可选接入和风天气官方预警（台风等）
- **实时数据**：接入 WAQI（World Air Quality Index）实时监测，未配置 Token 时自动降级为历史最新时点
- **智能问答**：接入 FastGPT / Dify 大模型（数据上下文注入），未配置 API Key 时降级为内置规则引擎；前端内置轻量 Markdown 渲染（粗体/列表/代码/换行），回答排版统一

## 技术栈

Python · FastAPI · Scikit-learn · Pandas · Chart.js · FastGPT · WAQI API · Open-Meteo API

## 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| 历史空气质量 | Open-Meteo Air Quality API（CAMS 再分析） | 6 污染物，小时级，近一年 |
| 历史气象 | Open-Meteo Archive API（ERA5 再分析） | 温度/湿度/风速，同一经纬度对齐 |
| 实时空气质量 | WAQI API | 地面监测站实时数据，10 分钟缓存 |

历史数据管道见 `fetch_data.py`，输出 `data/air_quality_cities.csv`（6 城市 × 8,784 小时 ≈ 5.3 万条）。

> 注：CAMS 为卫星同化再分析数据，浓度量级与地面站存在系统性偏差（如 SO2 偏高），
> 适合模式分析与建模演示；如需地面站实测历史数据，可替换为 OpenAQ（需注册免费 API Key）。

## 模型效果（时间序列 80/20 划分，无随机泄漏）

6 城市 × 4 污染物共 24 个模型，R² 介于 0.92 ~ 0.98。以北京为例：

| 目标污染物 | R² | MAE (μg/m³) |
|-----------|------|------|
| PM2.5 | 0.935 | 5.15 |
| PM10  | 0.917 | 6.11 |
| O3    | 0.968 | 7.46 |
| NO2   | 0.951 | 5.15 |

特征重要性（北京 PM2.5 模型）：滞后 1h 浓度 0.81，SO2 0.14，其余 < 0.02。

**预警回测**（全年滚动，24h 窗口，PM2.5 阈值 75 μg/m³）：

| 城市 | 超标事件数 | 召回率 | 精确率 |
|------|-----------|--------|--------|
| 北京 | 300 | 76.7% | 96.6% |
| 成都 | 234 | 79.9% | 80.6% |

## 特征工程

每个目标污染物使用 12~13 维输入特征：

- 其他污染物浓度（**剔除与目标存在整体-部分包含关系的交叉项**：预测 PM2.5 时不用 PM10，反之亦然，避免伪相关）
- 气象因子：温度、湿度、风速
- 周期因子：时刻（hour）、月份（month）
- 时序特征：目标污染物滞后 1h、滞后 24h、6h 滚动均值

AQI 按 HJ 633-2012 各污染物 IAQI 断点分段线性计算后取最大值。

## 快速开始

```bash
pip install -r requirements.txt

# 可选: 配置 FastGPT 问答（不配置则使用内置规则引擎）
set LLM_PROVIDER=fastgpt
set FASTGPT_API_KEY=fastgpt-xxxxxxxx
# 注意: FASTGPT_BASE_URL 以 FastGPT「账号 → API密钥」页面显示的 API 根地址为准
# 国内新版(cloud.fastgpt.cn):  set FASTGPT_BASE_URL=https://cloud.fastgpt.cn/api
# 国际版(fastgpt.in):          默认为 https://api.fastgpt.in/api 无需设置
set FASTGPT_BASE_URL=https://cloud.fastgpt.cn/api
# 若 API Key 为「账号级/全局密钥」(在 账号→API密钥 页创建), 还需传入应用 ID:
# 打开应用详情页, 从浏览器地址栏复制 /app/detail/{appId}/ 中间那串
set FASTGPT_APP_ID=你的应用ID
# 若使用应用内「API访问」页创建的应用级密钥, 则无需设置 FASTGPT_APP_ID

# 可选: 配置 WAQI 实时数据（不配置则降级为历史最新时点）
# 免费 Token 注册: https://aqicn.org/data-platform/token/
set WAQI_TOKEN=your_token

# 可选: 配置和风天气官方预警（台风/暴雨等气象部门正式发布）
# 免费 Key 注册: https://console.qweather.com
set QWEATHER_KEY=your_key

python main.py
```

访问 http://localhost:8000

模型首次使用时自动训练；也可手动触发：

```bash
curl -X POST "http://localhost:8000/api/model/train?city=beijing"
```

## API 接口（均支持 `?city=` 指定城市，图表类接口支持 `&pollutant=` 指定分析污染物）

| 接口 | 说明 |
|------|------|
| `GET /api/health` | 服务健康诊断（问答模式 `chat_mode`、最近错误 `chat_error`、WAQI 配置状态） |
| `GET /api/overview?city=` | 城市汇总指标 |
| `GET /api/cities` / `GET /api/cities/latest` / `GET /api/cities/realtime` | 城市对比 / 各城市最新记录 / 各城市实时数据（与仪表盘同源） |
| `GET /api/season-analysis?city=` | 季节分析 |
| `GET /api/aqi-distribution?city=` | AQI 等级分布 |
| `GET /api/correlation?city=` | 相关性矩阵 |
| `GET /api/charts/pm25-monthly?city=` | 月度趋势（不带 city 为各城市对比） |
| `GET /api/charts/diurnal-pattern?city=` | 日变化模式 |
| `GET /api/charts/season-boxplot?city=` | 季节分布 |
| `GET /api/charts/pollutant-means?city=` | 各污染物均值对比 |
| `GET /api/charts/primary-pollutant?city=` | 首要污染物占比 |
| `POST /api/model/train?city=` | 训练模型 |
| `GET /api/model/feature-importance?pollutant=&city=` | 特征重要性 |
| `POST /api/predict?pollutant=&city=` | 单条预测 |
| `GET /api/predict/prefill?pollutant=&city=` | 用最新实测值预填预测输入 |
| `POST /api/predict/batch?pollutant=&city=` | 批量预测 |
| `GET /api/alerts?city=` | 当前超标告警 |
| `GET /api/alerts/forecast?pollutant=&city=` | 未来 24h 预测预警（含预测序列） |
| `GET /api/alerts/comprehensive?city=` | 综合预警中心（污染物分级 + 高温/大风/暴雨多级预警 + 官方预警） |
| `GET /api/realtime?city=` | 实时空气质量（WAQI 或降级） |
| `POST /api/chat` | 智能问答（body: `{"question": "...", "city": "beijing"}`） |

## 项目结构

```
├── main.py                # FastAPI 入口与全部 API 接口
├── fetch_data.py          # 数据管道: Open-Meteo 6城市一年数据拉取与 AQI 计算
├── backend/
│   ├── engine.py          # 数据清洗、特征工程、模型训练、预测预警、规则问答
│   ├── realtime.py        # WAQI 实时接入（IAQI 按 HJ 633-2012 反解浓度）
│   ├── weather_alerts.py  # Open-Meteo 预报驱动的高温/大风/暴雨多级预警 + 和风官方预警
│   ├── llm_chat.py        # FastGPT / Dify 接入（支持全局密钥 appId）与规则兜底
│   └── models/            # 训练好的模型缓存（{city}_{pollutant}_model.pkl）
├── static/
│   └── index.html         # 前端界面（城市切换 / 仪表盘 / 综合预警 / 城市对比 / 预测 / 问答）
└── data/
    └── air_quality_cities.csv
```

## 许可证

MIT
