"""
RAG 模块 - 环保知识库检索（关键词匹配 + 相关度打分，零外部依赖）
后续可升级为 ChromaDB + BGE-M3 向量检索

幻觉控制第一道防线：每条检索结果带 0~1 归一化相关度分数，
最佳分数低于 MIN_SCORE 阈值时判定为"低置信"，返回空结果 + 低置信标记，
由上层（query_policy 工具 / orchestrator）执行拒答，避免编造标准条款。
"""
import os
import json
import re
from typing import List, Dict, Any

# 知识库文件路径
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "documents", "knowledge.json")

# 低置信拒答阈值（归一化相关度 0~1）
# 取值理由：单个查询词仅在正文中偶然命中约贡献 0.1 分左右；
# 0.15 可以过滤掉这类偶然命中，同时保留"关键词命中但覆盖不全"的弱相关结果。
# 阈值过低会把不相关问题（如"垃圾分类政策"）误判为命中，过高则会误杀真实标准查询。
MIN_SCORE = 0.15

# 默认知识库：真实常用环保标准与健康防护条款（可扩展）
DEFAULT_KNOWLEDGE = [
    {
        "id": "hj633-1",
        "category": "标准",
        "title": "HJ 633-2012 环境空气质量指数(AQI)技术规定",
        "content": """AQI 分级标准：0-50优，51-100良，101-150轻度污染，151-200中度污染，201-300重度污染，>300严重污染。
PM2.5 24小时均值二级标准：75μg/m³。PM10：150μg/m³。O3：160μg/m³。NO2：80μg/m³。""",
        "keywords": ["AQI", "标准", "分级", "HJ633", "空气质量指数"],
    },
    {
        "id": "hj633-2",
        "category": "标准",
        "title": "HJ 633-2012 首要污染物与 IAQI 计算方法",
        "content": """各污染物的空气质量分指数(IAQI)按分段线性插值计算，AQI 取各污染物 IAQI 最大值。
当 AQI 大于 50 时，IAQI 最大的污染物为首要污染物；IAQI 并列最大时并列为首要污染物。""",
        "keywords": ["IAQI", "首要污染物", "AQI", "计算", "HJ633", "分指数"],
    },
    {
        "id": "gb3095-1",
        "category": "标准",
        "title": "GB 3095-2012 环境空气质量标准 浓度限值",
        "content": """环境空气污染物基本项目浓度限值（二级/24小时均值）：
PM2.5 75μg/m³（一级35），PM10 150μg/m³（一级50），SO2 150μg/m³（一级50），
NO2 80μg/m³（一级40），CO 4mg/m³，O3 日最大8小时均值 160μg/m³（一级100）。
二类区（居住区、商业交通居民混合区等）执行二级标准。""",
        "keywords": ["GB3095", "限值", "浓度", "标准", "二级", "一级", "PM2.5", "PM10"],
    },
    {
        "id": "health-1",
        "category": "健康",
        "title": "PM2.5 对健康的影响",
        "content": """PM2.5 可深入肺泡，进入血液循环，长期暴露增加心血管疾病和呼吸系统疾病风险。
敏感人群（儿童、老人、心肺疾病患者）应减少户外活动。AQI>100时建议佩戴口罩。""",
        "keywords": ["PM2.5", "健康", "影响", "危害", "疾病", "口罩"],
    },
    {
        "id": "health-2",
        "category": "健康",
        "title": "臭氧(O3)对健康的影响",
        "content": """地面臭氧刺激呼吸道，引起咳嗽、胸闷，加重哮喘。夏季午后浓度最高。
建议敏感人群避免在午后高温时段进行户外运动。""",
        "keywords": ["臭氧", "O3", "健康", "哮喘", "呼吸道", "夏季"],
    },
    {
        "id": "health-3",
        "category": "健康",
        "title": "PM10 对健康的影响与防护",
        "content": """PM10（可吸入颗粒物）可沉积在上呼吸道，引发咳嗽、咽炎，加重慢性阻塞性肺病。
沙尘天气是 PM10 短时飙升的主要原因，外出建议佩戴防尘口罩，回家后清洗面部和鼻腔。""",
        "keywords": ["PM10", "可吸入颗粒物", "健康", "沙尘", "防护", "口罩"],
    },
    {
        "id": "health-4",
        "category": "健康",
        "title": "NO2/SO2/CO 的健康影响与防护建议",
        "content": """NO2 刺激呼吸道，长期暴露降低肺功能，交通干道附近浓度偏高；
SO2 诱发支气管痉挛，哮喘患者尤为敏感；CO 与血红蛋白结合造成缺氧，高浓度可致命。
防护建议：污染时段减少户外停留，远离交通主干道，室内注意通风与净化。""",
        "keywords": ["NO2", "SO2", "CO", "二氧化氮", "二氧化硫", "一氧化碳", "健康", "防护"],
    },
    {
        "id": "emergency-1",
        "category": "应急",
        "title": "重污染天气应急预案分级响应",
        "content": """重污染天气预警由轻到重分为黄色、橙色、红色三级：
黄色预警（预测重度污染持续2天）启动III级响应，建议性减排+健康防护提示；
橙色预警（预测重度污染持续3天）启动II级响应，工业企业限产停产、停止土石方作业；
红色预警（预测严重污染持续3天）启动I级响应，机动车单双号限行、中小学停课。""",
        "keywords": ["重污染", "应急", "预案", "预警", "分级", "响应", "限行", "停课"],
    },
    {
        "id": "forecast-1",
        "category": "预测",
        "title": "空气质量预测方法",
        "content": """本系统使用随机森林模型进行预测，特征包括：其他污染物浓度、温度、湿度、风速、
时刻、月份、目标污染物滞后1h/24h与6h滚动均值。预测精度 R² 0.92-0.98。""",
        "keywords": ["预测", "模型", "随机森林", "方法", "精度"],
    },
    {
        "id": "season-1",
        "category": "季节",
        "title": "中国主要城市季节污染特征",
        "content": """北方城市冬季PM2.5污染最重（供暖+静稳天气），夏季最轻。
南方城市夏季臭氧污染突出（高温+强光照）。春秋季受沙尘和秸秆焚烧影响。""",
        "keywords": ["季节", "冬季", "夏季", "污染", "特征", "供暖", "臭氧"],
    },
    {
        "id": "action-1",
        "category": "应对",
        "title": "空气污染应对措施",
        "content": """轻度污染：减少户外运动，关闭门窗。中度污染：佩戴N95口罩，开启空气净化器。
重度污染：避免外出，儿童老人留室内。严重污染：停止一切户外活动，学校停课。""",
        "keywords": ["应对", "措施", "口罩", "空气净化器", "停课", "建议"],
    },
]


def _load_knowledge() -> List[Dict[str, Any]]:
    """
    加载知识库：
    - 若 documents/knowledge.json 存在则以其为准；
    - 同时把默认知识库中缺失的条目（按 id 判断）合并进来并落盘，
      保证知识库升级后老部署也能自动获得新条款，且不覆盖用户手工添加的内容。
    """
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
        existing_ids = {d.get("id") for d in knowledge}
        missing = [d for d in DEFAULT_KNOWLEDGE if d["id"] not in existing_ids]
        if missing:
            knowledge.extend(missing)
            _save_knowledge(knowledge)
        return knowledge
    return list(DEFAULT_KNOWLEDGE)


def _save_knowledge(knowledge: List[Dict[str, Any]]):
    """保存知识库"""
    os.makedirs(os.path.dirname(KNOWLEDGE_PATH), exist_ok=True)
    with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)


# 初始化知识库
_knowledge_base = _load_knowledge()


def add_document(doc: Dict[str, Any]):
    """添加文档到知识库"""
    global _knowledge_base
    _knowledge_base.append(doc)
    _save_knowledge(_knowledge_base)


def _tokenize(text: str) -> List[str]:
    """简单分词：连续中文为一段、连续字母数字为一段"""
    return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())


def _is_match(query_token: str, doc_term: str) -> bool:
    """
    查询词与文档词（关键词/标题/正文片段）的匹配判定：
    - 完全相等；
    - 或一方是另一方的子串（长度≥2才允许，避免单字/单字母误匹配，
      如查询词 "PM2" 可命中正文 "PM2.5"，关键词 "标准" 可命中查询片段 "标准是多少"）。
    """
    qt, dt = query_token.lower(), doc_term.lower()
    if qt == dt:
        return True
    if len(qt) >= 2 and qt in dt:
        return True
    if len(dt) >= 2 and dt in qt:
        return True
    return False


def _score_doc(query_tokens: List[str], doc: Dict[str, Any]) -> float:
    """
    计算单条文档与查询的归一化相关度（0~1）：
      score = 0.5 * 查询词覆盖率 + 0.3 * 文档关键词命中率 + 0.2 * 标题命中
    - 查询词覆盖率：命中（标题/关键词/正文任一）的查询词占全部查询词比例
    - 文档关键词命中率：被查询词命中的文档关键词占该文档关键词比例
    - 标题命中：标题中包含任一查询词记 1，否则 0
    """
    if not query_tokens:
        return 0.0

    title = doc.get("title", "").lower()
    content = doc.get("content", "").lower()
    doc_keywords = [k.lower() for k in doc.get("keywords", [])]

    matched_tokens = 0
    for qt in query_tokens:
        hit_kw = any(_is_match(qt, dk) for dk in doc_keywords)
        hit_text = qt in title or qt in content
        if hit_kw or hit_text:
            matched_tokens += 1
    coverage = matched_tokens / len(query_tokens)

    if doc_keywords:
        kw_hit = sum(1 for dk in doc_keywords if any(_is_match(qt, dk) for qt in query_tokens))
        kw_ratio = kw_hit / len(doc_keywords)
    else:
        kw_ratio = 0.0

    title_hit = 1.0 if any(qt in title for qt in query_tokens) else 0.0

    return round(0.5 * coverage + 0.3 * kw_ratio + 0.2 * title_hit, 3)


def retrieve(query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> List[Dict[str, Any]]:
    """
    关键词匹配检索 + 相关度打分（简化版 RAG，后续可升级为向量检索）

    Args:
        query: 查询文本
        top_k: 返回最相关的 top_k 条
        min_score: 低置信阈值，最佳分数低于该值时判定为不相关，返回空列表

    Returns:
        相关知识条目列表（每条附带 score 字段，0~1）；
        低置信时返回空列表——配合 retrieve_detailed 可拿到低置信标记与最佳分数。
    """
    return retrieve_detailed(query, top_k, min_score)["results"]


def retrieve_detailed(query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> Dict[str, Any]:
    """
    检索并返回完整诊断信息，供需要"拒答"逻辑的上层使用。

    Returns:
        {
            "results": [...],        # 带分数的结果（低置信时为 []）
            "low_confidence": bool,  # 是否低置信（最佳分数 < min_score）
            "best_score": float,     # 全部候选中的最高分（无候选为 0）
        }
    """
    query_tokens = _tokenize(query)

    scored = []
    for doc in _knowledge_base:
        s = _score_doc(query_tokens, doc)
        if s > 0:
            scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score = scored[0][0] if scored else 0.0
    low_confidence = best_score < min_score

    results = []
    if not low_confidence:
        for s, doc in scored[:top_k]:
            item = dict(doc)
            item["score"] = s
            results.append(item)

    return {
        "results": results,
        "low_confidence": low_confidence,
        "best_score": best_score,
    }


def retrieve_and_format(query: str, top_k: int = 3) -> str:
    """
    检索并格式化为文本，供 LLM 使用。
    低置信时返回空字符串（上层应据此提示"未检索到相关标准条款"）。
    """
    docs = retrieve(query, top_k)
    if not docs:
        return ""

    lines = ["## 相关知识库信息"]
    for doc in docs:
        lines.append(f"### {doc.get('title', '')}（id: {doc.get('id', '')}，相关度: {doc.get('score', 0)}）")
        lines.append(doc.get("content", ""))
        lines.append("")

    return "\n".join(lines)


# 初始化时保存默认知识库
if not os.path.exists(KNOWLEDGE_PATH):
    _save_knowledge(DEFAULT_KNOWLEDGE)
