
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
CARS_PATH = APP_DIR / "cars.json"
FAQ_PATH = APP_DIR / "faq.json"

st.set_page_config(
    page_title="AutoChoice Agent｜智能选车助手",
    page_icon="🚗",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem;}
.hero {
    padding: 28px 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15,23,42,.98), rgba(30,64,175,.90));
    color: white;
    margin-bottom: 22px;
}
.hero h1 {margin: 0 0 8px 0; font-size: 2.05rem;}
.hero p {margin: 0; opacity: .9; font-size: 1.02rem;}
.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,.15);
    margin-right: 6px;
    margin-top: 12px;
    font-size: .83rem;
}
.card {
    border: 1px solid rgba(148,163,184,.25);
    border-radius: 16px;
    padding: 16px 18px;
    background: rgba(248,250,252,.55);
    margin-bottom: 12px;
}
.small {font-size: .87rem; opacity:.76;}
.metric-label {font-size:.80rem; color:#64748b; margin-bottom:2px;}
.trace-step {
    border-left: 4px solid #2563eb;
    padding: 10px 14px;
    margin: 8px 0;
    background: rgba(37,99,235,.055);
    border-radius: 0 12px 12px 0;
}
.tradeoff {color:#b45309;}
.ok {color:#047857;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with open(CARS_PATH, "r", encoding="utf-8") as f:
    CARS = json.load(f)
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ = json.load(f)

ENERGY_LABEL = {
    "bev": "纯电",
    "phev": "插混",
    "hev": "油混",
    "ice": "燃油",
}

BODY_LABEL = {
    "sedan": "轿车",
    "suv": "SUV",
}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def extract_constraints(text: str) -> dict:
    t = normalize_text(text)
    result = {
        "budget_max": 20.0,
        "city_commute": False,
        "long_trip": False,
        "fixed_parking": None,
        "family": False,
        "prefer_suv": False,
        "prefer_sedan": False,
        "energy_pref": [],
        "low_cost": False,
        "tech_focus": False,
        "space_focus": False,
    }

    # Budget extraction: 20万 / 20w / 15-20万 / 20以内
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)\s*(?:万|w)", t)
    if range_match:
        result["budget_max"] = float(range_match.group(2))
    else:
        budget_match = re.search(r"(?:预算|价格|落地|裸车)?\s*(\d+(?:\.\d+)?)\s*(?:万|w)", t)
        if budget_match:
            result["budget_max"] = float(budget_match.group(1))

    city_words = ["通勤", "市区", "城区", "上班", "城市"]
    long_words = ["长途", "高速", "跨城", "自驾", "回老家", "远途"]
    no_parking_words = ["没固定车位", "没有固定车位", "无固定车位", "不能装桩", "没法装桩", "没有家充"]
    parking_words = ["固定车位", "家充", "充电桩"]
    family_words = ["家用", "家庭", "孩子", "父母", "一家", "后排"]
    suv_words = ["suv", "suv车型", "通过性"]
    sedan_words = ["轿车", "三厢"]
    low_cost_words = ["省钱", "省心", "低成本", "用车成本", "经济", "便宜"]
    tech_words = ["智能", "智驾", "科技", "车机", "辅助驾驶"]
    space_words = ["空间", "后备箱", "大五座", "宽敞"]

    result["city_commute"] = any(w in t for w in city_words)
    result["long_trip"] = any(w in t for w in long_words)
    if any(w in t for w in no_parking_words):
        result["fixed_parking"] = False
    elif any(w in t for w in parking_words):
        result["fixed_parking"] = True

    result["family"] = any(w in t for w in family_words)
    result["prefer_suv"] = any(w in t for w in suv_words)
    result["prefer_sedan"] = any(w in t for w in sedan_words)
    result["low_cost"] = any(w in t for w in low_cost_words)
    result["tech_focus"] = any(w in t for w in tech_words)
    result["space_focus"] = any(w in t for w in space_words)

    if any(w in t for w in ["纯电", "电车", "bev"]):
        result["energy_pref"].append("bev")
    if any(w in t for w in ["插混", "混动", "phev", "增程"]):
        result["energy_pref"].append("phev")
    if any(w in t for w in ["油混", "hev"]):
        result["energy_pref"].append("hev")
    if any(w in t for w in ["燃油", "油车"]):
        result["energy_pref"].append("ice")

    return result

def score_car(car: dict, c: dict):
    score = 50.0
    reasons, tradeoffs = [], []
    hard_pass = True

    # Budget
    if car["price_demo_wan"] <= c["budget_max"]:
        diff = c["budget_max"] - car["price_demo_wan"]
        score += 18 if diff <= 4 else 14
        reasons.append(f"示例价格约 {car['price_demo_wan']:.1f} 万，在预算内")
    else:
        hard_pass = False
        over = car["price_demo_wan"] - c["budget_max"]
        score -= 35 + over * 2
        tradeoffs.append(f"示例价格高于预算约 {over:.1f} 万")

    energy = car["energy"]

    # Parking / charging
    if c["fixed_parking"] is False:
        if energy in ["phev", "hev"]:
            score += 18
            reasons.append("无固定车位时，对补能基础设施依赖较低")
        elif energy == "bev":
            score -= 14
            tradeoffs.append("无固定车位时，纯电日常补能便利性需要单独确认")
        else:
            score += 6
    elif c["fixed_parking"] is True and energy == "bev":
        score += 14
        reasons.append("有家充条件时，纯电通勤成本优势更容易兑现")

    # Use case
    if c["city_commute"]:
        if energy in ["bev", "phev", "hev"]:
            score += 8
            reasons.append("城市通勤场景下能源效率与平顺性更占优")
    if c["long_trip"]:
        if energy in ["phev", "hev", "ice"]:
            score += 12
            reasons.append("长途场景的补能灵活性更好")
        elif energy == "bev":
            score -= 5
            tradeoffs.append("长途体验更依赖高速补能网络和路线规划")

    if c["family"]:
        score += car["family_score"] * 1.6
        if car["family_score"] >= 8:
            reasons.append("家庭乘坐与日常实用性较强")

    if c["prefer_suv"]:
        score += 14 if car["body"] == "suv" else -8
    if c["prefer_sedan"]:
        score += 12 if car["body"] == "sedan" else -6

    if c["space_focus"]:
        score += car["space_score"] * 1.6
        if car["space_score"] >= 8:
            reasons.append("空间评分较高")

    if c["tech_focus"]:
        score += car["tech_score"] * 1.8
        if car["tech_score"] >= 8:
            reasons.append("智能化/车机体验在示例库中评分较高")

    if c["low_cost"]:
        score += car["cost_score"] * 1.7
        if car["cost_score"] >= 8:
            reasons.append("日常用车成本评分较优")

    if c["energy_pref"]:
        if energy in c["energy_pref"]:
            score += 18
            reasons.append(f"符合能源类型偏好：{ENERGY_LABEL[energy]}")
        else:
            score -= 10

    return round(score, 2), hard_pass, reasons[:5], tradeoffs[:3]

def search_catalog(constraints):
    scored = []
    for car in CARS:
        score, hard_pass, reasons, tradeoffs = score_car(car, constraints)
        scored.append({
            **car,
            "score": score,
            "hard_pass": hard_pass,
            "reasons": reasons,
            "tradeoffs": tradeoffs,
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)

def estimate_monthly_cost(car: dict, monthly_km: int = 1200):
    # Illustrative demo assumptions, not real purchase advice
    assumptions = {
        "bev": 0.16,
        "phev": 0.33,
        "hev": 0.42,
        "ice": 0.58,
    }
    variable = assumptions[car["energy"]] * monthly_km
    insurance_maint = {
        "bev": 520,
        "phev": 590,
        "hev": 620,
        "ice": 680,
    }[car["energy"]]
    return {
        "monthly_km": monthly_km,
        "energy_cost_demo": round(variable),
        "insurance_maintenance_demo": insurance_maint,
        "monthly_total_demo": round(variable + insurance_maint),
    }

def retrieve_faq(query: str, top_k: int = 3):
    docs = [f"{x['title']} {x['content']} {' '.join(x['tags'])}" for x in FAQ]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X = vectorizer.fit_transform(docs + [query])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    order = sims.argsort()[::-1][:top_k]
    result = []
    for i in order:
        result.append({**FAQ[i], "similarity": round(float(sims[i]), 3)})
    return result

def constraint_summary(c):
    items = [f"预算 ≤ {c['budget_max']:.1f} 万"]
    if c["city_commute"]: items.append("城市通勤")
    if c["long_trip"]: items.append("偶尔/经常长途")
    if c["fixed_parking"] is False: items.append("无固定车位/无家充")
    if c["fixed_parking"] is True: items.append("有固定车位/家充")
    if c["family"]: items.append("家庭使用")
    if c["prefer_suv"]: items.append("偏好SUV")
    if c["prefer_sedan"]: items.append("偏好轿车")
    if c["low_cost"]: items.append("关注用车成本")
    if c["tech_focus"]: items.append("关注智能化")
    if c["space_focus"]: items.append("关注空间")
    if c["energy_pref"]:
        items.append("能源偏好：" + "/".join(ENERGY_LABEL[x] for x in c["energy_pref"]))
    return items

def make_result_payload(query, constraints, top, rag_docs, trace):
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "constraints": constraints,
        "recommendations": [
            {
                "rank": i + 1,
                "model": x["model"],
                "score": x["score"],
                "price_demo_wan": x["price_demo_wan"],
                "energy": ENERGY_LABEL[x["energy"]],
                "reasons": x["reasons"],
                "tradeoffs": x["tradeoffs"],
                "monthly_cost_demo": estimate_monthly_cost(x),
            }
            for i, x in enumerate(top)
        ],
        "rag_sources": [
            {"id": x["id"], "title": x["title"], "similarity": x["similarity"]}
            for x in rag_docs
        ],
        "agent_trace": trace,
        "disclaimer": "车型和价格为面试 Demo 样例数据，仅用于展示 Agent 产品逻辑，不构成真实购车建议。",
    }

st.markdown(
    """
    <div class="hero">
        <h1>🚗 AutoChoice Agent｜智能选车助手</h1>
        <p>一个用于展示 Agent 产品逻辑的小型原型：约束抽取 → 车型检索 → 工具调用 → RAG → 推荐解释 → 结果评估</p>
        <span class="badge">Agent Workflow</span>
        <span class="badge">Tool Calling</span>
        <span class="badge">RAG</span>
        <span class="badge">Evaluation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("产品说明")
    st.write(
        "这个 Demo 不追求“车型数据库有多全”，而是展示："
        "如何把自然语言需求拆成结构化约束，并由 Agent 调用检索、成本计算和知识库工具。"
    )
    st.caption("车型、价格及成本参数均为样例快照，仅用于产品原型演示。")
    st.divider()
    st.subheader("建议测试")
    examples = [
        "预算20万，上海通勤，周末偶尔长途，家里没固定车位，希望省心。",
        "预算18万，主要城市通勤，有固定车位和家充，想要智能一点的纯电轿车。",
        "预算22万，家庭用车，经常带父母孩子，希望空间大，偶尔长途。",
    ]
    for idx, ex in enumerate(examples, start=1):
        if st.button(f"示例 {idx}", use_container_width=True):
            st.session_state["query_text"] = ex

default_query = st.session_state.get(
    "query_text",
    "预算20万，上海通勤，周末偶尔长途，家里没固定车位，希望省心。"
)

query = st.text_area(
    "告诉 Agent 你的选车需求",
    value=default_query,
    height=110,
    placeholder="例如：预算20万，主要城市通勤，偶尔长途，没有固定车位，希望省心……",
)

col_run, col_km = st.columns([1, 1])
with col_run:
    run = st.button("开始 Agent 分析", type="primary", use_container_width=True)
with col_km:
    monthly_km = st.slider("每月里程（成本工具参数）", 500, 3000, 1200, 100)

if run:
    constraints = extract_constraints(query)
    trace = []

    trace.append({
        "step": 1,
        "tool": "ConstraintExtractor",
        "action": "从自然语言中抽取预算、用车场景、补能条件与偏好",
        "status": "success",
    })

    catalog = search_catalog(constraints)
    trace.append({
        "step": 2,
        "tool": "VehicleCatalogSearch",
        "action": f"在 {len(CARS)} 条样例车型中进行硬约束过滤与加权排序",
        "status": "success",
    })

    top = catalog[:3]
    for x in top:
        x["cost"] = estimate_monthly_cost(x, monthly_km)
    trace.append({
        "step": 3,
        "tool": "OwnershipCostCalculator",
        "action": f"按每月 {monthly_km} km 估算三款候选车的示例月均用车成本",
        "status": "success",
    })

    rag_docs = retrieve_faq(query, top_k=3)
    trace.append({
        "step": 4,
        "tool": "FAQRetriever",
        "action": "从选车知识库检索与充电、长途、家庭场景相关的证据",
        "status": "success",
    })

    trace.append({
        "step": 5,
        "tool": "RecommendationComposer",
        "action": "融合约束、车型排序、成本工具和 RAG 证据，生成可解释推荐",
        "status": "success",
    })

    st.subheader("1. Agent 理解了什么")
    tags = constraint_summary(constraints)
    st.markdown(" ".join([f"`{x}`" for x in tags]))

    st.subheader("2. 推荐结果")
    cols = st.columns(3)
    for i, (col, car) in enumerate(zip(cols, top), start=1):
        with col:
            st.markdown(f"### #{i} {car['model']}")
            st.metric("Agent Score", f"{car['score']:.1f}")
            st.write(
                f"**{ENERGY_LABEL[car['energy']]} · {BODY_LABEL[car['body']]} · "
                f"样例价 {car['price_demo_wan']:.1f} 万**"
            )
            for r in car["reasons"][:3]:
                st.markdown(f"<div class='ok'>✓ {r}</div>", unsafe_allow_html=True)
            for t in car["tradeoffs"][:2]:
                st.markdown(f"<div class='tradeoff'>△ {t}</div>", unsafe_allow_html=True)
            st.caption(
                f"示例月均成本：约 ¥{car['cost']['monthly_total_demo']} "
                f"（按 {monthly_km} km/月）"
            )

    st.subheader("3. 对比表")
    table = pd.DataFrame([
        {
            "车型": x["model"],
            "能源": ENERGY_LABEL[x["energy"]],
            "车身": BODY_LABEL[x["body"]],
            "样例价格(万)": x["price_demo_wan"],
            "Agent Score": x["score"],
            "空间评分": x["space_score"],
            "智能化评分": x["tech_score"],
            "成本评分": x["cost_score"],
            "示例月均成本(元)": x["cost"]["monthly_total_demo"],
        }
        for x in top
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("4. RAG 证据")
    for doc in rag_docs:
        with st.expander(f"[{doc['id']}] {doc['title']} · 检索相似度 {doc['similarity']:.3f}"):
            st.write(doc["content"])
            st.caption("标签：" + " / ".join(doc["tags"]))

    st.subheader("5. Agent Tool Trace")
    for step in trace:
        st.markdown(
            f"""
            <div class="trace-step">
              <b>Step {step['step']} · {step['tool']}</b><br/>
              {step['action']}<br/>
              <span class="small">Status: {step['status']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("6. 产品评估视角")
    hard_pass_rate = sum(1 for x in top if x["hard_pass"]) / max(len(top), 1)
    tool_success_rate = sum(1 for x in trace if x["status"] == "success") / len(trace)
    rag_coverage = 1.0 if rag_docs else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("预算硬约束通过率", f"{hard_pass_rate*100:.0f}%")
    m2.metric("Tool Success Rate", f"{tool_success_rate*100:.0f}%")
    m3.metric("RAG Source Coverage", f"{rag_coverage*100:.0f}%")
    m4.metric("推荐候选数", len(top))

    st.info(
        "面试可继续讨论的线上指标：约束满足率、事实准确率、工具调用成功率、"
        "用户二次修改率、推荐点击率/收藏率，以及 Bad Case 分类。"
    )

    payload = make_result_payload(query, constraints, top, rag_docs, trace)
    st.download_button(
        "下载本次 Agent 结果 JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="autochoice_agent_result.json",
        mime="application/json",
    )

st.divider()
st.caption(
    "AutoChoice Agent · Product Prototype · "
    "Demo data only. 重点展示 Agent orchestration / Tool Calling / RAG / Evaluation，而非真实车型数据库完整性。"
)
