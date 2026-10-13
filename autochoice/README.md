# AutoChoice Agent｜智能选车助手

一个用于 AI Product Manager 面试/作品集的小型 Agent 产品 Demo。

## 这个 Demo 展示什么

不是为了证明“我做了一个比汽车平台更全的选车库”，而是展示一个技术型 AI PM 如何把用户自然语言需求拆解成：

1. **Constraint Extraction**：预算、场景、补能条件、偏好  
2. **Vehicle Catalog Search**：结构化车型库检索与排序  
3. **Tool Calling**：用车成本计算器  
4. **RAG**：从选车 FAQ 知识库检索证据  
5. **Recommendation Composer**：可解释推荐  
6. **Evaluation**：约束满足率、工具成功率、RAG 覆盖率、Bad Case 思路  

## 为什么适合 AI Agent 产品岗位

Agent 产品的关键不只是“聊天”，而是：
- 什么时候应该让模型理解意图；
- 什么时候应该调用结构化工具；
- 哪些事实必须从数据库 / API / RAG 返回；
- 如何保留来源和不确定性；
- 如何定义线上评估指标与 Bad Case。

这个 Demo 把这些产品决策显式展示出来。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

把以下文件放到 GitHub 仓库根目录：
- `app.py`
- `cars.json`
- `faq.json`
- `requirements.txt`

然后在 Streamlit Community Cloud 选择仓库并部署即可。

## 推荐测试语句

### 场景 1：没有固定车位
> 预算20万，上海通勤，周末偶尔长途，家里没固定车位，希望省心。

### 场景 2：纯电 + 家充
> 预算18万，主要城市通勤，有固定车位和家充，想要智能一点的纯电轿车。

### 场景 3：家庭空间
> 预算22万，家庭用车，经常带父母孩子，希望空间大，偶尔长途。

## 面试时怎么讲

不要说：
> “我做了一个完整的智能选车系统。”

建议说：
> “我用一个小型 MVP 验证了 Agent 产品的核心编排逻辑：自然语言约束抽取、结构化检索、工具调用、RAG 和结果评估。为了避免把 LLM 当黑盒，我把价格、成本和车型参数放到工具层，把模型角色限定在需求理解与结果解释；如果继续产品化，我会接真实车型 API，并把约束满足率、事实准确率、工具成功率和用户二次修改率作为核心指标。”

## 数据声明

`cars.json` 中的车型与价格为**面试 Demo 样例快照**，用于展示 Agent 产品流程，不构成真实购车建议，也不代表实时市场价格。
