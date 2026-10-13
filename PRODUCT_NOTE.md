# AutoChoice Agent｜1页产品说明

## 1. 用户问题
传统选车页要求用户自己理解大量车型参数，但真实用户往往只会说：
> “预算20万、城市通勤、周末长途、没固定车位，希望省心。”

产品问题不是“如何列更多车型”，而是：
**如何把模糊自然语言需求转换成可执行的产品约束，并给出可解释推荐。**

## 2. Agent Workflow
User Query  
→ Constraint Extractor  
→ Vehicle Catalog Search  
→ Ownership Cost Tool  
→ FAQ RAG  
→ Recommendation Composer  
→ User Feedback / Refine

## 3. 为什么要 Tool Calling
车型价格、能源类型、成本参数属于结构化事实。
不应该让大模型“凭印象生成”。

因此：
- 车型事实 → Database / API
- 用车成本 → Calculator Tool
- 场景知识 → RAG
- 用户需求理解 / 解释 → Agent

## 4. MVP 指标
- 约束满足率
- 车型事实准确率
- Tool Call Success Rate
- RAG Source Coverage
- 用户二次修改率
- 推荐结果点击/收藏率
- Bad Case 分类与回流率

## 5. MVP 之后怎么做
1. 接入实时车型参数、价格和经销商 API
2. 增加多轮澄清：预算、充电条件、空间、智能化权重
3. 引入真实 LLM 做 Tool Selection / Reasoning
4. 加入用户反馈闭环
5. 建立评测集：约束冲突、信息缺失、极端需求、数据过期
