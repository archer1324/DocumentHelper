# Enterprise ReAct Documentation Assistant

注：本项目参考https://github.com/emarco177/langchain-course课程

企业级智能文档助手，基于 ReAct (Reasoning + Acting) 推理范式构建，面向企业内部知识问答、流程咨询、故障排查、培训赋能等核心业务场景。

本项目目标是解决以下常见问题：
- 知识分散在 PDF、Word、PPT、Markdown、网页等多源文档中，检索成本高。
- 传统关键词搜索命中率不稳定，语义检索容易遗漏精确条款。
- 大模型回答缺乏可追溯依据，存在幻觉风险。
- 不同部门和角色的数据隔离不足，权限治理困难。

系统通过交织推理与行动，让模型先思考再调用工具，并在回答中强制附带来源引用，从而实现可追溯、低幻觉、可运营的企业级文档问答能力。

## Core Capabilities

### 1) 文档接入与处理层
- 多格式支持：PDF / Word / Markdown / PPT / HTML / TXT。
- 智能分块：递归语义切分，默认 chunk_size=1000、overlap=200。
- 元数据抽取：文档类型、所属部门、版本、生效时间、权限标签。
- 增量监听：文件哈希比对，仅重建新增或变更内容，避免全量重训。

### 2) 知识存储与检索层
- 双索引架构：向量索引 (Chroma) + 关键词索引 (BM25)。
- 混合召回：并行执行语义检索和关键词检索，加权融合。
- 重排序优化：Top-50 候选可接入 Cross-Encoder 二次排序。
- 权限过滤：按用户 departments / roles 在检索阶段进行隔离。

### 3) 智能问答与推理层 (ReAct Core)
- 意图识别：事实查询、流程咨询、对比分析、故障排查。
- 工具路由：知识库检索、网页检索、计算器、事实校验。
- ReAct 循环：Thought -> Action -> Observation -> Iteration -> Final Answer。
- 可信回答：强制来源标注 [文档名#章节]，低置信度时明确提示信息不足。

### 4) 交互与运营层
- 聊天界面：多轮上下文、追问、流式交互体验。
- 来源展示：答案下方引用折叠展示，便于审计与追溯。
- 反馈闭环：有用/无用/报错反馈沉淀为优化数据。
- 运营能力：文档管理、热词分析、A/B 参数配置、效果观测。

## Project Layout

```
documentation-helper-main/
  backend/
    enterprise/
      __init__.py
      config.py              # 企业版配置
      types.py               # 领域类型定义
      ingestion.py           # 增量接入与切分
      bm25_index.py          # BM25 索引实现
      retrieval.py           # 混合检索与重排
      react_core.py          # ReAct 推理核心
      ops.py                 # 反馈、热词、A/B 配置
  enterprise_ingest.py       # 入库脚本入口
  main_enterprise.py         # 企业聊天 UI (Streamlit)
  enterprise_admin_api.py    # 运营后台 API (FastAPI)
```

## Quick Start

### 1. 环境准备
- Python 3.11+
- 本地可用的 Ollama 服务 (用于 embedding 和对话模型)
- 可选：Tavily API Key (用于网页检索工具)

### 2. 安装依赖

```bash
pipenv install
```

### 3. 配置环境变量

在项目根目录创建 .env：

```env
# 可选：网页检索工具
TAVILY_API_KEY=your_tavily_api_key

# 可选：Cross-Encoder 模型
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Pinecone 向量库配置
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=enterprise-docs-2026
PINECONE_NAMESPACE=enterprise
```

### 4. 放入企业文档

将文档放入目录：

```text
enterprise_data/docs
```

PDF 可直接放在该目录或其子目录下，例如：

```text
enterprise_data/docs/人事制度.pdf
enterprise_data/docs/legal/合同管理制度.pdf
```

说明：
- 支持扩展名为 .pdf（以及 .doc/.docx/.ppt/.pptx/.md/.html/.txt）。
- 扫描版 PDF（图片型）可能抽取文本较少，建议优先上传可复制文本的 PDF。

可为每个文档附加 sidecar 元数据文件（可选）：

```text
合同管理制度.docx
合同管理制度.docx.meta.json
```

示例：

```json
{
  "department": "legal",
  "version": "2.3",
  "effective_time": "2026-01-15",
  "departments": ["legal", "finance"],
  "roles": ["manager", "employee"],
  "permission_tags": ["internal", "policy"]
}
```

### 5. 执行增量入库

```bash
python enterprise_ingest.py
```

入库后会自动读取 enterprise_data/docs 下新增或变更的 PDF 并写入 Pinecone。可重复执行该命令进行增量更新。

### 6. 启动企业问答 UI

```bash
streamlit run main_enterprise.py
```

### 7. 启动运营后台 API (可选)

```bash
uvicorn enterprise_admin_api:app --reload --port 8010
```

## Admin API

- GET /health
- GET /documents
- POST /ingest
- GET /hotwords?top_n=20
- GET /ab-test
- POST /ab-test

## Retrieval and Reasoning Flow

1. 用户提问 + 用户身份上下文 (user_id / departments / roles)
2. 意图识别 (fact/process/comparison/troubleshooting)
3. ReAct 规划动作并调用工具
4. 混合召回 (Vector + BM25) + 融合 + 重排
5. 权限过滤后生成可追溯回答
6. 输出引用与置信度，并记录反馈和运营指标

## Security and Governance

- 权限控制在检索层执行，防止跨部门文档泄漏。
- 答案默认携带引用，支持审计与复核。
- 低置信度显式告警，降低幻觉误导风险。
- 反馈数据用于持续优化检索与提示策略。

## Typical Enterprise Scenarios

- 新员工培训：快速查询制度、流程、规范与术语。
- 技术支持：基于历史文档定位故障排查步骤。
- 管理决策：对制度版本、部门规则进行对比分析。
- 合规审计：追溯结论来源，核验生效时间和版本。

## Current Limitations

- Cross-Encoder 依赖 sentence-transformers 环境，未安装时自动降级。
- Office/PDF 解析质量取决于源文档结构与 unstructured 依赖情况。
- 权限控制当前基于传入上下文，生产环境建议对接企业 SSO/JWT。

## Roadmap

- 引用一键跳转原文定位 (段落级高亮)。
- 更细粒度权限模型 (文档级 / 章节级 / 字段级)。
- 在线评测集与自动回归评估。
- 多模型策略路由与成本优化。

## License

MIT
