<h1 align="center">Agentic RAG for Docmind</h1>

<p align="center">
  <img alt="Agentic RAG for Docmind Logo" src="assets/logo-1.png" width="400px">
</p>

<p align="center">
  基于 <strong>LangGraph</strong> 构建的智能体检索增强生成（Agentic RAG）系统，支持任务拆解、
跨文档检索与答案聚合。
</p>

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [依赖管理](#依赖管理)
- [快速启动](#快速启动)
- [配置指南](#配置指南)
- [常见问题](#常见问题)

---

## 项目概述

核心特性包括：

- **父子分块策略（Parent-Child Chunking）**：将文档拆分为小粒度子块（用于精确检索）并关联到大粒度父块（用于提供丰富上下文）
- **混合检索（Hybrid Search）+ Rerank**：结合稠密向量嵌入与稀疏 BM25 检索并经过Cross-Encoder重排，实现最优召回效果
- **LangGraph 智能体**：编排查询重写、检索、上下文压缩与答案聚合的完整工作流
- **多 LLM 提供商支持**：无缝切换 Ollama（本地）、OpenAI 等
- **向量存储**：基于 Qdrant 实现高效相似度检索
- **Web 交互界面**：基于 Gradio 提供文档上传与对话式问答体验

### 数据流

```
PDF → Markdown 转换 → 父子分块 → 向量索引 → 智能体检索 → LLM 生成回答
```

---

## 系统架构

系统采用 **主图 + 子图** 的双层 LangGraph 架构：

### Main Graph
协调完整的问答流程：

| 节点 | 功能 |
|------|------|
| `rewrite_query` | 查询重写 — 将用户输入改写为适合 AI 处理的格式 |
| `request_clarification` | 请求澄清 — 当问题不明确时向用户请求补充信息 |
| `agent`（子图） | 问答子图 — 对每个重写后的子问题并行执行智能体流程 |
| `aggregate_answers` | 答案聚合 — 将多个子问题的答案合并为最终回答 |
| `summarize_history` | 历史压缩 — 将对话历史压缩为摘要并保留最近 N 条消息 |

### Subgraph
处理单个查询的完整生命周期：

| 节点 | 功能 |
|------|------|
| `orchestrator` | 核心调度器 — 决定调用工具、降级响应或生成答案 |
| `tools` | 工具节点 — 执行子块检索与父块召回 |
| `compress_context` | 上下文压缩器 — 基于 token 阈值动态压缩上下文 |
| `fallback_response` | 降级响应器 — 达到最大迭代后基于已有信息生成最佳答案 |
| `collect_answer` | 答案聚合器 — 收集并输出最终回答 |

---

## 项目结构

```
├── project/                        # 核心应用代码
│   ├── app.py                      # 应用入口，启动 Gradio UI
│   ├── config.py                   # 中央配置
│   ├── document_chunker.py         # 父子分块逻辑
│   ├── utils.py                    # PDF 转 Markdown 及工具函数
│   ├── .env.example                # 环境变量模板
│   │
│   ├── core/                       # 核心系统
│   │   ├── rag_system.py           # 系统引导
│   │   ├── document_manager.py     # 文档摄入管线
│   │   ├── chat_interface.py       # 智能体图交互封装
│   │   └── observability.py        # Langfuse 可观测性集成（测试中...）
│   │
│   ├── db/                         # 数据库层
│   │   ├── vector_db_manager.py    # Qdrant 向量数据库客户端封装
│   │   └── parent_store_manager.py # 父块文件存储管理
│   │
│   ├── rag_agent/                  # RAG 智能体
│   │   ├── graph.py                # 图构建与编译
│   │   ├── graph_state.py          # 图状态定义
│   │   ├── nodes.py                # 节点实现
│   │   ├── edges.py                # 条件路由逻辑
│   │   ├── tools.py                # 检索工具（子块搜索、父块召回）
│   │   ├── prompts.py              # 系统提示词
│   │   └── schemas.py              # 结构化输出 Schema
│   │
│   └── ui/                         # 用户界面
│       ├── gradio_app.py           # Gradio UI 实现
│       └── css.py                  # 自定义样式
│
├── parent_store/                   # 父块持久化存储（JSON）
├── qdrant_db/                      # Qdrant 本地向量数据库
└── requirements.txt                # Python 依赖
```

---

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.11+ |
| Ollama | 本地部署时需要（或使用云端 API Key） |
| Docker | Docker 部署时需要 |

---

## 依赖管理

### 核心依赖

项目依赖通过 `requirements.txt` 统一管理，主要包含：

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `langgraph` | 1.1.9 | Agentic 工作流编排 |
| `langchain-openai` | 1.2.1 | OpenAI / Ollama LLM 集成 |
| `langchain-ollama` | 1.1.0 | Ollama 本地模型集成 |
| `langchain-qdrant` | 1.1.0 | Qdrant 向量数据库集成 |
| `langchain-huggingface` | 1.2.2 | HuggingFace 嵌入模型 |
| `langchain-text-splitters` | 1.1.2 | 文档分块工具 |
| `fastembed` | 0.8.0 | 稀疏向量嵌入（BM25） |
| `sentence-transformers` | 5.4.1 | 稠密向量嵌入模型 |
| `gradio` | 6.13.0 | Web 交互界面 |
| `pymupdf4llm` | 1.27.2.3 | PDF 文档解析 |
| `langfuse` | 4.5.1 | 可观测性与追踪 |
| `tiktoken` | 0.12.0 | Token 计数 |
| `python-dotenv` | 1.2.2 | 环境变量管理 |

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/JYf12/Docmind-apr.git
cd agentic-rag-docmind
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板并根据需要修改：

```bash
cp project/.env.example project/.env
```

编辑 `project/.env` 文件，配置 LLM API 密钥等参数（可选，Langfuse 追踪默认关闭）。

### 4. 配置 LLM 提供商

编辑 `project/config.py`，修改 `ACTIVE_LLM_CONFIG` 切换 LLM 提供商：

```python
# 可选值: "ollama", "openai"
ACTIVE_LLM_CONFIG = "ollama"
```

如需使用云端 API（如 OpenAI），需在 `config.py` 中配置对应的 API Key 和模型名称。

### 5. 启动应用

```bash
python project/app.py
```

启动后访问 `http://localhost:7860` 即可使用 Gradio 界面进行文档上传与对话式问答。

---

## 配置指南

所有核心配置集中在 `project/config.py`。

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| "Model not found" 错误 | 模型名称与提供商不匹配 | 检查 `LLM_MODEL` 是否与提供商 API 一致 |
| 检索质量低 | 嵌入模型不佳或分块配置不当 | 更换嵌入模型或调整分块参数后重新索引 |
| 响应速度慢 | 嵌入模型过大 | 使用更轻量的嵌入模型（如 `all-MiniLM-L6-v2`） |
| API 速率限制 | 外部提供商请求过多 | 添加重试逻辑或切换至本地 Ollama 模型 |
| 内存不足 | 文档集过大或嵌入模型过大 | 使用更小的嵌入模型或启用 GPU 加速 |
| 检索结果为空 | 未索引文档或集合名称错误 | 确认文档已上传且 `CHILD_COLLECTION` 名称匹配 |
| 切换提供商后导入错误 | 缺少对应 SDK | 安装对应包：`pip install langchain-{provider}` |
| 多次运行答案不一致 | Temperature 设置过高 | 将 `LLM_TEMPERATURE` 设为 `0` |

---

> 💡 **Tip** 🚀 更多需求与优化正在持续跟进中，敬请期待！✨🔧
