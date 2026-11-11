# 私人 AI 助理

依据《私人助理项目开题文档》搭建：Gradio + FastAPI + 单 ReAct Agent + FAISS/BM25 混合检索 + SQLite。模型与 Embedding 通过官方 OpenAI Python SDK 调用远程兼容 API。

## 当前状态

RAG 已实现 PDF 分页解析、按需本地 OCR、统一清洗、切块、Embedding、FAISS + BM25+ 双路召回、RRF、可选远程 Rerank，以及带来源的终端问答。正文和向量保存在 FAISS + JSON，SQLite 管理知识库、文件和构建记录。Agent 已接入知识库检索、会话历史、心知天气、Tavily 网页搜索与网页抓取，以及项目工作区文本文件读写；FastAPI 提供 SSE 流式聊天，Gradio 页面逐段显示回答。文件工具仅访问 `data/workspace/` 下的 txt、md、json、csv 文件。

阅读顺序：[Agent 架构设计](docs/AGENT_ARCHITECTURE.md) → [Agent 接入说明](docs/AGENT_INTEGRATION.md) → [项目架构与数据流](docs/ARCHITECTURE.md) → [RAG 使用指南](docs/RAG_QUICKSTART.md)。数据库操作见 [SQLite 元数据指南](docs/SQLITE_METADATA.md)。

检索评测包括 `evaluations/rag_retrieval.jsonl` 中的 12 条手工样本，以及从现有三本资料分层抽样生成的 `evaluations/rag_retrieval_300.jsonl` 候选集。运行 `python -m scripts.evaluate_retrieval --dataset evaluations/rag_retrieval_300.jsonl --mode bm25` 建立无模型费用的基线；`--query-stopwords` 可运行实验性停用词过滤对照，`--mode all` 会额外调用 Embedding API 比较向量和混合检索。指标与逐题命中明细按数据集分别保存在 `evaluations/retrieval-*.json`。自动生成题仍需人工抽查来源支持关系；检索召回评测不等价于完整答案正确性评测。

第一阶段数据审核：运行 `python -m scripts.audit_retrieval_dataset --total 100` 生成三本资料均衡抽样的 100 条人工审核清单 `evaluations/rag_retrieval_review_100.jsonl`。审核 `query` 是否能由 `anchor_text` 回答、`chunk_id` 是否为正确证据，并将 `review_status` 改为 `approved`、`rejected` 或 `needs_relabel`；不要直接修改原始候选集。

自动审核完成后，运行 `python -m scripts.export_approved_eval` 将 `auto_verdict=approved` 的记录导出为 `evaluations/rag_retrieval_approved_76.jsonl`；q204 因答案跨多个相邻切片暂不纳入，需补充多个 `chunk_id` 后再加入。

## 目录结构

~~~text
PythonProject1/
├── app/
│   ├── config.py                 # 项目配置与路径
│   ├── main.py                   # FastAPI + 同源 Gradio 入口
│   ├── llm/                      # Agent、RAG 共用的模型访问层
│   │   ├── client.py             # 官方 OpenAI SDK 适配器
│   │   └── responses.py          # ModelReply、ToolCall
│   ├── rag/                      # 可供 Agent 调用的 RAG 子系统
│   │   ├── __init__.py           # 导出 RagService、SearchRequest、SearchResponse
│   │   ├── service.py            # 对外检索服务，统一组装召回与精排
│   │   ├── __main__.py            # python -m app.rag
│   │   ├── cli.py                # 预览、入库、检索、问答命令
│   │   ├── domain/               # 基础数据类型与对外契约
│   │   │   ├── models.py         # Document、Chunk、SearchResult
│   │   │   └── contracts.py      # SearchRequest、SearchResponse
│   │   ├── ingestion/            # PDF、OCR、清洗、切块
│   │   │   ├── loaders.py
│   │   │   ├── pdf_ocr.py
│   │   │   ├── cleaners.py
│   │   │   ├── chunkers.py
│   │   │   └── pipeline.py
│   │   ├── indexing/             # 向量化、FAISS/JSON 与入库编排
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   └── pipeline.py
│   │   ├── retrieval/            # 双路召回、RRF 和精排
│   │   │   ├── retrievers.py
│   │   │   ├── bm25.py
│   │   │   ├── fusion.py
│   │   │   └── reranker.py
│   │   ├── services/             # 终端独立问答辅助
│   │   │   └── chat_service.py
│   │   └── db/                   # SQLite 业务管理
│   │       ├── models.py
│   │       ├── session.py
│   │       ├── repository.py
│   │       └── __main__.py        # python -m app.rag.db
│   ├── api/                      # Agent 聊天、知识库及记忆 HTTP API
│   ├── agent/                    # NanoClaw 风格 Agent 编排
│   │   ├── __main__.py           # python -m app.agent
│   │   ├── service.py            # AgentService 应用入口
│   │   ├── loop.py               # 模型/工具闭环与引用
│   │   ├── context.py            # 上下文与记忆组装
│   │   ├── compression.py        # 四层压缩与预算
│   │   ├── memory.py             # 长期记忆读写
│   │   ├── prompts.py            # 中文助理提示词
│   │   └── session.py            # SQLite 会话与版本检查
│   └── tools/                    # Agent 工具适配层
│       ├── base.py               # Tool 基类
│       ├── registry.py           # 工具注册与参数路由
│       ├── knowledge.py          # 知识工具：返回片段与来源
│       ├── history.py            # 本会话原始历史分页回查
│       ├── files.py              # 项目工作区内受限文本读写
│       ├── memory.py             # Agent 长期记忆管理
│       ├── time.py               # 当前时间与 IANA 时区
│       ├── web.py                # Tavily 网页搜索
│       ├── web_fetch.py          # 公开网页抓取与正文提取
│       ├── weather.py            # 心知天气
│       └── code.py
├── scripts/
│   ├── index_document.py         # 单份 PDF 全流程入库验收
│   ├── inspect_documents.py      # 本地文档预览
│   └── validate_rag.py           # 真实检索与问答验收
├── tests/                        # 离线自动测试
├── docs/                         # 架构、使用和数据库说明
├── frontend/                     # Gradio 聊天页面
├── identity.md                   # 每轮注入 system prompt 的身份与语气设定
├── data/
│   ├── uploads/                  # 原始 PDF
│   ├── indexes/                  # CURRENT 与版本化 FAISS/JSON
│   └── assistant.db              # RAG 元数据与 Agent 完整会话轮次
├── artifacts/                    # 报告、图表等运行产物
├── .env.example                  # 配置模板
├── requirements.txt              # 直接依赖
└── requirements.lock.txt         # Windows/Python 3.11 环境快照
~~~

SQLite 元数据路径为 data/assistant.db，build_index() 自动同步知识库、文件处理报告和索引构建记录。旧索引可用 python -m app.rag.db sync --kb 教育资料 导入；查看命令和表结构见 [SQLite 元数据指南](docs/SQLITE_METADATA.md)。上传删除 HTTP 接口仍待实现。代码包用 __init__.py 标识；data、artifacts 是数据目录，不是 Python 包。

## RAG 基础数据类型

内部处理链路统一使用数据对象，字典只用于处理报告、JSON 保存和 API 输入输出：

~~~text
PDF → ingestion.loaders.load_pdf() → Document（清洗后一页）
    → ingestion.chunkers.split_document() → Chunk（片段）
    → indexing.pipeline.build_index() → FAISS + JSON，并登记 SQLite
    → retrieval.retrievers.KnowledgeIndex.search() → SearchResult
    → services.chat_service.answer_question() → 带来源答案
~~~

Document 与 Chunk 是独立的数据类，片段通过 source、page 关联来源，不使用继承。当前 Document 对应一页；跨页切片仍未实现。

PDF 加载默认启用本地 OCR：清洗后有效字符不足 40 且含图片的页面标记为疑似扫描页；未解决的英文碎字页面也会尝试识别。RapidOCR + pypdfium2 在本机完成渲染和中英文识别，不上传 PDF、不修改原文件。OCR 后重新统一清洗，再生成 Document。识别文字、双栏顺序、表格和代码缩进需要人工复核。

报告中 suspected_scanned_pages 是初始疑似扫描页，ocr_pages 是采用 OCR 结果的页，needs_ocr_pages 是尚未恢复有效文字的候选页，ocr_failed_pages 是识别或依赖失败的页。仅查看文字层可传 ocr_enabled=False，终端使用 --no-ocr。先限制 max_pages 验证效果，OCR 会增加处理时间。示例见 tests/test_loader.py；自动回归见 tests/test_ocr.py。

~~~python
from app.rag.ingestion.pipeline import document_chunks

chunks, reports = document_chunks("教育资料", max_pages=10, max_chunks=20)
for chunk in chunks[:3]:
    print(chunk.source, chunk.page, chunk.text)
    # 需要 JSON 字典时显式转换：chunk.to_dict()
~~~

document_chunks() 仍保留原参数和两项返回值，但第一项现在是 list[Chunk]；旧式 chunk["text"] 改为 chunk.text。search() 返回 list[SearchResult]，用 hit.chunk.text、hit.score 访问。

磁盘 chunks.json 仍保存原字段，旧索引无需因本次数据类型重构而重建。处理报告 processed_pages 表示已加载、清洗的页数，即使 max_chunks 导致部分页面没有输出片段；chunks 表示实际输出的片段数。

文档预览运行 `python -m scripts.inspect_documents`；单份 PDF 从解析、OCR、清洗、切块到 FAISS/JSON 和 SQLite 的手动验收运行 `python -m scripts.index_document`。顶部参数默认前 20 页，不限制切片数；MAX_PAGES=0 可处理整本。RAG 处理与存储实现在 app/rag 中，共享模型访问位于 app/llm；导入模块不会自动读取资料或调用模型。

混合检索默认开启，精排通过 `--rerank` 启用；需要在 .env 补充独立的 RERANK_API_BASE、RERANK_API_KEY、RERANK_MODEL。详见 [混合检索与精排指南](docs/HYBRID_RETRIEVAL.md)。

## 模块边界与数据流

- API 层校验请求并调用服务，不重复实现检索和 Agent 逻辑。
- Agent → tools.knowledge_search → RagService → 检索；工具返回证据，最终答案由 Agent 生成。
- API 检索路由后续共用 RagService；入库仍调用 indexing.pipeline.build_index。
- SQLite 保存知识库及文件信息；FAISS 保存向量；文本块保留文件名、页码（格式支持时）和稳定标识，用于引用与删除。
- 文档入库：上传 → 解析 → 切片 → Embedding → FAISS 持久化，并保存元数据。
- 问答流程：Gradio → FastAPI → Agent → 工具 → 带来源的最终答案。
- 混合检索：FAISS + jieba/BM25+ → 两路等权 RRF → 可选远程 Rerank → 上下文。
- 后续文件更新规划：删除后重新上传；删除时同步移除相应检索数据。
- 后续代码执行通过隔离服务完成。

## Agent 启动

```powershell
cd D:\Python\PythonProject1
& "D:\miniconda\envs\personal_assistant\python.exe" -m app.agent --kb 教育资料 --session study --rerank
```

输入 `/exit` 退出，相同 `--session` 恢复对话。`/remember 名称 内容` 保存长期记忆，`/memory` 查看，`/forget 名称` 删除；同一 `--owner` 的会话共享长期记忆。模型需支持工具调用。记忆与预算配置见 [记忆与四层压缩](docs/AGENT_MEMORY.md)。完整设计见 [Agent 架构说明](docs/AGENT_ARCHITECTURE.md)。

## 网页/API 启动

```powershell
cd D:\Python\PythonProject1
& "D:\miniconda\envs\personal_assistant\python.exe" -m pip install -r requirements.txt
& "D:\miniconda\envs\personal_assistant\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

页面：http://127.0.0.1:8000/；API 文档：http://127.0.0.1:8000/docs。聊天接口为 `POST /api/chat`（完整 JSON）和 `POST /api/chat/stream`（SSE）；页面使用同一 AgentService 流式显示回答。其他接口包括知识库列表和长期记忆管理。

要启用心知天气，在本机 `.env` 配置 `SENIVERSE_API_KEY`；开发者入口为 https://www.seniverse.com/。网页搜索使用 Tavily Search API，在 `.env` 配置 `TAVILY_API_KEY`；免费额度以服务商控制台为准，所在网络需要代理时可配置 `WEB_SEARCH_PROXY`。搜索后可用 `web_fetch` 抓取相关公开网页正文。抓取只允许公网 HTTP/HTTPS URL，最大响应 2 MB、传给模型的正文最多 6000 字符。文件读写工作目录为 `data/workspace/`。

服务默认只绑定本机 `127.0.0.1`。当前 API 没有用户认证，不应绑定 `0.0.0.0` 或直接暴露到局域网/公网。

## 环境与启动

Conda 环境名称：personal_assistant；Python 3.11。

在项目根目录的 PowerShell 执行：

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -m pip install -r requirements.txt
& "D:\miniconda\envs\personal_assistant\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

访问 http://127.0.0.1:8000/health，预期响应：

~~~json
{"status":"ok"}
~~~

API 文档地址：http://127.0.0.1:8000/docs 。业务 HTTP 提供 Agent 聊天、知识库只读列表和长期记忆接口；RAG 通过 `python -m app.rag` 独立调用。

模型连通阶段将 .env.example 复制为 .env 并填写实际参数；已有 .env 时保留原文件。健康检查不需要密钥；RAG 模型调用会加载项目 .env。真实密钥与私有资料不提交 Git。

## 依赖策略

requirements.txt 包含 RAG、FastAPI、Gradio 和 SQLAlchemy 元数据直接依赖；requirements.lock.txt 保存 Windows / Python 3.11 环境快照。jieba、rank-bm25 已接入；Agent 当前直接复用官方 SDK 的工具调用，无新增框架依赖；Gradio 页面与 JSON API 共用 AgentService。

开题文档列举 create_react_agent、AgentExecutor、EnsembleRetriever 等接口；本版按照 NanoClaw 的显式主循环组织，实现相应模型与工具闭环。Embedding 模型、向量维度和切片方案确定后保留记录；变更时重建索引。

## 开发阶段与验收

1. 基础后端：选择正确解释器，安装启动依赖；/health 返回 ok。
2. 模型连通：独立验证聊天回复、Embedding 向量及错误提示。
3. 文档入库：先选一个行业和 3～5 份资料；能解析、切片、保存，重启后能读取索引。
4. RAG 问答：双路召回，答案带来源；无依据时明确说明，删除文件后不召回旧块。
5. Agent 编排：已接入知识查询、历史、时间、长期记忆、天气、网页搜索与工作区文件工具；代码执行工具仍待实现。
6. 页面与流式：已提供 SSE、Gradio 逐段更新和历史会话恢复；上传管理和图表展示仍待实现。
7. 集成演示：测试超时、无答案、会话隔离、重启恢复；整理锁定依赖、启动说明和演示案例。

行业、模型名称、网络搜索提供商和 CodeBox 部署方案尚待确定；本骨架不预填这些业务选择。

## 接口规划

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | /health | 后端健康检查 |
| POST | /api/chat | Agent 问答（同步 JSON） |
| POST | /api/chat/stream | Agent 问答（SSE 流式输出） |
| GET | /api/sessions | 列出短期会话 |
| GET | /api/sessions/{session_id} | 读取会话轮次 |
| GET / POST | /knowledgebase | 列表 / 创建知识库 |
| DELETE | /knowledgebase/{kb_id} | 删除知识库及其数据 |
| POST | /knowledgebase/{kb_id}/files | 上传并入库 |
| DELETE | /knowledgebase/{kb_id}/files/{file_id} | 删除文件与相应索引 |
| GET | /api/knowledgebases | 返回已发布知识库列表 |
| GET / PUT / DELETE | /api/memories | 查看、保存、删除长期记忆 |
