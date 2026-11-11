# 项目架构与数据流

项目由 Agent 编排、工具适配、RAG 子系统和共享模型访问层组成。RAG 内部按领域、文档处理、索引、检索、终端问答和业务元数据分层；知识存储采用 FAISS + JSON，SQLite 管理业务元数据。
Agent 使用公共检索服务获取资料，后续自行组织最终答案。

Agent 的 NanoClaw 参考映射、会话表、循环与上下文策略见 [Agent 架构设计](AGENT_ARCHITECTURE.md)。

## 目录与职责

```text
PythonProject1/
├── app/
│   ├── config.py                 # 项目配置与路径
│   ├── main.py                   # FastAPI 入口，目前提供 /health
│   ├── llm/                      # Agent、RAG 共用的模型访问层
│   │   └── client.py             # 官方 OpenAI SDK 适配器
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
│   ├── api/                      # Agent 聊天、知识库列表和记忆 HTTP 路由
│   ├── agent/                    # AgentLoop、ContextBuilder、SessionManager
│   └── tools/                    # Agent 工具适配层
│       ├── knowledge.py          # 已实现：返回知识片段与来源
│       ├── time.py               # 当前时间与 IANA 时区
│       ├── memory.py             # 跨会话长期记忆管理
│       ├── web.py                # Tavily 网页搜索
│       ├── weather.py            # 心知天气
│       └── code.py
├── scripts/
│   ├── index_document.py         # 单份 PDF 全流程入库验收
│   ├── inspect_documents.py      # 本地文档预览
│   └── validate_rag.py           # 真实检索与问答验收
├── tests/                        # 离线自动测试
├── docs/                         # 架构、使用和数据库说明
├── frontend/                     # Gradio 聊天页面
├── data/
│   ├── uploads/                  # 原始 PDF
│   ├── indexes/                  # CURRENT 与版本化 FAISS/JSON
│   └── assistant.db              # SQLite 数据库
├── artifacts/                    # 报告、图表等运行产物
├── .env.example                  # 配置模板
├── requirements.txt              # 直接依赖
└── requirements.lock.txt         # Windows/Python 3.11 环境快照
```

## 文档处理流水线

`app.rag.ingestion.pipeline.ingest_document()` 处理单份 PDF，返回三个结果：

```python
from app.rag.ingestion.pipeline import ingest_document

documents, chunks, report = ingest_document(
    "data/uploads/教育资料/Python参考手册(第4版).pdf",
    source="Python参考手册(第4版).pdf",
    max_pages=20,
    chunk_size=400,
    overlap=60,
)
```

处理顺序为 PDF 文字提取、页面质量判断、按需 OCR、统一清洗和切块。
该函数不创建数据库或索引、不请求 Embedding。路径参数按 Python 常规规则解析，
需要脱离项目目录运行时可传入 `PROJECT_ROOT / 相对路径` 或绝对路径。

`document_chunks(name, ...)` 遍历指定知识库中的 PDF 并复用单文件流水线，
返回 `list[Chunk]` 与文件报告列表。`max_pages` 针对每份 PDF，`max_chunks` 是整个知识库总额度。
报告的 `processed_pages` 描述已加载范围，即使片段额度使部分页面没有输出切片。

Document 沿用当前项目的逐页含义；与 rag_service 的完整文档对象并非相同结构。
跨页切片和新的完整文档对象可作为后续独立变更，不能只改类名而忽略偏移、引用和旧索引兼容。

## 入库与存储

`app.rag.indexing.pipeline.build_index(name, client, ...)` 是知识库入库入口：

```text
登记 building 构建记录
    → ingestion.pipeline.document_chunks
    → indexing.embeddings.embed_chunks
    → indexing.vector_store.create_index
    → 写入新版本 index.faiss、chunks.json、metadata.json
    → db.repository.publish_index
    → 更新 SQLite 快照并切换 CURRENT
```

Embedding 客户端由调用者提供，既可以是真实 API 客户端，也可以是测试替身。
编码按批执行，FAISS 行号与 JSON 中的 Chunk 顺序一致。
建库和查询都进行 L2 归一化，IndexFlatIP 的内积结果对应余弦相似度。

| 位置 | 内容 |
|---|---|
| data/uploads/知识库/ | 原始 PDF |
| data/indexes/知识库/CURRENT | 当前已发布版本号 |
| data/indexes/知识库/版本/index.faiss | 向量索引 |
| 同版本 chunks.json | 片段正文、来源、页码、偏移和清洗版本 |
| 同版本 metadata.json | 模型信息、维度、参数及文件处理报告 |
| data/assistant.db | 知识库、文件当前快照和索引构建历史 |

SQLite 不重复保存正文和向量。数据库与文件系统不能共享事务，原有版本目录、
指针补偿和 sync 恢复逻辑继续使用；失败任务与上一版可用索引分开记录。
JSON 继续采用 schema_version=1，原来的 FAISS/JSON 索引不要求因路径迁移而重建。

## 检索与问答

```text
AgentLoop
  → app.tools.knowledge.knowledge_search
  → app.rag.RagService.search(SearchRequest(...))
  → KnowledgeIndex：FAISS + BM25+ → RRF → 可选 Rerank
  → SearchResponse：正文、文件名、页码、片段 ID、分数与索引版本
  → Agent 汇总工具结果并生成最终答案
```

外部调用统一使用 `from app.rag import RagService, SearchRequest`。工具层不导入 RAG 的索引、数据库和检索实现。`RagService` 每次读取 CURRENT 并加载本次使用的固定版本，按需创建 Embedding 和 Rerank 客户端，并在成功或失败后关闭连接。纯 BM25 且不精排时不读取模型配置、不请求模型。

`knowledge_search` 返回可 JSON 序列化的字典，不调用聊天模型。独立终端 `search` 复用同一服务；`ask` 在检索后通过 `services/chat_service.py` 生成答案。BM25 从当前版本 chunks.json 构建内存索引，已有向量无需重建。AgentLoop、Gradio 聊天页面和 FastAPI JSON 路由已实现。

接入示例与返回字段见 [Agent 接入说明](AGENT_INTEGRATION.md)，检索原理与验收见 [混合检索指南](HYBRID_RETRIEVAL.md)。

## 运行与验收

在项目根目录运行：

```powershell
python -m app.rag inspect --kb 教育资料 --max-pages 3 --max-chunks 10
python -m app.rag ingest --kb 教育资料 --max-pages 3 --max-chunks 10
python -m app.rag search --kb 教育资料 --question "Python 列表是什么？"
python -m app.rag.db info --kb 教育资料
python -m unittest discover -s tests -v
```

真实单文件入库验收：直接运行 `scripts/index_document.py`，或 `python -m scripts.index_document`。
顶部修改 PDF_PATH、MAX_PAGES、START_PAGE、CHUNK_SIZE、OVERLAP；默认前 20 页且不限制切片数，
MAX_PAGES=0 表示整本。脚本只请求 Embedding，不需要聊天模型，报告写入 artifacts/pdf-ingestion-test/report.json。
它独立读取 SQLite 核验三张业务表，并检查索引数量、版本、页码、处理报告及原 PDF 未被修改。

真实问答验收使用 `python -m scripts.validate_rag --rebuild`，需要 Embedding 和聊天配置。
相对 PDF 路径以项目根目录解析；默认测试正文第 16～20 页，不截断其生成的切片。
OCR、引用和答案是否正确仍需人工核对；离线假模型测试不能代替真实模型效果验收。

## 包边界与调用方式

RAG 的内部模块统一位于 `app/rag/`，共享模型客户端位于 `app/llm/`。每个 Python 包由 `__init__.py` 标识。
`app/rag/__main__.py` 是 RAG 自身的终端入口，`app/rag/db/__main__.py` 是业务元数据管理入口。

外部消费方优先使用公共入口；以下内部接口用于文档处理、入库与维护脚本：

```python
from app.rag import RagService, SearchRequest, SearchResponse
from app.rag.domain.models import Document, Chunk, SearchResult
from app.rag.ingestion.pipeline import ingest_document, document_chunks
from app.rag.indexing.pipeline import build_index
from app.rag.retrieval.retrievers import KnowledgeIndex
from app.rag.services.chat_service import answer_question
from app.rag.db.repository import knowledgebase_details
```

文档处理层负责生成 Document、Chunk 和处理报告；索引流水线调用文档处理、向量化和数据库发布；
检索层加载已发布索引；问答服务消费检索结果。SQLite db 管理此 RAG 子系统的业务元数据。
共享项目配置仍由 `app/config.py` 提供，所有数据路径仍指向项目根目录下的 data。

测试中的 patch 应指向实际依赖位置，例如 `app.config.DATA_DIR`、
`app.rag.ingestion.loaders.ocr_page` 和 `app.rag.indexing.pipeline.document_chunks`。
手动测试脚本集中在 scripts，自动测试集中在 tests。

## 官方 OpenAI SDK 客户端

`app/llm/client.py` 使用官方 `openai.OpenAI`，业务入口仍是
`embed(texts)`、`chat(messages)`、`rerank(query, documents)`。
`api_key`、`base_url`、`model` 继续来自原来的 ModelSettings；`.env` 配置项不变。

- Embedding 使用 `embeddings.with_raw_response.create`，通过 SDK 发送请求，读取原始 JSON 严格校验编号和数值，避免类型转换把布尔值误当合法向量。
- 聊天使用 `chat.completions.create`，保留当前兼容服务的非流式协议。
- Rerank 是服务商扩展，使用 SDK 的 `post("/rerank", ...)`，仍要求服务商支持既有 results/index/relevance_score 格式。
- 保留连接 15 秒、其他阶段 90 秒超时；`max_retries=0` 保持原来不自动重试的行为。
- SDK 异常统一转换成脱敏提示；不打印原始响应、密钥或请求正文；`with` 退出时关闭连接。

保留 httpx 依赖用于 SDK 传输配置、扩展响应类型、离线 MockTransport 及 FastAPI 测试。
切换 SDK 不改变模型名称、向量维度、切片和存储格式，无需仅因本次迁移重建已有索引。
测试使用内存 HTTP 传输，不代表真实提供商接口已经完成联调。
