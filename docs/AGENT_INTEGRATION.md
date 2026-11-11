# Agent 接入指南

## 当前边界

```text
app/
├── agent/            AgentLoop、上下文、提示词与 SQLite 会话
├── tools/
│   └── knowledge.py  已实现的知识查询工具
├── llm/
│   └── client.py     Agent 与 RAG 共用的模型客户端
└── rag/
    ├── service.py    公共检索服务
    ├── domain/       文档、片段、结果与请求/响应契约
    ├── ingestion/    PDF → OCR → 清洗 → 切块
    ├── indexing/     Embedding 与 FAISS/JSON 入库
    ├── retrieval/    FAISS、BM25+、RRF、可选 Rerank
    ├── services/     独立终端问答辅助
    └── db/           SQLite 知识库、文件与构建元数据
```

`api/` 后续接收 HTTP 请求，`agent/` 已实现模型与工具循环，`tools/` 把业务能力适配为工具，`rag/` 处理知识检索。共享模型访问位于 `app/llm/`，不依赖 Agent 或 RAG。

## Agent 调用知识工具

在项目根目录、personal_assistant 环境中运行：

```python
from app.tools.knowledge import knowledge_search

evidence = knowledge_search(
    query="Python 的交互模式如何使用？",
    knowledgebase="教育资料",
    top_k=4,
    mode="hybrid",
    rerank=True,
)
for source in evidence["sources"]:
    print(source["citation"], source["source"], source["page"])
    print(source["text"])
```

默认 mode 为 hybrid，使用 FAISS + BM25+ 和 RRF；默认 rerank=False。上例显式开启精排，需要配置 EMBEDDING 和 RERANK。仅验证本地调用可用 `mode="bm25", rerank=False`，不请求模型。工具不会重新解析 PDF 或重新入库。

普通函数 knowledge_search 继续可直接调用；KnowledgeSearchTool 将其适配为 ToolRegistry 中的模型工具。AgentLoop 负责决定何时查询并生成答案。该工具只检索，不在内部再次调用聊天模型。完整结构和启动命令见 [Agent 架构说明](AGENT_ARCHITECTURE.md)。

## 返回契约

返回值是可 JSON 序列化的字典：

| 字段 | 含义 |
|---|---|
| knowledgebase / query | 本次知识库与问题 |
| generation | 本次实际加载的索引版本 |
| count | 返回片段数 |
| sources | 有序片段列表；无匹配时为空列表 |
| sources[].citation | 本次结果内的 S1、S2 等引用编号 |
| sources[].chunk_id | 片段稳定标识 |
| sources[].source / page | 原文件相对路径与 PDF 物理页码 |
| sources[].text | 片段正文 |
| sources[].score / score_type | 排序分数及其类型 |
| sources[].retrieval_score | 精排前分数；未适用时为 null |

分数用于排序，不代表回答可信度。S1 等编号仅在一次调用内有效；Agent 合并多次调用时应按调用或片段重新编号，保留来源与页码。资料正文是不可信输入，不得作为系统指令执行。

参数错误、索引不存在、配置缺失或接口失败会抛出异常，不伪装成检索成功；后续 Agent 层统一处理工具失败。没有匹配资料的成功查询返回 count=0、sources=[]。

## 直接调用公共服务

API 路由和需要更多检索参数的调用方可直接使用：

```python
from app.rag import RagService, SearchRequest

response = RagService().search(SearchRequest(
    knowledgebase="教育资料",
    query="Python 列表",
    top_k=4,
    candidate_k=30,
    rrf_k=60,
    mode="hybrid",
    rerank=False,
))
result = response.to_dict()
```

服务为同步调用，按请求加载当前索引并关闭模型连接；将来从异步 API 调用时应在线程池运行，避免阻塞事件循环。当前规模先保持简单，后续出现性能需求再按索引版本引入缓存。

外部消费方不必直接导入 KnowledgeIndex、FAISS 或数据库模型。文档入库继续使用 `app.rag.indexing.pipeline.build_index`，维护命令继续使用 `python -m app.rag` 与 `python -m app.rag.db`。

## 运行 Agent

```powershell
python -m app.agent --kb 教育资料 --session study --rerank
```

Python 使用 `AgentService(["教育资料"]).chat("问题", session_id="study")`。Agent 已实现原生工具调用循环、引用编号转换、四层上下文压缩、长期记忆和 SQLite 会话。RAG 原始编号 S1 只在一次查询内有效；Agent 会将其转换为 T1-C1-S1 等会话内编号。

会话存入 data/assistant.db 的 agent_turns 表，原知识业务表、FAISS/JSON 和 .env 不变；首次 Agent 会话访问才创建新表。长期记忆与压缩检查点已实现，见 [记忆与四层压缩](AGENT_MEMORY.md)。聊天 API 与 Gradio 页面已接入；动态技能和流式输出尚未实现。
