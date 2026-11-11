# SQLite 知识库与文件元数据

数据库位置：`D:\Python\PythonProject1\data\assistant.db`。使用 SQLAlchemy 2.x；首次调用数据库功能或 `build_index()` 时自动创建表，导入模块不会建库。

## 三张表

| 表 | 用途 |
|---|---|
| knowledgebases | 名称、`kb_info` 简介、当前索引版本、当前片段数、创建和更新时间 |
| knowledge_files | 所属知识库、相对文件名、同步时文件大小、当前版本页数和片段数、处理报告（包括 OCR） |
| index_builds | 版本号、building/ready/failed 状态、模型、维度、参数、时间，以及失败异常类型 |

文件状态 `indexed` 表示当前版本包含该文件的片段，不代表整份 PDF 已全部入库；实际范围见 `processing_report` 中的 start_page、processed_pages 和 chunks。`no_chunks` 表示处理过但没有输出片段，`pending` 表示未被当前版本覆盖。限制 max_chunks 或重建只覆盖部分文件时，旧文件的当前片段数清零，不继续显示为已入库。

重建状态保存在 index_builds；knowledge_files 保持上一版已发布快照，不会因为失败重建丢失之前的页数和片段数。失败仅保存异常类型，不保存完整异常消息或 API Key。报告可能含 PDF 原文，应按原资料管理数据库文件。

## 终端命令

进入项目根目录，使用 personal_assistant 环境解释器：

```powershell
cd D:\Python\PythonProject1
python -m app.rag.db init
python -m app.rag.db save --kb 教育资料 --info "Python 教材和编程知识，用于学习辅导"
python -m app.rag.db list
python -m app.rag.db info --kb 教育资料
```

`save` 创建知识库或更新简介，不扫描文件、不调用模型、不修改索引。`info` 返回知识库、当前文件快照和构建历史。Agent 已实现知识工具调用；kb_info 可供后续自动知识库选择使用，目前知识库由应用侧指定。

原入库命令不变：

```powershell
python -m app.rag ingest --kb 教育资料 --max-pages 3 --max-chunks 10
```

这会做本地 OCR 并调用配置的 Embedding 服务；成功后 SQLite 自动同步。仅运行 inspect、load_pdf() 或 document_chunks() 不写业务数据库。

## 旧索引导入和恢复

```powershell
python -m app.rag.db sync --kb 教育资料
```

根据 CURRENT 与该版本的 metadata.json 导入旧索引，不重新 OCR、不调用 Embedding。重复同步不新增重复记录，保留已填写简介。

SQLite 和文件系统不共享原子事务。先写齐索引文件，再在短数据库事务内更新记录并切换 CURRENT；普通提交失败恢复旧指针。如果进程在两种存储之间崩溃，可用 sync 以磁盘 CURRENT 为准恢复数据库。未知任务可能保留 building 状态，不会被随意标记失败。

## 存储分工

```text
data/assistant.db                          知识库、文件和构建元数据
data/uploads/<知识库>/                     原始文件
data/indexes/<知识库>/<版本>/index.faiss     向量索引
data/indexes/<知识库>/<版本>/chunks.json     片段正文和来源
data/indexes/<知识库>/<版本>/metadata.json   版本快照和处理报告
```

保留 JSON 以兼容旧索引和恢复数据库，不在 SQLite 重复保存完整 Chunk 正文或向量。

本次提供元数据操作、入库同步及终端查询。上传、删除 HTTP 接口尚未接入，手动删除数据库行不能代替从索引删除内容。create_all() 负责初始建表，今后修改表结构时需要显式迁移。

## 自动测试

```powershell
python -m unittest discover -s tests -p test_metadata.py -v
python -m unittest discover -s tests -v
```

覆盖持久化、简介更新、失败重建、数据库提交失败的指针恢复、部分文件重建及旧索引同步；使用临时目录与假向量，不调用真实模型。

## Agent 短期会话

Agent 通过 app/agent/session.py 使用同一个 assistant.db，首次会话访问时创建 agent_turns 表：session_id 与 turn_number 为联合主键，messages_json 保存完整成功轮次，created_at 保存 UTC 时间。此表由 SessionManager 管理，不加入 RAG 的 ORM Base，避免两个子系统互相依赖。

RAG 的 db info/sync 命令只管理知识业务数据，不会列出或导入 Agent 历史。会话消息包括用户问题、工具请求、工具正文和最终回答；失败未完成轮次不保存。详见 [Agent 架构](AGENT_ARCHITECTURE.md)。

Agent 现另有 agent_checkpoints（压缩摘要与来源）、agent_memories（按 owner_id 的长期记忆）、agent_session_owners（会话归属）三张表；由 Agent 子系统惰性创建，不改变 RAG ORM 表。详见 [记忆与四层压缩](AGENT_MEMORY.md)。

学习记录使用 learning_quizzes、learning_attempts、learning_mistakes 保存练习题组、每次作答和复习任务。`learning_error_reasons` 保存学生在答错后自选的原因，按 `owner_id + attempt_id + question_id` 关联，只允许为错误答案记录。学习画像由历史作答和自选错因实时汇总，不另存一份可与事实脱节的 AI 判断；画像接口为 `GET /api/learning/profile`，聊天 Agent 只接收样本达到门槛的精简信号。
