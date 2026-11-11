# Gradio 页面与 FastAPI

## 启动

在项目根目录执行：

```powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -m pip install -r requirements.txt
& "D:\miniconda\envs\personal_assistant\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

聊天页：http://127.0.0.1:8000/；交互 API 文档：http://127.0.0.1:8000/docs。页面启动时读取已登记且有已发布索引的知识库，可刷新列表、选择向量/关键词/混合检索及开启精排。回答中的引用附带文件名和 PDF 物理页码。

会话 ID 在 Gradio 页面生成；“新对话”建立新会话，刷新历史列表并选择旧记录可恢复并继续对话。页面使用默认 owner_id=local，共用本地 Agent 的长期记忆。

### 短期会话接口

- `GET /api/sessions?owner_id=local&limit=50`：按最近活动时间列出会话，返回预览和轮数。
- `GET /api/sessions/{session_id}?owner_id=local`：返回完整轮次，包含模型工具调用记录。

## HTTP API

### `GET /api/knowledgebases`

返回已登记且已发布的知识库：

```json
[{"name":"教育资料","kb_info":"","chunk_count":487,"generation":"..."}]
```

### `POST /api/chat`

同步 JSON 聊天接口。模型调用在线程池中运行，因此不会阻塞 FastAPI 事件循环。

```json
{
  "message": "请按教材介绍 Python 列表",
  "session_id": "study-01",
  "knowledgebases": ["教育资料"],
  "mode": "hybrid",
  "rerank": false,
  "owner_id": "local"
}
```

session_id 与 owner_id 均限 1～64 位 ASCII 字母、数字、下划线或连字符。knowledgebases 可传 1～8 个名称。mode 为 hybrid、vector 或 bm25；rerank 为布尔值。返回 AgentResult：最终答案、已引用来源、模型调用/工具调用次数及上下文压缩信息。参数错误为 422，模型或 Agent 服务错误为 502。

### 长期记忆接口

- `GET /api/memories?owner_id=local`：列出当前所有者的长期记忆。
- `PUT /api/memories?owner_id=local`，正文 `{"key":"语言","content":"偏好中文"}`：创建或更新一项记忆。
- `DELETE /api/memories/语言?owner_id=local`：删除一项记忆。

记忆单项限制由 MemoryStore 执行；每个所有者最多 20 条、单项不超过 500 字符。

### 学生学习画像

- `GET /api/learning/profile`：按已登录账号返回各知识点的作答数、错误数、正确率和证据等级，以及答错后学生自选的错因统计。
- `POST /api/learning/attempts/{attempt_id}/error-reasons`，正文 `{"reasons":{"123":"condition_missed"}}`：为该次作答中答错的题保存可选错因。原因可为 `concept_unclear`、`condition_missed`、`method_confused`、`calculation_error`、`guessed` 或 `other`。

知识点至少累计 3 次作答后才标为需要关注；错因至少由学生自选 3 次后才标为重复自我报告。画像不推断性格或固定能力，低样本只显示“样本不足”。Agent 仅收到满足门槛的摘要，用于调整讲解和练习顺序。

## 处理路径

```text
Gradio generator ─┐
                  ├→ AgentService → AgentLoop → 工具与记忆 → SQLite / RagService
POST /api/chat ───┤
POST /api/chat/stream ┘
```

Gradio 挂载在 FastAPI 同一服务下，回调直接调用共享业务函数，不对本机 HTTP 发环回请求。浏览器的新对话与 HTTP 请求使用相同 AgentService、工具、记忆和 SQLite 会话契约。

## 本机安全边界

服务绑定 `127.0.0.1`。API 尚无登录认证、CSRF 防护或多用户授权，不能把监听地址改为 `0.0.0.0` 或直接暴露在局域网/公网。owner_id 和 session_id 用于本地记忆/会话分区，不是认证凭据。部署到共享网络前需接入可靠身份认证，并以服务端身份确定 owner_id，禁止信任客户端自报身份。

Gradio analytics 已关闭。UI/API 测试使用临时 SQLite 与模型替身，不调用远程模型，也不写实际私有会话。
