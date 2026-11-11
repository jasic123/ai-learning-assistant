# Agent 记忆与四层上下文压缩

## 三种状态各自保存什么

| 状态 | 内容 | 保存位置 | 跨会话 |
|---|---|---|---|
| 短期记忆 | 每个成功轮次的用户消息、工具请求、工具完整结果与最终回答 | SQLite agent_turns | 按 session_id 恢复 |
| 长期记忆 | 用户明确要求保存或有长期价值的稳定偏好/项目约定 | SQLite agent_memories | 同一 owner_id 的会话共享 |
| 压缩检查点 | 历史摘录或模型摘要、已覆盖轮次、来源定位信息 | SQLite agent_checkpoints | 在原 session_id 内恢复 |

数据库仍为 data/assistant.db。额外的 agent_session_owners 表约束会话归属；同名会话不能在不同 owner_id 间复用。原版未绑定且已有历史的会话归 local。owner_id 是本机数据隔离标签，不代替未来 API 的身份认证。

RAG 的原 PDF、FAISS、JSON 和知识业务表继续使用原路径与格式。记忆和摘要表在首次访问时惰性创建，本次代码升级不需要重建索引。

## 短期记忆

Agent 读取检查点及检查点之后的全部完整轮次，不再固定丢弃六轮以前的信息。SessionManager.load 统一返回检查点后的全部轮次，ContextBuilder 只组装消息，压缩和预算只由 ContextCompressor 负责。

Gradio 页面提供历史会话下拉列表；刷新列表后选择一条记录可恢复完整可见对话，并沿用原 session_id 继续聊天。HTTP 客户端也可用 `GET /api/sessions` 列表及 `GET /api/sessions/{session_id}` 读取记录。数据库保留所有工具调用和原文，即使 UI 只显示用户消息与最终回答。

原始轮次始终保留。压缩只处理发给模型的副本，最后成功时才将新轮次与新摘要检查点放在同一个事务中提交。模型失败、迭代耗尽、超限重试失败或会话版本冲突，都不会覆盖原摘要或追加未完成轮次。

## 长期记忆

交互终端命令：

```text
/remember 语言 偏好中文，代码需要中文注释
/remember 项目 私人助理，RAG 用 FAISS+JSON，SQLite 管理业务
/memory
/forget 语言
```

`/remember` 相同名称会更新原记录。默认 owner_id=local，`--owner` 选择其他所有者。同一所有者不同 --session 自动共享长期记忆，其他所有者看不到它。

Python 直接使用：

```python
from app.agent.service import AgentService

service = AgentService(["教育资料"], owner_id="local")
service.memory.remember("语言", "偏好中文，代码需要中文注释")
result = service.chat("继续设计 Agent", session_id="agent-design")
print(service.memory.list())
service.memory.forget("语言")
```

当前每位所有者最多 20 条，每条最多 500 字符；这是便于每轮注入的轻量偏好记忆，不承担文档知识库职责。每轮会把长期记忆放进 system prompt 的独立资料区，并明确其不能覆盖系统规则。`skills/memory/SKILL.md` 说明记忆价值判断、内容整理和保存边界，由 `ContextBuilder` 每轮读入上下文。Agent 可用 `manage_long_term_memory` 响应用户的记住/忘记请求；也可通过终端命令或 API 管理。对稳定且未来有帮助的信息，模型应主动判断是否保存；一次性任务和敏感信息不保存。模型只有收到保存工具成功结果后才能告知已保存。技能文件能指导模型行为，但不能保证模型每次都遵循；需要确认实际写入时，可查看长期记忆 API 或 `agent_memories` 表。

学生作答画像与上述用户偏好记忆分开保存。`learning_attempts` 是历史作答事实，`learning_error_reasons` 只保存学生答错后自选的原因；`GET /api/learning/profile` 根据这些记录汇总知识点正确率与错因频次。只有至少 3 次作答的知识点才会标为需要关注，只有至少 3 次自选记录才会作为重复错因传给 Agent。画像作为用户资料消息注入，而不是系统指令；Agent 被明确要求将其视为有限证据，不对学生作性格或固定能力判断。

项目根目录 `identity.md` 是可编辑的身份与语气配置，每轮请求都会重新读取；不存在或无法读取时使用内置私人助理身份。文件可以描述角色和表达风格，不能更改代码中的安全规则。当前时间与日期也会每轮注入 system prompt，默认时区为 `Asia/Shanghai`；`get_current_time` 可查询当前时间或指定 IANA 时区。

## 四层上下文防护

下表是本项目借鉴的四层分工，不宣称复刻 Claude Code 的内部实现或官方固定阈值。

| 层 | 触发 | 执行 | 模型调用 |
|---|---|---|---|
| 1. Micro compaction | 旧工具结果含长正文 | 最近两轮之前的引用正文缩为 500 字符摘录，保留调用 ID、来源、页码和引用编号；历史回查输出也可清理 | 无 |
| 2. Session memory | 估算输入超过输入预算的 80% | 将较早完整轮次改为有轮次编号的本地问答摘录，保留最近两轮详细消息，并复用已保存检查点 | 无 |
| 3. Auto compaction | 摘录后仍接近预算，或摘录本身超过 4000 字符 | 模型生成目标、约束、已完成、未完成事项摘要，合并旧检查点 | 有，单用户问题最多四次 |
| 4. Reactive compaction | 提供商明确返回上下文超限代码 | 更积极压缩历史和当前工具正文，然后对同一次模型请求重试一次 | 最多一次主请求重试，可能先生成摘要 |

一、二层不会额外请求摘要模型。第二层是有标记的摘录，并非完整语义总结；截断的事实不能凭空补全，必要时回查原文。

第三层摘要请求本身也要先检查输入预算；较长摘录分批合并，产物最多 3000 字符。摘要异常、空内容或过长结果都不会被当作有效摘要。失败时保留已有本地摘录；若它仍能装入硬预算，可继续，否则停止并提示缩小范围。

第四层仅识别 context_length_exceeded、context_window_exceeded、prompt_too_long 等明确错误代码，不把所有 HTTP 400、401、429 当成上下文超限。压缩只改变模型请求副本，不重新执行工具。第二次仍超限时抛出异常，不无限重试。当前轮次用户问题与 assistant/tool 对应关系不能被拆开；必要内容本身仍过大时直接停止。

## 预算与可观测信息

```powershell
python -m app.agent --kb 教育资料 --session study --owner local --context-tokens 16000 --output-reserve 2000
```

默认上下文窗口按 16000 配置，预留输出与协议空间 2000，允许输入 14000，主动压缩阈值约为输入预算的 80%。这只是项目默认值，必须按所选模型调整，不能据此认为所有服务都支持 16000。

估算包含系统提示、长期记忆、摘要、正文、工具参数和工具定义。没有为未知兼容服务强行套用特定 tokenizer：先按 UTF-8 字节估算，模型返回 usage.prompt_tokens 时向上校准本轮的估算比例。因此这是保守预算控制，不是精确 token 计数，实际超限由第四层兜底。输出预留是预算空间，不是对提供商实际最大输出的强制参数。

AgentResult 增加：

- compression：本轮触发的清理、摘录、摘要、重用、失败与超限恢复事件。
- summary_calls：摘要请求次数，失败尝试也计入。
- estimated_input_tokens：最后一次主请求的估算输入数。
- model_calls：主模型尝试数，含超限后的那次重试；摘要次数单独统计。

## 原始历史与引用回查

注册的 recall_history 工具绑定本次 owner 与 session，模型不能传入别人的会话名。参数为 turn_number、start、max_chars；最多每页 6000 字符，next_start 用于继续读。返回的是原始 JSON 文本的分页，单页不保证构成完整 JSON，合并各页可恢复完整原文。

第一层保留来源与页码；检查点最多保留最近 32 条来源定位信息，完整来源仍在 agent_turns。摘要中遗失的事实需要回查，不把模型生成的摘要当成新的权威文档。回查页出现旧引用时会补充来源元数据，方便最终答案返回来源。

引用仍使用 T轮次-C调用-S片段编号。长期记忆、摘要和工具正文都是数据，不允许覆盖系统指令。自动压缩不自动修改长期记忆。

## 文件与测试

- app/agent/memory.py：长期记忆 CRUD 与数量限制。
- app/agent/session.py：原始轮次、会话归属、压缩检查点原子提交。
- app/agent/compression.py：预算与四层压缩策略。
- app/agent/context.py：把记忆、摘要和消息组装为请求。
- app/tools/history.py：当前会话原文分页回查。
- tests/test_agent_memory.py：层级触发、摘要失败、重试限制、原文保留、来源恢复、归属隔离和事务回滚。

测试使用临时 SQLite 与离线模型替身；不会访问真实模型，也不会把测试内容写入你的实际记忆库。真实服务下的摘要质量和模型上下文容量仍需联调。
