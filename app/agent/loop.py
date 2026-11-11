"""NanoClaw 风格 AgentLoop：模型选择工具，工具结果回填，再由模型回答。"""
from collections import Counter
from dataclasses import dataclass
import json
import re
from uuid import uuid4

from app.agent.context import ContextBuilder
from app.agent.compression import ContextCompressor
from app.agent.session import SessionManager, validate_session_id
from app.tools.registry import ToolRegistry
from app.llm.responses import ContextWindowExceeded


@dataclass(frozen=True)
class AgentResult:
    session_id: str
    answer: str
    sources: tuple[dict, ...]
    model_calls: int
    tool_calls: int
    compression: tuple[str, ...] = ()
    summary_calls: int = 0
    estimated_input_tokens: int = 0

    def to_dict(self):
        return {"session_id": self.session_id, "answer": self.answer,
                "sources": list(self.sources), "model_calls": self.model_calls,
                "tool_calls": self.tool_calls, "compression": list(self.compression),
                "summary_calls": self.summary_calls, "estimated_input_tokens": self.estimated_input_tokens}


class AgentLoop:
    def __init__(self, client, tools: ToolRegistry, context: ContextBuilder,
                 sessions: SessionManager, *, max_iterations=8, memory=None,
                 context_tokens=16000, output_reserve=2000, learning_profile=None):
        if type(max_iterations) is not int or not 1 <= max_iterations <= 32:
            raise ValueError("模型迭代上限须为 1～32")
        self.client, self.tools, self.context, self.sessions = client, tools, context, sessions
        self.max_iterations = max_iterations
        self.memory = memory or []
        self.learning_profile = learning_profile or {}
        self.context_tokens, self.output_reserve = context_tokens, output_reserve

    def run(self, user_message: str, *, session_id="default", on_delta=None, on_status=None) -> AgentResult:
        validate_session_id(session_id)
        if not isinstance(user_message, str) or not user_message.strip() or len(user_message) > 8000:
            raise ValueError("用户问题须为 1～8000 字符")
        snapshot = self.sessions.load(session_id)
        current = [{"role": "user", "content": user_message}]
        definitions = self.tools.get_definitions()
        compressor = ContextCompressor(self.client, self.context, snapshot, definitions, memory=self.memory,
                                       context_tokens=self.context_tokens, output_reserve=self.output_reserve,
                                       learning_profile=self.learning_profile)
        # 每一轮重新计数，用户在下一轮再次询问相同问题不会触发熔断。
        signatures, call_ids = Counter(), set()
        tool_count = 0
        request_count = 0
        first_url = self._first_link_request(user_message, snapshot.turns)
        if first_url:
            call_id = uuid4().hex
            arguments = json.dumps({"url": first_url}, ensure_ascii=False)
            current.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": "web_fetch", "arguments": arguments,
                }}
            ]})
            if on_status:
                on_status({"tool": "web_fetch", "state": "running", "message": "正在读取第一个搜索结果…"})
            result = self.tools.execute("web_fetch", arguments)
            tool_count += 1
            current.append({"role": "tool", "tool_call_id": call_id,
                            "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
            if on_status:
                on_status({"tool": "web_fetch", "state": "failed" if not result.get("ok") or
                           result.get("data", {}).get("error") else "finished",
                           "message": "网页读取失败，正在整理原因…" if not result.get("ok") or
                           result.get("data", {}).get("error") else "网页已读取，正在整理内容…"})
        for iteration in range(1, self.max_iterations + 1):
            messages = compressor.prepare(current)
            request_count += 1
            try:
                reply = self._chat(messages, definitions, on_delta)
            except ContextWindowExceeded:
                # 只对明确的上下文超限压缩重试一次；不重复执行工具，不重试认证/限流错误。
                messages = compressor.prepare(current, reactive=True)
                request_count += 1
                reply = self._chat(messages, definitions, on_delta)
            compressor.observe(reply.input_tokens, messages)
            if not reply.tool_calls:
                if not isinstance(reply.content, str) or not reply.content.strip():
                    raise RuntimeError("模型没有返回有效回答")
                current.append(reply.to_message())
                originals = [message for turn in snapshot.turns for message in turn] + current
                sources = self._sources(originals, reply.content, snapshot.sources)
                # 只有完整成功轮次落库；异常和迭代耗尽不污染下次对话。
                self.sessions.append_turn(session_id, current, expected_version=snapshot.version,
                                          checkpoint=compressor.checkpoint)
                return AgentResult(session_id, reply.content.strip(), sources, request_count, tool_count,
                                   tuple(compressor.events), compressor.summary_calls, compressor.last_estimate)
            if iteration == self.max_iterations:
                break
            if len(reply.tool_calls) > 8:
                raise RuntimeError("单轮工具调用过多，已停止本轮")
            # 在执行任何工具前检查全部 ID，避免出现重复或无法配对的消息。
            ids = [call.id for call in reply.tool_calls]
            if (any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids) or call_ids.intersection(ids)):
                raise RuntimeError("模型工具调用 ID 重复或缺失")
            call_ids.update(ids)
            current.append(reply.to_message())
            for call in reply.tool_calls:
                tool_count += 1
                signature = self._signature(call)
                signatures[signature] += 1
                if signatures[signature] >= 3:
                    raise RuntimeError("工具相同参数反复调用，已停止本轮")
                if signatures[signature] == 2:
                    result = {"ok": False, "error": "repeated_call", "message": "请使用已有结果或修改检索问题"}
                else:
                    if on_status:
                        on_status({"tool": call.name, "state": "running",
                                   "message": self._tool_status(call.name, "running")})
                    result = self.tools.execute(call.name, call.arguments)
                    if on_status:
                        failed = (not result.get("ok") or
                                  (isinstance(result.get("data"), dict) and result["data"].get("error")))
                        on_status({"tool": call.name, "state": "failed" if failed else "finished",
                                   "message": self._tool_status(call.name, "failed" if failed else "finished")})
                if call.name == "knowledge_search" and result.get("ok"):
                    # RAG 的 S1 仅在一次查询内有效；Agent 加上轮次和调用序号。
                    for source in result["data"]["sources"]:
                        source["citation"] = f"T{snapshot.version + 1}-C{tool_count}-{source['citation']}"
                current.append({"role": "tool", "tool_call_id": call.id,
                                "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
        raise RuntimeError(f"已达到模型迭代上限（{self.max_iterations}），本轮未保存")

    @staticmethod
    def _tool_status(name, state):
        labels = {"web_search": "网页搜索", "web_fetch": "网页读取", "knowledge_search": "知识库检索",
                  "get_weather": "天气查询", "recall_history": "历史回查", "workspace_file": "文件处理",
                  "manage_long_term_memory": "长期记忆更新", "get_current_time": "时间查询"}
        label = labels.get(name, "工具处理")
        return f"{label}{'失败' if state == 'failed' else '完成' if state == 'finished' else '中'}…"

    @staticmethod
    def _first_link_request(message, turns):
        request = re.search(r"(?:第一个|第1个|首个).{0,8}(?:链接|搜索结果).{0,12}(?:打开|读取|阅读|抓取|看看|看一下)|"
                            r"(?:打开|读取|阅读|抓取|看看|看一下).{0,12}(?:第一个|第1个|首个).{0,8}(?:链接|搜索结果)",
                            message)
        if not request:
            return None
        for turn in reversed(turns):
            calls = {}
            for item in turn:
                if item.get("role") == "assistant":
                    calls.update({call.get("id"): call.get("function", {}).get("name")
                                  for call in item.get("tool_calls", [])})
            for item in reversed(turn):
                if item.get("role") != "tool" or calls.get(item.get("tool_call_id")) != "web_search":
                    continue
                try:
                    payload = json.loads(item.get("content", ""))
                    results = payload.get("data", {}).get("results", []) if payload.get("ok") else []
                    url = results[0].get("url") if results else None
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        return url
                except (ValueError, TypeError, AttributeError):
                    continue
        return None

    def _chat(self, messages, definitions, on_delta):
        if on_delta is None:
            return self.client.chat_with_tools(messages, definitions)
        return self.client.chat_with_tools(messages, definitions, on_delta=on_delta)

    @staticmethod
    def _signature(call):
        try:
            arguments = json.dumps(json.loads(call.arguments), sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            arguments = str(call.arguments)
        return call.name + ":" + arguments

    @staticmethod
    def _sources(messages, answer, remembered_sources=()):
        """提取最终答案引用过的实际证据，不把全文重复放进对外响应。"""
        cited = set(re.findall(r"\[(T\d+-C\d+-S\d+)\]", answer))
        sources = {source["citation"]: source for source in remembered_sources if source["citation"] in cited}
        for message in messages:
            if message.get("role") != "tool":
                continue
            payload = json.loads(message["content"])
            data = payload.get("data", {}) if payload.get("ok") else {}
            for source in data.get("sources", []):
                if source["citation"] in cited:
                    sources[source["citation"]] = {
                        **{key: value for key, value in source.items() if key != "text"},
                        "knowledgebase": source.get("knowledgebase", data.get("knowledgebase")),
                        "generation": source.get("generation", data.get("generation")),
                    }
        return tuple(sources.values())
