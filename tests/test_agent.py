"""Agent 离线验收：真实 SDK 协议、工具闭环、SQLite 会话及失败隔离。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from app import config
from app.agent.context import ContextBuilder
from app.agent.loop import AgentLoop
from app.agent.service import AgentService
from app.agent.session import SessionManager
from app.config import ModelSettings
from app.llm.client import ModelClient
from app.llm.responses import ModelReply, ToolCall
from app.rag.indexing.pipeline import build_index
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.registry import ToolRegistry
from test_rag import FakeModel, make_pdf


def completion(*, text=None, calls=None, finish=None):
    return {"choices": [{"index": 0, "finish_reason": finish or ("tool_calls" if calls else "stop"),
                         "message": {"role": "assistant", "content": text,
                                     **({"tool_calls": calls} if calls else {})}}]}


def function_call(call_id="call-1", query="Python lists"):
    return {"id": call_id, "type": "function", "function": {
        "name": "knowledge_search", "arguments": json.dumps({"query": query,
        "knowledgebase": "教育资料", "top_k": 1}, ensure_ascii=False)}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.sessions = SessionManager(self.data)
        self.registry = ToolRegistry()
        self.registry.register(KnowledgeSearchTool(["教育资料"], mode="bm25"))
        self.context = ContextBuilder(["教育资料"])

    def loop(self, replies, **kwargs):
        client = Mock()
        client.chat_with_tools.side_effect = replies
        return AgentLoop(client, self.registry, self.context, self.sessions, **kwargs), client

    def test_real_pdf_rag_sdk_agent_and_sqlite_end_to_end(self):
        """PDF 真实入库，用本地 BM25；仅聊天传输使用内存替身。"""
        with patch.object(config, "DATA_DIR", self.data):
            uploads = self.data / "uploads/教育资料"
            uploads.mkdir(parents=True)
            make_pdf(uploads / "lesson.pdf")
            meta = build_index("教育资料", FakeModel())
            requests = []

            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                self.assertEqual(body["tools"][0]["function"]["name"], "knowledge_search")
                if len(requests) == 1:
                    return httpx.Response(200, json=completion(calls=[function_call()]))
                self.assertEqual(body["messages"][-2]["tool_calls"][0]["id"], "call-1")
                tool_message = body["messages"][-1]
                self.assertEqual(tool_message["tool_call_id"], "call-1")
                evidence = json.loads(tool_message["content"])
                self.assertTrue(evidence["ok"])
                self.assertEqual(evidence["data"]["generation"], meta["generation"])
                self.assertEqual(evidence["data"]["sources"][0]["citation"], "T1-C1-S1")
                return httpx.Response(200, json=completion(text="教材介绍 Python lists。[T1-C1-S1]"))

            settings = ModelSettings("https://example.invalid/v1", "fake-model", "fake-key")
            transport = httpx.Client(transport=httpx.MockTransport(handler))
            with ModelClient(settings, http_client=transport) as client, \
                 patch("app.agent.service.ModelClient", return_value=client), \
                 patch("app.agent.service.model_settings", return_value=settings), \
                 patch("app.rag.service.model_settings", side_effect=AssertionError("BM25 不需要模型")):
                service = AgentService(["教育资料"], data_dir=self.data, mode="bm25")
                result = service.chat("按教材介绍 Python lists", session_id="lesson")
            self.assertTrue(transport.is_closed)
        self.assertEqual((result.model_calls, result.tool_calls), (2, 1))
        self.assertEqual(result.sources[0]["source"], "lesson.pdf")
        self.assertEqual(result.sources[0]["page"], 1)
        self.assertNotIn("text", result.sources[0])
        history = SessionManager(self.data).load("lesson")
        self.assertEqual(history.version, 1)
        self.assertEqual([item["role"] for item in history.turns[0]], ["user", "assistant", "tool", "assistant"])
        self.assertNotIn("fake-key", json.dumps(history.turns))
        # 新轮次可以直接回答，并携带上一轮的完整工具消息。
        loop, client = self.loop([ModelReply("上一轮的材料见 [T1-C1-S1]")])
        followup = loop.run("刚才是哪一页？", session_id="lesson")
        self.assertEqual(followup.sources[0]["page"], 1)
        self.assertEqual(self.sessions.load("lesson").version, 2)
        self.assertTrue(any(message["role"] == "tool" for message in client.chat_with_tools.call_args.args[0]))
        self.assertEqual(self.sessions.load("other").turns, [])

    def test_direct_answer_and_failed_model_do_not_pollute_history(self):
        loop, _ = self.loop([ModelReply("你好"), RuntimeError("模型暂时不可用")])
        first = loop.run("你好", session_id="one")
        self.assertEqual((first.model_calls, first.tool_calls, first.sources), (1, 0, ()))
        before = self.sessions.load("one")
        with self.assertRaises(RuntimeError):
            loop.run("第二问", session_id="one")
        self.assertEqual(self.sessions.load("one"), before)

    def test_agent_saves_durable_name_when_model_calls_memory_tool(self):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)
        arguments = json.dumps({"action": "remember", "key": "姓名", "content": "吴昂展"}, ensure_ascii=False)
        client.chat_with_tools.side_effect = [
            ModelReply(tool_calls=(ToolCall("memory-1", "manage_long_term_memory", arguments),)),
            ModelReply("已记住你的名字。"),
        ]
        with patch("app.agent.service.ModelClient", return_value=client), \
             patch("app.agent.service.model_settings", return_value=ModelSettings(
                 "https://example.invalid/v1", "model", "test-key")):
            service = AgentService(["教育资料"], data_dir=self.data, mode="bm25")
            service.chat("我叫吴昂展", session_id="remember-name")

        self.assertEqual(service.memory.list(), [{"key": "姓名", "content": "吴昂展"}])
        self.assertIn("长期记忆技能", client.chat_with_tools.call_args_list[0].args[0][0]["content"])
        tool_result = json.loads(client.chat_with_tools.call_args_list[1].args[0][-1]["content"])
        self.assertEqual(tool_result, {"ok": True, "data": {"saved": True, "key": "姓名"}})

    def test_registry_rejects_invalid_unknown_and_disallowed_inputs(self):
        with self.assertRaises(ValueError):
            self.registry.register(KnowledgeSearchTool(["教育资料"]))
        for arguments in ["{broken", "[]", "null", '{"query":NaN}', "x" * 8001]:
            with self.subTest(arguments=arguments[:20]):
                self.assertEqual(self.registry.execute("knowledge_search", arguments)["error"], "invalid_arguments")
        self.assertEqual(self.registry.execute("unknown", "{}") ["error"], "unknown_tool")
        for arguments in [{"query": "x", "knowledgebase": "私人资料"},
                          {"query": "x", "knowledgebase": "教育资料", "top_k": True},
                          {"query": "x", "knowledgebase": "教育资料", "unexpected": "private-key"}]:
            result = self.registry.execute("knowledge_search", json.dumps(arguments))
            self.assertFalse(result["ok"])
            self.assertNotIn("private-key", json.dumps(result))

    def test_tool_failure_is_feedback_and_can_be_answered(self):
        call = ToolCall("call-1", "knowledge_search", function_call()["function"]["arguments"])
        loop, client = self.loop([ModelReply(tool_calls=(call,)), ModelReply("知识查询失败，请稍后重试。")])
        with patch("app.tools.knowledge.knowledge_search", side_effect=RuntimeError("private-api-key")):
            result = loop.run("查教材")
        payload = json.loads(client.chat_with_tools.call_args.args[0][-1]["content"])
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertNotIn("private-api-key", json.dumps(self.sessions.load("default").turns))
        self.assertIn("失败", result.answer)

    def test_open_first_search_result_fetches_it_before_model_answer(self):
        search_turn = [
            {"role": "user", "content": "搜索资料"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "search-1", "type": "function", "function": {
                    "name": "web_search", "arguments": "{}",
                }}]},
            {"role": "tool", "tool_call_id": "search-1", "content": json.dumps({
                "ok": True, "data": {"results": [
                    {"title": "第一条", "url": "https://example.com/first", "summary": "摘要"},
                    {"title": "第二条", "url": "https://example.com/second", "summary": "摘要"},
                ]}})},
            {"role": "assistant", "content": "找到两条链接。"},
        ]
        self.sessions.append_turn("open-link", search_turn, expected_version=0)

        class FetchTool:
            name = "web_fetch"

            def to_definition(self):
                return {"type": "function", "function": {"name": self.name, "parameters": {}}}

            def execute(self, url):
                return {"url": url, "title": "第一条", "content": "页面正文"}

        tool = FetchTool()
        self.registry.register(tool)
        loop, client = self.loop([ModelReply("第一个链接已读取，内容是页面正文。")])
        statuses = []
        result = loop.run("第一个链接帮我打开", session_id="open-link", on_status=statuses.append)

        sent = client.chat_with_tools.call_args.args[0]
        fetch_call = next(call for message in sent for call in message.get("tool_calls", [])
                          if call["function"]["name"] == "web_fetch")
        self.assertEqual(json.loads(fetch_call["function"]["arguments"]),
                         {"url": "https://example.com/first"})
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual([status["state"] for status in statuses], ["running", "finished"])
        self.assertIn("页面正文", json.loads(sent[-1]["content"])["data"]["content"])

    def test_empty_search_is_success_with_zero_sources(self):
        call = ToolCall("call-1", "knowledge_search", function_call()["function"]["arguments"])
        loop, client = self.loop([ModelReply(tool_calls=(call,)), ModelReply("资料中没有匹配内容。")])
        with patch("app.tools.knowledge.knowledge_search", return_value={
                "knowledgebase": "教育资料", "generation": "g", "count": 0, "sources": []}):
            result = loop.run("查教材")
        payload = json.loads(client.chat_with_tools.call_args.args[0][-1]["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(result.sources, ())

    def test_two_knowledge_calls_keep_distinct_citations(self):
        calls = tuple(ToolCall(f"call-{i}", "knowledge_search", function_call(query=query)["function"]["arguments"])
                      for i, query in enumerate(("列表", "字典"), 1))
        loop, client = self.loop([ModelReply(tool_calls=calls), ModelReply("比较见 [T1-C1-S1] 和 [T1-C2-S1]。")])

        def evidence(query, *args, **kwargs):
            return {"knowledgebase": "教育资料", "generation": "g", "count": 1,
                    "sources": [{"citation": "S1", "chunk_id": query, "source": "lesson.pdf",
                                 "page": 3, "text": query, "score": 1.0, "score_type": "bm25"}]}

        with patch("app.tools.knowledge.knowledge_search", side_effect=evidence):
            result = loop.run("比较列表和字典")
        self.assertEqual([source["citation"] for source in result.sources], ["T1-C1-S1", "T1-C2-S1"])
        messages = client.chat_with_tools.call_args.args[0]
        self.assertEqual([message["tool_call_id"] for message in messages if message["role"] == "tool"],
                         ["call-1", "call-2"])
        self.assertEqual(len(messages[-3]["tool_calls"]), 2)

    def test_repeated_calls_and_iteration_limit_rollback_whole_turn(self):
        replies = [ModelReply(tool_calls=(ToolCall(f"call-{i}", "unknown", '{"a":1}'),)) for i in range(3)]
        loop, client = self.loop(replies)
        with patch.object(self.registry, "execute", wraps=self.registry.execute) as execute:
            with self.assertRaisesRegex(RuntimeError, "反复调用"):
                loop.run("查资料")
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(json.loads(client.chat_with_tools.call_args.args[0][-1]["content"])["error"], "repeated_call")
        self.assertEqual(self.sessions.load("default").turns, [])
        loop, _ = self.loop(replies, max_iterations=1)
        with patch.object(self.registry, "execute") as execute, self.assertRaisesRegex(RuntimeError, "迭代上限"):
            loop.run("查资料")
        execute.assert_not_called()
        self.assertEqual(self.sessions.load("default").turns, [])

    def test_duplicate_call_ids_fail_before_tools_execute(self):
        call = ToolCall("duplicate", "unknown", "{}")
        loop, _ = self.loop([ModelReply(tool_calls=(call, call))])
        with patch.object(self.registry, "execute") as execute, self.assertRaisesRegex(RuntimeError, "ID"):
            loop.run("查资料")
        execute.assert_not_called()
        self.assertEqual(self.sessions.load("default").turns, [])

    def test_context_preserves_complete_turns_and_current_chain(self):
        history = [[{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "x" * 5000}]]
        current = [{"role": "user", "content": "新问题"},
                   {"role": "assistant", "content": None, "tool_calls": [function_call()]},
                   {"role": "tool", "tool_call_id": "call-1", "content": "{}"}]
        before = deepcopy(history)
        messages = ContextBuilder(["教育资料"]).build_messages(history, current)
        self.assertEqual(messages[1:], history[0] + current)
        messages[1]["content"] = "修改返回副本"
        self.assertEqual(history, before)

    def test_session_version_conflicts_and_recent_complete_turns(self):
        snapshot = self.sessions.load("one")
        messages = [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]
        self.sessions.append_turn("one", messages, expected_version=snapshot.version)
        with self.assertRaisesRegex(RuntimeError, "其他请求"):
            self.sessions.append_turn("one", messages, expected_version=snapshot.version)
        for version in range(1, 8):
            self.sessions.append_turn("one", messages, expected_version=version)
        restored = SessionManager(self.data).load("one")
        self.assertEqual((restored.version, len(restored.turns)), (8, 8))
        with self.assertRaises(ValueError):
            self.sessions.load("../escape")

    def test_sdk_rejects_incomplete_tool_responses_and_keeps_reasoning_field(self):
        settings = ModelSettings("https://example.invalid/v1", "model", "fake-key")
        payload = completion(calls=[function_call()])
        payload["choices"][0]["message"]["reasoning_content"] = "opaque-provider-state"
        with ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)))) as client:
            reply = client.chat_with_tools([], self.registry.get_definitions())
            self.assertEqual(reply.to_message()["reasoning_content"], "opaque-provider-state")
        for payload in [completion(text="partial", finish="length"), completion(finish="tool_calls"),
                        completion(text=" "), {"choices": []}]:
            with self.subTest(payload=payload), ModelClient(settings, http_client=httpx.Client(
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))) as client:
                with self.assertRaises(RuntimeError):
                    client.chat_with_tools([], self.registry.get_definitions())

    def test_agent_cli_help_and_invalid_service_input_need_no_credentials(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "app.agent", "--help"], cwd=root,
                                capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--session", result.stdout)
        with patch("app.agent.service.model_settings") as settings:
            service = AgentService(["教育资料"], data_dir=self.data)
            for message, session_id in [(" ", "one"), ("x", "../bad")]:
                with self.assertRaises(ValueError):
                    service.chat(message, session_id=session_id)
            settings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
