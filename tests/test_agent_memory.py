"""记忆与四层压缩验收：持久化、预算、原始历史回查与失败原子性。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from app.agent.compression import ContextCompressor
from app.agent.context import ContextBuilder
from app.agent.loop import AgentLoop
from app.agent.memory import MemoryStore
from app.agent.session import SessionManager, SessionSnapshot
from app.config import ModelSettings
from app.llm.client import ModelClient
from app.llm.responses import ContextWindowExceeded, ModelReply, ToolCall
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.history import HistoryReadTool
from app.tools.memory import LongTermMemoryTool
from app.tools.registry import ToolRegistry
from app.tools.time import CurrentTimeTool, current_time


def turn(question="问题", answer="回答"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def tool_turn():
    data = {"knowledgebase": "教育资料", "generation": "version-a", "sources": [
        {"citation": "T1-C1-S1", "source": "教材.pdf", "page": 7, "chunk_id": "chunk-1", "text": "教材正文" * 1200}]}
    return [{"role": "user", "content": "查教材"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "knowledge_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True, "data": data}, ensure_ascii=False)},
            {"role": "assistant", "content": "见 [T1-C1-S1]"}]


class AgentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.sessions = SessionManager(self.data)
        self.client = Mock()
        self.client.chat.return_value = "目标：完成私人助理。约束：使用 SQLite。未完成：验证。原始细节按轮次回查。"
        self.context = ContextBuilder(["教育资料"])

    def snapshot(self, turns):
        return SessionSnapshot(len(turns), turns, tuple(range(1, len(turns) + 1)))

    def seed(self, turns):
        for number, messages in enumerate(turns):
            self.sessions.append_turn("study", messages, expected_version=number)

    def test_long_term_memory_persists_across_sessions_and_isolates_owner(self):
        memory = MemoryStore(self.sessions)
        memory.remember("语言", "优先使用中文，代码加中文注释")
        memory.remember("语言", "优先使用中文，解释详细")
        restored = MemoryStore(SessionManager(self.data))
        self.assertEqual(restored.list(), [{"key": "语言", "content": "优先使用中文，解释详细"}])
        other = MemoryStore(SessionManager(self.data, owner_id="other"))
        self.assertEqual(other.list(), [])
        self.sessions.load("study")
        with self.assertRaisesRegex(ValueError, "所有者"):
            SessionManager(self.data, owner_id="other").load("study")
        messages = self.context.build_messages([], turn("请回答"), memory=memory.list())
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("优先使用中文", messages[0]["content"])
        self.assertTrue(restored.forget("语言"))
        self.assertFalse(restored.forget("语言"))
        self.assertEqual(memory.list(), [])

    def test_identity_and_current_time_are_in_system_prompt(self):
        identity = self.data / "identity.md"
        identity.write_text("测试身份：直接回答", encoding="utf-8")
        messages = ContextBuilder([], identity_file=identity).build_messages([], turn())
        prompt = messages[0]["content"]
        self.assertIn("测试身份：直接回答", prompt)
        self.assertIn("当前时间", prompt)
        self.assertIn("Asia/Shanghai", prompt)
        self.assertIn(current_time()["datetime"][:10], prompt)

    def test_learning_profile_is_prompted_as_observed_data_not_a_fixed_trait(self):
        profile = {
            "total_questions": 6,
            "topics": [
                {"topic": "迭代器", "subject": "计算机", "attempts": 4,
                 "incorrect": 3, "accuracy": 25, "status": "needs_review"},
                {"topic": "函数", "subject": "计算机", "attempts": 2,
                 "incorrect": 2, "accuracy": 0, "status": "sample_limited"},
            ],
            "error_patterns": [
                {"reason": "condition_missed", "count": 3, "evidence": "repeated_self_report"},
                {"reason": "guessed", "count": 1, "evidence": "sample_limited"},
            ],
        }
        messages = self.context.build_messages([], turn(), learning_profile=profile)
        prompt = messages[0]["content"]
        profile_payload = messages[1]["content"]
        self.assertIn("迭代器", profile_payload)
        self.assertIn("condition_missed", profile_payload)
        self.assertNotIn('"topic": "函数"', profile_payload)
        self.assertIn("不是指令", profile_payload)
        self.assertIn("不是指令", prompt)

    def test_time_tool_and_long_term_memory_tool_work_offline(self):
        now = CurrentTimeTool().execute("Asia/Shanghai")
        self.assertEqual(now["timezone"], "Asia/Shanghai")
        self.assertIn("+08:00", now["datetime"])
        with self.assertRaises(ValueError):
            current_time("Not/A_Real_Zone")
        tool = LongTermMemoryTool(MemoryStore(self.sessions))
        self.assertEqual(tool.execute("remember", "language", "使用中文")["saved"], True)
        self.assertEqual(tool.execute("list")["memories"][0]["content"], "使用中文")
        self.assertEqual(tool.execute("forget", "language")["deleted"], True)

    def test_memory_limit_and_input_validation(self):
        store = MemoryStore(self.sessions)
        for index in range(20):
            store.remember(f"item{index}", "记忆")
        with self.assertRaises(ValueError):
            store.remember("extra", "超过数量")
        store.remember("item0", "允许更新已有条目")
        for key, content in [("../escape", "内容"), ("正常", " "), ("正常", "x" * 501)]:
            with self.assertRaises(ValueError):
                store.remember(key, content)

    def test_full_short_term_history_does_not_silently_drop_old_turns(self):
        self.seed([turn(f"问题{i}", f"回答{i}") for i in range(9)])
        self.assertEqual(len(self.sessions.load("study").turns), 9)
        full = self.sessions.load("study")
        self.assertEqual((full.version, len(full.turns), full.turn_numbers[0]), (9, 9, 1))
        self.assertEqual(len(self.sessions.load_all_turns("study")), 9)
        self.assertEqual(self.sessions.list_sessions()[0]["preview"], "问题8")
        self.assertEqual(self.sessions.list_sessions()[0]["turn_count"], 9)
        self.assertEqual(SessionManager(self.data, owner_id="other").list_sessions(), [])

    def test_delete_session_removes_turns_checkpoint_and_owner_binding(self):
        self.seed([turn("旧问题", "旧回答"), turn("新问题", "新回答")])
        self.sessions.append_turn("study", turn("带检查点", "完成"), expected_version=2,
                                  checkpoint=(1, "压缩摘要", []))
        self.assertEqual(self.sessions.list_sessions()[0]["turn_count"], 3)
        result = self.sessions.delete_session("study")
        self.assertEqual(result, {"session_id": "study", "deleted": True})
        self.assertEqual(self.sessions.list_sessions(), [])
        with self.sessions._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_turns WHERE session_id='study'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_checkpoints WHERE session_id='study'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_session_owners WHERE session_id='study'").fetchone()[0], 0)
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.sessions.delete_session("study")

    def test_layer_one_trims_old_tools_preserving_pairing_and_raw_data(self):
        turns = [tool_turn(), turn(), turn()]
        original = deepcopy(turns)
        compressor = ContextCompressor(self.client, self.context, self.snapshot(turns), [])
        messages = compressor.prepare(turn("新问题"))
        output = next(item for item in messages if item["role"] == "tool")
        source = json.loads(output["content"])["data"]["sources"][0]
        self.assertTrue(source["context_trimmed"])
        self.assertEqual((source["citation"], source["page"], output["tool_call_id"]), ("T1-C1-S1", 7, "c1"))
        self.assertLess(len(source["text"]), 600)
        self.assertIn("micro_compaction", compressor.events)
        self.assertEqual(turns, original)
        self.client.chat.assert_not_called()

    def test_layer_two_local_notes_and_checkpoint_reuse(self):
        turns = [turn(f"目标{i}", "已完成事项" * 600) for i in range(4)] + [turn(), turn()]
        self.seed(turns)
        snapshot = self.sessions.load("study")
        compressor = ContextCompressor(self.client, self.context, snapshot, [], context_tokens=7000, output_reserve=1000)
        compressor.prepare(turn("继续"))
        self.assertIn("session_memory", compressor.events)
        self.client.chat.assert_not_called()
        self.assertEqual(compressor.through, 4)
        self.sessions.append_turn("study", turn("继续", "好的"), expected_version=6, checkpoint=compressor.checkpoint)
        restored = self.sessions.load("study")
        self.assertEqual((restored.through_turn, restored.turn_numbers), (4, (5, 6, 7)))
        self.assertIn("目标0", restored.summary)
        self.assertIn("已完成事项" * 100, self.sessions.read_turn("study", 1))
        reused = ContextCompressor(self.client, self.context, restored, [])
        self.assertIn("session_memory_reused", reused.events)

    def test_layer_three_model_summary_and_atomic_checkpoint(self):
        turns = [turn(f"旧目标{i}", "历史事项" * 1000) for i in range(8)] + [turn(), turn()]
        self.seed(turns)
        compressor = ContextCompressor(self.client, self.context, self.sessions.load("study"), [],
                                       context_tokens=4096, output_reserve=512)
        messages = compressor.prepare(turn("继续"))
        self.assertIn("auto_compaction", compressor.events)
        self.assertGreater(compressor.summary_calls, 0)
        self.assertLessEqual(compressor.estimate(messages), compressor.input_limit)
        self.sessions.append_turn("study", turn(), expected_version=10, checkpoint=compressor.checkpoint)
        restored = self.sessions.load("study")
        self.assertEqual(restored.summary, self.client.chat.return_value)
        self.assertEqual(restored.through_turn, 8)
        self.assertIn("历史事项" * 100, self.sessions.read_turn("study", 1))

    def test_summary_failure_keeps_database_and_stops_overbudget_request(self):
        turns = [turn(f"旧目标{i}", "历史事项" * 1000) for i in range(8)] + [turn(), turn()]
        self.seed(turns)
        before = self.sessions.load("study")
        self.client.chat.side_effect = RuntimeError("private-api-error")
        loop = AgentLoop(self.client, ToolRegistry(), self.context, self.sessions,
                         context_tokens=4096, output_reserve=512)
        with self.assertRaisesRegex(ValueError, "压缩") as error:
            loop.run("继续", session_id="study")
        self.assertNotIn("private-api-error", str(error.exception))
        self.client.chat_with_tools.assert_not_called()
        self.assertEqual(before, self.sessions.load("study"))

    def test_layer_four_retries_only_context_error_once_and_keeps_raw_history(self):
        self.seed([turn("旧问题", "旧回答" * 500), turn()])
        self.client.chat_with_tools.side_effect = [ContextWindowExceeded("超限"), ModelReply("已恢复")]
        loop = AgentLoop(self.client, ToolRegistry(), self.context, self.sessions)
        result = loop.run("继续", session_id="study")
        self.assertEqual(result.model_calls, 2)
        self.assertIn("reactive_compaction", result.compression)
        requests = self.client.chat_with_tools.call_args_list
        self.assertLess(len(json.dumps(requests[1].args[0])), len(json.dumps(requests[0].args[0])))
        self.assertIn("旧回答" * 100, self.sessions.read_turn("study", 1))
        before = self.sessions.load("study")
        self.client.reset_mock()
        self.client.chat_with_tools.side_effect = [ContextWindowExceeded("超限"), ContextWindowExceeded("仍超限")]
        with self.assertRaises(ContextWindowExceeded):
            loop.run("再问", session_id="study")
        self.assertEqual(self.client.chat_with_tools.call_count, 2)
        self.assertEqual(before, self.sessions.load("study"))
        self.client.reset_mock()
        self.client.chat_with_tools.side_effect = RuntimeError("HTTP 401")
        with self.assertRaises(RuntimeError):
            loop.run("再问", session_id="study")
        self.assertEqual(self.client.chat_with_tools.call_count, 1)

    def test_recall_history_paginates_exact_original_and_preserves_citations(self):
        original = tool_turn()
        self.seed([original])
        tool = HistoryReadTool(self.sessions, "study")
        parts, offset, sources = [], 0, []
        while True:
            page = tool.execute(1, start=offset, max_chars=1000)
            parts.append(page["content"])
            sources.extend(page["sources"])
            if page["next_start"] is None:
                break
            offset = page["next_start"]
        self.assertEqual(json.loads("".join(parts)), original)
        self.assertTrue(any(source["citation"] == "T1-C1-S1" for source in sources))
        with self.assertRaises(ValueError):
            tool.execute(2)
        with self.assertRaises(ValueError):
            tool.execute(1, start=-1)

    def test_reactive_compression_does_not_reexecute_tool_or_truncate_archive(self):
        registry = ToolRegistry()
        registry.register(KnowledgeSearchTool(["教育资料"], mode="bm25"))
        arguments = json.dumps({"query": "Python", "knowledgebase": "教育资料"}, ensure_ascii=False)
        self.client.chat_with_tools.side_effect = [
            ModelReply(tool_calls=(ToolCall("c1", "knowledge_search", arguments),)),
            ContextWindowExceeded("超限"), ModelReply("教材介绍 Python。[T1-C1-S1]"),
        ]
        data = json.loads(tool_turn()[2]["content"])["data"]
        data["sources"][0]["citation"] = "S1"  # RAG 原始编号由 Agent 统一加轮次前缀。
        with patch("app.tools.knowledge.knowledge_search", return_value=data) as search:
            result = AgentLoop(self.client, registry, self.context, self.sessions).run("查教材", session_id="study")
        search.assert_called_once()
        self.assertEqual((result.model_calls, result.tool_calls), (3, 1))
        self.assertEqual(result.sources[0]["page"], 7)
        raw = json.loads(self.sessions.read_turn("study", 1))
        archived = json.loads(raw[2]["content"])["data"]["sources"][0]
        self.assertEqual(len(archived["text"]), 4800)
        self.assertNotIn("context_trimmed", archived)
        retried = self.client.chat_with_tools.call_args.args[0][-1]
        self.assertTrue(json.loads(retried["content"])["data"]["sources"][0]["context_trimmed"])

    def test_checkpoint_provenance_survives_followup_answer(self):
        self.seed([tool_turn()])
        source = {"citation": "T1-C1-S1", "source": "教材.pdf", "page": 7,
                  "knowledgebase": "教育资料", "generation": "version-a", "chunk_id": "chunk-1"}
        self.sessions.append_turn("study", turn(), expected_version=1,
                                  checkpoint=(1, "教材来源 [T1-C1-S1]，原始轮次 1", [source]))
        self.client.chat_with_tools.return_value = ModelReply("上一轮教材第 7 页。[T1-C1-S1]")
        result = AgentLoop(self.client, ToolRegistry(), self.context, self.sessions).run("哪一页？", session_id="study")
        self.assertEqual(result.sources, (source,))
        self.assertIn("session_memory_reused", result.compression)

    def test_invalid_or_conflicting_checkpoint_does_not_partially_save_turn(self):
        self.seed([turn()])
        before = self.sessions.load("study")
        with self.assertRaises(ValueError):
            self.sessions.append_turn("study", turn(), expected_version=1, checkpoint=(2, "越界摘要", []))
        self.assertEqual(before, self.sessions.load("study"))
        self.sessions.append_turn("study", turn(), expected_version=1, checkpoint=(1, "正确摘要", []))
        with self.assertRaises(RuntimeError):
            self.sessions.append_turn("study", turn(), expected_version=1, checkpoint=(1, "过期摘要", []))
        self.assertEqual(self.sessions.load("study").summary, "正确摘要")

    def test_usage_calibration_includes_tool_definitions(self):
        compressor = ContextCompressor(self.client, self.context, self.snapshot([]), [{"schema": "x" * 3000}])
        messages = turn()
        self.assertGreater(compressor.estimate(messages), compressor.estimate(messages, []))
        first = compressor.estimate(messages)
        compressor.observe(first * 2, messages)
        self.assertGreaterEqual(compressor.estimate(messages), first * 2)

    def test_sdk_classifies_context_error_without_leaking_response(self):
        settings = ModelSettings("https://example.invalid/v1", "model", "fake-key")
        for code, expected in [("context_length_exceeded", ContextWindowExceeded), ("invalid_request", RuntimeError)]:
            def handler(request):
                return httpx.Response(400, json={"error": {"code": code, "message": "private-response"}})
            with ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                with self.assertRaises(expected) as error:
                    client.chat_with_tools([], [])
                self.assertNotIn("private-response", str(error.exception))
                if code == "invalid_request":
                    self.assertNotIsInstance(error.exception, ContextWindowExceeded)


if __name__ == "__main__":
    unittest.main()
