"""Agent 接入边界：公共契约、只检索工具、配置选择及模型资源清理。"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app import config
from app.config import ModelSettings
from app.rag import RagService, SearchRequest, SearchResponse
from app.rag.domain.models import Chunk, SearchResult
from app.rag.indexing.pipeline import build_index
from app.tools.knowledge import knowledge_search
from test_rag import FakeModel, make_pdf


class RagServiceTests(unittest.TestCase):
    def test_invalid_requests_fail_before_io(self):
        cases = [dict(knowledgebase="../outside"), dict(query=" "), dict(query=None),
                 dict(top_k=True), dict(top_k=21), dict(mode=[]), dict(mode="invalid"),
                 dict(candidate_k=2), dict(rrf_k=0), dict(rerank="true"),
                 dict(vector_weight=0.8, keyword_weight=0.8), dict(vector_weight=float("nan"))]
        with patch("app.rag.service.KnowledgeIndex") as index, patch("app.rag.service.ModelClient") as client:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    SearchRequest(**{"knowledgebase": "测试", "query": "问题", **options})
            with self.assertRaises(TypeError):
                RagService().search({"query": "问题"})
            index.assert_not_called()
            client.assert_not_called()

    def test_real_pdf_to_agent_tool_is_json_serializable_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(config, "DATA_DIR", Path(temporary)):
            uploads = Path(temporary) / "uploads/教育资料"
            uploads.mkdir(parents=True)
            make_pdf(uploads / "lesson.pdf")
            meta = build_index("教育资料", FakeModel())
            before = {path: path.read_bytes() for path in Path(temporary).rglob("*") if path.is_file()}
            with patch("app.rag.service.model_settings") as settings, patch("app.rag.service.ModelClient") as client:
                result = knowledge_search("Python lists", "教育资料", top_k=1, mode="bm25")
                settings.assert_not_called()
                client.assert_not_called()
            self.assertEqual((result["knowledgebase"], result["generation"], result["count"]),
                             ("教育资料", meta["generation"], 1))
            source = result["sources"][0]
            self.assertEqual((source["citation"], source["source"], source["page"]), ("S1", "lesson.pdf", 1))
            self.assertEqual(source["score_type"], "bm25")
            self.assertIn("Python lists", source["text"])
            self.assertTrue(source["chunk_id"])
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("not-a-real-key", serialized)
            self.assertNotIn("answer", result)
            self.assertEqual(before, {path: path.read_bytes() for path in Path(temporary).rglob("*") if path.is_file()})

    def test_service_wires_hybrid_and_rerank_then_closes_clients(self):
        hit = SearchResult(Chunk("id1", "lesson.pdf", 3, "教材正文"), 0.9, "rerank", 0.01)
        settings = {kind: ModelSettings("https://example.invalid/v1", kind, "test-key")
                    for kind in ("EMBEDDING", "RERANK")}
        with patch("app.rag.service.KnowledgeIndex") as indexes, \
             patch("app.rag.service.model_settings", side_effect=lambda kind: settings[kind]) as load, \
             patch("app.rag.service.ModelClient") as clients, patch("app.rag.service.RerankerService") as rerankers:
            indexes.return_value.metadata = {"generation": "a" * 32}
            indexes.return_value.search.return_value = [hit]
            embedding, ranking = MagicMock(), MagicMock()
            clients.side_effect = [embedding, ranking]
            request = SearchRequest("测试", "问题", candidate_k=40, rrf_k=50,
                                    vector_weight=0.7, keyword_weight=0.3, rerank=True)
            result = RagService().search(request)
            self.assertEqual([call.args[0] for call in load.call_args_list], ["EMBEDDING", "RERANK"])
            indexes.return_value.search.assert_called_once_with(
                "问题", embedding.__enter__.return_value, 4, mode="hybrid", candidate_k=40,
                rrf_k=50, vector_weight=0.7, keyword_weight=0.3,
                reranker=rerankers.return_value)
            embedding.__exit__.assert_called_once()
            ranking.__exit__.assert_called_once()
            self.assertIsInstance(result, SearchResponse)
            self.assertEqual(result.hits, (hit,))
            self.assertEqual(result.to_dict()["sources"][0]["retrieval_score"], 0.01)

    def test_service_closes_clients_and_propagates_search_failure(self):
        with patch("app.rag.service.KnowledgeIndex") as indexes, \
             patch("app.rag.service.model_settings") as settings, patch("app.rag.service.ModelClient") as clients:
            indexes.return_value.search.side_effect = RuntimeError("精排失败")
            with self.assertRaisesRegex(RuntimeError, "精排失败"):
                RagService().search(SearchRequest("测试", "问题"))
            clients.return_value.__exit__.assert_called_once()
            self.assertEqual([call.args[0] for call in settings.call_args_list], ["EMBEDDING"])

    def test_empty_tool_result_does_not_invent_an_answer(self):
        with patch("app.rag.service.KnowledgeIndex") as indexes, patch("app.rag.service.ModelClient") as clients:
            indexes.return_value.metadata = {"generation": "b" * 32}
            indexes.return_value.search.return_value = []
            result = knowledge_search("没找到的词", "测试", mode="bm25")
            self.assertEqual((result["count"], result["sources"]), (0, []))
            self.assertNotIn("answer", result)
            clients.assert_not_called()

    def test_knowledge_tool_passes_configured_fusion_weights(self):
        with patch("app.rag.service.KnowledgeIndex") as indexes, \
             patch("app.rag.service.model_settings") as settings, \
             patch("app.rag.service.ModelClient") as clients:
            indexes.return_value.metadata = {"generation": "c" * 32}
            indexes.return_value.search.return_value = []
            request = SearchRequest("测试", "问题", vector_weight=0.25, keyword_weight=0.75)
            RagService().search(request)
            indexes.return_value.search.assert_called_once_with(
                "问题", clients.return_value.__enter__.return_value, 4, mode="hybrid",
                candidate_k=30, rrf_k=60, vector_weight=0.25, keyword_weight=0.75,
                reranker=None,
            )

    def test_knowledge_tool_and_shared_client_keep_dependency_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "app/rag/services/model_client.py").exists())
        targets = (root / "app/tools/knowledge.py", root / "app/llm/client.py")
        for path in targets:
            modules = []
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Import):
                    modules.extend(item.name for item in node.names)
            for module in modules:
                if path.name == "knowledge.py":
                    self.assertFalse(module.startswith(("app.rag.retrieval", "app.rag.indexing", "app.rag.db", "app.llm")))
                else:
                    self.assertFalse(module.startswith(("app.rag", "app.agent", "app.tools")))
