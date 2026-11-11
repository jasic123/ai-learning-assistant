"""混合召回、RRF 和精排离线回归；不使用真实密钥或模型网络。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from app import config
from app.config import ModelSettings
from app.rag import cli
from app.rag.domain.models import Chunk, SearchResult
from app.rag.indexing.pipeline import build_index
from app.rag.retrieval.bm25 import BM25Retriever, tokenize, query_terms
from app.rag.retrieval.fusion import rrf_fuse
from app.rag.retrieval.reranker import RerankerService
from app.rag.retrieval.retrievers import KnowledgeIndex
from app.llm.client import ModelClient


def chunk(identifier, text):
    return Chunk(identifier, "教材.pdf", 1, text)


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_chinese_english_and_no_match(self):
        chunks = [chunk("a", "Python 列表支持追加元素。"), chunk("b", "数据库事务支持回滚。")]
        retriever = BM25Retriever(chunks)
        for query in ("列表", "ＰＹＴＨＯＮ"):
            hits = retriever.search(query)
            self.assertEqual([hit.chunk.id for hit in hits], ["a"])
            self.assertGreater(hits[0].score, 0)
            self.assertEqual(hits[0].score_type, "bm25")
        self.assertEqual(retriever.search("unseenkeyword"), [])
        self.assertEqual(retriever.search("！？"), [])
        self.assertEqual(BM25Retriever([chunk("empty", "！？")]).search("Python"), [])
        self.assertIn("python", tokenize("ＰＹＴＨＯＮ列表"))

    def test_bm25_ignores_low_information_query_phrasing(self):
        chunks = [chunk("generic", "如何使用 Python 进行文本处理的基本方法。"),
                  chunk("specific", "Python open 文件逐行读取 readlines。")]
        hits = BM25Retriever(chunks).search("Python 怎样打开文件并逐行读取？")
        self.assertEqual(hits[0].chunk.id, "specific")

    def test_bm25_query_stopword_filter_is_optional(self):
        retriever = BM25Retriever([chunk("a", "如何使用 Python 进行文本处理。")])
        self.assertEqual(retriever.search("如何使用", filter_query_stop_words=True), [])
        self.assertEqual([hit.chunk.id for hit in retriever.search("如何使用")], ["a"])

    def test_query_expansion_adds_domain_terms(self):
        terms = query_terms("defaultdict 找零 词性标签 SQL")
        for term in ("default_factory", "denom", "tag", "pos", "load_parser"):
            self.assertIn(term, terms)

    def test_rrf_uses_ranks_deduplicates_and_keeps_inputs_immutable(self):
        a, b, c = [chunk(name, name) for name in "abc"]
        vectors = [SearchResult(a, -0.9), SearchResult(b, -0.99), SearchResult(a, 10)]
        keywords = [SearchResult(b, 1000, "bm25"), SearchResult(c, 500, "bm25")]
        hits = rrf_fuse(vectors, keywords, rrf_k=60)
        self.assertEqual([hit.chunk.id for hit in hits], ["b", "a", "c"])
        self.assertAlmostEqual(hits[0].score, 0.5 / 62 + 0.5 / 61)
        self.assertAlmostEqual(hits[1].score, 0.5 / 61)
        self.assertTrue(all(hit.score_type == "rrf" for hit in hits))
        self.assertEqual(vectors[0].score, -0.9)
        self.assertNotIn("score", a.to_dict())

    def test_rrf_weights_change_hybrid_ranking_and_validate_sum(self):
        a, b = chunk("a", "向量优先"), chunk("b", "关键词优先")
        vectors = [SearchResult(a, 0.9), SearchResult(b, 0.8)]
        keywords = [SearchResult(b, 12, "bm25"), SearchResult(a, 9, "bm25")]
        vector_favored = rrf_fuse(vectors, keywords, vector_weight=0.8, keyword_weight=0.2)
        keyword_favored = rrf_fuse(vectors, keywords, vector_weight=0.2, keyword_weight=0.8)
        self.assertEqual([hit.chunk.id for hit in vector_favored], ["a", "b"])
        self.assertEqual([hit.chunk.id for hit in keyword_favored], ["b", "a"])
        with self.assertRaises(ValueError):
            rrf_fuse(vectors, keywords, vector_weight=0.8, keyword_weight=0.8)

    def test_reranker_reorders_candidates_and_preserves_sources(self):
        candidates = [SearchResult(chunk("a", "第一条"), 0.02, "rrf"),
                      SearchResult(chunk("b", "第二条"), 0.01, "rrf")]
        client = Mock()
        client.rerank.return_value = [0.1, 0.9]
        hits = RerankerService(client).rerank("问题", candidates, top_k=1)
        client.rerank.assert_called_once_with("问题", ["第一条", "第二条"])
        self.assertIs(hits[0].chunk, candidates[1].chunk)
        self.assertEqual((hits[0].score_type, hits[0].score, hits[0].retrieval_score), ("rerank", 0.9, 0.01))
        self.assertEqual(candidates[1].score_type, "rrf")
        self.assertEqual(RerankerService(client).rerank("问题", []), [])

    def test_reranker_rejects_bad_scores_and_propagates_failures(self):
        client = Mock()
        service = RerankerService(client)
        candidates = [SearchResult(chunk("a", "正文"), 0.01, "rrf")]
        for scores in ([], [float("nan")], [float("inf")], [True], ["0.5"]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                client.rerank.return_value = scores
                service.rerank("问题", candidates)
        client.rerank.side_effect = RuntimeError("接口不可用")
        with self.assertRaisesRegex(RuntimeError, "接口不可用"):
            service.rerank("问题", candidates)

    def test_rerank_http_contract_restores_out_of_order_scores(self):
        settings = ModelSettings("https://example.invalid/v1", "test-reranker", "fake-secret")

        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(request.url.path, "/v1/rerank")
            self.assertEqual(body, dict(model="test-reranker", query="问题", documents=["甲", "乙"],
                                        top_n=2, return_documents=False))
            return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9},
                                                          {"index": 0, "relevance_score": 0.1}]})

        with ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            self.assertEqual(client.rerank("问题", ["甲", "乙"]), [0.1, 0.9])

    def test_rerank_http_rejects_invalid_mapping_and_redacts_response(self):
        settings = ModelSettings("https://example.invalid/v1", "test-reranker", "fake-secret")
        invalid = [[], [{"index": 0, "relevance_score": 0.5}] * 2,
                   [{"index": True, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.5}],
                   [{"index": 0, "relevance_score": True}, {"index": 1, "relevance_score": 0.5}],
                   [{"index": 0, "relevance_score": "response-secret"}, {"index": 1, "relevance_score": 0.5}]]
        for results in invalid:
            with self.subTest(results=results), ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"results": results})))) as client:
                with self.assertRaisesRegex(RuntimeError, "响应数量、编号或分数格式异常") as error:
                    client.rerank("问题", ["甲", "乙"])
                self.assertNotIn("response-secret", str(error.exception))

    def test_real_faiss_json_hybrid_recovers_keyword_candidate_and_reranks(self):
        # 把正确关键词放在向量召回范围之外，验证双路组合确实扩大召回，而不是只更改分数标签。
        chunks = [chunk("dense-a", "无关语义主题"), chunk("keyword", "ZQX_7788 是目标函数。"),
                  chunk("dense-b", "另一段背景内容")]
        embedding = Mock()
        embedding.settings = ModelSettings("https://example.invalid/v1", "test-embedding", "fake-secret")
        embedding.embed.side_effect = lambda texts: ([[1, 0], [0, 1], [0.8, 0.2]] if len(texts) == 3 else [[1, 0]])
        with tempfile.TemporaryDirectory() as temporary, patch.object(config, "DATA_DIR", Path(temporary)), \
             patch("app.rag.indexing.pipeline.document_chunks", return_value=(chunks, [])):
            build_index("测试", embedding)
            index = KnowledgeIndex("测试")
            data_files = [path for path in Path(temporary).rglob("*") if path.is_file()]
            before = {path: path.read_bytes() for path in data_files}
            self.assertEqual(index.search("ZQX_7788", embedding, 1, mode="vector", candidate_k=2)[0].chunk.id, "dense-a")
            hybrid = index.search("ZQX_7788", embedding, 2, candidate_k=3)
            self.assertIn("keyword", [hit.chunk.id for hit in hybrid])
            self.assertTrue(all(hit.score_type == "rrf" for hit in hybrid))
            ranking = Mock()
            ranking.rerank.side_effect = lambda query, texts: [1.0 if "ZQX_7788" in text else 0.0 for text in texts]
            hits = index.search("ZQX_7788", embedding, 1, candidate_k=2, reranker=RerankerService(ranking))
            self.assertEqual((hits[0].chunk.id, hits[0].score_type), ("keyword", "rerank"))
            embedding.embed.reset_mock()
            self.assertEqual(index.search("ZQX_7788", top_k=1, mode="bm25")[0].chunk.id, "keyword")
            embedding.embed.assert_not_called()
            self.assertEqual(index.search("notinthecorpus", mode="bm25"), [])
            self.assertEqual(before, {path: path.read_bytes() for path in data_files})

    def test_invalid_options_fail_before_embedding_request(self):
        index = object.__new__(KnowledgeIndex)
        client = Mock()
        for options in (dict(mode="unknown"), dict(top_k=True), dict(candidate_k=2), dict(rrf_k=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                index.search("问题", client, **options)
        client.embed.assert_not_called()

    def test_cli_bm25_requires_no_model_configuration(self):
        with patch.object(sys, "argv", ["rag", "search", "--kb", "测试", "--question", "问题", "--mode", "bm25"]), \
             patch.object(cli, "RagService") as factory, patch.object(cli, "model_settings") as settings, \
             patch.object(cli, "ModelClient") as client, redirect_stdout(io.StringIO()):
            factory.return_value.search.return_value.hits = ()
            self.assertEqual(cli.main(), 0)
            settings.assert_not_called()
            client.assert_not_called()
            self.assertEqual(factory.return_value.search.call_args.args[0].mode, "bm25")

    def test_cli_validates_rerank_configuration_before_model_requests(self):
        def settings(kind):
            if kind == "RERANK":
                raise ValueError("请填写 RERANK_API_KEY")
            return ModelSettings("https://example.invalid/v1", "test-model", "fake-key")

        with patch.object(sys, "argv", ["rag", "search", "--kb", "测试", "--question", "问题", "--rerank"]), \
             patch("app.rag.service.KnowledgeIndex") as factory, patch("app.rag.service.model_settings", side_effect=settings), \
             patch("app.rag.service.ModelClient") as client, patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(cli.main(), 1)
            client.assert_not_called()
            factory.return_value.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
