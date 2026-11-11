"""Offline checks for the curated retrieval benchmark and its ranking metrics."""
import unittest
from app.rag.domain.models import Chunk, SearchResult
from app.rag.retrieval.context import expand_neighbor_chunks
from scripts.generate_retrieval_eval import generate, parse_questions, select_anchors
from scripts.evaluate_retrieval import DEFAULT_DATASET, evaluate, load_cases


class RetrievalEvaluationTests(unittest.TestCase):
    def test_curated_dataset_has_unique_cases_and_sources(self):
        cases = load_cases(DEFAULT_DATASET)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({item["id"] for item in cases}), len(cases))
        self.assertTrue(all(item["relevant"] for item in cases))

    def test_metrics_reward_relevant_pages_ranked_at_top(self):
        cases = [{"id": "a", "query": "q", "relevant": [
            {"source": "book.pdf", "page": 1}, {"source": "book.pdf", "page": 2}]}]

        class FakeIndex:
            def search(self, *_args, **_kwargs):
                return [
                    SearchResult(Chunk("x", "book.pdf", 2, "relevant"), 0.9),
                    SearchResult(Chunk("y", "other.pdf", 9, "noise"), 0.8),
                    SearchResult(Chunk("z", "book.pdf", 1, "relevant"), 0.7),
                ]

        report = evaluate(FakeIndex(), cases, mode="bm25", top_k=3, candidate_k=3,
                          rrf_k=60, vector_weight=0.5, keyword_weight=0.5)
        metrics = report["metrics"]
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["hit_rate@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertLess(metrics["ndcg_at_k"], 1.0)

    def test_chunk_label_does_not_count_another_chunk_from_same_page(self):
        cases = [{"id": "chunk", "query": "q", "relevant": [
            {"source": "book.pdf", "page": 1, "chunk_id": "gold-chunk"}]}]

        class FakeIndex:
            def search(self, *_args, **_kwargs):
                return [SearchResult(Chunk("wrong-chunk", "book.pdf", 1, "same page, wrong chunk"), 0.9)]

        report = evaluate(FakeIndex(), cases, mode="bm25", top_k=1, candidate_k=1,
                          rrf_k=60, vector_weight=0.5, keyword_weight=0.5)
        self.assertEqual(report["metrics"]["hit_rate@1"], 0.0)
        self.assertFalse(report["cases"][0]["retrieved"][0]["relevant"])

    def test_neighbor_expansion_stays_within_source_and_page(self):
        chunks = [Chunk("a", "book.pdf", 1, "before", offset=0),
                  Chunk("b", "book.pdf", 1, "anchor", offset=100),
                  Chunk("c", "book.pdf", 1, "after", offset=200),
                  Chunk("d", "book.pdf", 2, "next page", offset=0),
                  Chunk("e", "other.pdf", 1, "other source", offset=0)]
        hits = [SearchResult(chunks[1], 1.0)]

        expanded = expand_neighbor_chunks(hits, chunks, window=1)

        self.assertEqual([chunk.id for chunk in expanded], ["a", "b", "c"])

    def test_neighbor_expansion_follows_explicit_links_across_pages(self):
        previous = Chunk("a", "book.pdf", 1, "previous", offset=0, next_chunk_id="b")
        anchor = Chunk("b", "book.pdf", 2, "anchor", offset=0,
                       prev_chunk_id="a", next_chunk_id="c")
        following = Chunk("c", "book.pdf", 2, "following", offset=100, prev_chunk_id="b")

        expanded = expand_neighbor_chunks([SearchResult(anchor, 1.0)],
                                          [previous, anchor, following], window=1)

        self.assertEqual([chunk.id for chunk in expanded], ["a", "b", "c"])

    def test_page_context_can_include_adjacent_page_without_crossing_source(self):
        chunks = [Chunk("a", "book.pdf", 1, "previous page", offset=0),
                  Chunk("b", "book.pdf", 2, "anchor page", offset=0),
                  Chunk("c", "book.pdf", 3, "next page", offset=0),
                  Chunk("d", "other.pdf", 1, "other source", offset=0)]
        hits = [SearchResult(chunks[1], 1.0)]

        expanded = expand_neighbor_chunks(hits, chunks, window=0, page_window=1)

        self.assertEqual([chunk.id for chunk in expanded], ["a", "b", "c"])

    def test_context_coverage_is_reported_separately_from_exact_retrieval(self):
        gold = Chunk("gold", "book.pdf", 1, "missing answer", offset=100)
        anchor = Chunk("anchor", "book.pdf", 1, "retrieved context", offset=0)
        cases = [{"id": "neighbor", "query": "q", "relevant": [
            {"source": "book.pdf", "page": 1, "chunk_id": "gold"}]}]

        class FakeIndex:
            chunks = [anchor, gold]

            def search(self, *_args, **_kwargs):
                return [SearchResult(anchor, 0.9)]

        report = evaluate(FakeIndex(), cases, mode="bm25", top_k=1, candidate_k=1,
                          rrf_k=60, vector_weight=0.5, keyword_weight=0.5,
                          context_window=1)

        self.assertEqual(report["metrics"]["hit_rate@1"], 0.0)
        self.assertEqual(report["metrics"]["context_hit_rate@1"], 1.0)
        self.assertEqual(report["cases"][0]["expanded_context_chunk_ids"], ["anchor", "gold"])

    def test_adjacent_page_context_reports_coverage_and_size(self):
        gold = Chunk("gold", "book.pdf", 1, "missing answer", offset=0)
        anchor = Chunk("anchor", "book.pdf", 2, "retrieved context", offset=0)
        cases = [{"id": "page-neighbor", "query": "q", "relevant": [
            {"source": "book.pdf", "page": 1, "chunk_id": "gold"}]}]

        class FakeIndex:
            chunks = [gold, anchor]

            def search(self, *_args, **_kwargs):
                return [SearchResult(anchor, 0.9)]

        report = evaluate(FakeIndex(), cases, mode="bm25", top_k=1, candidate_k=1,
                          rrf_k=60, vector_weight=0.5, keyword_weight=0.5,
                          page_window=1)

        self.assertEqual(report["metrics"]["hit_rate@1"], 0.0)
        self.assertEqual(report["metrics"]["page_context_hit_rate@1"], 1.0)
        self.assertEqual(report["metrics"]["page_context_chunks@1"], 2.0)
        self.assertEqual(report["metrics"]["page_context_chars@1"], 31.0)

    def test_anchor_sampling_spreads_cases_across_sources_and_pages(self):
        chunks = [Chunk(f"{source}-{page}", source, page, "说明" * 60)
                  for source in ("a.pdf", "b.pdf") for page in range(1, 11)]
        anchors = select_anchors(chunks, count=8)
        self.assertEqual(len({chunk.source for chunk in anchors}), 2)
        for source in ("a.pdf", "b.pdf"):
            pages = [chunk.page for chunk in anchors if chunk.source == source]
            self.assertEqual(len(pages), 4)
            self.assertGreaterEqual(len(set(pages)), 4)

    def test_question_parser_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "重复或未知"):
            parse_questions('[{"id":"q1","query":"这是一个有意义的检索问题"},'
                            '{"id":"q1","query":"这是另一个有效的检索问题"}]', {"q1", "q2"})

    def test_question_parser_marks_missing_batch_item_for_anchor_fallback(self):
        parsed = parse_questions('[{"id":"q1","query":"这是锚点一对应的问题"}]', {"q1", "q2"})
        self.assertEqual(parsed["q1"], "这是锚点一对应的问题")
        self.assertIsNone(parsed["q2"])

    def test_generator_keeps_source_page_and_chunk_provenance(self):
        chunks = [Chunk(f"chunk-{i}", f"book-{i % 2}.pdf", i + 1, "教材说明内容" * 30)
                  for i in range(4)]

        class FakeClient:
            def chat(self, messages):
                payload = __import__("json").loads(messages[1]["content"])
                return __import__("json").dumps([
                    {"id": item["id"], "query": f"围绕教材概念{item['id']}应当理解哪些关键内容？"}
                    for item in payload], ensure_ascii=False)

        cases = generate(type("Index", (), {"chunks": chunks})(), FakeClient(), count=4, batch_size=2)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0]["anchor_chunk_id"], "chunk-0")
        self.assertEqual(cases[0]["relevant"][0]["source"], "book-0.pdf")


if __name__ == "__main__":
    unittest.main()
