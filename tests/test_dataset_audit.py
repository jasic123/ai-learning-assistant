import json
import tempfile
import unittest
from pathlib import Path

from app.rag.domain.models import Chunk
from scripts.audit_retrieval_dataset import audit_case, stratified_sample, stratified_sample_total


class DatasetAuditTests(unittest.TestCase):
    def test_audit_flags_low_coverage_and_generic_question(self):
        case = {"id": "q1", "query": "这本书主要讲解什么内容？", "relevant": [
            {"source": "book.pdf", "page": 1, "chunk_id": "chunk-1"}],
            "anchor_chunk_id": "chunk-1"}
        row = audit_case(case, {"chunk-1": Chunk("chunk-1", "book.pdf", 1, "Python 列表和字典的操作")})
        self.assertIn("generic_question", row["quality_flags"])
        self.assertIn("low_term_coverage", row["quality_flags"])

    def test_stratified_sample_keeps_each_source_balanced(self):
        cases = [{"id": f"{source}-{i}", "query": "问题", "relevant": [{"source": source, "page": i}]}
                 for source in ("a.pdf", "b.pdf") for i in range(6)]
        selected = stratified_sample(cases, per_source=3)
        self.assertEqual(len(selected), 6)
        self.assertEqual({source: sum(item["relevant"][0]["source"] == source for item in selected)
                          for source in ("a.pdf", "b.pdf")}, {"a.pdf": 3, "b.pdf": 3})

    def test_total_sample_is_balanced_across_sources(self):
        cases = [{"id": f"{source}-{i}", "query": "问题", "relevant": [{"source": source, "page": i}]}
                 for source in ("a.pdf", "b.pdf", "c.pdf") for i in range(50)]
        selected = stratified_sample_total(cases, total=10)
        self.assertEqual(len(selected), 10)
        self.assertEqual({source: sum(item["relevant"][0]["source"] == source for item in selected)
                          for source in ("a.pdf", "b.pdf", "c.pdf")}, {"a.pdf": 4, "b.pdf": 3, "c.pdf": 3})


if __name__ == "__main__":
    unittest.main()
