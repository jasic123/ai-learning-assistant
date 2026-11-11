import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_approved_eval import export


class ExportApprovedEvalTests(unittest.TestCase):
    def test_exports_only_approved_rows_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "review.jsonl"
            output = root / "approved.jsonl"
            source.write_text("\n".join([
                json.dumps({"id": "q1", "query": "问题一", "source": "a.pdf", "page": 1,
                            "chunk_id": "c1", "auto_verdict": "approved", "auto_reason": "正文支持"}),
                json.dumps({"id": "q2", "query": "问题二", "source": "b.pdf", "page": 2,
                            "chunk_id": "c2", "auto_verdict": "rejected"}),
            ]), encoding="utf-8")
            cases = export(source, output)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["relevant"][0]["chunk_id"], "c1")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["id"], "q1")


if __name__ == "__main__":
    unittest.main()
