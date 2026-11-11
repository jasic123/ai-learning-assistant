"""验证入库验收脚本本身会发现数据库记录异常；使用合成 PDF 和假向量，不访问网络。"""
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.index_document import run_pipeline
from test_rag import FakeModel, make_pdf


class PdfPipelineRunnerTests(unittest.TestCase):
    def test_checks_real_persistence_and_detects_bad_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            source = data / "lesson.pdf"
            make_pdf(source)
            with patch("app.config.DATA_DIR", data), \
                 patch("scripts.index_document.DATA_DIR", data), \
                 redirect_stdout(io.StringIO()):
                result = run_pipeline(source, FakeModel(), max_pages=0)
                self.assertTrue(result["passed"])
                self.assertEqual(result["processed_pages"], 2)
                self.assertEqual(result["chunk_count"], 2)
                self.assertTrue(Path(result["database"]).is_file())
                # 后续验收必须独立读取 SQLite，不能只信任 build_index 返回的内存元数据。
                def corrupt_before_read(*args, **kwargs):
                    from app.rag.retrieval.retrievers import KnowledgeIndex
                    index = KnowledgeIndex(*args, **kwargs)
                    with closing(sqlite3.connect(data / "assistant.db")) as connection:
                        connection.execute("UPDATE knowledge_files SET chunk_count=999")
                        connection.commit()
                    return index
                with patch("scripts.index_document.KnowledgeIndex", side_effect=corrupt_before_read):
                    with self.assertRaisesRegex(AssertionError, "文件入库状态或切片数不一致"):
                        run_pipeline(source, FakeModel(), max_pages=0)


if __name__ == "__main__":
    unittest.main()
