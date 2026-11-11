"""分层迁移回归：验证新文档入口、跨文件额度、RAG 包布局和依赖方向。"""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.rag.domain.models import Chunk, Document
from app.rag.ingestion.pipeline import document_chunks, ingest_document
from test_rag import make_pdf


class ArchitectureTests(unittest.TestCase):
    def test_single_file_pipeline_preserves_page_report_without_writing_database(self):
        """单文件处理只返回 Document、Chunk 和报告，不应隐式创建 SQLite 或索引。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pdf = folder / "lesson.pdf"
            make_pdf(pdf)
            data = folder / "data"
            with patch("app.config.DATA_DIR", data):
                documents, chunks, report = ingest_document(pdf, source="子目录/lesson.pdf", max_chunks=1)
            self.assertEqual(len(documents), 2)
            self.assertTrue(all(isinstance(item, Document) for item in documents))
            self.assertEqual(len(chunks), 1)
            self.assertIsInstance(chunks[0], Chunk)
            self.assertEqual(chunks[0].source, "子目录/lesson.pdf")
            self.assertEqual(report["processed_pages"], 2)
            self.assertEqual(report["chunks"], 1)
            self.assertFalse(data.exists())

    def test_chunk_limit_is_shared_across_files_and_reports_keep_loaded_page_counts(self):
        """总额度为 3 时，首文件输出 2 块、次文件输出 1 块，不读取第三份资料。"""
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            uploads = data / "uploads" / "教育资料"
            uploads.mkdir(parents=True)
            for name in ("a.pdf", "b.pdf", "c.pdf"):
                make_pdf(uploads / name)
            with patch("app.config.DATA_DIR", data):
                chunks, reports = document_chunks("教育资料", max_chunks=3)
            self.assertEqual([item.source for item in chunks], ["a.pdf", "a.pdf", "b.pdf"])
            self.assertEqual([report["chunks"] for report in reports], [2, 1])
            self.assertEqual([report["processed_pages"] for report in reports], [2, 2])
            self.assertFalse((data / "assistant.db").exists())

    def test_rag_layers_have_one_home_without_legacy_import_shims(self):
        """各层只位于 app/rag 下；删除并列旧目录、单文件转发和旧示例入口。"""
        project = Path(__file__).resolve().parents[1]
        app = project / "app"
        rag = app / "rag"
        for layer in ("domain", "ingestion", "indexing", "retrieval", "services", "db"):
            self.assertTrue((rag / layer / "__init__.py").is_file())
            self.assertFalse((app / layer).exists())
        for module in ("models", "loaders", "ocr", "preprocessing", "splitter", "ingestion", "retrieval", "client"):
            self.assertFalse((rag / f"{module}.py").exists())
        self.assertFalse((app / "cli.py").exists())
        # 旧目录可能只剩 Python 字节码缓存；检查实际源码，避免把缓存误判为兼容实现。
        self.assertEqual(list((project / "examples").rglob("*.py")), [])

    def test_rag_dependency_layers_remain_separate(self):
        """RAG 各层使用统一包路径，文档处理层不直接依赖索引或业务数据库。"""
        root = Path(__file__).resolve().parents[1] / "app/rag"
        for layer in ("domain", "ingestion", "indexing", "retrieval", "services", "db"):
            for path in (root / layer).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    elif isinstance(node, ast.Import):
                        names = [item.name for item in node.names]
                    else:
                        continue
                    for name in names:
                        self.assertFalse(name.startswith(("app.domain", "app.ingestion", "app.indexing", "app.retrieval", "app.services", "app.db", "app.cli")), str(path))
                        if layer in {"domain", "ingestion"}:
                            self.assertFalse(name.startswith(("app.rag.indexing", "app.rag.db", "app.rag.retrieval", "app.rag.services")), str(path))


if __name__ == "__main__":
    unittest.main()
