# 持久化回归：在临时目录中模拟成功与失败，检查 SQLite 和 CURRENT 指针是否恢复一致。
"""SQLite 与索引发布一致性测试，使用临时目录和假向量，不调用真实模型。"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import func, select

from app.config import ModelSettings
from app.rag.db.models import IndexBuild, KnowledgeBase, KnowledgeFile
from app.rag.db.repository import knowledgebase_details, save_knowledgebase, sync_index
from app.rag.db.session import session_scope
from app.rag.indexing.pipeline import build_index
from app.rag.domain.models import Chunk


# 固定返回二维向量的测试替身，隔离真实模型服务与网络费用。
class FakeModel:
    settings = ModelSettings("https://example.invalid/v1", "test-model", "fake-secret")

    # 对每个输入返回固定非零向量，仅用于验证存储和事务流程。
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


# 覆盖索引发布、失败补偿、文件状态重置与旧索引同步。
class MetadataTests(unittest.TestCase):
    # 为每个测试创建独立目录并替换数据入口；清理由 unittest 在测试结束时执行。
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        self.uploads = self.data / "uploads" / "教育资料"
        self.uploads.mkdir(parents=True)
        (self.uploads / "lesson.pdf").write_bytes(b"fixture")
        self.root = self.data / "indexes" / "教育资料"
        self.addCleanup(patch.stopall)
        # 入库路径改到临时目录，所有数据库与索引写入都与正式数据隔离。
        patch("app.config.DATA_DIR", self.data).start()
        self.chunks = [Chunk(id="c1", source="lesson.pdf", page=16, text="中文正文" * 20)]
        self.reports = [dict(source="lesson.pdf", total_pages=100, processed_pages=5, chunks=1,
                             ocr_pages=[16], needs_ocr_pages=[], start_page=16)]
        patch("app.rag.indexing.pipeline.document_chunks", side_effect=lambda *a, **k: (self.chunks, self.reports)).start()

    # 从临时数据库重新读取详情，避免只检查测试中的内存变量。
    def details(self):
        return knowledgebase_details("教育资料", data_dir=self.data)

    # 验证成功发布会保存简介、文件报告和构建信息，且不保存测试密钥。
    def test_publication_persists_knowledgebase_files_and_versions(self):
        save_knowledgebase("教育资料", "Python 教材，用于编程辅导", data_dir=self.data)
        meta = build_index("教育资料", FakeModel(), start_page=16, max_pages=5)
        # 每次查询重新建连接，验证重启后仍可持久化读取，而非仅检查内存对象。
        detail = self.details()
        self.assertEqual(detail["knowledgebase"]["kb_info"], "Python 教材，用于编程辅导")
        self.assertEqual(detail["knowledgebase"]["current_generation"], meta["generation"])
        self.assertEqual(detail["files"][0]["source"], "lesson.pdf")
        self.assertEqual(detail["files"][0]["size_bytes"], 7)
        self.assertEqual(detail["files"][0]["processing_report"]["ocr_pages"], [16])
        self.assertEqual(detail["files"][0]["chunk_count"], 1)
        self.assertEqual(detail["files"][0]["status"], "indexed")
        self.assertEqual(detail["builds"][0]["status"], "ready")
        self.assertNotIn("fake-secret", json.dumps(detail))
        save_knowledgebase("教育资料", "更新后的简介", data_dir=self.data)
        self.assertEqual(self.details()["knowledgebase"]["current_generation"], meta["generation"])
        with session_scope(self.data) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(KnowledgeBase)), 1)

    # 模拟向量化失败，确认旧版本继续可用且失败记录仅包含异常类型。
    def test_failed_embedding_retains_active_metadata_and_records_safe_error(self):
        old = build_index("教育资料", FakeModel())
        with patch("app.rag.indexing.pipeline.document_chunks",
                   side_effect=lambda *a, **k: (self.chunks, self.reports)) as prepare, \
             patch.object(FakeModel, "embed", side_effect=RuntimeError("fake-secret must not be saved")):
            with self.assertRaises(RuntimeError):
                build_index("教育资料", FakeModel())
            with self.assertRaises(RuntimeError):
                build_index("教育资料", FakeModel())
            self.assertEqual(prepare.call_count, 1)
        detail = self.details()
        self.assertEqual(detail["knowledgebase"]["current_generation"], old["generation"])
        self.assertEqual((self.root / "CURRENT").read_text(), old["generation"])
        failed = next(row for row in detail["builds"] if row["status"] == "failed")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertNotIn("fake-secret", json.dumps(detail))

    # 部分资料参与重建时，旧文件不能继续保留 indexed 状态和过期切片数。
    def test_partial_rebuild_does_not_keep_stale_file_counts(self):
        build_index("教育资料", FakeModel())
        (self.uploads / "other.pdf").write_bytes(b"other")
        self.chunks = [Chunk(id="c2", source="other.pdf", page=1, text="另一份正文" * 10)]
        self.reports = [dict(source="other.pdf", total_pages=2, processed_pages=1, chunks=1)]
        build_index("教育资料", FakeModel())
        files = {row["source"]: row for row in self.details()["files"]}
        self.assertEqual(files["lesson.pdf"]["status"], "pending")
        self.assertEqual(files["lesson.pdf"]["chunk_count"], 0)
        self.assertEqual(files["other.pdf"]["status"], "indexed")

    # 在指针切换后模拟事务失败，确认恢复旧 CURRENT 和文件快照。
    def test_db_commit_failure_restores_current_and_previous_file_snapshot(self):
        old = build_index("教育资料", FakeModel())
        original_scope = session_scope
        calls = 0

        # 测试专用事务替身：第二次会话模拟发布提交失败，让补偿路径真实执行。
        @contextmanager
        def broken_scope(*args, **kwargs):
            nonlocal calls
            calls += 1
            fail_commit = calls == 2  # 第一次登记 building，第二次才是发布事务。
            with original_scope(*args, **kwargs) as session:
                yield session
                if fail_commit:
                    self.assertNotEqual((self.root / "CURRENT").read_text(), old["generation"])
                    raise RuntimeError("simulated commit failure")

        with patch("app.rag.db.repository.session_scope", broken_scope):
            with self.assertRaises(RuntimeError):
                build_index("教育资料", FakeModel())
        self.assertEqual((self.root / "CURRENT").read_text(), old["generation"])
        self.assertEqual(self.details()["knowledgebase"]["current_generation"], old["generation"])
        self.assertEqual(len(list(self.root.glob("*/metadata.json"))), 1)

    # 模拟指针替换失败，确认数据库事务回滚且旧索引保持激活。
    def test_pointer_failure_rolls_back_database_and_preserves_previous_index(self):
        old = build_index("教育资料", FakeModel())
        with patch.object(Path, "replace", side_effect=OSError("pointer failure")):
            with self.assertRaises(OSError):
                build_index("教育资料", FakeModel())
        self.assertEqual((self.root / "CURRENT").read_text(), old["generation"])
        detail = self.details()
        self.assertEqual(detail["knowledgebase"]["current_generation"], old["generation"])
        self.assertEqual(detail["files"][0]["generation"], old["generation"])
        self.assertTrue(any(row["status"] == "failed" for row in detail["builds"]))

    # 无可用切片时保留失败构建记录，不创建虚假的当前索引版本。
    def test_no_chunks_records_failed_attempt_without_active_generation(self):
        self.chunks = []
        with self.assertRaises(ValueError):
            build_index("教育资料", FakeModel())
        detail = self.details()
        self.assertIsNone(detail["knowledgebase"]["current_generation"])
        self.assertEqual(detail["builds"][0]["status"], "failed")
        self.assertFalse((self.root / "CURRENT").exists())

    # 清空管理记录后从磁盘同步，确认不调用模型、不重复记录并保留已有简介。
    def test_sync_imports_legacy_metadata_without_embedding(self):
        old = build_index("教育资料", FakeModel())
        # 模拟旧索引存在，但数据库没有文件/版本记录，或进程崩溃导致快照落后。
        with session_scope(self.data, write=True) as session:
            for model in (KnowledgeFile, IndexBuild, KnowledgeBase):
                session.query(model).delete()
        with patch.object(FakeModel, "embed", side_effect=AssertionError("不应调用模型")):
            sync_index("教育资料", self.root, self.uploads, data_dir=self.data)
            save_knowledgebase("教育资料", "简介应被保留", data_dir=self.data)
            sync_index("教育资料", self.root, self.uploads, data_dir=self.data)
        self.assertEqual(self.details()["knowledgebase"]["current_generation"], old["generation"])
        self.assertEqual(len(self.details()["files"]), 1)
        self.assertEqual(len(self.details()["builds"]), 1)
        self.assertEqual(self.details()["knowledgebase"]["kb_info"], "简介应被保留")


if __name__ == "__main__":
    unittest.main()
