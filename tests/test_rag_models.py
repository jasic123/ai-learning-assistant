# 数据模型回归：验证来源追溯、JSON 兼容、不可变属性及模块导入无副作用。
"""基础数据类型与模块边界测试，不调用模型、不读取私有 PDF。"""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from app.rag.domain.models import Chunk, Document, SearchResult
from app.rag.ingestion.chunkers import split_document


# 基础数据对象及持久化边界的回归测试集合。
class RagModelTests(unittest.TestCase):
    # 检查切片来源、偏移和稳定标识，确认检索分数独立且正文不可直接修改。
    def test_document_to_chunks_preserves_source_and_content(self):
        document = Document(source="子目录/教材.pdf", page=3, text="甲" * 850)
        chunks = split_document(document, chunk_size=400, overlap=60)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertEqual([chunk.offset for chunk in chunks], [0, 340, 680])
        for chunk in chunks:
            self.assertEqual((chunk.source, chunk.page), (document.source, 3))
            self.assertEqual(document.text[chunk.offset:chunk.offset + len(chunk.text)], chunk.text)
            expected_id = hashlib.sha256(
                f"{document.source}\0{document.page}\0{chunk.offset}\0{chunk.text}".encode()
            ).hexdigest()
            self.assertEqual(chunk.id, expected_id)
        self.assertEqual([chunk.position for chunk in chunks], [0, 1, 2])
        self.assertEqual(chunks[0].parent_id, chunks[1].parent_id)
        self.assertIsNone(chunks[0].prev_chunk_id)
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].id)
        self.assertEqual(chunks[1].prev_chunk_id, chunks[0].id)
        # 检索分数属于 SearchResult，不能意外覆盖来源 Chunk 的字段。
        result = SearchResult(chunk=chunks[0], score=0.9)
        self.assertIs(result.chunk, chunks[0])
        self.assertNotIn("score", result.chunk.to_dict())
        with self.assertRaises(FrozenInstanceError):
            chunks[0].text = "changed"

    # 验证对象与 JSON 双向转换，并兼容缺少新版可选字段的旧记录。
    def test_json_roundtrip_and_legacy_records(self):
        chunk = Chunk(id="id1", source="教材.pdf", page=2, text="资料正文",
                      offset=12, preprocessing_version="pdf-clean-v2")
        restored = Chunk.from_dict(json.loads(json.dumps(chunk.to_dict(), ensure_ascii=False)))
        self.assertEqual(restored, chunk)
        # 原 schema_version=1 索引只包含基本字段时，也能读取，不要求重建向量。
        legacy = Chunk.from_dict(dict(id="old", source="教材.pdf", page=1, text="旧片段"))
        self.assertEqual(legacy.offset, 0)
        self.assertEqual(legacy.preprocessing_version, "legacy")
        self.assertEqual(legacy.chunk_type, "prose")
        self.assertEqual(legacy.prev_chunk_id, None)
        self.assertEqual(legacy.to_dict()["text"], "旧片段")

    # 非法页码、空正文、负偏移等损坏记录在恢复对象前被拒绝。
    def test_invalid_persisted_records_are_rejected(self):
        valid = dict(id="id1", source="教材.pdf", page=1, text="正文")
        invalid = [None, [], dict(valid, page=True), dict(valid, page=0),
                   dict(valid, text=""), dict(valid, offset=-1),
                   dict(valid, preprocessing_version=123), dict(valid, position=-1),
                   dict(valid, chunk_type="unknown")]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                Chunk.from_dict(record)

    # 在子进程拦截 PDF 读取，确认导入入库模块不会自动读取资料或打印内容。
    def test_import_does_not_read_pdf_or_print(self):
        code = """from unittest.mock import patch
with patch("pypdf.PdfReader", side_effect=RuntimeError("unexpected read")):
    import app.rag.indexing.pipeline
"""
        # 使用当前解释器启动全新进程，避免模块缓存掩盖导入时副作用。
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "导入模块不应自动读取资料")
        self.assertTrue(result.stdout == "" and result.stderr == "", "导入模块不应输出内容")


if __name__ == "__main__":
    unittest.main()
