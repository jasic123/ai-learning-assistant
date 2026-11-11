# RAG 回归：用假模型和内存 HTTP 传输验证数据链路，不请求真实模型服务。
"""不访问真实模型、不发送私有资料的 RAG 回归测试。"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("AUTH_REQUIRED", "false")

import httpx
import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import ModelSettings, model_settings
from app.main import app
from app.llm.client import ModelClient
from app.config import kb_paths
from app.rag.ingestion.chunkers import split_text
from app.rag.ingestion.pipeline import document_chunks
from app.rag.indexing.embeddings import float_vectors
from app.rag.indexing.embeddings import embed_chunks
from app.rag.indexing.pipeline import build_index
from app.rag.retrieval.retrievers import KnowledgeIndex
from app.rag.services.chat_service import answer_question
from app.rag.domain.models import Chunk, Document, SearchResult
from app.rag.ingestion.loaders import load_pdf


# 生成含两类主题文字的最小 PDF，供解析和检索链路使用。
def make_pdf(path):
    writer = PdfWriter()
    for text in ("Python lists are mutable sequences. Python lists support append and remove.",
                 "Cooking recipes use ingredients and instructions. Cooking takes practice."):
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 750 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as handle:
        writer.write(handle)


# 可预测的模型替身：用主题关键词区分向量，并返回带引用的固定答案。
class FakeModel:
    settings = ModelSettings('https://example.invalid/v1', 'test-embedding', 'not-a-real-key')

    # 将含 Python 关键词的文字映射到一个方向，其余映射到另一个方向。
    def embed(self, texts):
        return [[1, 0] if 'python' in text.lower() else [0, 1] for text in texts]

    # 保存输入消息以便检查上下文格式，返回固定答案，不验证真实模型能力。
    def chat(self, messages):
        self.messages = messages
        return 'Python lists are mutable [S1].'


# 验证切片边界、入库检索、失败重建、接口契约和健康检查。
class RagTests(unittest.TestCase):
    # 检查重叠偏移及尾部覆盖，并拒绝零向量、非有限数和越界知识库路径。
    def test_chunk_boundaries(self):
        text = 'a' * 1100
        parts = list(split_text(text, 400, 60))
        self.assertEqual(parts[0], ('a' * 400, 0))
        self.assertEqual(parts[1][1], 340)
        self.assertEqual(parts[-1][1] + len(parts[-1][0]), len(text))
        with self.assertRaises(ValueError):
            list(split_text(text, 400, 400))
        with self.assertRaises(ValueError):
            float_vectors([[0, 0]], 1)
        with self.assertRaises(ValueError):
            float_vectors([[float('nan'), 1]], 1)
        with self.assertRaises(ValueError):
            kb_paths('../outside')

    def test_embedding_retries_http_400_batches_and_splits_long_structural_chunks(self):
        chunks = [Chunk("short", "lesson.pdf", 1, "short text"),
                  Chunk("long", "lesson.pdf", 1, "abcdefghij " * 40)]

        class LimitedClient:
            def embed(self, texts):
                if len(texts) > 1 or any(len(text) > 80 for text in texts):
                    raise RuntimeError("embeddings 返回 HTTP 400")
                return [[float(len(text)), 1.0] for text in texts]

        vectors = list(embed_chunks(chunks, LimitedClient(), batch_size=2))[0]
        self.assertEqual(vectors.shape, (2, 2))
        self.assertTrue(np.isfinite(vectors).all())

    # 走通 PDF 到带引用答案的离线链路，兼容旧片段记录并保留失败前可用版本。
    def test_pdf_index_search_and_failed_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp, patch('app.config.DATA_DIR', Path(tmp)):
            source = Path(tmp) / 'uploads' / '教育资料'
            source.mkdir(parents=True)
            pdf = source / 'lesson.pdf'
            make_pdf(pdf)
            # 保留原始字节，在处理结束后验证 PDF 未被清洗或索引流程改写。
            original = pdf.read_bytes()
            documents, load_report = load_pdf(pdf, source="lesson.pdf")
            self.assertEqual(len(documents), 2)
            self.assertTrue(all(isinstance(document, Document) for document in documents))
            self.assertEqual(load_report["processed_pages"], 2)
            self.assertIn("Python lists", documents[0].text)
            chunks, reports = document_chunks('教育资料')
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
            self.assertEqual([c.page for c in chunks], [1, 2])
            self.assertEqual(chunks[0].next_chunk_id, chunks[1].id)
            self.assertEqual(chunks[1].prev_chunk_id, chunks[0].id)
            model = FakeModel()
            meta = build_index('教育资料', model, batch_size=1)
            self.assertEqual(meta['count'], 2)
            self.assertNotIn('not-a-real-key', json.dumps(meta))
            index = KnowledgeIndex('教育资料')
            hits = index.search('Python list', model, top_k=1, mode='vector')
            self.assertIsInstance(hits[0], SearchResult)
            self.assertEqual(hits[0].chunk.page, 1)
            self.assertEqual(hits[0].chunk.source, 'lesson.pdf')
            self.assertGreater(hits[0].score, 0.99)
            self.assertIn('[S1]', answer_question('Python list', hits, model))
            self.assertIn('retrieved_sources', model.messages[1]['content'])
            # 模拟旧版磁盘索引：缺少可选字段的原始 JSON 记录仍可恢复为 Chunk。
            _, index_root = kb_paths('教育资料')
            records_path = index_root / meta['generation'] / 'chunks.json'
            records = json.loads(records_path.read_text(encoding='utf-8'))
            for record in records:
                record.pop('offset_basis', None)
                record.pop('preprocessing_version', None)
            records_path.write_text(json.dumps(records), encoding='utf-8')
            legacy_index = KnowledgeIndex('教育资料')
            self.assertEqual(legacy_index.chunks[0].preprocessing_version, 'legacy')
            self.assertEqual(legacy_index.search('Python list', model, 1)[0].chunk.page, 1)
            with patch.object(model, 'embed', side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError):
                    build_index('教育资料', model)
            self.assertEqual(KnowledgeIndex('教育资料').metadata['generation'], meta['generation'])
            self.assertEqual(pdf.read_bytes(), original)
            model.settings = ModelSettings('https://example.invalid/v1', 'changed-model', 'secret')
            with self.assertRaisesRegex(ValueError, '模型'):
                index.search('Python', model)

    # 用内存 HTTP 响应检查向量顺序恢复、聊天解析和错误信息脱敏。
    def test_client_contract_and_redacted_error(self):
        settings = FakeModel.settings
        # 模拟乱序的 Embedding 响应和聊天响应，不经过真实网络。
        def handler(request):
            if request.url.path.endswith('/embeddings'):
                return httpx.Response(200, json={'data': [{'index': 1, 'embedding': [0, 1]}, {'index': 0, 'embedding': [1, 0]}]})
            return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})
        with ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as model:
            self.assertEqual(model.embed(['one', 'two']), [[1, 0], [0, 1]])
            self.assertEqual(model.chat([{'role': 'user', 'content': 'question'}]), 'answer')
        with ModelClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(401, text='secret response')))) as model:
            with self.assertRaises(RuntimeError) as error:
                model.embed(['test'])
            self.assertNotIn('secret response', str(error.exception))
            self.assertIn('401', str(error.exception))
        self.assertNotIn('not-a-real-key', repr(settings))

    # 缺配置应明确报错，而不依赖密钥的健康检查仍能正常响应。
    def test_config_missing_and_health(self):
        with patch.dict(os.environ, {}, clear=True), patch('app.config.load_dotenv'):
            with self.assertRaisesRegex(ValueError, 'LLM_API_KEY'):
                model_settings('LLM')
        # 通过 ASGI 内存传输访问应用，无需启动监听端口。
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.get('/health')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'status': 'ok'})
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
