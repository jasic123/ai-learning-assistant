# OCR 回归：自动生成测试 PDF 并替换识别结果，验证渲染和处理分支，不依赖真实私有资料。
"""OCR 流水线回归测试：模拟识别结果，不读取用户资料、不调用外部服务。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from app.rag.ingestion.loaders import load_pdf
from app.rag.ingestion.pdf_ocr import OcrUnavailable, ocr_page, open_ocr_document
from app.rag.ingestion.chunkers import split_document


def make_pdf(path, text="www.TopSage.com", image=True, pages=1):
    """生成含文字层和图片对象的最小 PDF；这里只模拟扫描页结构。"""
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        if image:
            bitmap = DecodedStreamObject()
            bitmap.update({NameObject("/Type"): NameObject("/XObject"),
                           NameObject("/Subtype"): NameObject("/Image"),
                           NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1),
                           NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                           NameObject("/BitsPerComponent"): NumberObject(8)})
            # 用一个像素构造合法图片对象，只为触发扫描候选逻辑，不模拟真实识别难度。
            bitmap.set_data(b"\xff")
            resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): writer._add_object(bitmap)})
        stream = DecodedStreamObject()
        commands = "q 500 0 0 700 50 50 cm /Im0 Do Q\n" if image else ""
        commands += "BT /F1 12 Tf 50 750 Td " + " ".join(f"({line}) Tj 0 -15 Td" for line in text.splitlines()) + " ET"
        stream.set_data(commands.encode("ascii"))
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as stream:
        writer.write(stream)


# 覆盖 OCR 候选识别、禁用、缺依赖、单页失败和质量不足的处理。
class OcrTests(unittest.TestCase):
    # 每个测试独立生成临时 PDF，结束时自动清理，避免测试互相污染。
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "扫描样本.pdf"
        make_pdf(self.path)

    # 实际执行页面渲染，检查 OCR 收到像素矩阵并正确聚合文字与平均置信度。
    def test_pdf_renderer_passes_pixels_to_local_engine(self):
        # 真正执行 PDFium 页面渲染，只替换耗时的识别模型。
        with patch("app.rag.ingestion.pdf_ocr._engine") as factory:
            engine = factory.return_value
            engine.return_value = ([([], "中文正文", 0.98), ([], "Python examples", 0.96)], None)
            text, confidence = ocr_page(self.path, 0)
        engine.assert_called_once()
        image = engine.call_args.args[0]
        self.assertEqual(image.ndim, 3)
        self.assertGreater(image.shape[0], 0)
        self.assertEqual(text, "中文正文\nPython examples")
        self.assertAlmostEqual(confidence, 0.97)

    # 验证扫描候选经过 OCR 后重新清洗并切片，同时保持原始 PDF 字节不变。
    def test_scanned_page_is_marked_and_ocr_text_rejoins_cleaning_and_chunks(self):
        original = self.path.read_bytes()
        raw = "Python 编程参考手册中的列表支持增加删除和修改元素。" * 5 + "\nwww.TopSage.com"
        with patch("app.rag.ingestion.loaders.ocr_page", return_value=(raw, 0.97)) as ocr:
            documents, report = load_pdf(self.path, source=self.path.name)
        ocr.assert_called_once()
        self.assertEqual(ocr.call_args.args, (self.path, 0))
        self.assertIsNotNone(ocr.call_args.kwargs["document"])
        self.assertEqual(report["suspected_scanned_pages"], [1])
        self.assertEqual(report["ocr_pages"], [1])
        self.assertEqual(report["needs_ocr_pages"], [])
        self.assertEqual(report["low_text_pages"], [])
        self.assertEqual(report["page_diagnostics"][0]["extraction_method"], "ocr")
        self.assertEqual(report["page_diagnostics"][0]["ocr_confidence"], 0.97)
        self.assertNotIn("www.TopSage.com", documents[0].text)
        chunks = split_document(documents[0])
        self.assertTrue(chunks)
        self.assertEqual((chunks[0].source, chunks[0].page), (self.path.name, 1))
        self.assertEqual(self.path.read_bytes(), original)

    # 关闭 OCR 后不调用引擎，但保留待识别页记录，避免把失败当作空白。
    def test_disabling_ocr_keeps_pending_marker(self):
        with patch("app.rag.ingestion.loaders.ocr_page") as ocr:
            documents, report = load_pdf(str(self.path), source=self.path.name, ocr_enabled=False)
        ocr.assert_not_called()
        self.assertEqual(documents, [])
        self.assertEqual(report["needs_ocr_pages"], [1])
        self.assertEqual(report["page_diagnostics"][0]["ocr_status"], "disabled")

    # 检查物理页码从 1 开始、渲染索引从 0 开始，并拒绝非法起始页。
    def test_start_page_keeps_physical_page_numbers_and_ocr_offsets(self):
        make_pdf(self.path, pages=3)
        with patch("app.rag.ingestion.loaders.ocr_page", return_value=("中文有效正文" * 15, 0.96)) as ocr:
            documents, report = load_pdf(self.path, source=self.path.name, start_page=2, max_pages=1)
        ocr.assert_called_once()
        self.assertEqual(ocr.call_args.args, (self.path, 1))
        self.assertEqual([doc.page for doc in documents], [2])
        self.assertEqual(report["ocr_pages"], [2])
        self.assertEqual(report["processed_pages"], 1)
        self.assertEqual(report["total_pages"], 3)
        for start in (0, True, 4):
            with self.subTest(start=start), self.assertRaises(ValueError):
                load_pdf(self.path, source=self.path.name, start_page=start)

    # 有正常文字或没有图片的普通空白、短文字页不应无条件触发 OCR。
    def test_text_pages_and_true_blank_pages_do_not_call_ocr(self):
        for text in ("Python lists support append remove and many other sequence operations.", "", "Short"):
            with self.subTest(text=text):
                make_pdf(self.path, text=text, image=False)
                with patch("app.rag.ingestion.loaders.ocr_page") as ocr:
                    _, report = load_pdf(self.path, source=self.path.name)
                ocr.assert_not_called()
                self.assertEqual(report["needs_ocr_pages"], [])

    # 缺依赖时只尝试一次初始化，同时给全部候选页保留失败和待识别标记。
    def test_missing_dependency_marks_all_candidates_without_repeated_initialization(self):
        make_pdf(self.path, pages=2)
        with patch("app.rag.ingestion.loaders.ocr_page", side_effect=OcrUnavailable("test")) as ocr:
            _, report = load_pdf(self.path, source=self.path.name)
        ocr.assert_called_once()
        self.assertEqual(report["ocr_failed_pages"], [1, 2])
        self.assertEqual(report["needs_ocr_pages"], [1, 2])

    # 单页识别异常不得阻断后续页面的成功识别。
    def test_failure_of_one_page_does_not_stop_next_page(self):
        make_pdf(self.path, pages=2)
        with patch("app.rag.ingestion.loaders.ocr_page", side_effect=[RuntimeError("test"), ("中文有效正文" * 15, 0.95)]) as ocr, \
             patch("app.rag.ingestion.loaders.open_ocr_document", wraps=open_ocr_document) as open_pdf:
            documents, report = load_pdf(self.path, source=self.path.name)
        self.assertEqual(report["ocr_failed_pages"], [1])
        self.assertEqual(report["ocr_pages"], [2])
        self.assertEqual([doc.page for doc in documents], [2])
        self.assertEqual(open_pdf.call_count, 1)
        self.assertIs(ocr.call_args_list[0].kwargs["document"], ocr.call_args_list[1].kwargs["document"])

    # 识别输出只有水印时不能算成功恢复正文，应继续保留待识别状态。
    def test_ocr_with_insufficient_text_stays_pending(self):
        with patch("app.rag.ingestion.loaders.ocr_page", return_value=("www.TopSage.com", 0.9)):
            documents, report = load_pdf(self.path, source=self.path.name)
        self.assertEqual(documents, [])
        self.assertEqual(report["needs_ocr_pages"], [1])
        self.assertEqual(report["page_diagnostics"][0]["ocr_status"], "insufficient_text")

    # 即使没有图片对象，未解决的英文碎字也可以触发整页 OCR 回退。
    def test_unresolved_english_fragments_can_use_ocr_without_images(self):
        make_pdf(self.path, text="\n".join(["a"] * 50), image=False)
        with patch("app.rag.ingestion.loaders.ocr_page", return_value=("Python examples contain complete sentences. " * 4, 0.98)) as ocr:
            documents, report = load_pdf(self.path, source=self.path.name)
        ocr.assert_called_once()
        self.assertEqual(report["suspected_scanned_pages"], [])
        self.assertEqual(report["ocr_pages"], [1])
        self.assertEqual(len(documents), 1)


if __name__ == "__main__":
    unittest.main()
