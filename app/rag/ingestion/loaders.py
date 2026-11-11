# 单文件读取入口：页码按原 PDF 物理页编号，质量报告独立返回，低质量页不会静默视为成功。
"""PDF 加载与清洗：每个可用页面转换为一个 Document。"""
from pathlib import Path
import re
from pypdf import PdfReader
from app.rag.domain.models import Document
from app.rag.ingestion.pdf_ocr import OcrUnavailable, ocr_page, open_ocr_document
from app.rag.ingestion.cleaners import PREPROCESS_VERSION, clean_page, extract_page, repeated_margins


def load_pdf(path: Path, *, source: str, max_pages: int = 0,
             ocr_enabled: bool = True, start_page: int = 1) -> tuple[list[Document], dict]:
    """读取单份 PDF，不切片、不调用 Embedding；原始文件只读。

    source 由调用方提供知识库内相对路径，避免向用户返回机器绝对路径。
    为检测重复页眉页脚，先解析所选页面，再逐页清洗。
    默认对低文字量且包含图片的页面，以及英文碎字页面尝试本地 OCR。
    ocr_enabled=False 仅标记待识别页，不执行 OCR。
    start_page 为从 1 开始的 PDF 物理页码，max_pages 从该页起限制解析数量。
    """
    if max_pages < 0:
        raise ValueError("页数限制不能为负数")
    if type(start_page) is not int or start_page < 1:
        raise ValueError("起始页码必须是从 1 开始的整数")
    path = Path(path)
    with path.open("rb") as stream:
        reader = PdfReader(stream)
        # 先尝试空密码；需要非空密码的文件会报错，当前接口不提供用户密码参数。
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"PDF 已加密：{source}")
        pages = len(reader.pages)
        if start_page > pages:
            raise ValueError("起始页码超出 PDF 总页数")
        selected = min(pages - start_page + 1, max_pages) if max_pages else pages - start_page + 1
        # PDF 对外页码从 1 开始，内部访问 reader.pages 时转换为从 0 开始的索引。
        page_indices = range(start_page - 1, start_page - 1 + selected)
        extracted = [extract_page(reader.pages[i]) for i in page_indices]
        # pypdf 支持识别嵌套图片和内联图片；这里只统计，不解码整页图片。
        has_images = [bool(len(reader.pages[i].images)) for i in page_indices]
    # 先跨页收集边缘重复候选，再清洗每页；只分析本次选中的页面范围。
    repeated = repeated_margins([item[0] for item in extracted])
    # 先用原文字层清洗结果判断 OCR 候选，避免把只含水印的扫描页误当正文。
    initial = [clean_page(raw, repeated, page_number_value=i + start_page)
               for i, (raw, _, _) in enumerate(extracted)]
    suspected = [i + start_page for i, (text, _) in enumerate(initial)
                 if has_images[i] and len(re.findall(r"\w", text)) < 40]
    candidates = [i + start_page for i, (text, quality) in enumerate(initial)
                  if i + start_page in suspected or "fragmented_text" in quality["warnings"]]
    # 用物理页码分别记录识别结果与状态，区分成功、禁用、缺依赖和单页失败。
    ocr_results, ocr_status = {}, {}
    unavailable = False
    if not ocr_enabled:
        ocr_status.update({page: "disabled" for page in candidates})
    elif candidates:
        try:
            # 一个文件只打开一次 PDFium；候选页仍逐页渲染和释放位图。
            with open_ocr_document(path) as ocr_document:
                for page in candidates:
                    if unavailable:
                        ocr_status[page] = "unavailable"
                        continue
                    try:
                        ocr_results[page] = ocr_page(path, page - 1, document=ocr_document)
                        ocr_status[page] = "completed"
                    except OcrUnavailable:
                        unavailable = True
                        ocr_status[page] = "unavailable"
                    except Exception:
                        # 单页渲染/识别失败不阻断全书；报告明确标记，不输出底层异常中的内容。
                        ocr_status[page] = "failed"
        except OcrUnavailable:
            unavailable = True
            ocr_status.update({page: "unavailable" for page in candidates})
        except Exception:
            # PDFium 无法打开文件时标记候选页失败，不把 OCR 失败伪装成空白页。
            ocr_status.update({page: "failed" for page in candidates})
    # OCR 的文字也先参与跨页边缘检测，再走相同的清洗函数。
    final_texts = [ocr_results[i + start_page][0] if i + start_page in ocr_results else item[0]
                   for i, item in enumerate(extracted)]
    repeated = repeated_margins(final_texts)
    # 总页数描述原文件，processed_pages 描述本次处理范围；列表记录需要复核的物理页码。
    report = dict(source=source, total_pages=pages, start_page=start_page, analyzed_pages=selected,
                  processed_pages=selected, empty_pages=[], low_text_pages=[], chunks=0,
                  ocr_enabled=ocr_enabled, suspected_scanned_pages=suspected,
                  ocr_candidate_pages=candidates, ocr_pages=[], ocr_failed_pages=[], needs_ocr_pages=[],
                  preprocessing_version=PREPROCESS_VERSION,
                  repeated_margin_lines=sorted(repeated), page_diagnostics=[])
    documents = []
    for page, (raw, method, extraction_warnings) in enumerate(extracted, start_page):
        status, confidence = ocr_status.get(page, "not_needed"), None
        text, quality = clean_page(raw, repeated, page_number_value=page)
        if page in ocr_results:
            ocr_text, confidence = ocr_results[page]
            cleaned, ocr_quality = clean_page(ocr_text, repeated, page_number_value=page)
            # OCR 结果达到最低文字量且碎行检查通过时才替换原文；否则保留原提取结果。
            if len(re.findall(r"\w", cleaned)) >= 40 and "fragmented_text" not in ocr_quality["warnings"]:
                text, quality, method, status = cleaned, ocr_quality, "ocr", "success"
                report["ocr_pages"].append(page)
                quality["warnings"].append("ocr_check_reading_order_and_code")
            else:
                status = "insufficient_text"
        needs_ocr = page in candidates and status != "success"
        if needs_ocr:
            report["needs_ocr_pages"].append(page)
            quality["warnings"].append("needs_ocr")
        if status in {"failed", "unavailable"}:
            report["ocr_failed_pages"].append(page)
            quality["warnings"].append("ocr_" + status)
        quality.update(page=page, extraction_method=method, extraction_warnings=extraction_warnings,
                       text_layer_chars=len(raw), text_layer_cleaned_chars=len(initial[page - start_page][0]),
                       has_images=has_images[page - start_page], suspected_scanned=page in suspected,
                       needs_ocr=needs_ocr, ocr_status=status, ocr_confidence=confidence)
        # 即使页面最终因文字量不足被跳过，仍保留诊断，方便定位漏页原因。
        report["page_diagnostics"].append(quality)
        if not text.strip():
            report["empty_pages"].append(page)
        # 保留现有低文字量规则；短代码页可能被跳过，需要复核质量报告。
        if len(re.findall(r"[\w]", text)) < 40:
            report["low_text_pages"].append(page)
            continue
        # 只有通过最低文字量检查的页面才进入后续切片；此处不做向量化。
        documents.append(Document(source=source, page=page, text=text))
    return documents, report
