"""文档处理流水线：只负责读取、OCR、清洗和切块，不写索引或 SQLite。"""
from pathlib import Path
from dataclasses import replace
import hashlib

from app.config import kb_paths
from app.rag.domain.models import Chunk, Document
from app.rag.ingestion.chunkers import split_document, split_text
from app.rag.ingestion.loaders import load_pdf


def ingest_document(path, *, source=None, max_pages=0, max_chunks=0,
                    chunk_size=400, overlap=60, ocr_enabled=True, start_page=1) -> tuple[list[Document], list[Chunk], dict]:
    """处理单份 PDF，返回逐页 Document、Chunk 列表和文件报告。

    source 是知识库内相对路径；未提供时使用文件名。max_pages 从 start_page 开始限制。
    max_chunks 只限制输出片段；Document 和 processed_pages 仍反映已加载的完整选定范围。
    """
    if max_pages < 0 or max_chunks < 0:
        raise ValueError("页数和切片限制不能为负数")
    # 生成器需实际遍历，才能在读取 PDF 前校验切片参数。
    list(split_text("", chunk_size, overlap))
    path = Path(path)
    documents, report = load_pdf(path, source=source if source is not None else path.name,
                                 max_pages=max_pages, ocr_enabled=ocr_enabled, start_page=start_page)
    chunks = []
    for document in documents:
        pieces = split_document(document, chunk_size=chunk_size, overlap=overlap)
        if max_chunks:
            pieces = pieces[:max_chunks - len(chunks)]
        chunks.extend(pieces)
        if max_chunks and len(chunks) >= max_chunks:
            break
    chunks = [replace(chunk,
                      position=i,
                      prev_chunk_id=chunks[i - 1].id if i else None,
                      next_chunk_id=chunks[i + 1].id if i + 1 < len(chunks) else None,
                      parent_id=hashlib.sha256(
                          f"{chunk.source}\0{chunk.page}".encode()).hexdigest())
              for i, chunk in enumerate(chunks)]
    report["chunks"] = len(chunks)
    return documents, chunks, report


def document_chunks(name: str, *, max_pages=0, max_chunks=0, chunk_size=400, overlap=60,
                    ocr_enabled=True, start_page=1) -> tuple[list[Chunk], list[dict]]:
    """遍历知识库的 PDF，复用单文件流水线；页数按文件限制，片段数按知识库合计限制。"""
    if max_pages < 0 or max_chunks < 0:
        raise ValueError("页数和切片限制不能为负数")
    list(split_text("", chunk_size, overlap))
    folder, _ = kb_paths(name)
    files = sorted(path for path in folder.rglob("*")
                   if path.is_file() and path.suffix.lower() == ".pdf") if folder.is_dir() else []
    if not files:
        raise ValueError(f"未找到 PDF，请放入 data/uploads/{name}/")
    chunks, reports = [], []
    for path in files:
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("PDF 路径不得指向知识库之外")
        # 剩余片段额度传给单文件流水线；来源始终保存知识库内的相对路径。
        remaining = max_chunks - len(chunks) if max_chunks else 0
        _, pieces, report = ingest_document(path, source=path.relative_to(folder).as_posix(),
                                            max_pages=max_pages, max_chunks=remaining,
                                            chunk_size=chunk_size, overlap=overlap,
                                            ocr_enabled=ocr_enabled, start_page=start_page)
        chunks.extend(pieces)
        reports.append(report)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks, reports
