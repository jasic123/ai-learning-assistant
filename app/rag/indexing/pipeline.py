"""索引流水线：文档处理 → Embedding → FAISS/JSON → 版本发布与 SQLite 同步。"""
import shutil
import hashlib
import json
import uuid
from datetime import datetime, timezone

from app import config
from app.config import kb_paths
from app.rag.db.repository import begin_build, fail_build, publish_index
from app.rag.domain.models import Chunk
from app.rag.ingestion.cleaners import PREPROCESS_VERSION
from app.rag.ingestion.pipeline import document_chunks
from app.rag.indexing.embeddings import embed_chunks
from app.rag.indexing.vector_store import create_index, save_index


def build_index(name, client, *, batch_size=16, progress=None, **options):
    """重建索引并登记 SQLite；失败尝试与上一版可用索引分开记录。"""
    if batch_size < 1:
        raise ValueError("批量大小必须大于 0")
    kb_paths(name)  # 在创建数据库记录前检查目录名与路径边界。
    # 每次构建分配新的 32 位版本号，既关联任务记录，也作为新索引目录名。
    generation = uuid.uuid4().hex
    begin_build(name, generation, data_dir=config.DATA_DIR)
    try:
        return _build_index(name, client, generation, batch_size=batch_size, progress=progress, **options)
    except Exception as error:
        try:
            fail_build(generation, error, data_dir=config.DATA_DIR)
        except Exception:
            # 数据库本身不可写时不覆盖原错误；可用 sync 根据 CURRENT 修复记录。
            pass
        raise


def _build_index(name, client, generation, *, batch_size, progress, **options):
    chunks, reports, cache_path = _load_or_prepare(name, generation, options)
    if not chunks:
        raise ValueError("没有可用文字，请检查报告中的 needs_ocr_pages、ocr_failed_pages 和低文字量页面")
    # 文档处理、Embedding、向量存储分别由各自模块完成，本层只负责串联与发布。
    index = create_index(embed_chunks(chunks, client, batch_size=batch_size, progress=progress))
    uploads, root = kb_paths(name)
    root.mkdir(parents=True, exist_ok=True)
    # 每次入库写入新的版本目录，旧索引保留，失败时不会破坏当前可用版本。
    folder = root / generation
    folder.mkdir()
    # 元数据记录模型、维度、清洗版本和实际切片参数，不保存 API Key。
    meta = dict(schema_version=1, knowledgebase=name, generation=generation,
                created_at=datetime.now(timezone.utc).isoformat(),
                embedding_model=client.settings.model, embedding_base_url=client.settings.base_url,
                dimension=index.d, count=len(chunks), metric="cosine",
                preprocessing_version=PREPROCESS_VERSION,
                options=dict(chunk_size=options.get("chunk_size", 400), overlap=options.get("overlap", 60),
                             max_pages=options.get("max_pages", 0), max_chunks=options.get("max_chunks", 0),
                             ocr_enabled=options.get("ocr_enabled", True), start_page=options.get("start_page", 1)),
                documents=reports)
    try:
        save_index(folder, index, chunks, meta)
        # 先记录 SQLite，再切换 CURRENT；提交异常时恢复旧指针。
        publish_index(meta, root, uploads, data_dir=config.DATA_DIR)
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
    except Exception:
        # 恢复指针也失败时保留正在被 CURRENT 引用的版本，供 sync 修复数据库。
        current = root / "CURRENT"
        if not current.is_file() or current.read_text(encoding="ascii").strip() != generation:
            shutil.rmtree(folder, ignore_errors=True)
        raise
    return meta


def _prepared_cache(name, options):
    uploads, _ = kb_paths(name)
    files = sorted(path for path in uploads.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    identity = {
        "preprocessing_version": PREPROCESS_VERSION,
        "options": options,
        "files": [[path.relative_to(uploads).as_posix(), path.stat().st_size,
                   path.stat().st_mtime_ns] for path in files],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return config.DATA_DIR / "prepared_chunks" / name / f"{key}.json"


def _load_or_prepare(name, generation, options):
    """Keep successfully parsed chunks on disk so embedding retries do not repeat OCR."""
    cache_path = _prepared_cache(name, options)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        reports = payload["reports"]
        if not isinstance(reports, list):
            raise ValueError("invalid prepared reports")
        return chunks, reports, cache_path
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass

    chunks, reports = document_chunks(name, **options)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.stem}.{generation}.tmp")
    temporary.write_text(json.dumps({
        "chunks": [chunk.to_dict() for chunk in chunks],
        "reports": reports,
    }, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return chunks, reports, cache_path
