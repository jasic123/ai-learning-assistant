# 业务访问层：集中管理知识库、文件快照和构建记录，供命令行与入库流程共用。
"""知识库元数据操作，以及 SQLite 与磁盘 CURRENT 的同步。"""
import json
import re

from sqlalchemy import select

from app.config import validate_name
from app.rag.db.models import IndexBuild, KnowledgeBase, KnowledgeFile, utc_now
from app.rag.db.session import session_scope


# 按名称取出知识库，不存在则创建；flush 取得主键，但最终提交仍由外层事务负责。
def _knowledgebase(session, name):
    validate_name(name)
    row = session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))
    if row is None:
        row = KnowledgeBase(name=name)
        session.add(row)
        # 把待插入或修改内容发送给数据库并检查约束，此时尚未提交外层事务。
        session.flush()
    return row


# 仅提取模型已声明的数据库列，避免把 ORM 内部状态放进输出字典。
def _dict(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def save_knowledgebase(name, kb_info="", *, data_dir=None):
    """创建知识库或更新简介，不删除资料、不触发 OCR/Embedding。"""
    if not isinstance(kb_info, str):
        raise ValueError("知识库简介必须是字符串")
    with session_scope(data_dir, write=True) as session:
        row = _knowledgebase(session, name)
        row.kb_info, row.updated_at = kb_info.strip(), utc_now()
        # 把待插入或修改内容发送给数据库并检查约束，此时尚未提交外层事务。
        session.flush()
        return _dict(row)


# 按名称排序返回数据库中已登记的知识库，不等于扫描磁盘上的所有资料目录。
def list_knowledgebases(*, data_dir=None):
    with session_scope(data_dir) as session:
        return [_dict(row) for row in session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name))]


# 返回知识库、文件快照与构建历史；知识库未登记时给出业务错误。
def knowledgebase_details(name, *, data_dir=None):
    validate_name(name)
    with session_scope(data_dir) as session:
        row = session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))
        if row is None:
            raise ValueError("知识库尚未登记，请先创建、入库或同步旧索引")
        return dict(knowledgebase=_dict(row),
                    files=[_dict(item) for item in session.scalars(select(KnowledgeFile).where(
                        KnowledgeFile.knowledgebase_id == row.id).order_by(KnowledgeFile.source))],
                    builds=[_dict(item) for item in session.scalars(select(IndexBuild).where(
                        IndexBuild.knowledgebase_id == row.id).order_by(IndexBuild.started_at.desc()))])


def record_uploaded_file(name, source, size_bytes, *, data_dir=None):
    """Register a newly uploaded PDF as pending until an index build publishes it."""
    validate_name(name)
    if not isinstance(source, str) or not source.strip() or type(size_bytes) is not int or size_bytes < 1:
        raise ValueError("上传文件元数据无效")
    with session_scope(data_dir, write=True) as session:
        kb = _knowledgebase(session, name)
        row = session.scalar(select(KnowledgeFile).where(
            KnowledgeFile.knowledgebase_id == kb.id, KnowledgeFile.source == source))
        if row is None:
            row = KnowledgeFile(knowledgebase_id=kb.id, source=source)
            session.add(row)
        row.size_bytes = size_bytes
        row.status, row.generation = "pending", None
        row.total_pages, row.processed_pages, row.chunk_count = None, 0, 0
        row.processing_report, row.updated_at = {}, utc_now()
        session.flush()
        return _dict(row)


# 先登记 building 状态，使后续向量化失败的尝试也有可追溯记录。
def begin_build(name, generation, *, data_dir=None):
    with session_scope(data_dir, write=True) as session:
        kb = _knowledgebase(session, name)
        session.add(IndexBuild(generation=generation, knowledgebase_id=kb.id))


# 把构建标记为 failed 并记录结束时间；只保存异常类名，不保存完整异常正文。
def fail_build(generation, error, *, data_dir=None):
    with session_scope(data_dir, write=True) as session:
        row = session.get(IndexBuild, generation)
        if row is not None:
            row.status, row.error_type, row.finished_at = "failed", type(error).__name__, utc_now()


def _record_index(session, meta, uploads):
    """一次事务更新当前快照；重建未覆盖的旧文件必须撤销 indexed 标记。"""
    kb = _knowledgebase(session, meta["knowledgebase"])
    kb.current_generation, kb.chunk_count, kb.updated_at = meta["generation"], meta["count"], utc_now()
    build = session.get(IndexBuild, meta["generation"])
    if build is None:
        build = IndexBuild(generation=meta["generation"], knowledgebase_id=kb.id,
                           started_at=meta["created_at"])
        session.add(build)
    build.status, build.error_type, build.finished_at = "ready", None, utc_now()
    build.embedding_model, build.embedding_base_url = meta["embedding_model"], meta["embedding_base_url"]
    build.dimension, build.chunk_count = meta["dimension"], meta["count"]
    build.preprocessing_version, build.options = meta["preprocessing_version"], meta["options"]
    old_files = {row.source: row for row in session.scalars(select(KnowledgeFile).where(
        KnowledgeFile.knowledgebase_id == kb.id))}
    reports = {item["source"]: item for item in meta["documents"]}
    # 把本次报告与历史文件合并，确保未参与本次重建的旧文件也会更新状态。
    sources = set(reports) | set(old_files)
    if uploads.is_dir():
        sources.update(path.relative_to(uploads).as_posix() for path in uploads.rglob("*")
                       if path.is_file() and path.suffix.lower() == ".pdf")
    for source in sorted(sources):
        path = uploads / source
        if not path.resolve().is_relative_to(uploads.resolve()):
            raise ValueError("文件元数据路径不得指向知识库之外")
        row = old_files.get(source)
        if row is None:
            row = KnowledgeFile(knowledgebase_id=kb.id, source=source)
            session.add(row)
        report = reports.get(source)
        row.size_bytes = path.stat().st_size if path.is_file() else None
        row.chunk_count = report.get("chunks", 0) if report else 0
        # 有报告且产生切片才是 indexed；未参与此次版本的文件重置为 pending。
        row.status = ("indexed" if row.chunk_count else "no_chunks") if report else "pending"
        row.generation = meta["generation"] if report else None
        row.total_pages = report.get("total_pages") if report else None
        row.processed_pages = report.get("processed_pages", 0) if report else 0
        row.processing_report, row.updated_at = report or {}, utc_now()


def publish_index(meta, root, uploads, *, data_dir=None):
    """数据库提交失败时恢复旧 CURRENT；进程崩溃的跨存储间隙用 sync 恢复。

    ponytail: SQLite 与文件系统不共享事务，CURRENT 为磁盘索引事实来源。
    不在事务内进行模型调用，数据库写锁只覆盖本地元数据和指针切换。
    """
    pointer = root / (meta["generation"] + ".tmp")
    current = root / "CURRENT"
    previous, switched = None, False
    try:
        with session_scope(data_dir, write=True) as session:
            # 保存旧指针内容，数据库提交异常时用于恢复原来可用的版本。
            previous = current.read_bytes() if current.is_file() else None
            _record_index(session, meta, uploads)
            session.flush()  # 先验证约束，避免明显数据库错误发生在指针切换之后。
            pointer.write_text(meta["generation"], encoding="ascii")
            # 通过同目录替换发布指针，避免读者看到只写了一半的版本编号。
            pointer.replace(current)
            switched = True
    except Exception:
        if switched and current.is_file() and current.read_text(encoding="ascii").strip() == meta["generation"]:
            if previous is None:
                current.unlink(missing_ok=True)
            else:
                pointer.write_bytes(previous)
                # 通过同目录替换发布指针，避免读者看到只写了一半的版本编号。
                pointer.replace(current)
        raise
    finally:
        pointer.unlink(missing_ok=True)


def sync_index(name, root, uploads, *, data_dir=None):
    """导入旧 JSON 索引元数据或修复崩溃后的数据库记录，不调用 Embedding。"""
    validate_name(name)
    with session_scope(data_dir, write=True) as session:
        # 同步以磁盘当前版本为依据；不会根据数据库中失败的构建记录猜测版本。
        generation = (root / "CURRENT").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("索引版本标记损坏")
        folder = root / generation
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("knowledgebase") != name or meta.get("generation") != generation:
            raise ValueError("索引元数据与当前知识库或版本不一致")
        if not (folder / "index.faiss").is_file() or not (folder / "chunks.json").is_file():
            raise ValueError("索引文件不完整，无法同步")
        _record_index(session, meta, uploads)
    return generation
