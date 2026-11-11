"""手动入库验收：真实 PDF → 解析/OCR → 清洗 → 切块 → FAISS/JSON → SQLite。

不使用命令行参数解析器，不调用聊天模型；直接运行本文件即可。
默认检查前 20 页，MAX_PAGES 改为 0 可处理整份 PDF。
SQLite 保存管理元数据，Chunk 正文和向量仍按项目现有设计保存到版本目录。
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

# 直接运行时用脚本位置定位项目根目录，不依赖 PyCharm 的工作目录。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT, model_settings
from app.llm.client import ModelClient
from app.config import kb_paths
from app.rag.indexing.pipeline import build_index
from app.rag.retrieval.retrievers import KnowledgeIndex


# 修改这里的参数即可运行；路径基于项目根目录，不受 PyCharm 工作目录影响。
PDF_PATH = PROJECT_ROOT / "data/uploads/教育资料/Python参考手册(第4版).pdf"
KNOWLEDGEBASE = "PDF入库测试"
START_PAGE = 1                  # 从 1 开始的 PDF 物理页码。
MAX_PAGES = 20                  # 0 表示处理整本；本参数不限制生成的切片数量。
CHUNK_SIZE = 400                # 单位为字符，不是 token。
OVERLAP = 60                    # 期望重叠字符数，具体边界以现有切片器为准。


def file_digest(path):
    """流式计算文件摘要，检查处理流程没有改写原始 PDF。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_pipeline(pdf_path, client, *, knowledgebase=KNOWLEDGEBASE,
                 max_pages=MAX_PAGES, start_page=START_PAGE):
    """调用真实入库服务，再通过独立 SQLite 连接核对落盘记录；client 允许离线测试注入。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.is_file():
        raise ValueError(f"找不到测试 PDF：{pdf_path}")
    uploads, index_root = kb_paths(knowledgebase)
    target = uploads / pdf_path.name
    # 本次验收只针对这一份 PDF，防止知识库中其他文件混入导致统计难以核对。
    others = [path for path in uploads.rglob("*") if path.is_file()
              and path.suffix.lower() == ".pdf" and path != target] if uploads.is_dir() else []
    if others:
        raise ValueError("测试知识库含有其他 PDF，请使用只包含目标文件的专用知识库名称")
    original_digest = file_digest(pdf_path)
    uploads.mkdir(parents=True, exist_ok=True)
    if pdf_path.resolve() != target.resolve():
        shutil.copyfile(pdf_path, target)
    print(f"[1/3] 已准备测试 PDF：{pdf_path.name}", flush=True)
    print("[2/3] 正在解析、OCR、清洗、切块，并通过传入的 Embedding 客户端入库。", flush=True)
    # 全部复用项目当前实现，测试中不另外写一套清洗、切片或数据库插入逻辑。
    meta = build_index(knowledgebase, client, start_page=start_page, max_pages=max_pages,
                       max_chunks=0, chunk_size=CHUNK_SIZE, overlap=OVERLAP, ocr_enabled=True,
                       progress=lambda done, total: print(f"Embedding：{done}/{total}", flush=True))
    print("[3/3] 重新打开磁盘索引和 SQLite，核对已提交结果。", flush=True)
    index = KnowledgeIndex(knowledgebase)
    generation = meta["generation"]
    assert (index_root / "CURRENT").read_text(encoding="ascii").strip() == generation, "CURRENT 版本不一致"
    assert index.metadata == meta, "落盘元数据与构建返回值不一致"
    assert len(index.chunks) == index.index.ntotal == meta["count"] > 0, "正文与向量数量不一致或为空"
    assert len({chunk.id for chunk in index.chunks}) == len(index.chunks), "切片标识重复"
    assert all(chunk.source == pdf_path.name and chunk.text.strip() for chunk in index.chunks), "切片来源或正文无效"
    assert len(meta["documents"]) == 1, "本次验收应只处理一个文件"
    document_report = meta["documents"][0]
    expected_pages = min(document_report["total_pages"] - start_page + 1, max_pages) if max_pages else document_report["total_pages"] - start_page + 1
    assert document_report["processed_pages"] == expected_pages, "处理页数与请求范围不一致"
    assert all(start_page <= chunk.page < start_page + expected_pages for chunk in index.chunks), "切片页码超出处理范围"

    # 使用标准库直接读已保存的数据库，不通过 repository 的返回值代替持久化验证。
    database = DATA_DIR / "assistant.db"
    # sqlite3 连接自己的 with 只处理事务，不会关闭连接；closing 确保 Windows 文件锁及时释放。
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        kb = connection.execute("SELECT * FROM knowledgebases WHERE name=?", (knowledgebase,)).fetchone()
        assert kb is not None, "SQLite 缺少知识库记录"
        files = connection.execute("SELECT * FROM knowledge_files WHERE knowledgebase_id=?", (kb["id"],)).fetchall()
        build = connection.execute("SELECT * FROM index_builds WHERE generation=?", (generation,)).fetchone()
        assert kb["current_generation"] == generation and kb["chunk_count"] == len(index.chunks), "知识库版本或切片数不一致"
        assert len(files) == 1, "SQLite 应存在且仅存在一个测试文件记录"
        file = files[0]
        assert file["source"] == pdf_path.name and file["generation"] == generation, "文件来源或版本不一致"
        assert file["status"] == "indexed" and file["chunk_count"] == len(index.chunks), "文件入库状态或切片数不一致"
        assert file["size_bytes"] == target.stat().st_size, "文件大小未正确保存"
        assert file["total_pages"] == document_report["total_pages"] and file["processed_pages"] == expected_pages, "页数未正确保存"
        assert json.loads(file["processing_report"]) == document_report, "清洗/OCR 报告未完整保存"
        assert build is not None and build["knowledgebase_id"] == kb["id"], "构建记录缺失或关联错误"
        assert build["status"] == "ready" and build["finished_at"], "构建未成功结束"
        assert build["chunk_count"] == len(index.chunks) and build["dimension"] == index.index.d, "构建数量或维度不一致"
        assert json.loads(build["options"]) == meta["options"], "构建参数未正确保存"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall(), "数据库存在外键引用错误"
    assert file_digest(pdf_path) == original_digest == file_digest(target), "原 PDF 或测试副本被意外修改"
    return dict(passed=True, knowledgebase=knowledgebase, generation=generation,
                database=str(database), index_directory=str(index_root / generation),
                total_pages=document_report["total_pages"], processed_pages=expected_pages,
                chunk_count=len(index.chunks), dimension=index.index.d, embedding_model=client.settings.model,
                ocr_pages=document_report["ocr_pages"], needs_ocr_pages=document_report["needs_ocr_pages"],
                low_text_pages=document_report["low_text_pages"],
                preview=[chunk.to_dict() for chunk in index.chunks[:3]],
                limitations="通过表示入库及持久化一致性检查通过；OCR 准确率、语义完整性需人工检查。")


def main():
    """只加载 Embedding 配置，不要求 LLM 配置，不进行检索和问答。"""
    output = ARTIFACTS_DIR / "pdf-ingestion-test"
    output.mkdir(parents=True, exist_ok=True)
    try:
        print("将调用 .env 配置的真实 Embedding 服务，正文片段会发送到该服务；不会调用聊天模型。", flush=True)
        with ModelClient(model_settings("EMBEDDING")) as client:
            report = run_pipeline(PDF_PATH, client)
    except Exception as error:
        # 不将数据库异常中的 SQL 参数或模型接口原始内容写进报告。
        report = dict(passed=False, error_type=type(error).__name__)
        print(f"入库测试失败：{type(error).__name__}", flush=True)
        if isinstance(error, (AssertionError, ValueError, RuntimeError)):
            print(str(error), flush=True)
    path = output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["passed"]:
        print(f"通过：处理 {report['processed_pages']} 页，生成 {report['chunk_count']} 个切片。", flush=True)
        print(f"SQLite：{report['database']}", flush=True)
        if report["needs_ocr_pages"]:
            print(f"仍需复核 OCR 的页：{report['needs_ocr_pages']}", flush=True)
    print(f"测试报告：{path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
