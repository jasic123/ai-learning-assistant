# 真实验收脚本：可能向模型服务发送正文；它与使用假模型的离线回归测试用途不同。
"""真实 RAG 验证：会调用已配置的 Embedding 和聊天服务，不属于离线测试。

从项目根目录运行 python -m scripts.validate_rag --help。
默认只检查手册第 16～20 页；报告不保存密钥或完整接口异常。
"""
from pathlib import Path
import sys

# 兼容直接运行脚本；所有业务路径仍以 app.config.PROJECT_ROOT 为基准。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import ExitStack
import json
import re
import shutil

from app.config import ARTIFACTS_DIR, PROJECT_ROOT, model_settings
from app.llm.client import ModelClient
from app.config import kb_paths
from app.rag.indexing.pipeline import build_index
from app.rag.retrieval.retrievers import KnowledgeIndex
from app.rag.retrieval.reranker import RerankerService
from app.rag.services.chat_service import answer_question


# 这些问题的答案已在测试正文中确认；不把 LLM 自评当作正确性依据。
QUESTIONS = [
    ("如何打开文件并逐行读取内容？", 20),
    ("Python 的交互模式可以用来做什么？", 17),
    ("Python 动态类型变量与值的关系是什么？", 18),
]


# 必要时重建小范围索引，再检查目标页召回与引用编号；结果写入本地验收报告。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/uploads/教育资料/Python参考手册(第4版).pdf"))
    parser.add_argument("--kb", default="Python手册验证")
    parser.add_argument("--rebuild", action="store_true", help="重新 OCR 并调用 Embedding；旧索引保留")
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank", action="store_true", help="调用 RERANK 配置的精排服务")
    args = parser.parse_args()
    # 相对文件路径以项目根目录为基准，不随运行目录变化。
    if not args.pdf.is_absolute():
        args.pdf = PROJECT_ROOT / args.pdf
    output = ARTIFACTS_DIR / "rag-validation"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(knowledgebase=args.kb, mode=args.mode, rerank=args.rerank,
                  candidate_k=args.candidate_k, rrf_k=args.rrf_k, cases=[], passed=False)
    try:
        if not 4 <= args.candidate_k <= 200 or args.rrf_k < 1:
            raise ValueError("候选数量须在 4～200 之间，RRF 常数须大于 0")
        uploads, root = kb_paths(args.kb)
        rebuild = args.rebuild or not (root / "CURRENT").is_file()
        embedding = model_settings("EMBEDDING") if rebuild or args.mode != "bm25" else None
        llm = model_settings("LLM")
        ranking = model_settings("RERANK") if args.rerank else None
        # 只有显式重建或缺少当前索引时才生成向量；其他情况下复用已保存版本。
        if rebuild:
            if not args.pdf.is_file():
                raise ValueError("指定 PDF 文件不存在")
            uploads.mkdir(parents=True, exist_ok=True)
            target = uploads / args.pdf.name
            if args.pdf.resolve() != target.resolve():
                shutil.copyfile(args.pdf, target)
            print("本次将正文片段发送到配置的 Embedding 服务；OCR 在本机执行。", flush=True)
            with ModelClient(embedding) as client:
                build_index(args.kb, client, start_page=16, max_pages=5, max_chunks=0,
                            progress=lambda done, total: print(f"向量化 {done}/{total}", flush=True))
        # 重新从磁盘加载，验证持久化结果；进程重启后也走同一个入口。
        index = KnowledgeIndex(args.kb)
        report.update(chunk_count=len(index.chunks), dimension=index.index.d,
                      source_pages=sorted({chunk.page for chunk in index.chunks}))
        with ExitStack() as stack:
            vectors = stack.enter_context(ModelClient(embedding)) if args.mode != "bm25" else None
            chat = stack.enter_context(ModelClient(llm))
            reranker = RerankerService(stack.enter_context(ModelClient(ranking))) if ranking else None
            for question, expected_page in QUESTIONS:
                hits = index.search(question, vectors, top_k=4, mode=args.mode,
                                    candidate_k=args.candidate_k, rrf_k=args.rrf_k, reranker=reranker)
                answer = answer_question(question, hits, chat)
                citations = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
                # 这里只核验引用编号存在且未越界，不验证引用是否真正支持答案结论。
                valid_citations = bool(citations) and all(1 <= value <= len(hits) for value in citations)
                found = any(hit.chunk.page == expected_page for hit in hits)
                case = dict(question=question, expected_page=expected_page, expected_page_retrieved=found,
                            citation_ids_valid=valid_citations, answer=answer,
                            sources=[dict(citation=f"S{i}", file=hit.chunk.source, page=hit.chunk.page,
                                          score=hit.score, score_type=hit.score_type, retrieval_score=hit.retrieval_score,
                                          text=hit.chunk.text) for i, hit in enumerate(hits, 1)])
                report["cases"].append(case)
                print(f"{question}：目标页召回={found}，引用编号有效={valid_citations}", flush=True)
        # 验收通过范围仅限这些固定问题的目标页召回和引用编号，不代表全面答案质量。
        report["passed"] = all(case["expected_page_retrieved"] and case["citation_ids_valid"]
                               for case in report["cases"])
    except (ValueError, RuntimeError, OSError) as error:
        report["error"] = str(error)
        print(f"验证未完成：{error}", flush=True)
    report["limitations"] = "只验证指定正文范围和引用编号；答案事实、引用对应关系、OCR 代码仍需人工核对。"
    path = output / f"{args.mode}{'-rerank' if args.rerank else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"验证报告：{path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
