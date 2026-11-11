# 命令分工：inspect 本地解析；ingest、向量/混合检索调用 Embedding；精排和聊天分别按需调用。
"""终端入口：python -m app.rag --help。"""
import argparse
import json
import sys

from app.config import model_settings
from app.llm.client import ModelClient
from app.rag.ingestion.pipeline import document_chunks
from app.rag.indexing.embeddings import float_vectors
from app.rag.indexing.pipeline import build_index
from app.rag import RagService, SearchRequest
from app.rag.services.chat_service import answer_question


# 解析子命令并调用已有服务；本地预览、建库、检索和回答分别走对应分支。
def main():
    parser = argparse.ArgumentParser(description="PDF RAG：本地预览、入库、检索、带来源问答")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="用非私有短文本测试模型接口")
    smoke.add_argument("--kind", choices=("all", "llm", "embedding"), default="all")
    for name in ("inspect", "ingest"):
        sub = commands.add_parser(name, help="本地解析，不调用 API" if name == "inspect" else "将文本发送至 Embedding API 并建库")
        sub.add_argument("--kb", required=True)
        sub.add_argument("--max-pages", type=int, default=0, help="每份 PDF 页数上限，0 为全部")
        sub.add_argument("--start-page", type=int, default=1, help="每份 PDF 的起始物理页码，从 1 开始")
        sub.add_argument("--max-chunks", type=int, default=0, help="总切片上限，0 为全部")
        sub.add_argument("--chunk-size", type=int, default=400)
        sub.add_argument("--overlap", type=int, default=60)
        sub.add_argument("--no-ocr", action="store_true", help="仅标记待 OCR 页面，不执行本地识别")
        if name == "ingest":
            sub.add_argument("--batch-size", type=int, default=16)
    for name in ("search", "ask"):
        sub = commands.add_parser(name)
        sub.add_argument("--kb", required=True)
        sub.add_argument("--question", required=True)
        sub.add_argument("--top-k", type=int, default=4)
        sub.add_argument("--mode", choices=("vector", "bm25", "hybrid"), default="hybrid",
                         help="默认 FAISS + BM25 的 RRF 融合；bm25 不调用 Embedding")
        sub.add_argument("--candidate-k", type=int, default=30, help="每路及融合后候选上限，top-k～200")
        sub.add_argument("--rrf-k", type=int, default=60, help="RRF 排名平滑常数")
        sub.add_argument("--vector-weight", type=float, default=0.5, help="混合检索中的向量召回权重；与关键词权重之和须为 1")
        sub.add_argument("--keyword-weight", type=float, default=0.5, help="混合检索中的 BM25 召回权重；与向量权重之和须为 1")
        sub.add_argument("--rerank", action="store_true", help="用独立 RERANK 配置调用精排接口")
    args = parser.parse_args()
    try:
        # 连通测试只使用非私有短文本；通过并不代表真实资料召回质量已通过。
        if args.command == "smoke":
            if args.kind in {"all", "embedding"}:
                with ModelClient(model_settings("EMBEDDING")) as client:
                    vector = float_vectors(client.embed(["Python 是一种编程语言。"]), 1)
                    print(f"Embedding 成功，维度：{vector.shape[1]}")
            if args.kind in {"all", "llm"}:
                with ModelClient(model_settings("LLM")) as client:
                    print("LLM 成功：", client.chat([dict(role="user", content="请只回复：连接成功")]))
            return 0
        if args.command in {"inspect", "ingest"}:
            options = {k: getattr(args, k) for k in ("max_pages", "max_chunks", "chunk_size", "overlap", "start_page")}
            # 命令行的禁用开关取反后，转换为服务函数的正向 OCR 参数。
            options["ocr_enabled"] = not args.no_ocr
            if args.command == "inspect":
                chunks, reports = document_chunks(args.kb, **options)
                # 数据类型在内部保持一致，仅在终端 JSON 输出时转成字典。
                preview = [chunk.to_dict() for chunk in chunks[:2]]
                print(json.dumps(dict(count=len(chunks), documents=reports, preview=preview), ensure_ascii=False, indent=2))
                if not chunks:
                    print("没有可用文字，请检查报告中的 OCR 状态和低文字量页面。", file=sys.stderr)
                    return 1
            else:
                with ModelClient(model_settings("EMBEDDING")) as client:
                    print("即将把资料文本发送至所配置的 Embedding 服务。建议首次限制页数和切片数。", flush=True)
                    meta = build_index(args.kb, client, batch_size=args.batch_size,
                                       progress=lambda n, total: print(f"向量化 {n}/{total}", flush=True), **options)
                print(json.dumps(meta, ensure_ascii=False, indent=2))
            return 0
        request = SearchRequest(args.kb, args.question, top_k=args.top_k, mode=args.mode,
                                candidate_k=args.candidate_k, rrf_k=args.rrf_k,
                                vector_weight=args.vector_weight, keyword_weight=args.keyword_weight,
                                rerank=args.rerank)
        # 独立终端问答才需要聊天配置；Agent 知识工具只查询，不生成另一份最终答案。
        llm = model_settings("LLM") if args.command == "ask" else None
        result = RagService().search(request)
        hits = result.hits
        if not hits:
            print("当前知识库未召回匹配资料。")
        if args.command == "ask":
            with ModelClient(llm) as client:
                print(answer_question(args.question, hits, client))
            print("\n检索来源（请核验回答引用是否恰当）：")
        for number, hit in enumerate(hits, 1):
            print(f"[S{number}] {hit.chunk.source} 第 {hit.chunk.page} 页；{hit.score_type} 分数 {hit.score:.6f}")
            if args.command == "search":
                print(hit.chunk.text, "\n")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 将命令结果转换为进程退出状态；导入模块时不运行命令。
    raise SystemExit(main())
