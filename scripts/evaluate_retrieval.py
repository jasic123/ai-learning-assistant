"""Run the curated retrieval benchmark; vector modes call the configured embedding API."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import model_settings
from app.llm.client import ModelClient
from app.rag.retrieval.reranker import RerankerService
from app.rag.retrieval.retrievers import KnowledgeIndex
from app.rag.retrieval.context import expand_neighbor_chunks


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "evaluations" / "rag_retrieval.jsonl"


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(not item.get("id") or not item.get("query") or not item.get("relevant") for item in cases):
        raise ValueError("评测集必须至少包含一题，且每题包含 id、query 和 relevant")
    if len({item["id"] for item in cases}) != len(cases):
        raise ValueError("评测集 id 不得重复")
    for case in cases:
        if any(not item.get("source") or type(item.get("page")) is not int or item["page"] < 1
               for item in case["relevant"]):
            raise ValueError(f"评测题 {case['id']} 的相关来源须包含 source 和正整数 page")
    return cases


def _matched_gold(hit, relevant):
    return _matched_chunk(hit.chunk, relevant)


def _matched_chunk(chunk, relevant):
    for item in relevant:
        if chunk.source != item["source"]:
            continue
        if item.get("chunk_id"):
            if chunk.id == item["chunk_id"]:
                return (item["source"], item["page"], item["chunk_id"])
        elif chunk.page == item["page"]:
            return (item["source"], item["page"], None)
    return None


def evaluate(index, cases, *, mode, top_k, candidate_k, rrf_k, vector_weight,
             keyword_weight, client=None, reranker=None, filter_query_stop_words=False,
             expand_query=True, context_window=0, page_window=0):
    results, reciprocal_ranks, ndcgs = [], [], []
    recalls = {cutoff: [] for cutoff in (1, 3, 5)}
    hit_rates = {cutoff: [] for cutoff in (1, 3, 5)}
    context_recalls = {cutoff: [] for cutoff in (1, 3, 5)}
    context_hit_rates = {cutoff: [] for cutoff in (1, 3, 5)}
    page_context_recalls = {cutoff: [] for cutoff in (1, 3, 5)}
    page_context_hit_rates = {cutoff: [] for cutoff in (1, 3, 5)}
    page_context_chunk_counts = {cutoff: [] for cutoff in (1, 3, 5)}
    page_context_char_counts = {cutoff: [] for cutoff in (1, 3, 5)}
    for case in cases:
        hits = index.search(case["query"], client, top_k, mode=mode, candidate_k=candidate_k,
                            rrf_k=rrf_k, vector_weight=vector_weight,
                            keyword_weight=keyword_weight, reranker=reranker,
                            filter_query_stop_words=filter_query_stop_words,
                            expand_query=expand_query)
        relevant = case["relevant"]
        relevant_keys = {(item["source"], item["page"], item.get("chunk_id")) for item in relevant}
        matched = [_matched_gold(hit, relevant) for hit in hits]
        rank = next((i + 1 for i, key in enumerate(matched) if key is not None), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        for cutoff in recalls:
            found = {key for key in matched[:cutoff] if key is not None}
            recalls[cutoff].append(len(found) / len(relevant))
            hit_rates[cutoff].append(float(bool(found)))
            if context_window:
                context_chunks = expand_neighbor_chunks(
                    hits[:cutoff], index.chunks, window=context_window)
                context_found = {_matched_chunk(chunk, relevant) for chunk in context_chunks}
                context_found.discard(None)
                context_recalls[cutoff].append(len(context_found) / len(relevant))
                context_hit_rates[cutoff].append(float(bool(context_found)))
            if page_window:
                page_chunks = expand_neighbor_chunks(
                    hits[:cutoff], index.chunks, window=0, page_window=page_window)
                page_found = {_matched_chunk(chunk, relevant) for chunk in page_chunks}
                page_found.discard(None)
                page_context_recalls[cutoff].append(len(page_found) / len(relevant))
                page_context_hit_rates[cutoff].append(float(bool(page_found)))
                page_context_chunk_counts[cutoff].append(len(page_chunks))
                page_context_char_counts[cutoff].append(sum(len(chunk.text) for chunk in page_chunks))
        seen_relevant = set()
        discounts = []
        for rank, key in enumerate(matched, 1):
            if key is not None and key not in seen_relevant:
                discounts.append(1 / math.log2(rank + 1))
                seen_relevant.add(key)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), top_k) + 1))
        ndcgs.append(sum(discounts) / ideal if ideal else 0.0)
        results.append({
            "id": case["id"], "query": case["query"], "relevant": case["relevant"],
            "retrieved": [{"source": hit.chunk.source, "page": hit.chunk.page,
                           "chunk_id": hit.chunk.id, "score": hit.score,
                           "score_type": hit.score_type,
                           "relevant": matched[i] is not None}
                          for i, hit in enumerate(hits)],
            "expanded_context_chunk_ids": ([chunk.id for chunk in expand_neighbor_chunks(
                hits, index.chunks, window=context_window)] if context_window else []),
            "page_context_chunk_ids": ([chunk.id for chunk in expand_neighbor_chunks(
                hits, index.chunks, window=0, page_window=page_window)] if page_window else []),
        })
    mean = lambda values: round(sum(values) / len(values), 4) if values else 0.0
    metrics = {f"recall@{k}": mean(values) for k, values in recalls.items()}
    metrics.update({f"hit_rate@{k}": mean(values) for k, values in hit_rates.items()})
    metrics.update(mrr=mean(reciprocal_ranks), ndcg_at_k=mean(ndcgs))
    if context_window:
        metrics.update({f"context_recall@{k}": mean(values)
                        for k, values in context_recalls.items()})
        metrics.update({f"context_hit_rate@{k}": mean(values)
                        for k, values in context_hit_rates.items()})
    if page_window:
        metrics.update({f"page_context_recall@{k}": mean(values)
                        for k, values in page_context_recalls.items()})
        metrics.update({f"page_context_hit_rate@{k}": mean(values)
                        for k, values in page_context_hit_rates.items()})
        metrics.update({f"page_context_chunks@{k}": mean(values)
                        for k, values in page_context_chunk_counts.items()})
        metrics.update({f"page_context_chars@{k}": mean(values)
                        for k, values in page_context_char_counts.items()})
    return {"metrics": metrics, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default="教育资料")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("bm25", "vector", "hybrid", "all"), default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--vector-weight", type=float, default=0.5)
    parser.add_argument("--keyword-weight", type=float, default=0.5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--query-stopwords", action="store_true",
                        help="实验性对照：过滤查询中的低信息套话词，默认关闭")
    parser.add_argument("--no-query-expansion", action="store_true",
                        help="实验性对照：关闭教材术语查询扩展")
    parser.add_argument("--context-window", type=int, default=0,
                        help="评测上下文扩展：每个命中块附加同页前后 N 个块，仅统计覆盖率")
    parser.add_argument("--page-window", type=int, default=0,
                        help="额外评测同一 PDF 相邻 N 页的全部块，仅统计覆盖率和上下文规模")
    args = parser.parse_args()
    try:
        if (not 1 <= args.top_k <= 20 or not args.top_k <= args.candidate_k <= 200
                or args.rrf_k < 1 or args.context_window < 0 or args.page_window < 0):
            raise ValueError("top-k 须为 1～20，candidate-k 须为 top-k～200，rrf-k 须为正数")
        if min(args.vector_weight, args.keyword_weight) < 0 or abs(args.vector_weight + args.keyword_weight - 1) > 1e-6:
            raise ValueError("vector-weight 与 keyword-weight 须非负且总和为 1")
        cases = load_cases(args.dataset)
        modes = ("bm25", "vector", "hybrid") if args.mode == "all" else (args.mode,)
        index = KnowledgeIndex(args.kb)
        report = {"knowledgebase": args.kb, "generation": index.metadata["generation"],
                  "dataset": str(args.dataset), "case_count": len(cases), "top_k": args.top_k,
                  "candidate_k": args.candidate_k, "rrf_k": args.rrf_k,
                  "vector_weight": args.vector_weight, "keyword_weight": args.keyword_weight,
                  "query_stopword_filter": args.query_stopwords,
                  "query_expansion": not args.no_query_expansion,
                  "context_window": args.context_window,
                  "page_window": args.page_window,
                  "runs": {}}
        with ExitStack() as stack:
            embedding = None
            if any(mode in {"vector", "hybrid"} for mode in modes):
                embedding = stack.enter_context(ModelClient(model_settings("EMBEDDING")))
            reranker = None
            if args.rerank:
                reranker = RerankerService(stack.enter_context(ModelClient(model_settings("RERANK"))))
            for mode in modes:
                run = evaluate(index, cases, mode=mode, top_k=args.top_k,
                               candidate_k=args.candidate_k, rrf_k=args.rrf_k,
                               vector_weight=args.vector_weight, keyword_weight=args.keyword_weight,
                               client=embedding if mode != "bm25" else None, reranker=reranker,
                               filter_query_stop_words=args.query_stopwords,
                               expand_query=not args.no_query_expansion,
                               context_window=args.context_window,
                               page_window=args.page_window)
                report["runs"][mode] = run
                print(mode, json.dumps(run["metrics"], ensure_ascii=False), flush=True)
        output = DEFAULT_DATASET.parent
        report_path = output / f"retrieval-{args.dataset.stem}-{args.mode}{'-query-stopwords' if args.query_stopwords else ''}{'-no-query-expansion' if args.no_query_expansion else ''}{'-rerank' if args.rerank else ''}{f'-context-{args.context_window}' if args.context_window else ''}{f'-pages-{args.page_window}' if args.page_window else ''}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入：{report_path.relative_to(Path(__file__).resolve().parents[1])}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"评测未完成：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
