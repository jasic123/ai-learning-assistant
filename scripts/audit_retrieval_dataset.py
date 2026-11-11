"""Prepare a human-review queue for the generated retrieval evaluation set."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.retrieval.bm25 import _QUERY_STOP_WORDS, tokenize
from app.rag.retrieval.retrievers import KnowledgeIndex
from scripts.evaluate_retrieval import load_cases


def normalize_query(text):
    return [term for term in tokenize(text) if term not in _QUERY_STOP_WORDS and len(term) > 1]


def audit_case(case, chunks):
    gold = case["relevant"][0]
    chunk_id = case.get("anchor_chunk_id") or gold.get("chunk_id")
    chunk = chunks.get(chunk_id)
    flags = list(case.get("quality_flags", []))
    if chunk is None:
        flags.append("missing_anchor_chunk")
        anchor_text = ""
    else:
        anchor_text = chunk.text
    terms = normalize_query(case["query"])
    anchor_terms = set(tokenize(anchor_text))
    covered = sorted(set(terms) & anchor_terms)
    coverage = len(covered) / len(set(terms)) if terms else 0.0
    if coverage < 0.25:
        flags.append("low_term_coverage")
    if re.search(r"这本书|本书|主要讲|主要内容|是什么内容", case["query"]):
        flags.append("generic_question")
    if not anchor_text.strip():
        flags.append("empty_anchor")
    return {
        "id": case["id"], "query": case["query"],
        "source": gold["source"], "page": gold["page"], "chunk_id": chunk_id,
        "anchor_text": anchor_text[:3000], "query_terms": terms,
        "covered_terms": covered, "term_coverage": round(coverage, 4),
        "quality_flags": sorted(set(flags)), "review_status": "pending",
    }


def stratified_sample(cases, per_source=100):
    groups = defaultdict(list)
    for case in cases:
        groups[case["relevant"][0]["source"]].append(case)
    selected = []
    for source in sorted(groups):
        items = groups[source]
        amount = min(per_source, len(items))
        indexes = sorted({round(i * (len(items) - 1) / max(1, amount - 1))
                          for i in range(amount)})
        selected.extend(items[index] for index in indexes)
    return selected


def stratified_sample_total(cases, total=100):
    groups = defaultdict(list)
    for case in cases:
        groups[case["relevant"][0]["source"]].append(case)
    sources = sorted(groups)
    if not sources or total < len(sources):
        raise ValueError("样本总数必须不小于来源数量")
    quotas = {source: total // len(sources) for source in sources}
    for source in sources[:total % len(sources)]:
        quotas[source] += 1
    selected = []
    for source in sources:
        selected.extend(stratified_sample(groups[source], quotas[source]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default="教育资料")
    parser.add_argument("--dataset", type=Path, default=Path("evaluations/rag_retrieval_300.jsonl"))
    parser.add_argument("--total", type=int, default=100,
                        help="审核总样本数，按来源尽量均衡；默认 100")
    parser.add_argument("--output", type=Path, default=Path("evaluations/rag_retrieval_review_100.jsonl"))
    args = parser.parse_args()
    try:
        if args.total < 3:
            raise ValueError("total 至少为 3")
        cases = load_cases(args.dataset)
        index = KnowledgeIndex(args.kb)
        chunks = {chunk.id: chunk for chunk in index.chunks}
        selected = stratified_sample_total(cases, args.total)
        rows = [audit_case(case, chunks) for case in selected]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                               encoding="utf-8")
        flagged = sum(bool(row["quality_flags"]) for row in rows)
        print(f"待审核样本：{len(rows)} 条；自动标记风险：{flagged} 条")
        print(f"输出：{args.output}")
        print("审核时将 review_status 改为 approved、rejected 或 needs_relabel；不要直接改原始评测集。")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"审核清单生成失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
