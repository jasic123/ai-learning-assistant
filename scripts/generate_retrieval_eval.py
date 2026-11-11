"""Generate source-grounded retrieval questions from an existing local index.

This sends sampled chunk text to the configured LLM API and may incur charges.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import model_settings
from app.llm.client import ModelClient
from app.rag.retrieval.retrievers import KnowledgeIndex


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "evaluations" / "rag_retrieval_300.jsonl"


def _quality(text):
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 100:
        return -1
    useful = len(re.findall(r"[\w\u3400-\u9fff]", compact))
    if useful / len(compact) < 0.55 or "�" in compact:
        return -1
    return min(useful, 800) + 15 * sum(mark in compact for mark in "。！？;{}()")


def select_anchors(chunks, count=300):
    """Choose content-rich chunks from evenly spaced pages in each source."""
    pages = defaultdict(lambda: defaultdict(list))
    for chunk in chunks:
        if _quality(chunk.text) >= 0:
            pages[chunk.source][chunk.page].append(chunk)
    sources = sorted(pages)
    if not sources:
        raise ValueError("索引里没有足够长且可读的切片")
    quotas = {source: count // len(sources) for source in sources}
    for source in sources[:count % len(sources)]:
        quotas[source] += 1

    selected = []
    for source in sources:
        available = sorted(pages[source])
        amount = min(quotas[source], len(available))
        if amount == 0:
            continue
        indexes = sorted({round(i * (len(available) - 1) / max(1, amount - 1))
                          for i in range(amount)})
        for index in indexes:
            page = available[index]
            selected.append(max(pages[source][page], key=lambda chunk: _quality(chunk.text)))
    if len(selected) < count:
        selected_ids = {chunk.id for chunk in selected}
        remainder = sorted((chunk for chunk in chunks if chunk.id not in selected_ids and _quality(chunk.text) >= 0),
                           key=lambda chunk: (chunk.source, chunk.page, chunk.offset))
        selected.extend(remainder[:count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"索引只有 {len(selected)} 个可用抽样切片，无法生成 {count} 题")
    return selected[:count]


def parse_questions(text, expected_ids):
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = "\n".join(line for line in candidate.splitlines()
                               if not line.strip().startswith("```"))
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("["), candidate.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("模型未返回 JSON 数组") from None
        payload = json.loads(candidate[start:end + 1])
    if not isinstance(payload, list):
        raise ValueError("模型返回的题目必须是 JSON 数组")
    by_id = {}
    for item in payload:
        if (not isinstance(item, dict) or item.get("id") not in expected_ids
            or item["id"] in by_id):
            raise ValueError("模型返回了重复或未知锚点编号")
        query = item.get("query")
        if query is not None and not isinstance(query, str):
            raise ValueError("模型问题字段须为字符串或 null")
        by_id[item["id"]] = query.strip() if query and query.strip() else None
    return {identifier: by_id.get(identifier) for identifier in expected_ids}


def _fallback_query(text):
    from app.rag.retrieval.bm25 import tokenize

    stop = {"python", "the", "and", "for", "with", "from", "this", "that", "可以", "如果", "一个", "以及"}
    terms, seen = [], set()
    for term in tokenize(text):
        if term not in stop and len(term) > 1 and term not in seen:
            terms.append(term)
            seen.add(term)
        if len(terms) == 4:
            break
    focus = "、".join(terms) or "本页知识点"
    return f"教材中关于{focus}的主要概念或处理方法是什么？"


def generate(index, client, *, count=300, batch_size=10, progress=None,
             initial_cases=(), on_batch=None):
    anchors = select_anchors(index.chunks, count)
    cases = list(initial_cases)
    seen = {re.sub(r"\W+", "", item["query"]).casefold() for item in cases}
    system = (
        "根据给定教材片段，为检索评测生成一个真实学生可能提出的中文问题。问题必须能由该片段直接回答，"
        "具体到该片段，不得抄写原句，不得引入片段之外的事实。每个输入编号必须恰好返回一次。"
        "问题写成完整自然问句，至少 12 个字符，不能只输出关键词。"
        '只返回 JSON 数组：[ {"id":"输入编号","query":"问题"} ]；不要答案、来源或 Markdown。'
    )
    if len(cases) > count:
        raise ValueError("续跑文件中的题目数超过目标数量")
    for start in range(len(cases), len(anchors), batch_size):
        batch = anchors[start:start + batch_size]
        identifiers = [f"q{start + i + 1:03d}" for i in range(len(batch))]
        payload = [{"id": identifier, "text": chunk.text[:1800]}
                   for identifier, chunk in zip(identifiers, batch)]
        response = client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        queries = parse_questions(response, set(identifiers))
        for identifier, chunk in zip(identifiers, batch):
            query = queries[identifier]
            flags = []
            if not query:
                query = _fallback_query(chunk.text)
                flags.append("fallback_query")
            if len(query) < 12:
                flags.append("short_query")
            if len(query) > 300:
                flags.append("long_query")
            if len(query) > 2000:
                query = query[:2000]
                flags.append("truncated_query")
            normalized = re.sub(r"\W+", "", query).casefold()
            if normalized in seen:
                query = _fallback_query(chunk.text)
                normalized = re.sub(r"\W+", "", query).casefold()
                flags.append("duplicate_replaced")
                if normalized in seen:
                    query = f"第{chunk.page}页中{query}"
                    normalized = re.sub(r"\W+", "", query).casefold()
            seen.add(normalized)
            cases.append({"id": identifier, "query": query,
                          "relevant": [{"source": chunk.source, "page": chunk.page,
                                        "chunk_id": chunk.id}],
                          "anchor_chunk_id": chunk.id, "generated": True,
                          "quality_flags": flags})
        if progress:
            progress(len(cases), count)
        if on_batch:
            on_batch(cases)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default="教育资料")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="从同名 .partial 文件续跑")
    args = parser.parse_args()
    try:
        if not 300 <= args.count <= 500 or not 1 <= args.batch_size <= 20:
            raise ValueError("count 须为 300～500，batch-size 须为 1～20")
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        if args.output.exists():
            raise ValueError(f"输出文件已存在，为避免覆盖而停止：{args.output}")
        if partial.exists() and not args.resume:
            raise ValueError(f"发现未完成文件，请加 --resume 续跑或自行检查：{partial}")
        initial_cases = []
        if partial.exists():
            initial_cases = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()
                             if line.strip()]
            expected = [f"q{i:03d}" for i in range(1, len(initial_cases) + 1)]
            if [item.get("id") for item in initial_cases] != expected:
                raise ValueError("续跑文件中的题号不连续或顺序异常")
            if any(not item.get("query") or not item.get("relevant") or not item.get("anchor_chunk_id")
                   for item in initial_cases):
                raise ValueError("续跑文件中存在不完整题目")
        index = KnowledgeIndex(args.kb)
        print(f"将从 {len(index.chunks)} 个切片中按书籍和页码抽样 {args.count} 条；每批 {args.batch_size} 条。")
        print("将把抽样正文发送给聊天模型；这里只调用 LLM，不调用 Embedding 或精排。", flush=True)

        def checkpoint(items):
            temporary = partial.with_suffix(partial.suffix + ".tmp")
            temporary.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in items),
                                 encoding="utf-8")
            temporary.replace(partial)

        with ModelClient(model_settings("LLM")) as client:
            cases = generate(
                index, client, count=args.count, batch_size=args.batch_size,
                initial_cases=initial_cases,
                progress=lambda done, total: print(f"已生成 {done}/{total}", flush=True),
                on_batch=checkpoint,
            )
        partial.replace(args.output)
        print(f"已生成 {len(cases)} 条候选题：{args.output}")
        flagged = sum(bool(case["quality_flags"]) for case in cases)
        print(f"长度需优先复核：{flagged} 条。自动生成集仍需抽查来源支持关系，再用于正式比较。")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"生成未完成：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
