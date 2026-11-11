"""Use the configured LLM to judge whether each anchor chunk answers its query."""
import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import model_settings
from app.llm.client import ModelClient


def parse_judgement(text, expected_ids):
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
        raise ValueError("模型返回结果不是 JSON 数组")
    allowed = {"approved", "rejected", "needs_relabel"}
    result = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("id") not in expected_ids or item["id"] in result:
            raise ValueError("模型返回了未知或重复的题目编号")
        verdict = item.get("verdict")
        reason = item.get("reason")
        if verdict not in allowed or not isinstance(reason, str) or not reason.strip():
            raise ValueError("模型返回的审核字段无效")
        result[item["id"]] = {"verdict": verdict, "reason": reason.strip()[:500]}
    for identifier in expected_ids:
        result.setdefault(identifier, {"verdict": "needs_relabel", "reason": "模型未返回该题的审核结果，需人工复核。"})
    return result


def judge(input_path, output_path, *, batch_size=10, initial_rows=()):
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("审核文件为空")
    system = (
        "你是 RAG 评测集审核员。判断给定 anchor_text 是否足以回答 query。"
        "只能依据 anchor_text，不要补充外部知识。"
        "approved：正文直接包含答案或足够支持答案；"
        "rejected：正文与问题无关，或明显无法回答；"
        "needs_relabel：问题有意义但当前切片不完整，需要换来源或补充相邻切片。"
        '只返回 JSON 数组：[{"id":"...","verdict":"approved/rejected/needs_relabel","reason":"简短中文理由"}]。'
    )
    judged = list(initial_rows)
    with ModelClient(model_settings("LLM")) as client:
        for start in range(len(judged), len(rows), batch_size):
            batch = rows[start:start + batch_size]
            payload = [{"id": row["id"], "query": row["query"],
                        "anchor_text": row["anchor_text"][:3000]} for row in batch]
            raw = client.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
            judgements = parse_judgement(raw, {row["id"] for row in batch})
            for row in batch:
                item = dict(row)
                item["auto_verdict"] = judgements[row["id"]]["verdict"]
                item["auto_reason"] = judgements[row["id"]]["reason"]
                item["review_status"] = judgements[row["id"]]["verdict"]
                judged.append(item)
            print(f"已审核 {len(judged)}/{len(rows)}", flush=True)
            temporary = output_path.with_suffix(output_path.suffix + ".partial")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judged), encoding="utf-8")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judged), encoding="utf-8")
    temporary.replace(output_path)
    return judged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evaluations/rag_retrieval_review_100.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluations/rag_retrieval_review_100_judged.jsonl"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if not 1 <= args.batch_size <= 20:
            raise ValueError("batch-size 必须为 1～20")
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        if args.output.exists():
            raise ValueError(f"输出文件已存在，为避免覆盖而停止：{args.output}")
        initial_rows = []
        if partial.exists():
            if not args.resume:
                raise ValueError(f"发现未完成文件，请加 --resume：{partial}")
            initial_rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = judge(args.input, args.output, batch_size=args.batch_size, initial_rows=initial_rows)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        partial.replace(args.output)
        counts = {}
        for row in rows:
            counts[row["auto_verdict"]] = counts.get(row["auto_verdict"], 0) + 1
        print(f"审核完成：{args.output}")
        print(json.dumps(counts, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"自动审核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
