"""Export approved rows from an LLM-reviewed retrieval audit file."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys


def export(input_path, output_path):
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    approved = [row for row in rows if row.get("auto_verdict") == "approved"
                and row.get("id") != "q204"]
    if not approved:
        raise ValueError("没有找到 auto_verdict=approved 的记录")
    ids = [row.get("id") for row in approved]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("审核集 ID 缺失或重复")
    cases = []
    for row in approved:
        if not row.get("source") or type(row.get("page")) is not int or not row.get("chunk_id"):
            raise ValueError(f"记录 {row.get('id')} 缺少来源页或 chunk_id")
        cases.append({
            "id": row["id"], "query": row["query"],
            "relevant": [{"source": row["source"], "page": row["page"],
                          "chunk_id": row["chunk_id"]}],
            "review_status": "approved",
            "review_reason": row.get("auto_reason", ""),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
                           encoding="utf-8")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evaluations/rag_retrieval_review_100_judged.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluations/rag_retrieval_approved_76.jsonl"))
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"输出文件已存在，为避免覆盖而停止：{args.output}")
        cases = export(args.input, args.output)
        print(f"已导出 {len(cases)} 条 approved 评测题：{args.output}")
        print(json.dumps(dict(Counter(case["relevant"][0]["source"] for case in cases)), ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
