"""Turn OCR text into editable fields without inventing missing answers."""
import json
import re

from app.config import model_settings
from app.llm.client import ModelClient


def structure_question_text(text, on_delta=None):
    if not isinstance(text, str) or not text.strip():
        return "其他", "", "", "", "", "", "", None, "", "请先粘贴或识别题目原文。"
    messages = [
        {"role": "system", "content": (
            "你负责把试题原文整理成结构化字段。只能提取原文明确給出的内容，不得编造缺失内容或猜答案。"
            '只返回 JSON：{"subject":"计算机/英语/数学/其他","topic":"知识点",'
            '"stem":"完整题干","options":{"A":"...","B":"...","C":"...","D":"..."},'
            '"answer":"A/B/C/D 或空字符串","explanation":"答案解析"}。'
            "根据内容判断学科：极限、导数、积分等归为数学；编程、算法和计算机系统归为计算机；"
            "英语语言题归为英语；无法判断时归为其他。尽可能完整保留题干，必须保留公式、数字、变量和条件。"
            "不得只输出‘求极限’等指令而遗漏其后的公式；答案未明确给出时留空，不要自行求解。"
        )},
        {"role": "user", "content": "整理以下试题原文：\n" + text[:30000]},
    ]
    try:
        with ModelClient(model_settings("LLM")) as client:
            raw = client.chat(messages, on_delta=on_delta) if on_delta else client.chat(messages)
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        if not isinstance(data, dict) or not isinstance(data.get("options"), dict):
            raise ValueError("模型没有返回结构化选项")
        options = data["options"]
        subject = data.get("subject")
        subject = subject if subject in {"计算机", "英语", "数学", "其他"} else "其他"
        subject = infer_question_subject(text, subject)
        answer = data.get("answer")
        answer = answer if answer in {"A", "B", "C", "D"} else None
        stem, restored_math = restore_math_lines(text, data.get("stem", ""), subject)
        note = "题目已自动整理，请核对学科、题干、选项、公式和答案后保存。"
        if restored_math:
            note += " 已从 OCR 原文补入模型可能遗漏的数学表达式。"
        if not answer:
            note += " 原文中没有明确答案，请自行选择。"
        return (subject, data.get("topic", ""), stem, options.get("A", ""),
                options.get("B", ""), options.get("C", ""), options.get("D", ""),
                answer, data.get("explanation", "") or "暂无解析，可在下方和 AI 讨论这道题。", note)
    except Exception as error:
        return "其他", "", "", "", "", "", "", None, "", f"自动整理失败：{error}。原文已保留，请重试或校对填写。"


def restore_math_lines(source, stem, subject):
    """Keep math expressions from OCR when the model omitted them from the stem."""
    stem = stem.strip()
    if subject != "数学" or not isinstance(source, str):
        return stem, False
    choice_line = re.compile(r"^\s*(?:[A-DＡ-Ｄ]\s*[.、．:：)]|[（(]\s*[A-DＡ-Ｄ]\s*[）)])")
    math_line = re.compile(
        r"(?:\\(?:lim|sin|cos|tan|int|sum)|\blim\b|[=<>≤≥≈∫∑√]|\w\s*(?:\+|-|\*|/|\^|->)\s*\w)",
        re.IGNORECASE,
    )
    candidates = []
    normalized_stem = re.sub(r"\s+", "", stem).lower()
    for line in source.splitlines():
        line = line.strip()
        if choice_line.match(line):
            break
        if math_line.search(line) and re.sub(r"\s+", "", line).lower() not in normalized_stem:
            candidates.append(line)
    if not candidates:
        return stem, False
    expression = "；".join(candidates)
    return (f"{stem.rstrip('：:；;，,')}：{expression}" if stem else expression), True


def infer_question_subject(text, model_subject="其他"):
    if re.search(r"高等数学|微积分|极限|导数|积分|微分方程|数列|级数|矩阵|线性代数", text):
        return "数学"
    return model_subject if model_subject in {"计算机", "英语", "数学", "其他"} else "其他"
