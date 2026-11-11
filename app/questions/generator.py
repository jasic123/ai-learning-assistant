"""根据一个 RAG 片段生成一道人审选择题。"""
import json

from app.rag.domain.models import Chunk
from app.questions.models import Question


class QuestionGenerationError(RuntimeError):
    """模型没有生成可用且符合格式的题目。"""


class QuestionGenerator:
    def __init__(self, client):
        self.client = client

    def generate(self, chunk: Chunk, *, topic="", subject="") -> Question:
        if not isinstance(chunk, Chunk):
            raise ValueError("出题输入必须是 RAG 文本片段")
        if not isinstance(topic, str) or len(topic) > 2000:
            raise ValueError("出题主题必须是 2000 字符以内的文本")
        if not isinstance(subject, str) or len(subject) > 100:
            raise ValueError("学科必须是 100 字符以内的文本")

        messages = [
            {"role": "system", "content": (
                "你是教师命题助手。只根据用户提供的资料生成一道四选一单选题。"
                "题目必须围绕用户指定的出题主题，不得退化成资料中的其他泛化知识点。"
                "题目、选项、答案和解析都必须能由资料支持；资料不足时不要猜测。"
                '只返回 JSON 对象，格式为 {"stem":"题干",'
                '"options":{"A":"选项A","B":"选项B","C":"选项C","D":"选项D"},'
                '"answer":"A","explanation":"解析"}。'
                '资料不足时只返回 {"error":"insufficient_evidence"}。不要输出 Markdown。'
            )},
            {"role": "user", "content": "学科：\n" + (subject.strip() or "未指定")
             + "\n出题主题：\n" + (topic.strip() or "根据资料核心知识点")
             + "\n\n教材资料：\n" + chunk.text},
        ]
        try:
            payload = json.loads(self.client.chat(messages))
        except (TypeError, ValueError):
            raise QuestionGenerationError("模型没有返回有效的题目 JSON") from None

        if payload == {"error": "insufficient_evidence"}:
            raise QuestionGenerationError("资料片段不足以生成有依据的题目")
        if (not isinstance(payload, dict)
            or set(payload) != {"stem", "options", "answer", "explanation"}
            or not isinstance(payload["options"], dict)
            or set(payload["options"]) != {"A", "B", "C", "D"}):
            raise QuestionGenerationError("模型返回的题目字段不符合约定格式")

        try:
            return Question(
                stem=payload["stem"],
                options=tuple(payload["options"][label] for label in "ABCD"),
                answer=payload["answer"],
                explanation=payload["explanation"],
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.id,
                subject=subject.strip(),
                topic=topic.strip(),
            )
        except ValueError as error:
            raise QuestionGenerationError("模型返回的题目内容无效") from error
