"""题目领域模型。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """基于单个知识库片段生成、等待教师审核的四选一题目。"""

    stem: str
    options: tuple[str, str, str, str]
    answer: str
    explanation: str
    source: str
    page: int
    chunk_id: str
    status: str = "draft"
    subject: str = ""
    topic: str = ""

    def __post_init__(self):
        if not isinstance(self.stem, str) or not self.stem.strip():
            raise ValueError("题干不能为空")
        if (not isinstance(self.options, tuple) or len(self.options) != 4
            or any(not isinstance(option, str) or not option.strip() for option in self.options)):
            raise ValueError("选项必须是四个非空字符串")
        if not isinstance(self.answer, str) or self.answer not in {"A", "B", "C", "D"}:
            raise ValueError("答案编号必须是 A、B、C 或 D")
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("解析不能为空")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("来源文件不能为空")
        if type(self.page) is not int or self.page < 1:
            raise ValueError("来源页码必须是正整数")
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("来源片段 ID 不能为空")
        if not isinstance(self.status, str) or self.status not in {"draft", "approved", "rejected"}:
            raise ValueError("题目状态必须是 draft、approved 或 rejected")
        if not isinstance(self.subject, str) or len(self.subject) > 100:
            raise ValueError("学科必须是 100 字符以内的文本")
        if not isinstance(self.topic, str) or len(self.topic) > 2000:
            raise ValueError("知识点必须是 2000 字符以内的文本")

    def to_dict(self) -> dict:
        return {
            "stem": self.stem,
            "options": dict(zip("ABCD", self.options)),
            "answer": self.answer,
            "explanation": self.explanation,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "status": self.status,
            "subject": self.subject,
            "topic": self.topic,
        }
