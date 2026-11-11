"""回查本会话完整历史，补偿摘要和工具清理造成的细节损失。"""
from app.tools.base import Tool
import json


class HistoryReadTool(Tool):
    name = "recall_history"
    description = "按轮次读取本会话原始记录；摘要遗漏细节或需要旧资料原文时使用，可分页。"

    def __init__(self, sessions, session_id):
        self.sessions, self.session_id = sessions, session_id

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "turn_number": {"type": "integer", "minimum": 1},
            "start": {"type": "integer", "minimum": 0, "default": 0},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 6000, "default": 4000},
        }, "required": ["turn_number"], "additionalProperties": False}

    def execute(self, turn_number, start=0, max_chars=4000):
        if type(start) is not int or start < 0 or type(max_chars) is not int or not 1 <= max_chars <= 6000:
            raise ValueError("历史分页参数无效")
        text = self.sessions.read_turn(self.session_id, turn_number)
        if start > len(text):
            raise ValueError("起点超出历史长度")
        end = min(len(text), start + max_chars)
        fragment = text[start:end]
        sources = []
        # 为分页中出现的引用补充定位信息，即使原引用已不在压缩检查点内。
        for message in json.loads(text):
            if message.get("role") != "tool":
                continue
            data = json.loads(message["content"]).get("data", {})
            for source in data.get("sources", []):
                if source["citation"] in fragment:
                    sources.append({**{key: value for key, value in source.items() if key != "text"},
                                    "knowledgebase": source.get("knowledgebase", data.get("knowledgebase")),
                                    "generation": source.get("generation", data.get("generation"))})
        return {"turn_number": turn_number, "total_chars": len(text), "start": start,
                "next_start": end if end < len(text) else None,
                "content": fragment, "sources": sources,
                "format": "原始JSON文本分页，单页可能不是完整JSON"}
