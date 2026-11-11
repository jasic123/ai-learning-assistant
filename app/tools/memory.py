"""通过 Agent 工具显式管理跨会话长期记忆。"""
from app.tools.base import Tool


class LongTermMemoryTool(Tool):
    name = "manage_long_term_memory"
    description = "查看、保存或删除长期记忆。只在用户明确要求记住/忘记，或稳定偏好对后续确有帮助时操作。"

    def __init__(self, store):
        self.store = store

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "remember", "forget"]},
            "key": {"type": "string", "minLength": 1, "maxLength": 64},
            "content": {"type": "string", "minLength": 1, "maxLength": 500},
        }, "required": ["action"], "additionalProperties": False}

    def execute(self, action, key=None, content=None):
        if action == "list":
            return {"memories": self.store.list()}
        if action == "remember":
            self.store.remember(key, content)
            return {"saved": True, "key": key}
        if action == "forget":
            return {"deleted": self.store.forget(key), "key": key}
        raise ValueError("未知记忆操作")
