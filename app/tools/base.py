"""工具契约：借鉴 NanoClaw，统一名称、参数定义和执行入口。"""
from abc import ABC, abstractmethod


class Tool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """给模型看的 JSON Schema；执行端仍须校验实际参数。"""

    @abstractmethod
    def execute(self, **arguments) -> dict:
        """返回业务结果，不调用聊天模型生成最终回答。"""

    def to_definition(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}
