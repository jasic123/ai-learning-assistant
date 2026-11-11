"""模型返回的数据类型：将 SDK 对象与 Agent 编排分开。"""
from dataclasses import dataclass


class ContextWindowExceeded(RuntimeError):
    """提供商明确报告上下文超限；仅此错误允许压缩后重试。"""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # 保留模型原始 JSON 参数；无效 JSON 由工具注册表反馈给模型修正。
    arguments: str


@dataclass(frozen=True)
class ModelReply:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None
    input_tokens: int | None = None

    def to_message(self):
        """还原 assistant 消息，后续 tool 结果必须关联这里的调用 ID。"""
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
            # 部分兼容服务要求下一轮保留此扩展字段，不在终端展示。
            if self.reasoning_content:
                message["reasoning_content"] = self.reasoning_content
        return message
