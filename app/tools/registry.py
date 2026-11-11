"""按工具名分发调用；错误转成结构化结果，原始异常不进入模型上下文。"""
import json

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError("工具名称重复")
        self._tools[tool.name] = tool

    def get_definitions(self):
        return [tool.to_definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": "unknown_tool"}
        try:
            if not isinstance(arguments, str) or len(arguments) > 8000:
                raise ValueError
            values = json.loads(arguments)
            if not isinstance(values, dict):
                raise ValueError
            # 严格拒绝 Python JSON 默认接受的 NaN/Infinity；不输出实际参数。
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_arguments"}
        try:
            data = tool.execute(**values)
            payload = {"ok": True, "data": data}
            if len(json.dumps(payload, ensure_ascii=False, allow_nan=False)) > 24000:
                return {"ok": False, "error": "tool_output_too_large"}
            return payload
        except Exception as error:
            # 工具可能访问远程模型；异常正文可能含凭据，只返回异常类型。
            return {"ok": False, "error": "tool_failed", "error_type": type(error).__name__}
