"""项目工作区内的受限文本文件读写。"""
from pathlib import Path

from app import config
from app.tools.base import Tool


class WorkspaceFileTool(Tool):
    name = "workspace_file"
    description = "在项目 data/workspace 工作区读取或写入文本文件，不访问工作区以外的路径。"
    allowed_suffixes = {".txt", ".md", ".json", ".csv"}
    max_bytes = 32_000
    read_chars = 3_000

    def __init__(self, data_dir=None):
        self.root = Path(data_dir) if data_dir is not None else config.DATA_DIR
        self.root = self.root / "workspace"

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["read", "write"]},
            "path": {"type": "string", "minLength": 1, "maxLength": 240},
            "content": {"type": "string", "maxLength": self.max_bytes},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
        }, "required": ["operation", "path"], "additionalProperties": False}

    def _path(self, name):
        if not isinstance(name, str) or not name.strip() or len(name) > 240:
            raise ValueError("文件路径无效")
        relative = Path(name)
        if relative.is_absolute() or relative.drive or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("只允许使用工作区内的相对路径")
        if relative.suffix.lower() not in self.allowed_suffixes:
            raise ValueError("仅支持 txt、md、json 和 csv 文本文件")
        target = self.root / relative
        if not target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("文件路径不得离开工作区")
        return target

    def execute(self, operation, path, content=None, offset=0):
        target = self._path(path)
        if operation == "read":
            if type(offset) is not int or offset < 0:
                raise ValueError("读取位置必须是非负整数")
            raw = target.read_bytes()
            if len(raw) > self.max_bytes:
                raise ValueError("文件超过单次读取上限 32000 字节")
            text = raw.decode("utf-8")
            if offset > len(text):
                raise ValueError("读取位置超过文件长度")
            end = min(len(text), offset + self.read_chars)
            return {"path": target.relative_to(self.root).as_posix(), "content": text[offset:end],
                    "total_chars": len(text), "offset": offset,
                    "next_offset": end if end < len(text) else None}
        if operation != "write":
            raise ValueError("操作只能是 read 或 write")
        if not isinstance(content, str):
            raise ValueError("写入内容必须是文本")
        raw = content.encode("utf-8")
        if len(raw) > self.max_bytes:
            raise ValueError("单次写入上限为 32000 字节")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("文件路径不得离开工作区")
        target.write_bytes(raw)
        return {"path": target.relative_to(self.root).as_posix(), "bytes": len(raw)}
