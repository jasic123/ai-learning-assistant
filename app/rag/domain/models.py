# 数据对象边界：Document 表示清洗后页面，Chunk 表示切片，SearchResult 表示一次检索命中。
"""RAG 基础数据类型：用明确的字段替代处理链路中零散的字典。"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    """清洗后的文档单元；当前每个对象对应 PDF 的一页，不代表整本书。"""

    source: str  # 知识库内的相对文件名
    page: int    # PDF 物理页码，从 1 开始
    text: str    # 该页清洗后的文字，保留代码缩进


@dataclass(frozen=True)
class Chunk:
    """从 Document 切分出的片段，与来源文档是关联关系，不使用继承。"""

    # 稳定切片标识由来源、页码、偏移和正文共同计算，便于追溯而非随机编号。
    id: str
    source: str
    page: int
    text: str
    offset: int = 0  # 在清洗后页面文字中的起始字符位置，不是 PDF 字节偏移
    # 明确 offset 的坐标系，避免将清洗后字符位置误当作原始 PDF 字节位置。
    offset_basis: str = "cleaned_page_text"
    # 兼容缺少版本字段的历史记录；新切片由切分器写入当前清洗版本。
    preprocessing_version: str = "legacy"
    parent_id: str = ""
    position: int = 0
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_type: str = "prose"

    def to_dict(self) -> dict:
        """只在 JSON 保存、终端输出等边界转换成字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """从索引文件恢复；旧索引缺少可选字段时使用默认值。"""
        if not isinstance(data, dict):
            raise ValueError("片段数据必须是 JSON 对象")
        if any(not isinstance(data.get(key), str) or not data[key].strip()
               for key in ("id", "source", "text")):
            raise ValueError("片段缺少有效的 id、source 或 text")
        # bool 在 Python 中属于 int，故使用 type 检查，拒绝 JSON true 作为页码。
        if type(data.get("page")) is not int or data["page"] < 1:
            raise ValueError("片段页码必须是从 1 开始的整数")
        offset = data.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("片段偏移必须是非负整数")
        for key in ("offset_basis", "preprocessing_version"):
            if key in data and not isinstance(data[key], str):
                raise ValueError(f"片段的 {key} 必须是字符串")
        if "parent_id" in data and not isinstance(data["parent_id"], str):
            raise ValueError("片段的 parent_id 必须是字符串")
        position = data.get("position", 0)
        if type(position) is not int or position < 0:
            raise ValueError("片段的 position 必须是非负整数")
        for key in ("prev_chunk_id", "next_chunk_id"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ValueError(f"片段的 {key} 必须是字符串或 null")
        chunk_type = data.get("chunk_type", "prose")
        if chunk_type not in {"prose", "code", "question", "mixed"}:
            raise ValueError("片段的 chunk_type 无效")
        return cls(
            id=data["id"], source=data["source"], page=data["page"], text=data["text"],
            offset=offset, offset_basis=data.get("offset_basis", "cleaned_page_text"),
            preprocessing_version=data.get("preprocessing_version", "legacy"),
            parent_id=data.get("parent_id", ""), position=position,
            prev_chunk_id=data.get("prev_chunk_id"), next_chunk_id=data.get("next_chunk_id"),
            chunk_type=chunk_type,
        )


@dataclass(frozen=True)
class SearchResult:
    """检索命中 = 原片段 + 本次排名分数；检索信息不写回持久化片段。"""

    chunk: Chunk
    # 当前阶段分数：余弦、BM25、RRF 或精排分数，均不表示答案正确率。
    score: float
    score_type: str = "cosine"  # cosine / bm25 / rrf / rerank，不能跨类型比较数值。
    retrieval_score: float | None = None  # 精排时保留上一阶段分数，方便排查排序变化。
