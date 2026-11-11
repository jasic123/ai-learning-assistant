"""RAG 对外契约：外层只提交检索请求、接收有来源的资料，不接触索引细节。"""
from dataclasses import dataclass
import math

from app.config import validate_name
from app.rag.domain.models import SearchResult


@dataclass(frozen=True)
class SearchRequest:
    """同步检索请求；创建时校验，避免无效参数触发文件访问或模型请求。"""

    knowledgebase: str
    query: str
    top_k: int = 4
    mode: str = "hybrid"
    candidate_k: int = 30
    rrf_k: int = 60
    vector_weight: float = 0.5
    keyword_weight: float = 0.5
    rerank: bool = False

    def __post_init__(self):
        validate_name(self.knowledgebase)
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("问题不能为空")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 20:
            raise ValueError("top_k 须为 1～20 的整数")
        if not isinstance(self.mode, str) or self.mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("检索模式须为 vector、bm25 或 hybrid")
        if type(self.candidate_k) is not int or not self.top_k <= self.candidate_k <= 200:
            raise ValueError("候选数量须为 top_k～200 的整数")
        if type(self.rrf_k) is not int or self.rrf_k < 1:
            raise ValueError("RRF 常数须为正整数")
        weights = (self.vector_weight, self.keyword_weight)
        if (any(type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
                for weight in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6)):
            raise ValueError("向量与关键词权重必须是非负有限数，且总和为 1")
        if type(self.rerank) is not bool:
            raise ValueError("rerank 须为布尔值")


@dataclass(frozen=True)
class SearchResponse:
    """保留实际读取的索引版本；来源编号只在本次响应内有效。"""

    knowledgebase: str
    query: str
    generation: str
    hits: tuple[SearchResult, ...]

    def to_dict(self):
        """转换为 Agent 工具可消费的 JSON 字典，不暴露 API 配置和磁盘绝对路径。"""
        return {
            "knowledgebase": self.knowledgebase,
            "query": self.query,
            "generation": self.generation,
            "count": len(self.hits),
            "sources": [
                {"citation": f"S{number}", "chunk_id": hit.chunk.id,
                 "source": hit.chunk.source, "page": hit.chunk.page, "text": hit.chunk.text,
                 "score": hit.score, "score_type": hit.score_type,
                 "retrieval_score": hit.retrieval_score}
                for number, hit in enumerate(self.hits, 1)
            ],
        }
