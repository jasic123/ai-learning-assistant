"""RAG 公共入口：提供检索服务与请求/响应类型，内部负责知识存储及召回。"""
from app.rag.domain.contracts import SearchRequest, SearchResponse
from app.rag.service import RagService

__all__ = ["RagService", "SearchRequest", "SearchResponse"]
