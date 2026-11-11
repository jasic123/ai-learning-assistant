"""RAG 对外服务：Agent、工具和终端通过此入口查询，不自行组装底层组件。"""
from contextlib import ExitStack

from app.config import model_settings
from app.llm.client import ModelClient
from app.rag.domain.contracts import SearchRequest, SearchResponse
from app.rag.retrieval.reranker import RerankerService
from app.rag.retrieval.retrievers import KnowledgeIndex


class RagService:
    """无连接状态的同步服务；创建或导入不加载模型、索引或数据库。"""

    def search(self, request: SearchRequest) -> SearchResponse:
        if not isinstance(request, SearchRequest):
            raise TypeError("请传入 SearchRequest 检索请求")
        # 每次读取 CURRENT 并加载固定版本，本次检索内部不跨版本混用。
        # 当前单机规模按请求加载；需要更高吞吐时再按版本加入缓存。
        index = KnowledgeIndex(request.knowledgebase)
        embedding = model_settings("EMBEDDING") if request.mode != "bm25" else None
        ranking = model_settings("RERANK") if request.rerank else None
        # 先校验全部配置再创建客户端；任何中间失败都会关闭已经创建的连接。
        with ExitStack() as stack:
            client = stack.enter_context(ModelClient(embedding)) if embedding else None
            reranker = RerankerService(stack.enter_context(ModelClient(ranking))) if ranking else None
            hits = index.search(
                request.query, client, request.top_k, mode=request.mode,
                candidate_k=request.candidate_k, rrf_k=request.rrf_k,
                vector_weight=request.vector_weight, keyword_weight=request.keyword_weight,
                reranker=reranker,
            )
        return SearchResponse(request.knowledgebase, request.query, index.metadata["generation"], tuple(hits))
