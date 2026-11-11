"""检索流水线：FAISS 与 BM25 双路召回 → RRF → 可选 Rerank。"""
import re

from app.config import kb_paths
from app.rag.domain.models import SearchResult
from app.rag.indexing.embeddings import float_vectors
from app.rag.indexing.vector_store import load_index
from app.rag.retrieval.fusion import rrf_fuse


class KnowledgeIndex:
    """在内存中加载固定的已发布版本；CURRENT 更新后需重新创建实例才会刷新。"""

    def __init__(self, name):
        _, root = kb_paths(name)
        if not (root / "CURRENT").is_file():
            raise ValueError("知识库尚未入库，请先执行 ingest")
        generation = (root / "CURRENT").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("索引版本标记损坏")
        self.index, self.chunks, self.metadata = load_index(root / generation)
        self._bm25 = None  # 基于本实例的固定版本延迟构建，避免与 CURRENT 的其他版本混用。

    def search(self, query: str, client=None, top_k: int = 4, *, mode="hybrid",
               candidate_k=30, rrf_k=60, vector_weight=0.5, keyword_weight=0.5,
               reranker=None, filter_query_stop_words=False, expand_query=True) -> list[SearchResult]:
        """candidate_k 是每路召回及融合后候选上限；top_k 是最终返回数量。

        bm25 模式完全本地运行；hybrid 按 vector_weight/keyword_weight 融合；传入 reranker 才执行精排。
        显式请求精排时让失败向上抛出，不用普通检索伪装精排成功。
        """
        if not query.strip() or type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("问题不能为空，top_k 须在 1～20 之间")
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("检索模式必须为 vector、bm25 或 hybrid")
        if type(candidate_k) is not int or not top_k <= candidate_k <= 200:
            raise ValueError("候选数量须不小于 top_k，且不超过 200")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("RRF 平滑常数必须为正整数")
        vectors = self._vector_search(query, client, candidate_k) if mode != "bm25" else []
        keywords = []
        if mode != "vector":
            if self._bm25 is None:
                from app.rag.retrieval.bm25 import BM25Retriever
                self._bm25 = BM25Retriever(self.chunks)
            keywords = self._bm25.search(query, candidate_k,
                                         filter_query_stop_words=filter_query_stop_words,
                                         expand_query=expand_query)
        if mode == "hybrid":
            candidates = rrf_fuse(vectors, keywords, rrf_k=rrf_k, limit=candidate_k,
                                  vector_weight=vector_weight, keyword_weight=keyword_weight)
        else:
            candidates = vectors if mode == "vector" else keywords
        if reranker is not None:
            return reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def _vector_search(self, query, client, limit):
        """文档和问题使用同一模型及归一化规则，保持 FAISS 行号与 Chunk 顺序一致。"""
        if client is None:
            raise ValueError("向量及混合检索需要 Embedding 客户端")
        if (client.settings.model != self.metadata.get("embedding_model")
            or client.settings.base_url != self.metadata.get("embedding_base_url")):
            raise ValueError("Embedding 服务或模型与索引不一致，请恢复配置或重新入库")
        vector = float_vectors(client.embed([query]), 1)
        if vector.shape[1] != self.index.d:
            raise ValueError("问题向量维度与索引不同")
        # 限制返回数不超过现有向量数；行号与保存 chunks 时的顺序对应。
        scores, ids = self.index.search(vector, min(limit, self.index.ntotal))
        # 相似度属于本次检索结果，不写回 Chunk，也不污染索引中的原始片段。
        return [SearchResult(chunk=self.chunks[int(i)], score=float(score))
                for score, i in zip(scores[0], ids[0]) if i >= 0]
