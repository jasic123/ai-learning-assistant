"""RRF 排名融合：避免直接相加 BM25 与余弦分数这两种不同量纲。"""
import math

from app.rag.domain.models import SearchResult


def rrf_fuse(vector_hits, keyword_hits, *, rrf_k=60, limit=30,
             vector_weight=0.5, keyword_weight=0.5):
    """按配置权重融合排名；每路贡献为 weight / (rrf_k + rank)。"""
    if type(rrf_k) is not int or rrf_k < 1 or type(limit) is not int or limit < 1:
        raise ValueError("RRF 平滑常数和候选数量必须为正整数")
    weights = (vector_weight, keyword_weight)
    if (any(type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
            for weight in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6)):
        raise ValueError("向量与关键词权重必须是非负有限数，且总和为 1")
    scores, chunks = {}, {}
    for hits, weight in ((vector_hits, vector_weight), (keyword_hits, keyword_weight)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            identifier = hit.chunk.id
            # 同一通道内的重复 ID 只计分一次；跨通道同一 ID 累加贡献。
            if identifier in seen:
                continue
            seen.add(identifier)
            chunks.setdefault(identifier, hit.chunk)
            scores[identifier] = scores.get(identifier, 0.0) + weight / (rrf_k + rank)
    # Python 稳定排序让同分时保持首次召回顺序，便于复现和排查。
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [SearchResult(chunks[identifier], scores[identifier], score_type="rrf") for identifier in ordered]
