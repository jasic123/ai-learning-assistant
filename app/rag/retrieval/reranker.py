"""精排：仅向模型发送已召回候选，保留片段来源与精排前分数。"""
from dataclasses import replace
import math


class RerankerService:
    """使用独立 Rerank 客户端，对问题与候选正文成对打分。"""

    def __init__(self, client):
        self.client = client

    def rerank(self, query, candidates, *, top_k=4):
        if not query.strip() or type(top_k) is not int or top_k < 1:
            raise ValueError("问题不能为空，精排返回数量必须为正整数")
        if not candidates:
            return []
        scores = self.client.rerank(query, [hit.chunk.text for hit in candidates])
        if (len(scores) != len(candidates)
            or any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                   for score in scores)):
            raise ValueError("精排分数数量或数值无效")
        ranked = [replace(hit, score=float(score), score_type="rerank", retrieval_score=hit.score)
                  for hit, score in zip(candidates, scores, strict=True)]
        return sorted(ranked, key=lambda hit: hit.score, reverse=True)[:top_k]
