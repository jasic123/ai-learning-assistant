"""Embedding 处理：校验数值、归一化和批量编码，不写索引或业务数据库。"""
import faiss
import numpy as np


def float_vectors(vectors, count):
    # 先统一为 FAISS 使用的 float32 二维矩阵，并检查数量、维度和数值。
    try:
        matrix = np.asarray(vectors, dtype="float32")
    except (TypeError, ValueError):
        raise ValueError("Embedding 不是等长数值向量") from None
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embedding 数量、维度或数值无效")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Embedding 不得为零向量")
    # FAISS 接收连续内存布局的 float32 数组，避免输入矩阵布局不兼容。
    matrix = np.ascontiguousarray(matrix)
    # 文档和问题都归一化后，内积检索结果相当于余弦相似度；该分数不是答案可信度。
    faiss.normalize_L2(matrix)
    return matrix


def _split_embedding_text(text):
    middle = len(text) // 2
    boundaries = [text.rfind(mark, 1, middle) for mark in ("\n", "。", ". ", " ")]
    boundary = max(boundaries)
    split_at = boundary + (1 if text[boundary:boundary + 1] == "\n" else 0) if boundary >= middle // 2 else middle
    return text[:split_at].strip(), text[split_at:].strip()


def _embed_one(client, text):
    try:
        return client.embed([text])[0]
    except RuntimeError as error:
        if "HTTP 400" not in str(error) or len(text) <= 64:
            raise
        left, right = _split_embedding_text(text)
        if not left or not right:
            raise
        vectors = [_embed_one(client, part) for part in (left, right)]
        return np.mean(vectors, axis=0).tolist()


def embed_chunks(chunks, client, *, batch_size=16, progress=None):
    """按 Chunk 原顺序逐批产出归一化向量，避免一次发送全部正文。

    调用真实客户端时会访问配置的远程服务；progress 在该批向量被消费后报告进度。
    """
    if batch_size < 1:
        raise ValueError("批量大小必须大于 0")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        try:
            vectors = client.embed([chunk.text for chunk in batch])
        except RuntimeError as error:
            if "HTTP 400" not in str(error):
                raise
            # A provider may reject one oversized/unsupported item; retry the small batch individually.
            vectors = [_embed_one(client, chunk.text) for chunk in batch]
        yield float_vectors(vectors, len(batch))
        if progress:
            progress(min(start + batch_size, len(chunks)), len(chunks))
