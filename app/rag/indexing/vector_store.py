"""FAISS 与 JSON 的物理存取；SQLite 业务元数据由 db.repository 负责。"""
import json

import faiss
import numpy as np

from app.rag.domain.models import Chunk


def create_index(vector_batches):
    """按顺序追加向量；首批确定维度，FAISS 行号与 Chunk 保存顺序一一对应。"""
    index = None
    for vectors in vector_batches:
        if index is None:
            # 向量已经归一化，精确内积检索对应余弦相似度。
            index = faiss.IndexFlatIP(vectors.shape[1])
        if index.d != vectors.shape[1]:
            raise ValueError("Embedding 各批次维度不一致")
        index.add(vectors)
    if index is None:
        raise ValueError("没有可写入索引的向量")
    return index


def save_index(folder, index, chunks, metadata):
    """写入已创建的版本目录，不切换 CURRENT；发布由索引流水线统一协调。"""
    # 经 Python 读写字节，避开 FAISS 原生文件接口在 Windows 中文路径上的编码问题。
    (folder / "index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    (folder / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False), encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(folder):
    """恢复向量、Chunk 和版本元数据，并校验三者数量与维度；兼容原 schema_version=1。"""
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        records = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("片段索引必须是列表")
        chunks = [Chunk.from_dict(record) for record in records]
        index = faiss.deserialize_index(np.frombuffer((folder / "index.faiss").read_bytes(), dtype="uint8"))
    except (OSError, RuntimeError, ValueError):
        raise ValueError("索引缺失或损坏，请重新入库") from None
    if (not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("dimension") != index.d
        or metadata.get("count") != index.ntotal
        or index.ntotal != len(chunks) or not chunks):
        raise ValueError("索引与元数据不一致，请重新入库")
    return index, chunks, metadata
