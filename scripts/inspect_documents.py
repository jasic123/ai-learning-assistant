# 手动预览示例：读取指定知识库的部分 PDF 和切片，不构建索引，也不调用远程模型。
"""从项目根目录运行：python -m scripts.inspect_documents。"""
from pathlib import Path
import sys

# 兼容直接运行脚本；所有业务路径仍以 app.config.PROJECT_ROOT 为基准。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.ingestion.pipeline import document_chunks


# 预览每个已处理文件的报告和前三个片段；参数在示例内指定，可按资料规模调整。
def main():
    # 每份 PDF 最多解析 2 页，知识库累计最多返回 50 个片段。
    chunks, reports = document_chunks("教育资料", max_pages=2, max_chunks=50,
                                     chunk_size=400, overlap=60)
    for report in reports:
        print("文件：", report["source"], "处理页数：", report["processed_pages"],
              "切片数量：", report["chunks"])
    # Chunk 是数据对象，直接用属性访问。
    for chunk in chunks[:3]:
        print(chunk.source, chunk.page, chunk.text)


if __name__ == "__main__":
    main()
