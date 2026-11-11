# 手动 PDF 检查脚本：main() 读取本机指定资料；unittest 自动发现时不会执行该入口。
from pathlib import Path
import sys

if __name__ == "__main__":
    # 兼容终端直接执行 python tests/test_loader.py。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.ingestion.loaders import load_pdf

# 手动测试使用本机固定路径，换机器或换资料时需要自行调整。
path = Path(
    r"D:\Python\PythonProject1\data\uploads\教育资料\Python参考手册(第4版).pdf"
)

# 手动加载指定 PDF 的前 20 页，打印 OCR 状态和清洗后的页面摘要。
def main():
    # 先检查 20 页；验证效果后可以增加页数，0 表示整本。扫描页默认自动 OCR。
    documents, report = load_pdf(path, source=path.name, max_pages=20)
    print("有效页面数：", len(documents))
    print("疑似扫描页：", report["suspected_scanned_pages"])
    print("OCR 成功页：", report["ocr_pages"])
    print("仍需 OCR 页：", report["needs_ocr_pages"])
    print("OCR 失败页：", report["ocr_failed_pages"])
    print("低文字量页：", report["low_text_pages"])
    # 仅预览每页前 500 个字符，不代表该页完整提取内容只有这些。
    for document in documents[:20]:
        print(f"\n文件：{document.source}，第 {document.page} 页")
        print(document.text[:500])


if __name__ == "__main__":
    main()
