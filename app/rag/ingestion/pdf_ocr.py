# OCR 依赖按需导入；没有候选页时不初始化模型，也不会把 PDF 发送到远程服务。
"""本地中文/英文 OCR：只渲染待识别页，不改写 PDF、不上传内容。"""
from contextlib import closing, contextmanager
from functools import lru_cache


class OcrUnavailable(RuntimeError):
    """未安装本地 OCR 所需依赖。"""


# 首次调用才初始化 CPU OCR 模型，并在当前进程内缓存一个实例。
# 缓存成功返回的引擎实例；初始化抛出的异常不会作为正常实例缓存。
@lru_cache(maxsize=1)
def _engine():
    # 模型较大，首次使用才加载；后续页面复用同一个 CPU 推理实例。
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise OcrUnavailable("请在项目环境安装 rapidocr-onnxruntime 和 pypdfium2") from None
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


@contextmanager
def open_ocr_document(path):
    """打开一个 PDFium 文档供同一文件的多个 OCR 页面复用。"""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise OcrUnavailable("请在项目环境安装 rapidocr-onnxruntime 和 pypdfium2") from None
    with pdfium.PdfDocument(str(path), password="") as pdf:
        yield pdf


def ocr_page(path, page_index, *, document=None):
    """返回一页的识别文字和平均识别置信度；page_index 从 0 开始。

    ponytail: 使用 OCR 引擎默认文本框顺序；复杂双栏、表格和代码需人工复核。
    置信度只代表模型对字符的判断，不代表事实或阅读顺序正确。
    """
    engine = _engine()
    # 独立调用时自己打开 PDF；批量加载时复用调用方已打开的文档。
    if document is None:
        with open_ocr_document(path) as pdf:
            image = _render_page(pdf, page_index)
    else:
        image = _render_page(document, page_index)
    result, _ = engine(image)
    if not result:
        return "", None
    return "\n".join(item[1] for item in result), sum(float(item[2]) for item in result) / len(result)


def _render_page(pdf, page_index):
    """Render and release one page before the next page is loaded."""
    # 每次只渲染一页并及时释放，避免整本扫描书转换为图片后占满内存。
    with closing(pdf[page_index]) as page:
        with closing(page.render(scale=2.5)) as bitmap:
            # 复制像素后再关闭 PDFium 页面和位图，避免后续 OCR 引用已释放的底层内存。
            return bitmap.to_numpy().copy()
