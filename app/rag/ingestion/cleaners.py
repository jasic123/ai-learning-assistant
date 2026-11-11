# 统一清洗入口：原文字层与 OCR 结果共用规则；质量诊断不通过时标记问题而非猜测正文。
"""保守清洗：保留代码缩进，不拼接猜测性的英文碎字。"""
from collections import Counter
import re
import unicodedata

# 清洗规则的版本号会写入索引，方便判断旧索引是否需要重新生成。
PREPROCESS_VERSION = "pdf-clean-v5-structured-chunks"


# 统一换行、等价字符和空白；保留代码行首缩进，不进行语义改写。
def normalize_text(text):
    # NFC 统一等价字符；保留全角标点等语义，不进行更激进的兼容字符转换。
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    # 将不换行空格和全角空格转为普通空格，便于后续统计和检索。
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    # 去掉不可见控制字符，但保留换行和制表符，避免破坏代码排版。
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\ufeff]", "", text)
    # 只删行尾空白，保留行首缩进；多余空行压缩后仍保留段落间隔。
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip("\n")


def fragment_ratio(text, *, min_lines=20):
    """估算英文碎行比例；中文及中英混排行不使用英文短行规则。"""
    # 这里只收集符合英文候选条件的行，中文行不进入英文碎行比例的分母。
    lines = []
    for line in text.splitlines():
        line = line.strip()
        letters = [char for char in line if char.isalpha()]
        # 只统计字母均为 ASCII 的候选行；中文、其他文字和混排行保留原文。
        # 这是英文提取质量的启发式指标，并非完整的语种识别。
        if letters and all(char.isascii() for char in letters):
            lines.append(line)
    # 默认至少有 20 条英文候选行，避免中文页中少量英文缩写触发误报。
    if not lines or len(lines) < min_lines:
        return 0.0
    # 纯数字、标点和代码括号不算英文碎字；仅 1～2 个英文字母的行计入。
    return sum(bool(re.fullmatch(r"[A-Za-z]{1,2}", line)) for line in lines) / len(lines)


# 返回提取文字、提取方式和警告；仅在英文碎行明显时尝试布局提取。
def extract_page(page):
    # 先尝试普通文字提取；扫描页由 ingestion.loaders 的质量检查决定是否调用 OCR。
    plain = normalize_text(page.extract_text() or "")
    method, warnings = "plain", []
    if fragment_ratio(plain) > 0.35:
        try:
            # 碎行较多才尝试布局模式，避免所有页面都额外解析一次。
            layout = normalize_text(page.extract_text(extraction_mode="layout") or "")
            plain_chars = len(re.sub(r"\s", "", plain))
            layout_chars = len(re.sub(r"\s", "", layout))
            # 只有文字量未明显丢失且碎行比例下降时才采用；阅读顺序仍须人工核对。
            # 比较布局结果时取消样本量门槛，避免仅因行数变少就误认为质量改善。
            if layout_chars >= 0.7 * plain_chars and fragment_ratio(layout, min_lines=1) < fragment_ratio(plain):
                plain, method = layout, "layout"
                warnings.append("layout_fallback_check_reading_order")
        except (TypeError, ValueError, RuntimeError):
            warnings.append("layout_fallback_failed")
    if fragment_ratio(plain) > 0.35:
        # 不猜测性拼接碎字，改为记录警告，交给诊断和后续 OCR 流程处理。
        warnings.append("fragmented_text_check_parser_or_ocr")
    return plain, method, warnings


# 生成页边缘比较用的规范化键，不将该键作为正文返回。
def margin_key(line):
    # 仅用于页边缘文本比对，不会拿这个压缩空格后的值覆盖正文或代码。
    return re.sub(r"\s+", " ", line).strip()


# 仅检查字符串形态是否像页码；是否删除还需要结合页面位置和物理页码。
def page_number(line):
    # 这里只判断是否像页码；实际删除时还要求数字与当前 PDF 物理页码一致。
    return bool(re.fullmatch(r"(?:第\s*)?\d{1,5}(?:\s*页)?|[-–—]\s*\d{1,5}\s*[-–—]", line.strip()))


def repeated_margins(pages):
    """只检测至少三页中高频出现的完全相同边缘短行。"""
    counts = Counter()
    if len(pages) < 3:
        # 样本过少时不推断重复页眉、页脚。
        return set()
    for text in pages:
        lines = [line for line in text.splitlines() if line.strip()]
        # 仅检查首尾各两条非空行，避免全页重复句子被当作页眉。
        candidates = lines[:2] + lines[-2:]
        # 排除常见代码特征和页码；集合确保每一页对同一文本最多计数一次。
        keys = {margin_key(line) for line in candidates
                if 3 <= len(margin_key(line)) <= 100
                and not re.search(r"[=(){}\[\]]|^\s*(?:>>>|def |class |import |from )", line)
                and not page_number(line)}
        counts.update(keys)
    # 至少三页且覆盖所选页面的 80% 才判定为候选；这是需要复核的启发式规则。
    return {line for line, count in counts.items() if count >= 3 and count / len(pages) >= 0.8}


# 统一清洗一页并返回诊断字典；repeated 是跨页重复候选，page_number_value 是物理页码。
def clean_page(text, repeated=(), page_number_value=None):
    text = normalize_text(text)
    lines = text.splitlines()
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    # 只允许在页首、页尾删除候选行，正文中相同的文字仍保留。
    # 删除范围限定为首尾两条非空行，防止重复标题或数字出现在正文时被误删。
    edges = set(nonempty[:2] + nonempty[-2:])
    kept, removed = [], []
    for i, line in enumerate(lines):
        key = margin_key(line)
        reason = None
        if i in edges:
            if key in repeated:
                reason = "repeated_margin"
            elif page_number(key) and (page_number_value is not None and re.search(r"\d+", key).group() == str(page_number_value)):
                reason = "margin_page_number"
            elif re.fullmatch(r"(?:https?://)?www\.topsage\.com/?", key, flags=re.I):
                # 这是当前资料中已观察到的边缘水印，不泛化删除所有网址。
                reason = "observed_watermark"
        if reason:
            # 保存删除文本及原因，方便人工检查是否误删。
            removed.append(dict(line=key, reason=reason))
        else:
            kept.append(line)
    cleaned = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", "\n".join(kept)).strip("\n")
    warnings = []
    if fragment_ratio(cleaned) > 0.35:
        warnings.append("fragmented_text")
    # 同时返回清洗结果和质量报告，供入库元数据、诊断命令使用。
    return cleaned, dict(raw_chars=len(text), cleaned_chars=len(cleaned), removed=removed, warnings=warnings)
