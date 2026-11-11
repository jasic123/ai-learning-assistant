# 纯文本处理模块：切片单位是字符，单页内切分；返回的偏移用于追溯清洗后原文。
"""文本切片：Document -> list[Chunk]，不读取文件、不访问模型。"""
import hashlib
import re
import unicodedata
from dataclasses import replace

from app.rag.domain.models import Chunk, Document
from app.rag.ingestion.cleaners import PREPROCESS_VERSION


_QUESTION_START = re.compile(
    r"^\s*(?:\d{1,3}[.．、)）]\s*|[（(]\d{1,3}[)）]\s*|(?:题目|习题|练习)\s*\d+\s*[.．、:：]?\s*)"
)
_OPTION = re.compile(r"^\s*(?:([A-D])[.、:：)]\s*|\(([A-D])\)\s*)", re.I)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_CODE_LINE = re.compile(
    r"^\s*(?:>>>|\.\.\.|(?:async\s+)?(?:def|class|if|elif|else|for|while|try|except|finally|with|return|yield|import|from|raise|assert|pass)\b|"
    r"[A-Za-z_]\w*(?:\[[^]]+\])?\s*(?:=|\+=|-=|\.)\s*\S)"
)


def _line_records(text):
    records, offset = [], 0
    for line in text.splitlines(keepends=True):
        records.append((offset, offset + len(line), line))
        offset += len(line)
    if text and (not records or offset < len(text)):
        records.append((offset, len(text), text[offset:]))
    return records


def _protected_spans(text):
    """Find fenced/code-prompt blocks and numbered questions with A-D choices."""
    lines = _line_records(text)
    spans = []
    i = 0

    while i < len(lines):
        start, _, line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            end_index = i
            for end_index in range(i + 1, len(lines)):
                if re.match(rf"^\s*{re.escape(marker)}{{3,}}\s*$", lines[end_index][2].rstrip("\r\n")):
                    break
            spans.append((start, lines[end_index][1]))
            i = end_index + 1
            continue

        if re.match(r"^\s*>>>", line):
            end_index = i
            for candidate_index in range(i + 1, len(lines)):
                candidate = lines[candidate_index][2]
                if not candidate.strip():
                    break
                end_index = candidate_index
            spans.append((start, lines[end_index][1]))
            i = end_index + 1
            continue

        # Indented source examples are protected when at least two lines look like code.
        if _CODE_LINE.match(line):
            end_index, code_lines = i, 1
            for candidate_index in range(i + 1, len(lines)):
                candidate = lines[candidate_index][2]
                if not candidate.strip():
                    continue
                if candidate.startswith(("    ", "\t")) or _CODE_LINE.match(candidate):
                    end_index = candidate_index
                    code_lines += bool(_CODE_LINE.match(candidate))
                    continue
                break
            if code_lines >= 2:
                spans.append((start, lines[end_index][1]))
                i = end_index + 1
                continue

        normalized = unicodedata.normalize("NFKC", line)
        is_question = bool(_QUESTION_START.match(normalized) or
                           re.search(r"[?？]\s*$", normalized) or
                           re.match(r"^\s*(?:下列|以下|哪一|哪项|何者)", normalized))
        if is_question:
            expected, option_indexes = "A", []
            end_index = i
            for candidate_index in range(i + 1, min(i + 32, len(lines))):
                candidate = unicodedata.normalize("NFKC", lines[candidate_index][2])
                option = _OPTION.match(candidate)
                if option:
                    label = (option.group(1) or option.group(2)).upper()
                    if label != expected:
                        break
                    option_indexes.append(candidate_index)
                    expected = chr(ord(expected) + 1)
                    end_index = candidate_index
                    if expected == "E":
                        # Keep wrapped D text and an immediately following answer/explanation line.
                        if end_index + 1 < len(lines) and re.match(
                            r"^\s*(?:答案|解析|解答)\s*[:：]", lines[end_index + 1][2]
                        ):
                            end_index += 1
                        spans.append((start, lines[end_index][1]))
                        break
                elif option_indexes and candidate.strip():
                    # A choice may wrap onto following indented/plain lines before the next label.
                    end_index = candidate_index
            if expected == "E":
                i = end_index + 1
                continue
        i += 1

    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_plain(text, size, overlap, offset=0):
    """Split ordinary prose at paragraph and sentence boundaries."""
    start = 0
    while start < len(text):
        limit = min(start + size, len(text))
        end = limit
        if limit < len(text):
            for separators in (("\n\n",), ("。", "！", "？", ". ", "! ", "? "), ("\n",), (" ", "；", "; ")):
                candidates = [pos + len(mark) for mark in separators
                              if (pos := text.rfind(mark, start + size // 2, limit)) >= 0]
                if candidates:
                    end = max(candidates)
                    break
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip("\n"))
        piece = raw.strip("\n")
        if piece.strip():
            yield piece, offset + start + leading
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        newline = text.find("\n", next_start, end)
        if newline >= 0 and newline + 1 < end:
            next_start = newline + 1
        start = next_start


def split_text(text, size=400, overlap=60):
    """Split prose normally while keeping code examples and multiple-choice items intact.

    offset 是清洗后本页字符偏移，页码保持 PDF 物理页码。
    结构块可能超过 size；页边界仍为硬边界。
    """
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("切片至少 100 字符，且 0 <= 重叠长度 < 切片长度")
    cursor = 0
    for start, end in _protected_spans(text):
        yield from _split_plain(text[cursor:start], size, overlap, cursor)
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip("\n"))
        piece = raw.strip("\n")
        if piece.strip():
            yield piece, start + leading
        cursor = end
    yield from _split_plain(text[cursor:], size, overlap, cursor)



def split_document(document: Document, *, chunk_size: int = 400, overlap: int = 60) -> list[Chunk]:
    """保留来源页码、页内顺序及结构类型；片段 ID 算法保持稳定。"""
    chunks = []
    parent_id = hashlib.sha256(f"{document.source}\0{document.page}".encode()).hexdigest()
    for text, offset in split_text(document.text, chunk_size, overlap):
        # 相同来源、页码、偏移和正文生成相同 ID；内容或位置变化时 ID 随之变化。
        identifier = hashlib.sha256(
            f"{document.source}\0{document.page}\0{offset}\0{text}".encode()
        ).hexdigest()
        chunks.append(Chunk(id=identifier, source=document.source, page=document.page,
                            text=text, offset=offset, preprocessing_version=PREPROCESS_VERSION,
                            parent_id=parent_id, position=len(chunks), chunk_type=_chunk_type(text)))
    return _link_chunks(chunks)


def _chunk_type(text):
    has_code = bool(re.search(r"(?m)^\s*(?:```|~~~|>>>|\.\.\.|(?:async\s+)?(?:def|class)\s+)", text))
    has_question = bool(re.search(r"(?m)^\s*(?:\d{1,3}[.．、)）]|[A-D][.、:：)]|题目\s*\d+)", text))
    if has_code and has_question:
        return "mixed"
    if has_code:
        return "code"
    if has_question:
        return "question"
    return "prose"


def _link_chunks(chunks):
    return [replace(chunk,
                    prev_chunk_id=chunks[i - 1].id if i else None,
                    next_chunk_id=chunks[i + 1].id if i + 1 < len(chunks) else None)
            for i, chunk in enumerate(chunks)]
