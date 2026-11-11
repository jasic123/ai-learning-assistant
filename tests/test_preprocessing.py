# 清洗回归：覆盖中文、英文碎行、页边缘清理、布局回退与切片偏移。
"""PDF 清洗和切片回归测试，不调用模型接口。"""
import unittest

from app.rag.ingestion.cleaners import clean_page, extract_page, fragment_ratio, repeated_margins
from app.rag.ingestion.chunkers import split_text


# 清洗规则与片段边界的离线回归测试集合。
class PreprocessingTests(unittest.TestCase):
    # 比较中文、中英混排、数字符号和英文碎行，验证样本量门槛与比例计算。
    def test_fragment_ratio_only_counts_sufficient_english_lines(self):
        chinese = '\n'.join(['中文', '标题', '一', '二'] * 10)
        mixed_lines = '\n'.join(['学习 Python', '第 1 章', '变量 x'] * 10)
        fragmented = '\n'.join(['P', 'yt', 'h', 'on'] * 5)
        cases = [
            ('empty', '', 0.0),
            ('chinese', chinese, 0.0),
            ('mixed_lines', mixed_lines, 0.0),
            ('numbers_and_symbols', '\n'.join(['1', '{}', '()', '。'] * 10), 0.0),
            ('few_english_lines', chinese + '\nAI\nI', 0.0),
            ('normal_english', '\n'.join(['This is a complete sentence.'] * 20), 0.0),
            ('fragmented_english', fragmented, 1.0),
            ('mixed_page', chinese + '\n' + fragmented, 1.0),
            ('partial_fragmentation', '\n'.join(['a'] * 10 + ['complete sentence'] * 10), 0.5),
        ]
        for name, text, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(fragment_ratio(text), expected)

    # 中文短行不应套用英文碎字规则，也不应触发布局提取。
    def test_chinese_short_lines_do_not_trigger_english_layout_fallback(self):
        raw = '\n'.join(['中文', '标题', '一', '二', '学习 Python'] * 10) + '\nAI'

        # 模拟 pypdf 页面对象，只提供本测试所需的文字提取接口。
        class Page:
            # 根据提取模式返回测试文本，让布局回退分支可重复验证。
            def extract_text(self, extraction_mode='plain'):
                if extraction_mode == 'layout':
                    raise AssertionError('中文短行不应触发英文布局回退')
                return raw

        text, method, warnings = extract_page(Page())
        self.assertEqual(text, raw)
        self.assertEqual(method, 'plain')
        self.assertEqual(warnings, [])
        cleaned, report = clean_page(text)
        self.assertEqual(cleaned, raw)
        self.assertNotIn('fragmented_text', report['warnings'])

    # 布局模式行数减少但仍然碎字时，不能误判为质量改善。
    def test_layout_with_fewer_but_still_fragmented_lines_is_rejected(self):
        # 模拟 pypdf 页面对象，只提供本测试所需的文字提取接口。
        class Page:
            # 根据提取模式返回测试文本，让布局回退分支可重复验证。
            def extract_text(self, extraction_mode='plain'):
                return '\n'.join(['a'] * (18 if extraction_mode == 'layout' else 24))

        text, method, warnings = extract_page(Page())
        self.assertEqual(len(text.splitlines()), 24)
        self.assertEqual(method, 'plain')
        self.assertIn('fragmented_text_check_parser_or_ocr', warnings)

    # 只清除可确认的页码、水印和控制字符，保留代码缩进及正文网址、数字。
    def test_cleaning_is_conservative_and_traceable(self):
        raw = "30\r\nTitle\n\ndef hello():\n    print('hi')\n\nwww.TopSage.com\x00\n"
        text, report = clean_page(raw, page_number_value=30)
        self.assertIn("def hello():\n    print('hi')", text)
        self.assertNotIn('www.TopSage.com', text)
        self.assertNotIn('\x00', text)
        self.assertEqual({item['reason'] for item in report['removed']},
                         {'margin_page_number', 'observed_watermark'})
        # 正文中的网址和独立数值应保持不变。
        body = "Results\n\n0\nwww.example.com\n\nConclusion"
        cleaned, _ = clean_page(body, page_number_value=30)
        self.assertIn('0', cleaned)
        self.assertIn('www.example.com', cleaned)

    # 跨页重复文本只允许从边缘删除，正文中相同句子仍需保留。
    def test_only_repeated_edge_lines_are_removed(self):
        pages = [f"Book title\n\nDifferent body {i}\n\nCopyright 2026" for i in range(4)]
        repeated = repeated_margins(pages)
        self.assertEqual(repeated, {'Book title', 'Copyright 2026'})
        cleaned, report = clean_page(pages[0], repeated)
        self.assertIn('Different body 0', cleaned)
        self.assertNotIn('Book title', cleaned)
        # 即使文字与页眉相同，位于正文中间时也保留。
        middle = 'One\nTwo\nThree\nBook title\nFour\nFive\nSix'
        self.assertIn('Book title', clean_page(middle, repeated)[0])
        self.assertEqual(repeated_margins(pages[:2]), set())

    # 布局模式确实改善碎行时采用结果，并提示人工核对阅读顺序。
    def test_layout_fallback_is_selected_for_fragmented_text(self):
        # 模拟 pypdf 页面对象，只提供本测试所需的文字提取接口。
        class Page:
            # 根据提取模式返回测试文本，让布局回退分支可重复验证。
            def extract_text(self, extraction_mode='plain'):
                if extraction_mode == 'layout':
                    return 'Python examples contain complete lines.\n    print("hello")'
                return '\n'.join('Python examples contain complete lines.')
        text, method, warnings = extract_page(Page())
        self.assertEqual(method, 'layout')
        self.assertIn('    print', text)
        self.assertIn('layout_fallback_check_reading_order', warnings)

    # 验证段落优先、长度上限、原文偏移和正文覆盖，避免切片丢字。
    def test_paragraph_priority_and_offsets(self):
        text = '甲' * 220 + '\n\n' + '乙' * 100 + '。' + '丙' * 300
        chunks = list(split_text(text, 400, 60))
        self.assertEqual(chunks[0][0], '甲' * 220)
        self.assertTrue(all(len(piece) <= 400 for piece, _ in chunks))
        for piece, start in chunks:
            self.assertEqual(text[start:start + len(piece)], piece)
        # 收集全部片段覆盖的原文位置，验证非空白正文没有因边界或重叠调整而丢失。
        covered = set()
        for piece, start in chunks:
            covered.update(range(start, start + len(piece)))
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        code = 'Introduction\n\n    def example():\n        return 123\n'
        code_chunks = [piece for piece, _ in split_text(code) if 'def example' in piece]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn('    def example():\n        return 123', code_chunks[0])

    def test_fenced_code_and_indented_code_are_not_split(self):
        fenced = "```python\ndef calculate():\n    return " + "1 + " * 80 + "1\n```"
        chunks = list(split_text("说明\n\n" + fenced + "\n\n结论", 120, 20))
        code_chunks = [piece for piece, _ in chunks if "def calculate" in piece]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn("```python", code_chunks[0])
        self.assertIn("```", code_chunks[0][len("```python"):])
        self.assertGreater(len(code_chunks[0]), 120)

        indented = "说明\n\n    def iterate():\n        first = next(items)\n        return first\n\n结论"
        code_chunks = [piece for piece, _ in split_text(indented, 120, 20) if "def iterate" in piece]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn("return first", code_chunks[0])
        self.assertNotIn("结论", code_chunks[0])

    def test_chunks_are_classified_without_changing_source_text(self):
        from app.rag.domain.models import Document
        from app.rag.ingestion.chunkers import split_document

        code = split_document(Document("lesson.pdf", 1, ">>> def answer():\n...     return 42"))
        question = split_document(Document("lesson.pdf", 2,
                                           "1. 哪个答案正确？\nA. 错误\nB. 正确"))

        self.assertEqual(code[0].chunk_type, "code")
        self.assertEqual(question[0].chunk_type, "question")
        self.assertEqual(code[0].text, ">>> def answer():\n...     return 42")

    def test_multiple_choice_stem_and_four_options_stay_together(self):
        question = (
            "1. 下列关于迭代器的说法，哪一项正确？\n"
            "A. " + "错误描述 " * 20 + "\n"
            "B. 迭代器实现了 __iter__ 和 __next__。\n"
            "C. 迭代器可以随意重复遍历。\n"
            "D. 迭代器必须预先保存所有元素。\n"
        )
        chunks = list(split_text("前置知识。\n\n" + question + "\n后续知识。", 120, 20))
        matches = [piece for piece, _ in chunks if "下列关于迭代器" in piece]
        self.assertEqual(len(matches), 1)
        self.assertTrue(all(f"{label}. " in matches[0] for label in "ABCD"))
        self.assertGreater(len(matches[0]), 120)


if __name__ == '__main__':
    unittest.main()
