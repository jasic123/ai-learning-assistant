import unittest
from unittest.mock import Mock

from app.questions.generator import QuestionGenerationError, QuestionGenerator
from app.questions.models import Question
from app.questions.store import QuestionStore
from app.rag.domain.models import Chunk


class QuestionModelTests(unittest.TestCase):
    def test_valid_question_defaults_to_draft(self):
        question = Question("题干", ("甲", "乙", "丙", "丁"), "B", "解析",
                            "教材.pdf", 3, "chunk-1")
        self.assertEqual(question.answer, "B")
        self.assertEqual(question.status, "draft")

    def test_invalid_question_fields_are_rejected(self):
        valid = dict(stem="题干", options=("甲", "乙", "丙", "丁"), answer="B",
                     explanation="解析", source="教材.pdf", page=3, chunk_id="chunk-1")
        invalid = [dict(valid, options=("甲", "乙", "丙")),
                   dict(valid, answer="E"), dict(valid, page=True),
                   dict(valid, status="published")]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Question(**fields)


class QuestionGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.chunk = Chunk("chunk-1", "教材.pdf", 3, "植物通过光合作用将光能转化为化学能。")
        self.client = Mock()
        self.generator = QuestionGenerator(self.client)

    def test_generated_question_uses_chunk_as_trusted_source(self):
        self.client.chat.return_value = (
            '{"stem":"光合作用将光能转化为什么？",'
            '"options":{"A":"热能","B":"化学能","C":"声能","D":"电能"},'
            '"answer":"B","explanation":"资料指出光能转化为化学能。"}'
        )
        question = self.generator.generate(self.chunk)
        self.assertEqual(question.options, ("热能", "化学能", "声能", "电能"))
        self.assertEqual(question.answer, "B")
        self.assertEqual((question.source, question.page, question.chunk_id),
                         ("教材.pdf", 3, "chunk-1"))
        self.assertEqual(question.status, "draft")
        self.assertIn("根据资料核心知识点", self.client.chat.call_args.args[0][1]["content"])

    def test_requested_topic_is_explicitly_passed_to_generation_prompt(self):
        self.client.chat.return_value = (
            '{"stem":"迭代器通过哪个方法获取下一个元素？",'
            '"options":{"A":"__next__","B":"__iter__","C":"__len__","D":"__str__"},'
            '"answer":"A","explanation":"教材说明 __next__ 返回下一个对象。"}'
        )
        question = self.generator.generate(self.chunk, topic="Python 迭代器")
        self.assertIn("迭代器", question.stem)
        messages = self.client.chat.call_args.args[0]
        self.assertIn("Python 迭代器", messages[1]["content"])
        self.assertIn("指定的出题主题", messages[0]["content"])

    def test_invalid_or_unsupported_model_output_is_rejected(self):
        for output in (
            "不是 JSON",
            '{"error":"insufficient_evidence"}',
            '{"stem":"题干","options":{"A":"甲"},"answer":"A","explanation":"解析"}',
        ):
            with self.subTest(output=output):
                self.client.chat.return_value = output
                with self.assertRaises(QuestionGenerationError):
                    self.generator.generate(self.chunk)

    def test_question_serializes_options_with_labels(self):
        question = Question("题干", ("甲", "乙", "丙", "丁"), "B", "解析",
                            "教材.pdf", 3, "chunk-1")
        self.assertEqual(question.to_dict()["options"],
                         {"A": "甲", "B": "乙", "C": "丙", "D": "丁"})


class QuestionStoreTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = QuestionStore(Path(self.temporary.name))

    def test_save_list_and_review_preserve_source(self):
        question = Question("题干", ("甲", "乙", "丙", "丁"), "B", "解析",
                            "教材.pdf", 3, "chunk-1")
        saved = self.store.save_many([question])[0]
        reviewed = self.store.review(saved["id"], stem="修改后的题干", options=["一", "二", "三", "四"],
                                     answer="A", explanation="修改后的解析", status="approved")
        self.assertEqual(reviewed["stem"], "修改后的题干")
        self.assertEqual(reviewed["options"]["A"], "一")
        self.assertEqual(reviewed["status"], "approved")
        self.assertEqual((reviewed["source"], reviewed["page"], reviewed["chunk_id"]),
                         ("教材.pdf", 3, "chunk-1"))
        self.assertEqual(len(self.store.list("approved")), 1)

    def test_list_shows_newest_questions_first(self):
        first = Question("旧题", ("甲", "乙", "丙", "丁"), "A", "解析",
                         "教材.pdf", 1, "chunk-1")
        second = Question("新题", ("甲", "乙", "丙", "丁"), "B", "解析",
                          "教材.pdf", 2, "chunk-2")
        self.store.save_many([first, second])
        self.assertEqual([item["stem"] for item in self.store.list("draft")], ["新题", "旧题"])


if __name__ == "__main__":
    unittest.main()
