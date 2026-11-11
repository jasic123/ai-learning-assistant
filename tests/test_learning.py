import tempfile
import unittest
from pathlib import Path

from app.learning.store import LearningStore
from app.questions.models import Question
from app.questions.store import QuestionStore


class LearningStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        self.questions = QuestionStore(self.data)
        self.learning = LearningStore(self.data)
        self.saved = self.questions.save_many([
            Question("迭代器返回下一个值的方法是？", ("__iter__", "__next__", "__len__", "__str__"),
                     "B", "Python 3 迭代器通过 __next__ 返回下一个元素。", "Python参考手册.pdf", 62, "chunk-a"),
            Question("迭代器协议初始化方法是？", ("__iter__", "__next__", "__len__", "__str__"),
                     "A", "__iter__ 返回一个迭代器对象。", "Python参考手册.pdf", 62, "chunk-b"),
        ])

    def test_submit_auto_grades_and_saves_only_wrong_answers(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="Python 迭代器", questions=self.saved)
        result = self.learning.submit_quiz(quiz["id"], {
            str(self.saved[0]["id"]): "A", str(self.saved[1]["id"]): "A",
        })
        self.assertEqual((result["correct_count"], result["total_count"], result["score"]), (1, 2, 50))
        self.assertEqual([item["correct"] for item in result["results"]], [False, True])
        mistakes = self.learning.list_mistakes(subject="计算机", query="迭代器")
        self.assertEqual(len(mistakes), 1)
        self.assertEqual(mistakes[0]["question_id"], self.saved[0]["id"])
        self.assertIn("__next__", mistakes[0]["explanation"])

    def test_question_metadata_filters_approved_teacher_bank_and_reviews(self):
        questions = self.questions.save_many([
            Question("虚拟语气题 A", ("A", "B", "C", "D"), "A", "解析 A", "教材.pdf", 1,
                     "mood-a", "approved", "英语", "虚拟语气"),
            Question("定语从句题", ("A", "B", "C", "D"), "B", "解析 B", "教材.pdf", 2,
                     "clause", "approved", "英语", "定语从句"),
            Question("待审虚拟语气题", ("A", "B", "C", "D"), "C", "解析 C", "教材.pdf", 3,
                     "mood-draft", "draft", "英语", "虚拟语气"),
        ])
        selected = self.questions.list(status="approved", subject="英语", topic="虚拟语气")
        self.assertEqual([item["id"] for item in selected], [questions[0]["id"]])
        reviewed = self.questions.review(questions[2]["id"], stem="已核对的虚拟语气题",
                                         options=["A", "B", "C", "D"], answer="C",
                                         explanation="核对后的解析", status="approved",
                                         subject="英语", topic="虚拟语气")
        self.assertEqual((reviewed["subject"], reviewed["topic"], reviewed["status"]),
                         ("英语", "虚拟语气", "approved"))

    def test_review_quality_updates_sm2_schedule_and_history(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="Python 迭代器", questions=self.saved[:1])
        self.learning.submit_quiz(quiz["id"], {str(self.saved[0]["id"]): "D"})
        before = self.learning.list_mistakes()[0]
        updated = self.learning.review_mistake(self.saved[0]["id"], 5)
        self.assertEqual(updated["repetitions"], 1)
        self.assertEqual(updated["interval_days"], 1)
        self.assertGreater(updated["ease_factor"], 2.5)
        self.assertNotEqual(updated["due_at"], before["due_at"])
        history = self.learning.history(7)
        self.assertEqual(history["timeline"][0]["score"], 0)
        self.assertEqual(history["mistakes_by_subject"], {"计算机": 1})
        self.assertEqual(len(history["daily"]), 1)

    def test_review_quiz_removes_original_only_when_pair_is_correct(self):
        practice = self.learning.create_quiz(subject="计算机", topic="Python 迭代器", questions=self.saved[:1])
        self.learning.submit_quiz(practice["id"], {str(self.saved[0]["id"]): "A"})
        review = self.learning.create_quiz(
            subject="计算机", topic="到期错题复习", questions=self.saved,
            quiz_type="review", review_pairs=[[self.saved[0]["id"], self.saved[1]["id"]]],
        )
        self.learning.submit_quiz(review["id"], {
            str(self.saved[0]["id"]): "B", str(self.saved[1]["id"]): "A",
        })
        self.assertEqual(self.learning.list_mistakes(), [])

        practice = self.learning.create_quiz(subject="计算机", topic="Python 迭代器", questions=self.saved[:1])
        self.learning.submit_quiz(practice["id"], {str(self.saved[0]["id"]): "A"})
        review = self.learning.create_quiz(
            subject="计算机", topic="到期错题复习", questions=self.saved,
            quiz_type="review", review_pairs=[[self.saved[0]["id"], self.saved[1]["id"]]],
        )
        self.learning.submit_quiz(review["id"], {
            str(self.saved[0]["id"]): "B", str(self.saved[1]["id"]): "B",
        })
        mistakes = self.learning.list_mistakes()
        self.assertEqual(len(mistakes), 1)
        self.assertEqual(mistakes[0]["topic"], "Python 迭代器")

    def test_invalid_submission_and_review_quality_are_rejected(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="迭代器", questions=self.saved[:1])
        with self.assertRaises(ValueError):
            self.learning.submit_quiz(quiz["id"], {})
        with self.assertRaises(ValueError):
            self.learning.review_mistake(self.saved[0]["id"], 6)

    def test_mistake_search_accepts_related_words_and_searches_options(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="Python 贪心算法", questions=self.saved[:1])
        self.learning.submit_quiz(quiz["id"], {str(self.saved[0]["id"]): "A"})
        self.assertEqual(len(self.learning.list_mistakes(query="相关贪心算法")), 1)
        self.assertEqual(len(self.learning.list_mistakes(query="__next__")), 1)

    def test_imported_student_question_enters_mistake_search_and_due_reviews(self):
        question = self.questions.save_many([
            Question("贪心算法每次选择什么？", ("局部最优", "全局最差", "随机值", "最大值"),
                     "A", "每一步选取当前看来最优的方案。", "学生上传题目", 1, "upload-test", "approved")
        ])[0]
        mistake = self.learning.add_mistake(question, subject="计算机", topic="贪心算法")
        self.assertEqual(mistake["question_id"], question["id"])
        self.assertEqual(len(self.learning.list_mistakes(subject="计算机", query="贪心")), 1)
        self.assertEqual(len(self.learning.due_reviews(subject="计算机")), 1)

    def test_learning_records_are_isolated_by_owner(self):
        question = self.saved[0]
        first = LearningStore(self.data, owner_id="student-one")
        second = LearningStore(self.data, owner_id="student-two")
        first.add_mistake(question, subject="计算机", topic="迭代器")
        self.assertEqual(len(first.list_mistakes()), 1)
        self.assertEqual(second.list_mistakes(), [])
        self.assertEqual(first.due_reviews(subject="计算机").__len__(), 1)
        self.assertEqual(second.due_reviews(subject="计算机"), [])

    def test_profile_aggregates_topics_and_only_uses_student_reported_reasons(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="Python 迭代器", questions=self.saved[:1])
        attempts = []
        for selected in ("A", "A", "B"):
            result = self.learning.submit_quiz(quiz["id"], {str(self.saved[0]["id"]): selected})
            attempts.append(result["attempt_id"])
        self.learning.save_error_reasons(attempts[0], {str(self.saved[0]["id"]): "concept_unclear"})
        self.learning.save_error_reasons(attempts[1], {str(self.saved[0]["id"]): "condition_missed"})
        profile = self.learning.learning_profile()

        self.assertEqual(profile["total_attempts"], 3)
        self.assertEqual(profile["total_questions"], 3)
        self.assertEqual(profile["topics"][0]["topic"], "Python 迭代器")
        self.assertEqual(profile["topics"][0]["incorrect"], 2)
        self.assertEqual(profile["topics"][0]["accuracy"], 33)
        self.assertEqual(profile["topics"][0]["status"], "needs_review")
        self.assertEqual(profile["reason_sample_count"], 2)
        self.assertTrue(all(item["evidence"] == "sample_limited" for item in profile["error_patterns"]))

    def test_profile_is_owner_scoped_and_rejects_reasons_for_correct_answers(self):
        first = LearningStore(self.data, owner_id="student-one")
        second = LearningStore(self.data, owner_id="student-two")
        quiz = first.create_quiz(subject="计算机", topic="迭代器", questions=self.saved)
        result = first.submit_quiz(quiz["id"], {
            str(self.saved[0]["id"]): "A", str(self.saved[1]["id"]): "A",
        })
        with self.assertRaisesRegex(ValueError, "只能为答错"):
            first.save_error_reasons(result["attempt_id"], {str(self.saved[1]["id"]): "guessed"})
        with self.assertRaises(LookupError):
            second.save_error_reasons(result["attempt_id"], {str(self.saved[0]["id"]): "guessed"})
        self.assertEqual(second.learning_profile()["total_attempts"], 0)

    def test_profile_marks_low_sample_topics_without_calling_them_weak(self):
        quiz = self.learning.create_quiz(subject="计算机", topic="单次练习", questions=self.saved[:1])
        self.learning.submit_quiz(quiz["id"], {str(self.saved[0]["id"]): "A"})
        profile = self.learning.learning_profile()
        self.assertEqual(profile["topics"][0]["status"], "sample_limited")
        self.assertEqual(profile["topics"][0]["evidence"], "sample_limited")


if __name__ == "__main__":
    unittest.main()
