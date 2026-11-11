import unittest

from scripts.judge_retrieval_dataset import parse_judgement


class DatasetJudgeTests(unittest.TestCase):
    def test_parse_judgement_requires_all_ids_and_allowed_verdicts(self):
        result = parse_judgement(
            '[{"id":"q1","verdict":"approved","reason":"正文直接解释了问题。"}]', {"q1"})
        self.assertEqual(result["q1"]["verdict"], "approved")
        with self.assertRaises(ValueError):
            parse_judgement('[{"id":"q1","verdict":"maybe","reason":"不确定"}]', {"q1"})

    def test_parse_judgement_fills_missing_id_for_manual_review(self):
        result = parse_judgement('[{"id":"q1","verdict":"approved","reason":"正文支持。"}]', {"q1", "q2"})
        self.assertEqual(result["q2"]["verdict"], "needs_relabel")


if __name__ == "__main__":
    unittest.main()
