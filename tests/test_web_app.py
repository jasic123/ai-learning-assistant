"""FastAPI/Gradio 集成、请求校验和前端共享 Agent 业务入口。"""
import unittest
import os
from unittest.mock import Mock, patch
from app.agent.session import SessionManager

os.environ.setdefault("AUTH_REQUIRED", "false")

from fastapi.testclient import TestClient

from app.main import app
from frontend.app import (chat, delete_history_session_ui, discuss_student_question, recognize_uploaded_question,
                          restore_session, structure_question_text, _readable_math_answer)


class WebAppTests(unittest.TestCase):
    def test_teacher_can_create_knowledgebase_and_upload_pdf(self):
        from pathlib import Path
        from app import config

        data = Path(self._temporary_directory())
        with patch.object(config, "DATA_DIR", data), \
             patch("app.api.knowledgebase.DATA_DIR", data), TestClient(app) as client:
            created = client.post("/api/knowledgebases", json={"name": "physics", "kb_info": "课程教材"})
            self.assertEqual(created.status_code, 200, created.text)
            uploaded = client.post("/api/knowledgebases/physics/files",
                                  files={"file": ("lesson.pdf", b"%PDF-1.4\nlesson", "application/pdf")})
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            details = client.get("/api/knowledgebases/physics")
            self.assertEqual(details.status_code, 200, details.text)
            self.assertEqual(details.json()["files"][0]["source"], "lesson.pdf")
            self.assertEqual(details.json()["files"][0]["status"], "pending")
            with patch("app.api.knowledgebase._build_knowledgebase", return_value={
                    "generation": "a" * 32, "count": 3, "knowledgebase": "physics"}):
                built = client.post("/api/knowledgebases/physics/build")
            self.assertEqual(built.status_code, 200, built.text)
            self.assertEqual(built.json()["count"], 3)
            invalid = client.post("/api/knowledgebases/physics/files",
                                  files={"file": ("fake.pdf", b"not pdf", "application/pdf")})
            self.assertEqual(invalid.status_code, 422)

    def test_student_practices_only_teacher_approved_questions(self):
        from pathlib import Path
        from app import config
        from app.questions.models import Question
        from app.questions.store import QuestionStore

        data = Path(self._temporary_directory())
        with patch.object(config, "DATA_DIR", data), TestClient(app) as client:
            saved = QuestionStore(data).save_many([
                Question("已批准题", ("正确", "错项", "错项", "错项"), "A", "教材依据解析",
                         "lesson.pdf", 3, "approved-1", "approved", "英语", "虚拟语气"),
                Question("待审题", ("正确", "错项", "错项", "错项"), "A", "尚未审核",
                         "lesson.pdf", 4, "draft-1", "draft", "英语", "虚拟语气"),
            ])
            started = client.post("/api/learning/quizzes", json={
                "subject": "英语", "topic": "虚拟语气", "count": 5, "source": "bank",
            })
            self.assertEqual(started.status_code, 200, started.text)
            self.assertEqual([item["id"] for item in started.json()["questions"]], [saved[0]["id"]])
            question_id = saved[0]["id"]
            submitted = client.post(f"/api/learning/quizzes/{started.json()['id']}/submit",
                                    json={"answers": {str(question_id): "B"}})
            self.assertEqual(submitted.status_code, 200, submitted.text)
            self.assertFalse(submitted.json()["results"][0]["correct"])
            self.assertEqual(client.get("/api/learning/mistakes?subject=%E8%8B%B1%E8%AF%AD").json()[0]["question_id"],
                             question_id)
            stats = client.get("/api/questions/analytics")
            self.assertEqual(stats.status_code, 200, stats.text)
            self.assertEqual(stats.json()[0]["attempts"], 1)
            self.assertEqual(stats.json()[0]["accuracy"], 0)

    def test_login_cookie_protects_api_and_sessions_are_per_user(self):
        from pathlib import Path
        folder = Path(self._temporary_directory())
        with patch.dict(os.environ, {"AUTH_REQUIRED": "true"}), \
             patch("app.auth.DATA_DIR", folder), \
             patch("app.agent.session.config.DATA_DIR", folder), TestClient(app) as client:
            self.assertEqual(client.get("/api/sessions").status_code, 401)
            registered = client.post("/api/auth/register", json={
                "username": "student_one", "password": "secure-password-1"})
            self.assertEqual(registered.status_code, 200, registered.text)
            user_one = registered.json()["user"]
            self.assertTrue(client.cookies.get("study_session"))
            self.assertEqual(client.get("/api/auth/me").json()["user"]["id"], user_one["id"])

            first = SessionManager(owner_id=user_one["id"])
            first.append_turn("my-session", [{"role": "user", "content": "私有问题"},
                                               {"role": "assistant", "content": "私有回答"}],
                              expected_version=0)
            self.assertEqual(client.get("/api/sessions?owner_id=local").json()[0]["session_id"], "my-session")

            client.post("/api/auth/logout")
            second_login = client.post("/api/auth/register", json={
                "username": "student_two", "password": "secure-password-2"})
            self.assertEqual(second_login.status_code, 200, second_login.text)
            self.assertEqual(client.get("/api/sessions").json(), [])
            self.assertEqual(client.get("/api/sessions/my-session").status_code, 404)

    def test_ui_and_health_are_served(self):
        with TestClient(app) as client:
            health = client.get("/health")
            page = client.get("/")
            self.assertEqual(health.json(), {"status": "ok"})
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.headers["content-type"])
            self.assertEqual(client.get("/openapi.json").json()["info"]["title"], "AI 学习助手")
            self.assertEqual(client.get("/api/knowledgebases").status_code, 200)

    def test_chat_route_validates_and_reuses_agent_service(self):
        result = {"session_id": "session-1", "answer": "答案 [T1-C1-S1]", "sources": [],
                  "model_calls": 2, "tool_calls": 1, "compression": [],
                  "summary_calls": 0, "estimated_input_tokens": 10}
        with patch("app.api.chat.AgentService") as factory, TestClient(app) as client:
            factory.return_value.chat.return_value.to_dict.return_value = result
            invalid = client.post("/api/chat", json={"message": "问题", "session_id": "../bad",
                "knowledgebases": ["教育资料"]})
            self.assertEqual(invalid.status_code, 422)
            response = client.post("/api/chat", json={"message": "问题", "session_id": "session-1",
                "knowledgebases": ["教育资料"], "mode": "hybrid",
                "vector_weight": 0.7, "keyword_weight": 0.3})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), result)
            factory.assert_called_once_with(["教育资料"], mode="hybrid", rerank=False,
                                            owner_id="local", study_mode="general",
                                            vector_weight=0.7, keyword_weight=0.3)
            factory.return_value.chat.assert_called_once_with("问题", session_id="session-1")

    def test_question_generation_route_uses_rag_evidence_and_returns_drafts(self):
        from pathlib import Path
        from app.rag.domain.contracts import SearchResponse
        from app.rag.domain.models import Chunk, SearchResult

        chunk = Chunk("chunk-1", "教材.pdf", 3, "光合作用将光能转化为化学能。")
        hit = SearchResult(chunk, 0.9, "bm25")
        response = SearchResponse("生物", "光合作用", "generation-1", (hit,))
        model_output = ('{"stem":"光合作用将光能转化为什么？",'
                        '"options":{"A":"热能","B":"化学能","C":"声能","D":"电能"},'
                        '"answer":"B","explanation":"资料指出光能转化为化学能。"}')
        with patch("app.api.questions.RagService") as rag_factory, \
             patch("app.api.questions.ModelClient") as client_factory, \
             patch("app.api.questions.model_settings", return_value=object()), \
             patch("app.questions.store.config.DATA_DIR", Path(self._temporary_directory())), \
             TestClient(app) as client:
            rag_factory.return_value.search.return_value = response
            client_factory.return_value.__enter__.return_value.chat.return_value = model_output
            result = client.post("/api/questions/generate", json={
                "knowledgebase": "生物", "topic": "光合作用", "count": 1, "mode": "bm25",
                "vector_weight": 0.3, "keyword_weight": 0.7})
            self.assertEqual(result.status_code, 200)
            question = result.json()[0]
            self.assertEqual(question["answer"], "B")
            self.assertEqual((question["source"], question["page"], question["chunk_id"]),
                             ("教材.pdf", 3, "chunk-1"))
            self.assertEqual(question["status"], "draft")
            prompt = client_factory.return_value.__enter__.return_value.chat.call_args.args[0][1]["content"]
            self.assertIn("光合作用", prompt)
            request = rag_factory.return_value.search.call_args.args[0]
            self.assertEqual((request.knowledgebase, request.query, request.top_k, request.mode),
                             ("生物", "光合作用", 1, "bm25"))
            self.assertEqual((request.vector_weight, request.keyword_weight), (0.3, 0.7))
            question_id = question["id"]
            drafts = client.get("/api/questions", params={"status": "draft"}).json()
            self.assertEqual([item["id"] for item in drafts], [question_id])
            reviewed = client.put(f"/api/questions/{question_id}", json={
                "stem": "老师修改后的题干", "options": ["热能", "化学能", "声能", "电能"],
                "answer": "B", "explanation": "老师确认了答案和依据。", "status": "approved",
            })
            self.assertEqual(reviewed.status_code, 200)
            self.assertEqual(reviewed.json()["status"], "approved")
            self.assertEqual((reviewed.json()["source"], reviewed.json()["page"]), ("教材.pdf", 3))

    def test_learning_api_submits_practice_and_lists_wrong_answers(self):
        from pathlib import Path
        from app.learning.store import LearningStore
        from app.questions.models import Question
        from app.questions.store import QuestionStore
        data = Path(self._temporary_directory())
        with patch("app.questions.store.config.DATA_DIR", Path(self._temporary_directory())), \
             patch("app.learning.store.config.DATA_DIR", data), TestClient(app) as client:
            question = QuestionStore(data).save_many([Question(
                "迭代器返回下一个值的方法？", ("__iter__", "__next__", "__len__", "__str__"),
                "B", "Python 3 使用 __next__。", "教材.pdf", 62, "chunk-1")])[0]
            quiz = LearningStore(data).create_quiz(subject="计算机", topic="迭代器", questions=[question])
            submitted = client.post(f"/api/learning/quizzes/{quiz['id']}/submit",
                                    json={"answers": {str(question["id"]): "A"}})
            self.assertEqual(submitted.status_code, 200)
            self.assertEqual(submitted.json()["score"], 0)
            attempt_id = submitted.json()["attempt_id"]
            reason = client.post(f"/api/learning/attempts/{attempt_id}/error-reasons",
                                 json={"reasons": {str(question["id"]): "condition_missed"}})
            self.assertEqual(reason.status_code, 200, reason.text)
            profile = client.get("/api/learning/profile")
            self.assertEqual(profile.status_code, 200, profile.text)
            self.assertEqual(profile.json()["error_patterns"][0]["reason"], "condition_missed")
            mistakes = client.get("/api/learning/mistakes", params={"subject": "计算机", "query": "迭代器"})
            self.assertEqual(mistakes.status_code, 200)
            self.assertEqual(len(mistakes.json()), 1)
            history = client.get("/api/learning/history", params={"days": 7})
            self.assertEqual(history.status_code, 200, history.text)
            self.assertEqual(history.json()["timeline"][0]["topic"], "迭代器")
            due = client.get("/api/learning/reviews/due")
            self.assertEqual(len(due.json()), 1)
            self.assertNotIn("answer", due.json()[0])

    def test_learning_profile_is_scoped_to_authenticated_owner(self):
        from pathlib import Path
        from app.learning.store import LearningStore
        from app.questions.models import Question
        from app.questions.store import QuestionStore
        data = Path(self._temporary_directory())
        with patch("app.auth.DATA_DIR", data), patch("app.learning.store.config.DATA_DIR", data), \
             patch("app.questions.store.config.DATA_DIR", data), TestClient(app) as client:
            registered = client.post("/api/auth/register", json={
                "username": "profile_student", "password": "secure-password-1"})
            self.assertEqual(registered.status_code, 200, registered.text)
            question = QuestionStore(data).save_many([Question(
                "概念题", ("A", "B", "C", "D"), "B", "解析", "教材.pdf", 1, "profile-1")])[0]
            foreign = LearningStore(data, owner_id="other-student")
            quiz = foreign.create_quiz(subject="数学", topic="函数", questions=[question])
            attempt = foreign.submit_quiz(quiz["id"], {str(question["id"]): "A"})
            response = client.post(f"/api/learning/attempts/{attempt['attempt_id']}/error-reasons",
                                   json={"reasons": {str(question["id"]): "guessed"}})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(client.get("/api/learning/profile").json()["total_attempts"], 0)

    def test_practice_generation_weight_routing_and_wrong_answer_persistence(self):
        from pathlib import Path
        from app.questions.models import Question
        from app.questions.store import QuestionStore
        from app.learning.store import LearningStore
        data = Path(self._temporary_directory())
        with patch("app.questions.store.config.DATA_DIR", data), \
             patch("app.learning.store.config.DATA_DIR", data), \
             patch("app.api.learning.generate_questions") as generate, TestClient(app) as client:
            question = QuestionStore(data).save_many([Question(
                "求极限题", ("1/6", "-1/6", "0", "1/3"), "B", "正弦展开得到 -1/6。",
                "教材.pdf", 12, "limit-chunk")])[0]
            generate.return_value = [question]
            response = client.post("/api/learning/quizzes", json={
                "knowledgebase": "数学资料", "subject": "数学", "topic": "极限",
                "count": 1, "mode": "hybrid", "vector_weight": 0.3, "keyword_weight": 0.7,
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["questions"]), 1)
            request = generate.call_args.args[0]
            self.assertEqual((request.vector_weight, request.keyword_weight), (0.3, 0.7))
            quiz_id = response.json()["id"]
            question_id = response.json()["questions"][0]["id"]
            graded = client.post(f"/api/learning/quizzes/{quiz_id}/submit",
                                 json={"answers": {str(question_id): "A"}})
            self.assertEqual(graded.status_code, 200, graded.text)
            self.assertFalse(graded.json()["results"][0]["correct"])
            mistakes = client.get("/api/learning/mistakes", params={"subject": "数学"}).json()
            self.assertEqual([item["question_id"] for item in mistakes], [question_id])

    def test_student_can_import_question_without_knowledgebase(self):
        from pathlib import Path
        data = Path(self._temporary_directory())
        payload = {
            "subject": "计算机", "topic": "贪心算法", "stem": "贪心算法每步选择什么？",
            "options": ["局部最优", "随机值", "全局最差", "任意值"],
            "answer": "A", "explanation": "每一步选择当前看来最优的解。",
            "ocr_text": "原始试卷文字",
        }
        with patch("app.questions.store.config.DATA_DIR", data), \
             patch("app.learning.store.config.DATA_DIR", data), TestClient(app) as client:
            response = client.post("/api/learning/mistakes/import", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            question_id = response.json()["question"]["id"]
            found = client.get("/api/learning/mistakes", params={"subject": "计算机", "query": "贪心"})
            self.assertEqual([item["question_id"] for item in found.json()], [question_id])
            due = client.get("/api/learning/reviews/due")
            self.assertEqual([item["question_id"] for item in due.json()], [question_id])

    def test_student_image_ocr_route_uses_local_engine(self):
        import io
        from PIL import Image
        from unittest.mock import patch
        image = Image.new("RGB", (8, 8), "white")
        content = io.BytesIO()
        image.save(content, format="PNG")
        fake_engine = Mock(return_value=([("题干", "贪心算法每一步如何选择？", 0.99)], None))
        with patch("app.api.learning._engine", return_value=fake_engine), TestClient(app) as client:
            response = client.post("/api/learning/mistakes/ocr", files={
                "file": ("question.png", content.getvalue(), "image/png")
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"text": "贪心算法每一步如何选择？"})
        fake_engine.assert_called_once()

    def test_student_question_structure_route_returns_reviewable_fields(self):
        model_json = ('{"subject":"计算机","topic":"贪心算法","stem":"每一步如何选择？",'
                      '"options":{"A":"局部最优","B":"随机","C":"全局最差","D":"任意"},'
                      '"answer":"A","explanation":"每一步选择当前最优方案。"}')
        with patch("app.learning.recognition.ModelClient") as factory, \
             patch("app.learning.recognition.model_settings", return_value=object()), TestClient(app) as client:
            factory.return_value.__enter__.return_value.chat.return_value = model_json
            response = client.post("/api/learning/mistakes/structure", json={"text": "OCR 原文"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["topic"], "贪心算法")
        self.assertEqual(response.json()["options"]["A"], "局部最优")
        self.assertEqual(response.json()["answer"], "A")

    def test_student_question_structure_stream_emits_deltas_then_fields(self):
        result = ("计算机", "贪心算法", "每一步如何选择？", "局部最优", "随机", "全局最差", "任意",
                  "A", "每一步选择当前最优方案。", "题目已自动整理")

        def structure(text, on_delta=None):
            on_delta("{\\\"title\\\":")
            on_delta("\\\"贪心\\\"}")
            return result

        with patch("app.api.learning.structure_question_text", side_effect=structure), TestClient(app) as client:
            response = client.post("/api/learning/mistakes/structure/stream", json={"text": "OCR 原文"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "token"', response.text)
        self.assertIn('"type": "done"', response.text)
        self.assertIn('"topic": "贪心算法"', response.text)

    def test_review_can_start_without_knowledgebase_and_removes_correct_mistake(self):
        from pathlib import Path
        from app.learning.store import LearningStore
        from app.questions.models import Question
        from app.questions.store import QuestionStore
        data = Path(self._temporary_directory())
        with patch("app.questions.store.config.DATA_DIR", data), \
             patch("app.learning.store.config.DATA_DIR", data), TestClient(app) as client:
            questions = QuestionStore(data).save_many([
                Question(f"高数极限题 {index}", ("1/6", "-1/6", "0", "1/3"), "B",
                         "正弦展开给出 -1/6。", "学生上传题目", 1, f"upload-review-{index}", "approved")
                for index in range(3)
            ])
            practice = LearningStore(data).create_quiz(
                subject="数学", topic="极限", questions=questions)
            LearningStore(data).submit_quiz(practice["id"], {str(question["id"]): "A" for question in questions})

            too_many = client.post("/api/learning/reviews/start", json={"subject": "数学", "count": 4})
            self.assertEqual(too_many.status_code, 422)
            self.assertIn("只有 3 道到期题", too_many.json()["detail"])

            started = client.post("/api/learning/reviews/start", json={"subject": "数学", "count": 1})
            self.assertEqual(started.status_code, 200, started.text)
            quiz = started.json()
            self.assertEqual(len(quiz["questions"]), 1)
            self.assertEqual((quiz["requested_count"], quiz["available_count"], quiz["selected_count"]),
                             (1, 3, 1))
            reviewed_question_id = quiz["questions"][0]["id"]
            submitted = client.post(f"/api/learning/quizzes/{quiz['id']}/submit",
                                    json={"answers": {str(reviewed_question_id): "B"}})
            self.assertEqual(submitted.status_code, 200, submitted.text)
            self.assertTrue(submitted.json()["results"][0]["correct"])
            self.assertEqual(len(client.get("/api/learning/mistakes").json()), 2)

    def test_question_ocr_callback_auto_fills_structured_fields(self):
        model_json = ('{"subject":"计算机","topic":"高数极限","stem":"求极限","options":{"A":"1/6",'
                      '"B":"-1/6","C":"0","D":"1/3"},"answer":"B",'
                      '"explanation":"利用正弦展开。"}')
        ocr_text = "高等数学\nlim (sin x - x) / x^3\n求极限：\nx->0\nA. 1/6\nB. -1/6"
        with patch("frontend.app.ocr_image", return_value=ocr_text) as ocr, \
             patch("app.learning.recognition.ModelClient") as factory, \
             patch("app.learning.recognition.model_settings", return_value=object()):
            factory.return_value.__enter__.return_value.chat.return_value = model_json
            result = recognize_uploaded_question("question.png")
        self.assertEqual(result[0], ocr_text)
        self.assertEqual(result[1:9], (
            "数学", "高数极限", "求极限：lim (sin x - x) / x^3；x->0",
            "1/6", "-1/6", "0", "1/3", "B",
        ))
        self.assertIn("补入模型可能遗漏的数学表达式", result[-1])
        self.assertIn("自动整理", result[-1])
        ocr.assert_called_once_with("question.png")

    def test_pasted_question_text_is_structured_without_manual_retyping(self):
        model_json = ('{"subject":"数学","topic":"极限","stem":"题干","options":{"A":"甲",'
                      '"B":"乙","C":"丙","D":"丁"},"answer":"", "explanation":""}')
        with patch("app.learning.recognition.ModelClient") as factory, \
             patch("app.learning.recognition.model_settings", return_value=object()):
            factory.return_value.__enter__.return_value.chat.return_value = model_json
            result = structure_question_text("一段 OCR 试题")
        self.assertEqual(result[:8], ("数学", "极限", "题干", "甲", "乙", "丙", "丁", None))
        self.assertIn("自行选择", result[-1])

    def test_gradio_callback_uses_http_chat_handler_and_returns_citations(self):
        result = {"answer": "列表见 [T1-C1-S1]。", "sources": [
            {"citation": "T1-C1-S1", "source": "教材.pdf", "page": 8}],
            "model_calls": 2, "tool_calls": 1}
        history = [{"role": "assistant", "content": "前文"}]
        with patch("frontend.app.run_chat", return_value=result) as send:
            updated, cleared = chat("列表是什么？", history, "session-1", ["教育资料"], "hybrid", False)
        self.assertEqual(cleared, "")
        self.assertEqual(len(updated), 3)
        self.assertTrue(updated[-1]["content"].endswith("[T1-C1-S1] 教材.pdf · 第 8 页"))
        self.assertEqual(updated[0], history[0])
        self.assertEqual(send.call_args.args[0].session_id, "session-1")

    def test_gradio_empty_selection_and_errors_are_displayed(self):
        history = []
        unchanged, cleared = chat("问题", history, "s", [], "hybrid", False)
        self.assertEqual(cleared, "")
        self.assertIn("选择", unchanged[-1]["content"])
        with patch("frontend.app.run_chat", side_effect=ValueError("知识库名称无效")):
            updated, _ = chat("问题", history, "s", ["../bad"], "hybrid", False)
        self.assertIn("知识库名称无效", updated[-1]["content"])
        skipped, text = chat(" ", history, "s", ["教育资料"], "hybrid", False)
        self.assertEqual((skipped, text), (history, ""))

    def test_math_discussion_strips_latex_answer_wrappers(self):
        raw = r"所以答案是 \[ \boxed{-\frac{1}{6}} \]。"
        self.assertEqual(_readable_math_answer(raw), "所以答案是 -1/6。")
        with patch("frontend.app.ModelClient") as factory, \
             patch("frontend.app.model_settings", return_value=object()):
            factory.return_value.__enter__.return_value.chat.return_value = raw
            history, _ = discuss_student_question("求极限", [], "答案是什么？")
        self.assertIn("答案是 -1/6", history[-1]["content"])
        self.assertNotIn("\\boxed", history[-1]["content"])
        prompt = factory.return_value.__enter__.return_value.chat.call_args.args[0][0]["content"]
        self.assertIn("不要输出 LaTeX 标记", prompt)
        self.assertEqual(
            _readable_math_answer(r"\frac{x+1}{x-2} \approx 3/2"),
            "(x+1) 除以 (x-2) 约等于 3/2",
        )

    def test_memory_routes_validate_and_persist_with_test_store(self):
        with patch("app.agent.session.config.DATA_DIR", __import__("pathlib").Path(self._temporary_directory())), \
             TestClient(app) as client:
            saved = client.put("/api/memories", params={"owner_id": "local"},
                               json={"key": "language", "content": "中文"})
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(client.get("/api/memories").json(), [{"key": "language", "content": "中文"}])
            self.assertEqual(client.delete("/api/memories/language").json(), {"deleted": True})
            self.assertEqual(client.put("/api/memories", json={"key": "../bad", "content": "x"}).status_code, 422)

    def test_session_routes_and_gradio_restore_persisted_short_term_history(self):
        from pathlib import Path
        folder = self._temporary_directory()
        with patch("app.agent.session.config.DATA_DIR", Path(folder)):
            sessions = SessionManager()
            sessions.append_turn("study-1", [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": None, "tool_calls": []},
                {"role": "assistant", "content": "第一答"},
            ], expected_version=0)
            with TestClient(app) as client:
                listed = client.get("/api/sessions").json()
                self.assertEqual(listed[0]["preview"], "第一问")
                detail = client.get("/api/sessions/study-1").json()
                self.assertEqual(detail["turns"][0]["messages"][-1]["content"], "第一答")
                self.assertEqual(client.get("/api/sessions?owner_id=other").json()[0]["session_id"], "study-1")
            with patch("frontend.app.SessionManager", return_value=sessions):
                restored, session_id = restore_session("study-1")
                self.assertEqual(session_id, "study-1")
                self.assertEqual(restored, [
                    {"role": "user", "content": "第一问"},
                    {"role": "assistant", "content": "第一答"},
                ])

    def test_session_delete_route_removes_owned_conversation(self):
        from pathlib import Path
        folder = self._temporary_directory()
        with patch("app.agent.session.config.DATA_DIR", Path(folder)):
            sessions = SessionManager()
            sessions.append_turn("delete-me", [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ], expected_version=0)
            sessions.append_turn("delete-me", [
                {"role": "user", "content": "第二问"},
                {"role": "assistant", "content": "第二答"},
            ], expected_version=1, checkpoint=(1, "摘要", []))
            with TestClient(app) as client:
                deleted = client.delete("/api/sessions/delete-me")
                self.assertEqual(deleted.status_code, 200, deleted.text)
                self.assertEqual(deleted.json(), {"session_id": "delete-me", "deleted": True})
                self.assertEqual(client.get("/api/sessions").json(), [])
                self.assertEqual(client.delete("/api/sessions/delete-me").status_code, 404)

    def test_locked_database_error_is_actionable_in_ui_and_api(self):
        from pathlib import Path
        import sqlite3
        folder = self._temporary_directory()
        with patch("frontend.app.SessionManager") as manager:
            manager.return_value.delete_session.side_effect = sqlite3.OperationalError("database is locked")
            result = delete_history_session_ui("session-1", True, [], "session-2")
        self.assertIn("关闭数据库", result[3])
        self.assertEqual(result[1], "session-2")

        with patch("app.agent.session.config.DATA_DIR", Path(folder)), \
             patch("app.api.sessions.SessionManager.delete_session",
                   side_effect=sqlite3.OperationalError("database is locked")), TestClient(app) as client:
            response = client.delete("/api/sessions/locked-session")
        self.assertEqual(response.status_code, 503)
        self.assertIn("关闭数据库浏览器", response.json()["detail"])

    def _temporary_directory(self):
        import tempfile
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return folder.name


if __name__ == "__main__":
    unittest.main()
