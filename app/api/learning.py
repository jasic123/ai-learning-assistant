"""Practice, mistake review and learning history endpoints."""
import sqlite3
import random
import tempfile
import json
from queue import Queue
from threading import Thread
from pathlib import Path as FilePath
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Path, Query, UploadFile, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse

from app.api.questions import GenerateQuestionsRequest, generate_questions
from app.learning.store import LearningStore
from app.learning.recognition import structure_question_text
from app.questions.store import QuestionStore
from app.questions.models import Question
from app.rag.ingestion.pdf_ocr import OcrUnavailable, _engine
from app.rag import RagService, SearchRequest


router = APIRouter(prefix="/api/learning", tags=["learning"])


class StartQuizRequest(BaseModel):
    knowledgebase: str = Field(default="", max_length=128)
    subject: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=2000)
    count: int = Field(default=5, ge=1, le=8)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = False
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    keyword_weight: float = Field(default=0.5, ge=0, le=1)
    source: Literal["library", "web", "auto", "bank"] = "auto"


class SubmitQuizRequest(BaseModel):
    answers: dict[str, Literal["A", "B", "C", "D"]]


class ReplaceQuestionRequest(BaseModel):
    question_id: int = Field(gt=0)
    knowledgebase: str = Field(default="", max_length=128)
    subject: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=2000)
    source: Literal["library", "web", "auto", "bank"] = "auto"


class ErrorReasonsRequest(BaseModel):
    reasons: dict[str, Literal[
        "concept_unclear", "condition_missed", "method_confused",
        "calculation_error", "guessed", "other",
    ]]


class StartReviewRequest(BaseModel):
    knowledgebase: str | None = Field(default=None, min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=100)
    count: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = False


class ImportMistakeRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=2000)
    stem: str = Field(min_length=1, max_length=4000)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1, max_length=8000)
    ocr_text: str = Field(default="", max_length=30000)


class StructureMistakeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)


class ReviewRequest(BaseModel):
    quality: int = Field(ge=0, le=5)


class ChatFeedbackRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    turn_number: int = Field(ge=1)
    rating: Literal[-1, 1]
    comment: str = Field(default="", max_length=1000)


def start_quiz(payload, owner_id="local"):
    if payload.source == "bank":
        questions = QuestionStore().list(status="approved", subject=payload.subject, topic=payload.topic)
        random.SystemRandom().shuffle(questions)
        questions = questions[:payload.count]
        if not questions:
            return {"id": None, "questions": []}
        try:
            return LearningStore(owner_id=owner_id).create_quiz(
                subject=payload.subject, topic=payload.topic, questions=questions)
        except (ValueError, OSError, sqlite3.Error) as error:
            if isinstance(error, ValueError):
                raise HTTPException(status_code=422, detail=str(error)) from None
            raise HTTPException(status_code=503, detail="学习记录暂时不可用。") from None
    questions = generate_questions(GenerateQuestionsRequest(
        knowledgebase=payload.knowledgebase, subject=payload.subject, topic=payload.topic, count=payload.count,
        mode=payload.mode, rerank=payload.rerank,
        vector_weight=payload.vector_weight, keyword_weight=payload.keyword_weight,
        source=payload.source,
    ))
    if not questions:
        return {"id": None, "questions": []}
    try:
        return LearningStore(owner_id=owner_id).create_quiz(subject=payload.subject, topic=payload.topic, questions=questions)
    except (ValueError, OSError, sqlite3.Error) as error:
        if isinstance(error, ValueError):
            raise HTTPException(status_code=422, detail=str(error)) from None
        raise HTTPException(status_code=503, detail="学习记录暂时不可用。") from None


def start_review(payload, owner_id="local"):
    store = LearningStore(owner_id=owner_id)
    due = store.due_reviews(limit=100, subject=payload.subject)
    available_count = len(due)
    if payload.count > available_count:
        raise HTTPException(
            status_code=422,
            detail=f"{payload.subject}当前只有 {available_count} 道到期题，请减少复习题量。",
        )
    random.SystemRandom().shuffle(due)
    due = due[:payload.count]
    if not due:
        return {"id": None, "questions": [], "review_pairs": [],
                "requested_count": payload.count, "available_count": available_count,
                "selected_count": 0}
    bank = {item["id"]: item for item in QuestionStore().list()}
    questions, pairs = [], []
    for item in due:
        original = bank.get(item["question_id"])
        if original is None:
            continue
        questions.append(original)
        # 暂不生成变式题；同一个 ID 成对表示“原错题答对即可移除”。
        pairs.append([original["id"], original["id"]])
    if not questions:
        return {"id": None, "questions": [], "review_pairs": [],
                "message": "当前到期错题没有可用题目。", "requested_count": payload.count,
                "available_count": available_count, "selected_count": 0}
    try:
        quiz = store.create_quiz(subject=payload.subject, topic="到期错题复习", questions=questions,
                                 quiz_type="review", review_pairs=pairs)
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="学习数据库不可写，请检查 data 目录权限或关闭占用数据库的进程。") from None
    return {**quiz, "review_pairs": pairs, "requested_count": payload.count,
            "available_count": available_count, "selected_count": len(questions)}


def submit_quiz(quiz_id, payload, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).submit_quiz(quiz_id, payload.answers)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="学习记录暂时不可用。") from None


def replace_quiz_question(quiz_id, payload, owner_id="local"):
    try:
        if payload.source == "bank":
            generated = [question for question in QuestionStore().list(
                status="approved", subject=payload.subject, topic=payload.topic)
                         if question["id"] != payload.question_id]
            if generated:
                random.SystemRandom().shuffle(generated)
                generated = generated[:1]
        else:
            generated = generate_questions(GenerateQuestionsRequest(
                knowledgebase=payload.knowledgebase, subject=payload.subject, topic=payload.topic, count=1,
                source=payload.source,
            ))
        if not generated:
            raise HTTPException(status_code=422, detail="没有找到可替换题目的资料。")
        replacement = generated[0]
        index = LearningStore(owner_id=owner_id).replace_quiz_question(
            quiz_id, payload.question_id, replacement, owner_id=owner_id)
        return {"index": index, "question": replacement}
    except HTTPException:
        raise
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="学习数据库暂时不可用。") from None


def save_error_reasons(attempt_id, payload, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).save_error_reasons(
            attempt_id, payload.reasons, owner_id=owner_id,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="错因记录暂时不可用。") from None


def learning_profile(owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).learning_profile(owner_id=owner_id)
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="学习画像暂时不可用。") from None


def list_mistakes(subject=None, query=None, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).list_mistakes(subject=subject, query=query)
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="错题本暂时不可用。") from None


def import_mistake(payload, owner_id="local"):
    try:
        question = Question(
            stem=payload.stem, options=tuple(payload.options), answer=payload.answer,
            explanation=payload.explanation, source="学生上传题目", page=1,
            chunk_id=f"upload-{uuid4().hex}", status="approved",
            subject=payload.subject, topic=payload.topic,
        )
        saved = QuestionStore().save_many([question])[0]
        LearningStore(owner_id=owner_id).add_mistake(saved, subject=payload.subject, topic=payload.topic)
        return {"question": saved, "message": "题目已保存到错题本。"}
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except HTTPException:
        raise
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="题库暂时不可写，请检查数据库权限。") from None


def ocr_image(file):
    if not file or not isinstance(file, (str, FilePath)):
        raise ValueError("请先上传图片")
    path = FilePath(file)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError("只支持 PNG、JPG、JPEG、WEBP 或 BMP 图片")
    if path.stat().st_size > 15 * 1024 * 1024:
        raise ValueError("图片不能超过 15 MB")
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError("图片分辨率不能超过 2500 万像素")
            image = np.asarray(source.convert("RGB"))
        result, _ = _engine()(image)
    except OcrUnavailable:
        raise
    except Exception as error:
        raise RuntimeError(f"图片 OCR 失败：{error}") from None
    return "\n".join(item[1] for item in (result or []))


def due_reviews(limit=20, owner_id="local"):
    try:
        items = LearningStore(owner_id=owner_id).due_reviews(limit)
        # Do not show the answer before the learner has attempted self-recall.
        return [{key: value for key, value in item.items() if key not in {"answer", "explanation"}}
                for item in items]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="复习任务暂时不可用。") from None


def review_mistake(question_id, payload, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).review_mistake(question_id, payload.quality)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="复习记录暂时不可用。") from None


def learning_history(days, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).history(days)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="学习历史暂时不可用。") from None


def chat_feedback(payload, owner_id="local"):
    try:
        return LearningStore(owner_id=owner_id).record_chat_feedback(
            payload.session_id, payload.turn_number, payload.rating, payload.comment,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="问答反馈暂时不可用。") from None


@router.post("/quizzes", summary="按学科和知识点生成练习")
async def create_quiz(payload: StartQuizRequest, request: Request):
    return await run_in_threadpool(start_quiz, payload, request.state.user["id"])


@router.post("/mistakes/import", summary="保存学生上传或手动录入的错题")
async def mistake_import(payload: ImportMistakeRequest, request: Request):
    return await run_in_threadpool(import_mistake, payload, request.state.user["id"])


@router.post("/mistakes/ocr", summary="识别学生上传的试题图片")
async def mistake_ocr(request: Request, file: UploadFile = File(...)):
    suffix = FilePath(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(status_code=422, detail="只支持 PNG、JPG、JPEG、WEBP 或 BMP 图片")
    temporary = FilePath(tempfile.gettempdir()) / f"mistake-{uuid4().hex}{suffix}"
    try:
        data = await file.read()
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(status_code=422, detail="图片不能超过 15 MB")
        temporary.write_bytes(data)
        return {"text": await run_in_threadpool(ocr_image, temporary)}
    except OcrUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    finally:
        temporary.unlink(missing_ok=True)


@router.post("/mistakes/structure", summary="将 OCR 试题原文整理为可校对字段")
async def mistake_structure(payload: StructureMistakeRequest, request: Request):
    result = await run_in_threadpool(structure_question_text, payload.text)
    subject, topic, stem, option_a, option_b, option_c, option_d, answer, explanation, message = result
    return {"subject": subject, "topic": topic, "stem": stem,
            "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d},
            "answer": answer, "explanation": explanation, "message": message}


@router.post("/mistakes/structure/stream", summary="流式整理 OCR 试题字段")
async def mistake_structure_stream(payload: StructureMistakeRequest, request: Request):
    events = Queue()

    def run():
        try:
            result = structure_question_text(payload.text, on_delta=lambda text: events.put({"type": "token", "content": text}))
            subject, topic, stem, option_a, option_b, option_c, option_d, answer, explanation, message = result
            events.put({"type": "done", "result": {
                "subject": subject, "topic": topic, "stem": stem,
                "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d},
                "answer": answer, "explanation": explanation, "message": message,
            }})
        except Exception:
            events.put({"type": "error", "detail": "题目整理失败，请检查模型服务后重试。"})

    def encode():
        Thread(target=run, daemon=True).start()
        while True:
            event = events.get()
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            if event["type"] in {"done", "error"}:
                return
        yield "data: [DONE]\n\n"

    return StreamingResponse(iterate_in_threadpool(encode()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/reviews/start", summary="随机生成到期错题及举一反三练习")
async def review_start(payload: StartReviewRequest, request: Request):
    return await run_in_threadpool(start_review, payload, request.state.user["id"])


@router.post("/quizzes/{quiz_id}/submit", summary="提交答案并自动判题")
async def quiz_submit(request: Request, quiz_id: int = Path(gt=0), payload: SubmitQuizRequest = ...):
    return await run_in_threadpool(submit_quiz, quiz_id, payload, request.state.user["id"])


@router.post("/quizzes/{quiz_id}/replace", summary="替换练习中的一道题")
async def quiz_replace(request: Request, quiz_id: int = Path(gt=0), payload: ReplaceQuestionRequest = ...):
    return await run_in_threadpool(replace_quiz_question, quiz_id, payload, request.state.user["id"])


@router.post("/attempts/{attempt_id}/error-reasons", summary="保存答错后的自选错因")
async def attempt_error_reasons(request: Request, attempt_id: int = Path(gt=0),
                                payload: ErrorReasonsRequest = ...):
    return await run_in_threadpool(save_error_reasons, attempt_id, payload, request.state.user["id"])


@router.get("/profile", summary="查看长期学习画像")
async def profile(request: Request):
    return await run_in_threadpool(learning_profile, request.state.user["id"])


@router.get("/mistakes", summary="搜索错题本")
async def mistakes(request: Request, subject: str | None = Query(default=None, max_length=100),
                   query: str | None = Query(default=None, max_length=200)):
    return await run_in_threadpool(list_mistakes, subject, query, request.state.user["id"])


@router.get("/reviews/due", summary="获取到期复习题")
async def reviews_due(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    return await run_in_threadpool(due_reviews, limit, request.state.user["id"])


@router.post("/mistakes/{question_id}/review", summary="自评复习难度并更新间隔")
async def mistake_review(request: Request, question_id: int = Path(gt=0), payload: ReviewRequest = ...):
    return await run_in_threadpool(review_mistake, question_id, payload, request.state.user["id"])


@router.get("/history", summary="查看 7/14/30 天学习轨迹")
async def history(request: Request, days: int = Query(default=7)):
    return await run_in_threadpool(learning_history, days, request.state.user["id"])


@router.post("/chat-feedback", summary="提交一轮问答反馈")
async def feedback(payload: ChatFeedbackRequest, request: Request):
    return await run_in_threadpool(chat_feedback, payload, request.state.user["id"])
