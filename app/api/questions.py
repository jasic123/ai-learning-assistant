"""从已入库教学资料生成待审核选择题。"""
from typing import Literal
import sqlite3

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import model_settings
from app.llm.client import ModelClient
from app.questions.generator import QuestionGenerationError, QuestionGenerator
from app.questions.store import QuestionStore
from app.rag import RagService, SearchRequest
from app.rag.domain.models import Chunk
from app.tools.web import WebSearchTool
import hashlib


router = APIRouter(prefix="/api/questions", tags=["questions"])


class GenerateQuestionsRequest(BaseModel):
    knowledgebase: str = Field(default="", max_length=128)
    subject: str = Field(default="", max_length=100)
    topic: str = Field(min_length=1, max_length=2000)
    count: int = Field(default=3, ge=1, le=8)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = False
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    keyword_weight: float = Field(default=0.5, ge=0, le=1)
    source: Literal["library", "web", "auto"] = "auto"


class ReviewQuestionRequest(BaseModel):
    stem: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1, max_length=4000)
    status: Literal["draft", "approved", "rejected"]
    subject: str | None = Field(default=None, max_length=100)
    topic: str | None = Field(default=None, max_length=2000)


def generate_questions(payload: GenerateQuestionsRequest) -> list[dict]:
    try:
        chunks = []
        if payload.source in {"library", "auto"}:
            if not payload.knowledgebase:
                if payload.source == "library":
                    raise ValueError("资料库模式必须选择资料库")
            else:
                results = RagService().search(SearchRequest(
                    payload.knowledgebase, payload.topic, top_k=payload.count,
                    mode=payload.mode, rerank=payload.rerank,
                    vector_weight=payload.vector_weight, keyword_weight=payload.keyword_weight,
                ))
                chunks.extend(hit.chunk for hit in results.hits)
        if payload.source in {"web", "auto"} and len(chunks) < payload.count:
            query = " ".join(item for item in (payload.subject, payload.topic, "选择题 练习") if item).strip()
            search = WebSearchTool().execute(query, max_results=max(payload.count - len(chunks), 3))
            if search.get("error"):
                if not chunks:
                    raise RuntimeError(search.get("message", "联网搜索失败"))
            for item in search.get("results", []):
                text = f"{item.get('title', '')}\n{item.get('summary', '')}".strip()
                if text:
                    digest = hashlib.sha256(item["url"].encode()).hexdigest()[:24]
                    chunks.append(Chunk(digest, item["url"], 1, text))
        if not chunks:
            return []
        with ModelClient(model_settings("LLM")) as client:
            generator = QuestionGenerator(client)
            questions = []
            failures = []
            for chunk in chunks:
                try:
                    question = generator.generate(chunk, topic=payload.topic,
                                                  subject=payload.subject)
                except QuestionGenerationError as error:
                    failures.append(str(error))
                    continue
                questions.append(question)
                if len(questions) >= payload.count:
                    break
            if not questions and failures:
                raise QuestionGenerationError(failures[0])
        return QuestionStore().save_many(questions)
    except QuestionGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except OSError:
        raise HTTPException(status_code=503, detail="本地资料或索引暂时不可用。") from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="题库数据库暂时不可用。") from None


def list_questions(status=None, subject=None, topic=None):
    try:
        return QuestionStore().list(status, subject=subject, topic=topic)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="题库数据库暂时不可用。") from None


def review_question(question_id, payload):
    try:
        return QuestionStore().review(
            question_id, stem=payload.stem, options=payload.options,
            answer=payload.answer, explanation=payload.explanation, status=payload.status,
            subject=payload.subject, topic=payload.topic,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="题库数据库暂时不可用。") from None


@router.post("/generate", summary="根据教学资料生成待审核选择题")
async def generate(payload: GenerateQuestionsRequest):
    return await run_in_threadpool(generate_questions, payload)


@router.get("", summary="列出题目，可按审核状态筛选")
async def questions(status: Literal["draft", "approved", "rejected"] | None = Query(default=None),
                    subject: str | None = Query(default=None, max_length=100),
                    topic: str | None = Query(default=None, max_length=2000)):
    return await run_in_threadpool(list_questions, status, subject, topic)


@router.put("/{question_id}", summary="编辑并审核一道题目")
async def review(question_id: int = Path(gt=0), payload: ReviewQuestionRequest = ...):
    return await run_in_threadpool(review_question, question_id, payload)


def question_analytics():
    from collections import defaultdict
    from contextlib import closing
    import json
    from app.learning.store import LearningStore

    questions = QuestionStore().list(status="approved")
    approved_ids = {question["id"] for question in questions}
    metrics = defaultdict(lambda: {"attempts": 0, "correct": 0, "learners": set()})
    with closing(LearningStore()._connect()) as db:
        rows = db.execute("""SELECT q.owner_id, a.results_json
                             FROM learning_attempts a
                             JOIN learning_quizzes q ON q.id=a.quiz_id""").fetchall()
    for owner_id, results_json in rows:
        for result in json.loads(results_json):
            question_id = result.get("question_id")
            if question_id in approved_ids:
                metrics[question_id]["attempts"] += 1
                metrics[question_id]["correct"] += int(bool(result.get("correct")))
                metrics[question_id]["learners"].add(owner_id)
    return [{"question_id": question["id"], "attempts": metrics[question["id"]]["attempts"],
             "correct_count": metrics[question["id"]]["correct"],
             "learner_count": len(metrics[question["id"]]["learners"]),
             "accuracy": round(metrics[question["id"]]["correct"] / metrics[question["id"]]["attempts"] * 100)
             if metrics[question["id"]]["attempts"] else None} for question in questions]


@router.get("/analytics", summary="查看已批准题目的匿名练习统计")
async def analytics():
    return await run_in_threadpool(question_analytics)
