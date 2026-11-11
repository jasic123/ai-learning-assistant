"""Agent 聊天 HTTP 接口；同步模型链路放在线程池运行。"""
from typing import Literal

import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.responses import StreamingResponse

from app.agent.service import AgentService


router = APIRouter(prefix="/api", tags=["agent"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    knowledgebases: list[str] = Field(min_length=1, max_length=8)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = False
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    keyword_weight: float = Field(default=0.5, ge=0, le=1)
    study_mode: Literal["general", "computer", "english", "math", "encouragement"] = "general"
    owner_id: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,64}$")


def run_chat(payload: ChatRequest) -> dict:
    """HTTP 与 Gradio 共用同一业务入口，不从应用内部回环请求 HTTP。"""
    try:
        service = AgentService(payload.knowledgebases, mode=payload.mode, rerank=payload.rerank,
                               owner_id=payload.owner_id, study_mode=payload.study_mode,
                               vector_weight=payload.vector_weight, keyword_weight=payload.keyword_weight)
        return service.chat(payload.message, session_id=payload.session_id).to_dict()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except OSError:
        raise HTTPException(status_code=503, detail="本地资料或数据库暂时不可用。") from None


def iter_chat(payload: ChatRequest):
    """HTTP SSE 和 Gradio 共用的同步 Agent token 事件。"""
    try:
        service = AgentService(payload.knowledgebases, mode=payload.mode, rerank=payload.rerank,
                               owner_id=payload.owner_id, study_mode=payload.study_mode,
                               vector_weight=payload.vector_weight, keyword_weight=payload.keyword_weight)
        for event in service.stream_chat(payload.message, session_id=payload.session_id):
            if event["type"] != "error":
                yield event
            else:
                error = event["error"]
                if isinstance(error, ValueError):
                    detail = str(error)
                elif isinstance(error, RuntimeError):
                    detail = str(error)
                elif isinstance(error, OSError):
                    detail = "本地资料或数据库暂时不可用。"
                else:
                    detail = "请求失败，请检查 API 配置和服务状态。"
                yield {"type": "error", "detail": detail}
                return
    except ValueError as error:
        yield {"type": "error", "detail": str(error)}
    except RuntimeError as error:
        yield {"type": "error", "detail": str(error)}
    except OSError:
        yield {"type": "error", "detail": "本地资料或数据库暂时不可用。"}


@router.post("/chat", summary="调用私人助理 Agent")
async def chat(payload: ChatRequest, request: Request):
    payload = payload.model_copy(update={"owner_id": request.state.user["id"]})
    return await run_in_threadpool(run_chat, payload)


@router.post("/chat/stream", summary="流式调用私人助理 Agent")
async def chat_stream(payload: ChatRequest, request: Request):
    payload = payload.model_copy(update={"owner_id": request.state.user["id"]})
    def encode():
        for event in iter_chat(payload):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(iterate_in_threadpool(encode()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
