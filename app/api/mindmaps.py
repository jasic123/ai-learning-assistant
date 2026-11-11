"""Generate and edit structured text mind maps."""
import json
import sqlite3

from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import model_settings
from app.llm.client import ModelClient
from app.mindmaps.store import MindMapStore


router = APIRouter(prefix="/api/mindmaps", tags=["mindmaps"])


class GenerateMindMapRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    context: str = Field(default="", max_length=12000)


class SaveMindMapRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    tree: dict
    id: int | None = Field(default=None, gt=0)


def _parse_tree_response(text):
    """Accept strict JSON and harmless Markdown code fences around JSON."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型没有返回导图内容")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = "\n".join(line for line in candidate.splitlines()
                               if not line.strip().startswith("```"))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(candidate[start:end + 1])


def generate_mindmap(payload, owner_id="local"):
    messages = [
        {"role": "system", "content": (
            "你是学习结构化助手。把主题整理成可编辑的思维导图。只返回 JSON，"
            '格式为 {"title":"主题","children":[{"title":"分支","children":[{"title":"要点"}]}]}。'
            "最多 8 个一级分支，每个分支最多 8 个子节点；不要输出 Markdown。"
        )},
        {"role": "user", "content": f"主题：{payload.topic}\n资料：{payload.context}"},
    ]
    try:
        with ModelClient(model_settings("LLM")) as client:
            tree = _parse_tree_response(client.chat(messages))
        MindMapStore._validate_tree(tree)
        return MindMapStore(owner_id=owner_id).save(payload.topic, tree)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"模型返回的导图格式无效：{error}") from None
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="导图数据库暂时不可用。") from None


def save_mindmap(payload, owner_id="local"):
    try:
        return MindMapStore(owner_id=owner_id).save(payload.title, payload.tree, payload.id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="导图数据库暂时不可用。") from None


@router.post("/generate", summary="生成结构化思维导图")
async def generate(payload: GenerateMindMapRequest, request: Request):
    return await run_in_threadpool(generate_mindmap, payload, request.state.user["id"])


@router.get("", summary="列出已保存导图")
async def maps(request: Request):
    return await run_in_threadpool(MindMapStore(owner_id=request.state.user["id"]).list)


@router.put("", summary="保存或编辑导图")
async def save(payload: SaveMindMapRequest, request: Request):
    return await run_in_threadpool(save_mindmap, payload, request.state.user["id"])
