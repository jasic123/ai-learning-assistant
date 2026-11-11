"""本地 Agent 长期记忆的列表、保存和删除接口。"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.memory import MemoryStore
from app.agent.session import SessionManager


router = APIRouter(prefix="/api/memories", tags=["memory"])


class MemoryRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=500)


def store_for(owner_id):
    return MemoryStore(SessionManager(owner_id=owner_id))


@router.get("")
def list_memories(request: Request):
    return store_for(request.state.user["id"]).list()


@router.put("")
def save_memory(payload: MemoryRequest, request: Request):
    try:
        store_for(request.state.user["id"]).remember(payload.key, payload.content)
        return {"key": payload.key, "content": payload.content.strip()}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@router.delete("/{key}")
def forget_memory(key: str, request: Request):
    return {"deleted": store_for(request.state.user["id"]).forget(key)}
