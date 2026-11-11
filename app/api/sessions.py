"""本地短期会话的列表与恢复接口。"""
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request

from app.agent.session import SessionManager


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def sessions_for(owner_id):
    return SessionManager(owner_id=owner_id)


@router.get("")
def list_sessions(request: Request, limit: int = Query(default=50, ge=1, le=100)):
    return sessions_for(request.state.user["id"]).list_sessions(limit)


@router.get("/{session_id}")
def load_session(session_id: str, request: Request):
    try:
        return {"session_id": session_id,
                "turns": sessions_for(request.state.user["id"]).load_all_turns(session_id)}
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@router.delete("/{session_id}", summary="删除历史会话及其压缩检查点")
def delete_session(session_id: str, request: Request):
    try:
        return sessions_for(request.state.user["id"]).delete_session(session_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except sqlite3.OperationalError as error:
        if "locked" in str(error).lower():
            raise HTTPException(status_code=503, detail="数据库正被其他程序占用，请关闭数据库浏览器后重试。") from None
        raise HTTPException(status_code=503, detail="数据库暂时无法完成删除操作。") from None
