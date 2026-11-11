"""知识库管理：学生读取已发布库，教师工作台可创建、上传和构建。"""
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import DATA_DIR, kb_paths, validate_name, model_settings
from app.llm.client import ModelClient
from app.rag.db.repository import (list_knowledgebases, knowledgebase_details,
                                   record_uploaded_file, save_knowledgebase)
from app.rag.indexing.pipeline import build_index


router = APIRouter(prefix="/api/knowledgebases", tags=["knowledgebases"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    kb_info: str = Field(default="", max_length=1000)


@router.get("")
def knowledgebases():
    return [{"name": row["name"], "kb_info": row["kb_info"],
             "chunk_count": row["chunk_count"], "generation": row["current_generation"]}
            for row in list_knowledgebases(data_dir=DATA_DIR)
            if row["current_generation"]]


@router.get("/manage")
def manage_knowledgebases():
    return list_knowledgebases(data_dir=DATA_DIR)


@router.post("")
def create_knowledgebase(payload: CreateKnowledgeBaseRequest):
    try:
        validate_name(payload.name)
        uploads, indexes = kb_paths(payload.name)
        uploads.mkdir(parents=True, exist_ok=True)
        indexes.mkdir(parents=True, exist_ok=True)
        return save_knowledgebase(payload.name, payload.kb_info, data_dir=DATA_DIR)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="知识库数据库暂时不可用。") from None


@router.post("/{name}/files")
async def upload_knowledge_file(name: str, file: UploadFile = File(...)):
    try:
        validate_name(name)
        if not file.filename or Path(file.filename.replace("\\", "/")).suffix.lower() != ".pdf":
            raise HTTPException(status_code=422, detail="目前只支持上传 PDF 文件。")
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF 文件不能超过 50 MB。")
        if not contents.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="文件内容不是有效的 PDF。")
        filename = Path(file.filename.replace("\\", "/")).name
        uploads, _ = kb_paths(name)
        uploads.mkdir(parents=True, exist_ok=True)
        destination = uploads / filename
        if destination.exists():
            raise HTTPException(status_code=409, detail="该资料库中已存在同名文件，请修改文件名后再上传。")
        temporary = uploads / f".{uuid4().hex}.upload"
        try:
            temporary.write_bytes(contents)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            save_knowledgebase(name, data_dir=DATA_DIR)
            record_uploaded_file(name, filename, len(contents), data_dir=DATA_DIR)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return {"knowledgebase": name, "filename": filename, "size_bytes": len(contents),
                "message": "PDF 已上传，下一步请构建索引。"}
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except OSError:
        raise HTTPException(status_code=503, detail="无法保存 PDF，请检查数据目录空间和权限。") from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="知识库数据库暂时不可用。") from None
    finally:
        await file.close()


def _build_knowledgebase(name):
    try:
        with ModelClient(model_settings("EMBEDDING")) as client:
            return build_index(name, client)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except OSError:
        raise HTTPException(status_code=503, detail="PDF 或索引文件暂时不可用。") from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="索引记录暂时无法写入数据库。") from None


@router.post("/{name}/build")
async def build_knowledgebase(name: str):
    return await run_in_threadpool(_build_knowledgebase, name)


@router.get("/{name}")
def knowledgebase_detail(name: str):
    try:
        return knowledgebase_details(name, data_dir=DATA_DIR)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="知识库详情暂时不可用。") from None
