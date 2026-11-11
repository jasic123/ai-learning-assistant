"""私人助理 API 与 Gradio 聊天页面入口。"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr

from app.auth import AuthStore, COOKIE_NAME, auth_required
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.knowledgebase import router as knowledgebase_router
from app.api.learning import router as learning_router
from app.api.mindmaps import router as mindmaps_router
from app.api.memory import router as memory_router
from app.api.questions import router as questions_router
from app.api.sessions import router as sessions_router
from frontend.app import CSS, build_ui

app = FastAPI(title="AI 学习助手", version="1.0.0")
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(knowledgebase_router)
app.include_router(learning_router)
app.include_router(mindmaps_router)
app.include_router(memory_router)
app.include_router(questions_router)
app.include_router(sessions_router)


@app.middleware("http")
async def require_login(request, call_next):
    path = request.url.path
    if path == "/health" or path.startswith("/api/auth/") or not auth_required():
        request.state.user = {"id": "local", "username": "本地模式"}
        return await call_next(request)
    user = AuthStore().user_for_token(request.cookies.get(COOKIE_NAME))
    if not user:
        return JSONResponse({"detail": "请先登录"}, status_code=401)
    request.state.user = user
    return await call_next(request)


# 健康检查只确认应用能够响应，不代表模型服务、OCR 或数据库已经通过验收。
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# 同一进程提供 UI 与 API，Gradio 回调直接复用 chat 业务函数。
app = gr.mount_gradio_app(app, build_ui(), path="/", css=CSS,
                          theme=gr.themes.Default(primary_hue="emerald", secondary_hue="orange",
                                                  neutral_hue="slate"),
                          server_name="127.0.0.1")
