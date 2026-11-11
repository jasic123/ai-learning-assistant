"""Account registration and opaque browser-session endpoints."""
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.auth import AuthStore, COOKIE_NAME, SESSION_DAYS, auth_required, cookie_secure


router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


def _set_session(response, token):
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_DAYS * 24 * 60 * 60,
                        httponly=True, secure=cookie_secure(), samesite="lax", path="/")


@router.get("/me")
def me(request: Request):
    if not auth_required():
        return {"authenticated": True, "auth_required": False,
                "user": {"id": "local", "username": "本地模式"}}
    user = AuthStore().user_for_token(request.cookies.get(COOKIE_NAME))
    return {"authenticated": bool(user), "auth_required": True, "user": user}


@router.post("/register")
def register(payload: Credentials, response: Response):
    try:
        user = AuthStore().register(payload.username, payload.password)
        token, user = AuthStore().login(payload.username, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    _set_session(response, token)
    return {"user": user}


@router.post("/login")
def login(payload: Credentials, response: Response):
    try:
        token, user = AuthStore().login(payload.username, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error)) from None
    _set_session(response, token)
    return {"user": user}


@router.post("/logout")
def logout(request: Request, response: Response):
    AuthStore().logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True,
                           secure=cookie_secure(), samesite="lax")
    return {"logged_out": True}
