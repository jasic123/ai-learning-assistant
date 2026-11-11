# 远程接口边界：官方 SDK 管理认证与请求，项目层保留数据校验和错误脱敏。
"""官方 OpenAI SDK 适配层：聊天、Embedding 和服务商扩展 Rerank。"""

from contextlib import contextmanager
import math
import re
import httpx
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from app.config import ModelSettings
from app.llm.responses import ContextWindowExceeded, ModelReply, ToolCall


class ModelClient:
    """业务使用 embed/chat/rerank；http_client 可注入离线测试传输。"""

    def __init__(self, settings: ModelSettings, *, http_client=None):
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url + "/",
            # 保留连接 15 秒、其他阶段 90 秒超时；迁移不引入额外自动重试。
            timeout=httpx.Timeout(90, connect=15),
            max_retries=0,
            http_client=http_client,
        )

    # 进入 with 时把当前客户端实例交给调用方。
    def __enter__(self):
        return self

    # 无论请求成功或失败都关闭 HTTP 客户端；不吞掉 with 内抛出的异常。
    def __exit__(self, *args):
        self._client.close()

    @contextmanager
    def _api_errors(self, endpoint):
        """SDK 原始异常可能含响应正文；只抛出可以安全显示的业务提示。"""
        try:
            yield
        except APIStatusError as error:
            body = error.body if isinstance(error.body, dict) else {}
            detail = body.get("error", body)
            if (endpoint == "chat/completions" and isinstance(detail, dict)
                and detail.get("code") in {"context_length_exceeded", "context_window_exceeded", "prompt_too_long"}):
                raise ContextWindowExceeded("模型上下文超限，需要压缩后重试。") from None
            code = detail.get("code") if isinstance(detail, dict) else None
            safe_code = f"（服务商错误码：{code}）" if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else ""
            raise RuntimeError(
                f"{endpoint} 返回 HTTP {error.status_code}{safe_code}；检查模型权限、配置、额度或限流。"
            ) from None
        except APIConnectionError:
            raise RuntimeError(f"{endpoint} 请求失败或超时，请检查网络和 API 基址。") from None
        except APIError:
            raise RuntimeError(f"{endpoint} SDK 响应处理失败，请检查接口兼容性。") from None

    # 批量文本转换为向量列表；按接口 index 恢复输入顺序，并校验数量及基础数值格式。
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding 输入不得为空")
        try:
            with self._api_errors("embeddings"):
                response = self._client.embeddings.with_raw_response.create(
                    model=self.settings.model, input=texts, encoding_format="float",
                )
                # SDK 类型转换可能把 true 转成 1.0；读原始 JSON，保留对非法向量的严格检查。
                payload = response.http_response.json()
            # 服务可能乱序返回向量，必须先恢复与输入文本一一对应的顺序。
            data = sorted(payload["data"], key=lambda item: item["index"])
            # 编号必须完整且不重复，避免向量和正文错配。
            if (any(type(item["index"]) is not int for item in data)
                or [item["index"] for item in data] != list(range(len(texts)))):
                raise ValueError
            vectors = [item["embedding"] for item in data]
            if any(not isinstance(v, list) or not v or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in v) for v in vectors):
                raise ValueError
            return vectors
        except (AttributeError, KeyError, TypeError, ValueError):
            raise RuntimeError("Embedding 响应数量、顺序或向量格式异常。") from None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """通过 SDK 自定义请求访问服务商 /rerank，按 index 恢复候选分数。"""
        if not query.strip() or not documents or any(not text.strip() for text in documents):
            raise ValueError("精排问题和候选正文不得为空")
        try:
            # /rerank 是服务商扩展，不假设 SDK 存在 rerank.create 标准资源。
            with self._api_errors("rerank"):
                response = self._client.post(
                    "/rerank", cast_to=httpx.Response, body={
                        "model": self.settings.model, "query": query, "documents": documents,
                        "top_n": len(documents), "return_documents": False,
                    },
                )
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            results = payload["results"]
            if not isinstance(results, list) or len(results) != len(documents):
                raise ValueError
            # 服务常按相关性顺序返回结果，不能把该顺序误当作输入片段顺序。
            if any(type(item["index"]) is not int for item in results):
                raise ValueError
            ordered = sorted(results, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(len(documents))):
                raise ValueError
            scores = [item["relevance_score"] for item in ordered]
            if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                   for score in scores):
                raise ValueError
            return [float(score) for score in scores]
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Rerank 响应数量、编号或分数格式异常。") from None

    def chat_with_tools(self, messages: list[dict], tools: list[dict], *, on_delta=None) -> ModelReply:
        """返回文本或工具调用；提供回调时按 SDK 流式增量输出文本。"""
        try:
            with self._api_errors("chat/completions"):
                response = self._client.chat.completions.create(
                    model=self.settings.model, messages=messages, tools=tools,
                    tool_choice="auto", stream=on_delta is not None,
                )
                if on_delta is not None:
                    content, content_deltas, calls = "", [], {}
                    finish_reason, reasoning, input_tokens = None, "", None
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        delta = choice.delta
                        if isinstance(delta.content, str) and delta.content:
                            content += delta.content
                            content_deltas.append(delta.content)
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if isinstance(reasoning_delta, str):
                            reasoning += reasoning_delta
                        for part in delta.tool_calls or []:
                            item = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                            if part.id:
                                item["id"] += part.id
                            if part.function:
                                item["name"] += part.function.name or ""
                                item["arguments"] += part.function.arguments or ""
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        usage = getattr(chunk, "usage", None)
                        tokens = getattr(usage, "prompt_tokens", None)
                        if type(tokens) is int and tokens > 0:
                            input_tokens = tokens
                    if finish_reason not in {"stop", "tool_calls"}:
                        raise ValueError
                    tool_calls = tuple(ToolCall(item["id"], item["name"], item["arguments"])
                                       for _, item in sorted(calls.items()))
                    if finish_reason == "tool_calls" and not tool_calls:
                        raise ValueError
                    if not tool_calls and not content.strip():
                        raise ValueError
                    # Tool-call rounds may contain provider control tokens in content; only stream final answers.
                    if not tool_calls:
                        for part in content_deltas:
                            on_delta(part)
                    return ModelReply(content or None, tool_calls, reasoning or None, input_tokens)
            choice = response.choices[0]
            if choice.finish_reason not in {"stop", "tool_calls"}:
                raise ValueError
            message = choice.message
            if message.content is not None and not isinstance(message.content, str):
                raise ValueError
            calls = []
            for call in message.tool_calls or []:
                if (call.type != "function" or not isinstance(call.id, str) or not call.id
                    or not isinstance(call.function.name, str) or not call.function.name
                    or not isinstance(call.function.arguments, str)):
                    raise ValueError
                calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
            if choice.finish_reason == "tool_calls" and not calls:
                raise ValueError
            if not calls and (not message.content or not message.content.strip()):
                raise ValueError
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "prompt_tokens", None)
            return ModelReply(message.content, tuple(calls), reasoning,
                              tokens if type(tokens) is int and tokens > 0 else None)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Agent 模型响应异常或未完整结束，请检查工具调用兼容性和输出长度。") from None

    # 提交非流式聊天消息并提取首个文本答案；RAG 独立问答继续使用此接口。
    def chat(self, messages: list[dict[str, str]], *, on_delta=None) -> str:
        try:
            with self._api_errors("chat/completions"):
                response = self._client.chat.completions.create(
                    model=self.settings.model, messages=messages, stream=on_delta is not None,
                )
                if on_delta is not None:
                    parts = []
                    finish_reason = None
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        content = choice.delta.content
                        if isinstance(content, str) and content:
                            parts.append(content)
                            on_delta(content)
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                    answer = "".join(parts)
                    if finish_reason != "stop" or not answer.strip():
                        raise ValueError
                    return answer.strip()
            # 按当前非流式文本协议读取首个候选答案，不接受缺失或空白文本。
            answer = response.choices[0].message.content
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError
            return answer.strip()
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("聊天接口未返回非空文本答案。") from None
