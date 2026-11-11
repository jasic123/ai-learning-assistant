"""通过 Tavily Search API 搜索互联网并返回网页标题、链接和摘要。"""
import json
import os

import httpx
from dotenv import load_dotenv

from app.config import PROJECT_ROOT
from app.tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "搜索互联网，返回网页标题、链接和摘要；适合获取最新信息或查找网页来源。"
    endpoint = "https://api.tavily.com/search"
    default_max_results = 5
    max_results = 10
    max_output_chars = 4_000

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 300},
            "max_results": {"type": "integer", "minimum": 1, "maximum": self.max_results,
                            "default": self.default_max_results},
        }, "required": ["query"], "additionalProperties": False}

    def execute(self, query, max_results=default_max_results):
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            raise ValueError("搜索词须为 1～300 个字符")
        if type(max_results) is not int or not 1 <= max_results <= self.max_results:
            raise ValueError("搜索结果数量须为 1～10")
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return {"query": query.strip(), "results": [], "error": "api_key_missing",
                    "message": "未配置 TAVILY_API_KEY；请在项目 .env 中填写 Tavily API Key。"}

        options = {"json": {"api_key": api_key, "query": query.strip(),
                            "search_depth": "basic", "topic": "general",
                            "max_results": max_results},
                   "timeout": httpx.Timeout(15, connect=5)}
        if proxy := os.getenv("WEB_SEARCH_PROXY", "").strip():
            options["proxy"] = proxy
        try:
            response = httpx.post(self.endpoint, **options)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            return {"query": query.strip(), "results": [], "error": "timeout",
                    "message": "搜索服务连接超时；请检查网络或 WEB_SEARCH_PROXY。"}
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                return {"query": query.strip(), "results": [], "error": "rate_limited",
                        "message": "搜索服务触发限流或额度不足，请检查 Tavily 账户额度。"}
            if error.response.status_code in {401, 403}:
                return {"query": query.strip(), "results": [], "error": "authentication_failed",
                        "message": "Tavily API Key 无效或当前账户无权调用搜索服务。"}
            return {"query": query.strip(), "results": [], "error": "search_unavailable",
                    "message": "Tavily 搜索服务暂不可用，请稍后重试。"}
        except httpx.RequestError:
            return {"query": query.strip(), "results": [], "error": "search_unavailable",
                    "message": "无法连接 Tavily 搜索服务；请检查网络或 WEB_SEARCH_PROXY。"}
        except (ValueError, TypeError):
            return {"query": query.strip(), "results": [], "error": "search_unavailable",
                    "message": "Tavily 返回了无法识别的响应。"}

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raw_results = []
        items = [{"title": str(item.get("title", ""))[:300],
                  "url": str(item.get("url", ""))[:1200],
                  "summary": str(item.get("content", ""))[:600]}
                 for item in raw_results[:max_results]
                 if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))]
        result = {"query": query.strip(), "results": items}
        while len(json.dumps(result, ensure_ascii=False)) > self.max_output_chars and result["results"]:
            result["results"].pop()
        return result
