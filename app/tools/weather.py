"""心知天气查询工具。"""
from datetime import date as calendar_date, datetime, timedelta
import os
import re
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from app.config import PROJECT_ROOT
from app.tools.base import Tool
from app.tools.web import WebSearchTool


class WeatherTool(Tool):
    name = "get_weather"
    description = ("查询指定地点的当前天气或日期天气预报。date 可用 current、today、tomorrow、"
                   "day_after_tomorrow，或 YYYY-MM-DD；地点可以是城市名或城市名,国家/地区。")

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "location": {"type": "string", "minLength": 1, "maxLength": 80},
            "date": {"type": "string", "description": "current、today、tomorrow、day_after_tomorrow 或 YYYY-MM-DD；省略时查询当前天气。"},
        }, "required": ["location"], "additionalProperties": False}

    def execute(self, location, date="current"):
        if not isinstance(location, str) or not location.strip() or len(location) > 80:
            raise ValueError("地点须为 1～80 个字符")
        if not isinstance(date, str):
            raise ValueError("日期须为 current、today、tomorrow、day_after_tomorrow 或 YYYY-MM-DD")
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        offsets = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}
        if date == "current":
            target_date = None
        elif date in offsets:
            target_date = today + timedelta(days=offsets[date])
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            try:
                target_date = calendar_date.fromisoformat(date)
            except ValueError:
                raise ValueError("日期无效，请使用 YYYY-MM-DD") from None
        else:
            raise ValueError("日期须为 current、today、tomorrow、day_after_tomorrow 或 YYYY-MM-DD")
        if target_date is not None and not 0 <= (target_date - today).days <= 2:
            raise ValueError("天气预报仅支持今天至后天")
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        key = os.getenv("SENIVERSE_API_KEY", "").strip()
        if key:
            try:
                if target_date is None:
                    endpoint = "https://api.seniverse.com/v3/weather/now.json"
                    params = {"key": key, "location": location.strip(), "language": "zh-Hans", "unit": "c"}
                else:
                    endpoint = "https://api.seniverse.com/v3/weather/daily.json"
                    params = {"key": key, "location": location.strip(), "language": "zh-Hans", "unit": "c",
                              "start": 0, "days": 3}
                response = httpx.get(endpoint, params=params, timeout=httpx.Timeout(12, connect=5))
                response.raise_for_status()
                result = response.json()["results"][0]
                place = result["location"]
                if target_date is None:
                    now = result["now"]
                    return {"location": place["name"], "date": today.isoformat(), "weather": now["text"],
                            "temperature_c": now["temperature"], "updated_at": result.get("last_update"),
                            "source": "心知天气"}
                forecast = next((item for item in result["daily"] if item.get("date") == target_date.isoformat()), None)
                if forecast is not None:
                    return {"location": place["name"], "date": forecast["date"],
                            "weather_day": forecast.get("text_day"), "weather_night": forecast.get("text_night"),
                            "high_c": forecast.get("high"), "low_c": forecast.get("low"),
                            "source": "心知天气"}
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                pass

        date_text = target_date.isoformat() if target_date else today.isoformat()
        query = f"{location.strip()} {date_text} 天气预报"
        search = WebSearchTool().execute(query, max_results=5)
        return {"location": location.strip(), "date": date_text, "source": "web_search",
                "results": search.get("results", []),
                **({"search_error": search["error"], "message": search["message"]}
                   if search.get("error") else {})}
