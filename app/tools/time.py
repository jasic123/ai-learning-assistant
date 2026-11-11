"""读取应用默认或指定 IANA 时区的当前时间。"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.base import Tool


DEFAULT_TIMEZONE = "Asia/Shanghai"
WEEKDAYS_ZH = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def current_time(timezone_name=DEFAULT_TIMEZONE):
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("时区必须是有效的 IANA 名称，例如 Asia/Shanghai")
    try:
        zone = ZoneInfo(timezone_name.strip())
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("未知时区，请使用 IANA 时区名称，例如 Asia/Shanghai") from None
    now = datetime.now(zone)
    return {"timezone": zone.key, "datetime": now.isoformat(timespec="seconds"),
            "weekday": WEEKDAYS_ZH[now.weekday()],
            "utc_datetime": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class CurrentTimeTool(Tool):
    name = "get_current_time"
    description = "获取当前日期和时间；默认使用中国标准时间，也可指定 IANA 时区。"

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "IANA 时区，如 Asia/Shanghai、America/New_York。"},
        }, "additionalProperties": False}

    def execute(self, timezone=DEFAULT_TIMEZONE):
        return current_time(timezone)
