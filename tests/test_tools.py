"""新 Agent 工具的离线工作区边界检查。"""
import tempfile
import unittest
from datetime import date as calendar_date
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from app.agent.service import AgentService
from app.llm.responses import ModelReply
from app.tools.files import WorkspaceFileTool
from app.tools.web_fetch import WebFetchTool
from app.tools.web import WebSearchTool
from app.tools.weather import WeatherTool


class WorkspaceFileToolTests(unittest.TestCase):
    def test_read_write_and_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = WorkspaceFileTool(directory)
            result = tool.execute("write", "notes/计划.md", "第一行\n第二行")
            self.assertEqual(tool.execute("read", "notes/计划.md")["content"], "第一行\n第二行")
            tool.execute("write", "notes/long.txt", "x" * 5_000)
            first = tool.execute("read", "notes/long.txt")
            second = tool.execute("read", "notes/long.txt", offset=first["next_offset"])
            self.assertEqual(first["next_offset"], 3_000)
            self.assertEqual(first["content"] + second["content"], "x" * 5_000)
            for path in ("../outside.txt", str(Path(directory) / "outside.txt"), "notes.exe"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    tool.execute("read", path)

    def test_reject_symlink_escape_and_oversized_content(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory) / "workspace"
            root.mkdir()
            try:
                (root / "link").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("当前 Windows 权限不允许创建目录符号链接")
            tool = WorkspaceFileTool(directory)
            with self.assertRaises(ValueError):
                tool.execute("write", "link/escape.txt", "x")
            with self.assertRaises(ValueError):
                tool.execute("write", "large.txt", "x" * (tool.max_bytes + 1))
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_agent_registers_tools_and_forwards_stream_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

            def reply(messages, definitions, *, on_delta=None):
                self.assertEqual({item["function"]["name"] for item in definitions}, {
                    "knowledge_search", "recall_history", "get_current_time", "manage_long_term_memory",
                    "get_weather", "web_search", "web_fetch", "workspace_file",
                })
                if on_delta:
                    on_delta("增量")
                return ModelReply("增量答案")

            client.chat_with_tools.side_effect = reply
            service = AgentService(["教育资料"], data_dir=directory, mode="bm25")
            with patch("app.agent.service.ModelClient", return_value=client), \
                 patch("app.agent.service.model_settings"):
                events = list(service.stream_chat("你好", session_id="stream"))
            self.assertEqual(events[0], {"type": "token", "content": "增量"})
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["result"]["answer"], "增量答案")


class WebFetchToolTests(unittest.TestCase):
    def test_extracts_readable_page_text_and_keeps_source_url(self):
        html = (
            "<html><head><title>官方指南</title><script>bad instruction</script></head>"
            "<body><nav>菜单</nav><h1>安装方法</h1><p>先下载应用。</p><p>再启动。</p></body></html>")
        response = Mock(status_code=200, headers={"content-type": "text/html; charset=utf-8"},
                        encoding="utf-8")
        response.iter_bytes.return_value = [html.encode("utf-8")]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.raise_for_status.return_value = None
        client = Mock()
        client.stream.return_value = response
        manager = Mock()
        manager.__enter__ = Mock(return_value=client)
        manager.__exit__ = Mock(return_value=None)
        with patch("app.tools.web_fetch.httpx.Client", return_value=manager):
            result = WebFetchTool().execute("https://1.1.1.1/guide")
        self.assertEqual(result["url"], "https://1.1.1.1/guide")
        self.assertEqual(result["title"], "官方指南")
        self.assertIn("先下载应用。", result["content"])
        self.assertNotIn("bad instruction", result["content"])
        self.assertNotIn("菜单", result["content"])

    def test_extracts_text_from_pdf_search_result(self):
        from test_rag import make_pdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exam.pdf"
            make_pdf(path)
            response = Mock(status_code=200, headers={"content-type": "application/pdf"}, encoding=None)
            response.iter_bytes.return_value = [path.read_bytes()]
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=None)
            response.raise_for_status.return_value = None
            client = Mock()
            client.stream.return_value = response
            manager = Mock()
            manager.__enter__ = Mock(return_value=client)
            manager.__exit__ = Mock(return_value=None)
            with patch("app.tools.web_fetch.httpx.Client", return_value=manager):
                result = WebFetchTool().execute("https://1.1.1.1/exam.pdf")
        self.assertEqual(result["title"], "PDF 文档")
        self.assertIn("Python lists are mutable sequences", result["content"])

    def test_rejects_private_hosts_and_non_web_schemes(self):
        tool = WebFetchTool()
        for url in ("file:///C:/secret.txt", "http://127.0.0.1/admin", "http://192.168.1.1/"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                tool.execute(url)


class WeatherToolTests(unittest.TestCase):
    def test_relative_future_date_uses_forecast_endpoint(self):
        payload = {"results": [{"location": {"name": "上海"}, "daily": [
            {"date": "2026-11-06", "text_day": "多云", "text_night": "晴", "high": "20", "low": "12"}
        ]}]}
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        with patch("app.tools.weather.datetime") as clock, \
             patch("app.tools.weather.load_dotenv"), \
             patch.dict("os.environ", {"SENIVERSE_API_KEY": "test-key"}), \
             patch("app.tools.weather.httpx.get", return_value=response) as get:
            clock.now.return_value.date.return_value = calendar_date(2026, 11, 4)
            result = WeatherTool().execute("上海", date="day_after_tomorrow")
        self.assertEqual(result["date"], "2026-11-06")
        self.assertEqual(result["weather_day"], "多云")
        self.assertEqual(get.call_args.args[0], "https://api.seniverse.com/v3/weather/daily.json")
        self.assertEqual(get.call_args.kwargs["params"]["days"], 3)

    def test_missing_weather_key_falls_back_to_web_search(self):
        search_result = {"query": "上海 2026-11-06 天气预报",
                         "results": [{"title": "天气预报", "url": "https://example.com/weather",
                                      "summary": "多云，12 至 20 度"}]}
        with patch("app.tools.weather.datetime") as clock, \
             patch("app.tools.weather.load_dotenv"), \
             patch.dict("os.environ", {"SENIVERSE_API_KEY": ""}), \
             patch("app.tools.weather.WebSearchTool.execute", return_value=search_result) as search:
            clock.now.return_value.date.return_value = calendar_date(2026, 11, 4)
            result = WeatherTool().execute("上海", date="day_after_tomorrow")
        self.assertEqual(result["source"], "web_search")
        self.assertEqual(result["results"], search_result["results"])
        search.assert_called_once_with("上海 2026-11-06 天气预报", max_results=5)

    def test_weather_provider_failure_falls_back_to_web_search(self):
        response = Mock()
        response.raise_for_status.side_effect = httpx.ConnectTimeout("provider timeout")
        with patch("app.tools.weather.datetime") as clock, \
             patch("app.tools.weather.load_dotenv"), \
             patch.dict("os.environ", {"SENIVERSE_API_KEY": "test-key"}), \
             patch("app.tools.weather.httpx.get", return_value=response), \
             patch("app.tools.weather.WebSearchTool.execute", return_value={"results": []}) as search:
            clock.now.return_value.date.return_value = calendar_date(2026, 11, 4)
            result = WeatherTool().execute("上海", date="day_after_tomorrow")
        self.assertEqual(result["source"], "web_search")
        search.assert_called_once_with("上海 2026-11-06 天气预报", max_results=5)

    def test_rejects_dates_outside_forecast_horizon_without_request(self):
        with patch("app.tools.weather.datetime") as clock, \
             patch("app.tools.weather.httpx.get") as get:
            clock.now.return_value.date.return_value = calendar_date(2026, 11, 4)
            with self.assertRaisesRegex(ValueError, "今天至后天"):
                WeatherTool().execute("上海", date="2026-11-10")
        get.assert_not_called()


class WebSearchToolTests(unittest.TestCase):
    def test_search_uses_tavily_and_returns_links(self):
        response = Mock()
        response.json.return_value = {"results": [
            {"title": "Result", "url": "https://example.com/page", "content": "Summary"},
            {"title": "Unsafe", "url": "file:///secret", "content": "ignored"},
        ]}
        response.raise_for_status.return_value = None
        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}), \
             patch("app.tools.web.httpx.post", return_value=response) as post:
            result = WebSearchTool().execute(" latest news ", max_results=2)
        self.assertEqual(result["query"], "latest news")
        self.assertEqual(result["results"], [{"title": "Result", "url": "https://example.com/page",
                                              "summary": "Summary"}])
        self.assertEqual(post.call_args.args[0], "https://api.tavily.com/search")
        self.assertEqual(post.call_args.kwargs["json"]["api_key"], "test-key")
        self.assertEqual(post.call_args.kwargs["json"]["query"], "latest news")

    def test_search_uses_configured_proxy(self):
        response = Mock()
        response.json.return_value = {"results": []}
        response.raise_for_status.return_value = None
        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "test-key",
                                       "WEB_SEARCH_PROXY": "http://127.0.0.1:7890"}), \
             patch("app.tools.web.httpx.post", return_value=response) as post:
            result = WebSearchTool().execute("query")
        self.assertEqual(result["results"], [])
        self.assertEqual(post.call_args.kwargs["proxy"], "http://127.0.0.1:7890")

    def test_search_requires_key_and_reports_timeout_rate_limit_and_auth_errors(self):
        tool = WebSearchTool()
        for query, count in [("", 1), ("x", 0), ("x", True)]:
            with self.subTest(query=query, count=count), self.assertRaises(ValueError):
                tool.execute(query, max_results=count)
        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": ""}), \
             patch("app.tools.web.httpx.post") as post:
            result = tool.execute("query")
        self.assertEqual(result["error"], "api_key_missing")
        post.assert_not_called()

        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}), \
             patch("app.tools.web.httpx.post", side_effect=httpx.ConnectTimeout("private-response")):
            result = tool.execute("query")
        self.assertEqual(result["error"], "timeout")
        self.assertIn("WEB_SEARCH_PROXY", result["message"])
        self.assertNotIn("private-response", str(result))

        response = Mock(status_code=429)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "private-response", request=Mock(), response=response)
        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}), \
             patch("app.tools.web.httpx.post", return_value=response):
            result = tool.execute("query")
        self.assertEqual(result["error"], "rate_limited")
        self.assertNotIn("private-response", str(result))

        response.status_code = 401
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "private-response", request=Mock(), response=response)
        with patch("app.tools.web.load_dotenv"), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}), \
             patch("app.tools.web.httpx.post", return_value=response):
            result = tool.execute("query")
        self.assertEqual(result["error"], "authentication_failed")


if __name__ == "__main__":
    unittest.main()
