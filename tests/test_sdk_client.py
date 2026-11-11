"""通过真实 SDK 与内存 HTTP 传输检查协议，不调用真实模型服务。"""
import json
import unittest

import httpx
from openai import OpenAI

from app.config import ModelSettings
from app.llm.client import ModelClient


class SdkClientTests(unittest.TestCase):
    settings = ModelSettings("https://example.invalid/provider/v1", "test-model", "fake-sdk-key")

    def test_sdk_authentication_paths_payloads_and_connection_cleanup(self):
        requests = []

        def handler(request):
            requests.append(request)
            self.assertEqual(request.headers["authorization"], "Bearer fake-sdk-key")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "test-model")
            if request.url.path.endswith("/embeddings"):
                self.assertEqual(body["input"], ["教材正文"])
                self.assertEqual(body["encoding_format"], "float")
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
            if request.url.path.endswith("/chat/completions"):
                self.assertFalse(body["stream"])
                return httpx.Response(200, json={"choices": [{"message": {"content": " 答案 [S1] "}}]})
            self.assertEqual(body["documents"], ["教材正文"])
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

        transport = httpx.Client(transport=httpx.MockTransport(handler))
        with ModelClient(self.settings, http_client=transport) as client:
            self.assertIsInstance(client._client, OpenAI)
            self.assertEqual(client._client.max_retries, 0)
            self.assertEqual(client._client.timeout.connect, 15)
            self.assertEqual(client._client.timeout.read, 90)
            self.assertEqual(client.embed(["教材正文"]), [[1.0, 0.0]])
            self.assertEqual(client.chat([{"role": "user", "content": "问题"}]), "答案 [S1]")
            self.assertEqual(client.rerank("问题", ["教材正文"]), [0.8])
        self.assertTrue(transport.is_closed)
        self.assertEqual([request.url.path for request in requests],
                         ["/provider/v1/embeddings", "/provider/v1/chat/completions", "/provider/v1/rerank"])

    def test_http_errors_are_redacted_and_not_retried(self):
        for status in (401, 429, 500):
            for operation in ("embed", "chat", "rerank"):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(status, json={"error": {"message": "private-provider-response"}})

                with self.subTest(status=status, operation=operation), ModelClient(
                        self.settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                    arguments = (["正文"],) if operation == "embed" else ("问题", ["正文"]) if operation == "rerank" else ([{"role": "user", "content": "问题"}],)
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as error:
                        getattr(client, operation)(*arguments)
                    self.assertNotIn("private-provider-response", str(error.exception))
                    self.assertNotIn("fake-sdk-key", str(error.exception))
                    self.assertEqual(len(requests), 1)

    def test_provider_error_code_is_visible_without_response_body(self):
        transport = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": {
                "code": "input_too_long", "message": "private-provider-response"}})))
        with ModelClient(self.settings, http_client=transport) as client:
            with self.assertRaisesRegex(RuntimeError, "input_too_long") as error:
                client.embed(["正文"])
        self.assertNotIn("private-provider-response", str(error.exception))

    def test_chat_tool_stream_emits_deltas_and_returns_complete_answer(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            chunks = [
                {"id": "stream-1", "object": "chat.completion.chunk", "created": 1,
                 "model": "test-model", "choices": [{"index": 0,
                 "delta": {"content": "流式"}, "finish_reason": None}]},
                {"id": "stream-1", "object": "chat.completion.chunk", "created": 1,
                 "model": "test-model", "choices": [{"index": 0,
                 "delta": {"content": "答案"}, "finish_reason": "stop"}]},
            ]
            body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        deltas = []
        with ModelClient(self.settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            reply = client.chat_with_tools([], [], on_delta=deltas.append)
        self.assertEqual(deltas, ["流式", "答案"])
        self.assertEqual(reply.content, "流式答案")
        self.assertTrue(requests[0]["stream"])

    def test_plain_chat_can_stream_text_deltas(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            chunks = [
                {"id": "plain-1", "object": "chat.completion.chunk", "created": 1,
                 "model": "test-model", "choices": [{"index": 0,
                 "delta": {"content": '{"title":'}, "finish_reason": None}]},
                {"id": "plain-1", "object": "chat.completion.chunk", "created": 1,
                 "model": "test-model", "choices": [{"index": 0,
                 "delta": {"content": '"题目"}'}, "finish_reason": "stop"}]},
            ]
            body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        deltas = []
        with ModelClient(self.settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            answer = client.chat([{"role": "user", "content": "整理题目"}], on_delta=deltas.append)
        self.assertEqual(answer, '{"title":"题目"}')
        self.assertEqual("".join(deltas), answer)
        self.assertTrue(requests[0]["stream"])

    def test_chat_tool_stream_hides_content_from_tool_call_round(self):
        def handler(request):
            chunks = [{
                "id": "stream-tool", "object": "chat.completion.chunk", "created": 1,
                "model": "test-model", "choices": [{"index": 0,
                    "delta": {
                        "content": "}\n</tool_call>",
                        "tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                            "function": {"name": "web_search", "arguments": json.dumps({"query": "today"})}}],
                    }, "finish_reason": "tool_calls"}],
            }]
            body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        deltas = []
        with ModelClient(self.settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            reply = client.chat_with_tools([], [], on_delta=deltas.append)
        self.assertEqual(deltas, [])
        self.assertEqual(reply.tool_calls[0].name, "web_search")

    def test_timeout_is_redacted_and_client_closes_on_failure(self):
        requests = []

        def handler(request):
            requests.append(request)
            raise httpx.ReadTimeout("private-timeout-detail", request=request)

        transport = httpx.Client(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(RuntimeError, "请求失败或超时") as error:
            with ModelClient(self.settings, http_client=transport) as client:
                client.embed(["正文"])
        self.assertNotIn("private-timeout-detail", str(error.exception))
        self.assertTrue(transport.is_closed)
        self.assertEqual(len(requests), 1)

    def test_invalid_model_payloads_still_fail_after_sdk_migration(self):
        cases = [
            ("embed", {"data": [{"index": 0, "embedding": [True]}]}),
            ("embed", {"data": [{"index": True, "embedding": [1.0]}]}),
            ("embed", {"data": [{"index": 0, "embedding": ["private-invalid-number"]}]}),
            ("embed", {"data": []}),
            ("embed", {"data": None}),
            ("chat", {"choices": []}),
            ("chat", {"choices": [{"message": {"content": None}}]}),
            ("chat", {"choices": [{"message": {"content": " "}}]}),
            ("rerank", ["private-invalid-object"]),
        ]
        for operation, payload in cases:
            with self.subTest(operation=operation, payload=payload), ModelClient(
                    self.settings, http_client=httpx.Client(transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json=payload)))) as client:
                arguments = (["正文"],) if operation == "embed" else ("问题", ["正文"]) if operation == "rerank" else ([{"role": "user", "content": "问题"}],)
                with self.assertRaises(RuntimeError) as error:
                    getattr(client, operation)(*arguments)
                self.assertNotIn("private-invalid", str(error.exception))


if __name__ == "__main__":
    unittest.main()
