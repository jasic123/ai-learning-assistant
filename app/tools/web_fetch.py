"""抓取公开网页并提取可供 Agent 阅读的正文。"""
from html.parser import HTMLParser
from io import BytesIO
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from app.tools.base import Tool


class _TextExtractor(HTMLParser):
    ignored = {"script", "style", "noscript", "svg", "nav", "footer", "header"}
    breaks = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._ignored, self._in_title = [], [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored:
            self._ignored += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.breaks and not self._ignored:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.ignored and self._ignored:
            self._ignored -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.breaks and not self._ignored:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title.append(data)
        if not self._ignored and not self._in_title:
            self.parts.append(data)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "抓取搜索结果中的公开网页并提取正文；通常先调用 web_search 找到相关页面，再抓取阅读详情。"
    max_output_chars = 6_000
    max_response_bytes = 2_000_000

    @staticmethod
    def _readable_pdf_text(text):
        """修正常见 Symbol 字体映射，避免 PDF 文字层显示为私有区字符。"""
        mapping = {
            "\uf028": "(", "\uf029": ")", "\uf02b": "+", "\uf02d": "-",
            "\uf03d": "=", "\uf0a2": "′", "\uf0f2": "∫", "\uf067": "γ",
        }
        return "".join(mapping.get(char, char) for char in text)

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "url": {"type": "string", "minLength": 8, "maxLength": 2000},
        }, "required": ["url"], "additionalProperties": False}

    @staticmethod
    def _public_url(url):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password):
            raise ValueError("只允许抓取不含凭据的 HTTP/HTTPS 公开网页 URL")
        try:
            addresses = {ipaddress.ip_address(parsed.hostname)}
        except ValueError:
            addresses = {ipaddress.ip_address(item[4][0])
                         for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("出于安全限制，只能抓取解析到公网地址的网页")
        return parsed

    def execute(self, url):
        if not isinstance(url, str) or not 8 <= len(url) <= 2000:
            raise ValueError("网页 URL 长度无效")
        current = url.strip()
        headers = {"User-Agent": "PersonalAssistant/1.0", "Accept": "text/html,text/plain;q=0.9"}
        try:
            with httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False,
                              headers=headers) as client:
                for _ in range(6):
                    self._public_url(current)
                    with client.stream("GET", current) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise ValueError("网页重定向缺少目标地址")
                            current = urljoin(current, location)
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        content = bytearray()
                        for chunk in response.iter_bytes():
                            content.extend(chunk)
                            if len(content) > self.max_response_bytes:
                                raise ValueError("网页超过 2 MB 抓取上限")
                        encoding = response.encoding or "utf-8"
                    break
                else:
                    raise ValueError("网页重定向次数过多")
        except httpx.TimeoutException:
            return {"url": current, "error": "timeout", "message": "读取网页超时，请稍后重试或直接打开链接。"}
        except httpx.HTTPStatusError as error:
            return {"url": current, "error": "http_error",
                    "message": f"网页服务器返回 HTTP {error.response.status_code}，未能读取正文。"}
        except httpx.RequestError:
            return {"url": current, "error": "network_error", "message": "无法连接该网页，请检查网络或直接打开链接。"}

        raw = bytes(content)
        if "application/pdf" in content_type or raw.startswith(b"%PDF-"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(BytesIO(raw))
                if reader.is_encrypted:
                    return {"url": current, "error": "encrypted_pdf", "message": "PDF 有密码保护，无法读取正文。"}
                body = self._readable_pdf_text(
                    "\n".join(page.extract_text() or "" for page in reader.pages)
                ).strip()
            except Exception:
                return {"url": current, "error": "invalid_pdf", "message": "PDF 下载成功，但解析失败。"}
            if not body:
                return {"url": current, "error": "pdf_no_text",
                        "message": "PDF 是扫描图片或没有文字层，当前抓取器无法提取文字。"}
            title = "PDF 文档"
        elif any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
            text = raw.decode(encoding, errors="replace")
            title, body = "", text
            if "html" in content_type or "<html" in text[:500].lower():
                parser = _TextExtractor()
                parser.feed(text)
                title = re.sub(r"\s+", " ", " ".join(parser.title)).strip()[:300]
                body = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
        else:
            return {"url": current, "error": "unsupported_content_type",
                    "message": "链接返回的不是可读取的网页或 PDF。"}
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body:
            return {"url": current, "error": "empty_content", "message": "网页没有提取到可读正文。"}
        return {"url": current, "title": title, "content": body[:self.max_output_chars],
                "truncated": len(body) > self.max_output_chars}
