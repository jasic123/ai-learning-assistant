# 配置回归：使用虚构环境变量验证配置加载与对象打印不会输出测试密钥。
"""配置加载不得向终端输出密钥；只使用虚构配置。"""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from app.config import model_settings


# 配置输出回归测试集合，始终使用虚构密钥。
class ConfigSecretTests(unittest.TestCase):
    # 隔离环境变量并捕获标准输出、错误输出，检查 repr 与加载过程不会泄露测试密钥。
    def test_model_settings_are_silent(self):
        values = {}
        for kind in ("LLM", "EMBEDDING", "RERANK"):
            values.update({f"{kind}_API_BASE": "https://example.invalid/v1",
                           f"{kind}_MODEL": "test-model",
                           f"{kind}_API_KEY": "fake-secret-never-log"})
        output = io.StringIO()
        # 同时隔离系统环境与 .env 加载，避免测试读取真实配置。
        with patch.dict(os.environ, values, clear=True), patch("app.config.load_dotenv"):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                for kind in ("LLM", "EMBEDDING", "RERANK"):
                    settings = model_settings(kind)
                    self.assertTrue(settings.api_key == "fake-secret-never-log")
                    self.assertNotIn("fake-secret-never-log", repr(settings))
        self.assertTrue(output.getvalue() == "", "配置加载不得输出内容")


if __name__ == "__main__":
    unittest.main()
