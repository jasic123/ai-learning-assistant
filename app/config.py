# 配置入口：模块导入只定义路径与配置类型；调用 model_settings 时才读取模型配置。
"""从项目 .env / 系统环境读取配置，密钥不会进入日志。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# 以配置文件位置确定项目根目录，不依赖终端当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# 不可变模型配置；api_key 参与请求认证，但从默认 repr 中排除，减少误打印。
@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    model: str
    # repr=False 仅隐藏默认对象打印；不会加密密钥，也不能阻止主动打印 api_key。
    api_key: str = field(repr=False)


# 分别读取聊天、向量和精排模型配置；缺项时只列出变量名，不输出密钥。
def model_settings(kind: str) -> ModelSettings:
    if kind not in {"LLM", "EMBEDDING", "RERANK"}:
        raise ValueError("未知的模型配置类型")
    # 系统已有环境变量优先；本地 .env 仅补充缺失项。
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    values = {suffix: os.getenv(f"{kind}_{suffix}", "").strip()
              for suffix in ("API_BASE", "MODEL", "API_KEY")}
    missing = [f"{kind}_{key}" for key, value in values.items() if not value]
    if missing:
        raise ValueError("请先在项目 .env 填写：" + ", ".join(missing))
    # 检查基址不含账号密码、查询参数或片段，避免把认证信息混进可保存的地址。
    parsed = urlsplit(values["API_BASE"])
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{kind}_API_BASE 必须是无凭据和查询参数的 http(s) API 基址")
    return ModelSettings(values["API_BASE"].rstrip("/"), values["MODEL"], values["API_KEY"])




# 知识库路径规则由各层共用，不触发建库或目录创建。
def validate_name(name):
    if (not isinstance(name, str) or not name.strip() or name in {".", ".."}
        or any(char in name for char in '/\\:*?"<>|') or name.endswith((".", " "))):
        raise ValueError("知识库名称必须是单个合法目录名，例如：教育资料")
    return name


def kb_paths(name):
    # 知识库名只允许单层目录，防止使用 ../ 等内容访问其他路径。
    validate_name(name)
    roots = (DATA_DIR / "uploads", DATA_DIR / "indexes")
    paths = tuple(root / name for root in roots)
    # 解析符号链接后的真实路径也必须留在 uploads 或 indexes 目录内。
    if any(not path.resolve().is_relative_to(root.resolve()) for root, path in zip(roots, paths)):
        raise ValueError("知识库目录不得指向数据目录之外")
    return paths
