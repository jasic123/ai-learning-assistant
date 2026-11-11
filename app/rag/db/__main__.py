# 独立元数据命令入口：创建或查看管理记录，不会自动解析 PDF 或生成向量。
"""元数据管理：python -m app.rag.db --help；不调用 OCR 或模型 API。"""
import argparse
import json

from sqlalchemy.exc import SQLAlchemyError

from app.config import DATA_DIR
from app.rag.db.repository import knowledgebase_details, list_knowledgebases, save_knowledgebase, sync_index
from app.rag.db.session import init_db
from app.config import kb_paths


# 解析 init、list、save、info 和 sync 命令；成功返回 0，已处理的操作失败返回 1。
def main():
    parser = argparse.ArgumentParser(description="SQLite 知识库与文件元数据管理")
    # 子命令必填；--help 仅展示用法，不进入数据库操作分支。
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="创建 data/assistant.db 和表")
    commands.add_parser("list", help="列出已登记知识库")
    save = commands.add_parser("save", help="创建知识库或更新简介")
    save.add_argument("--kb", required=True)
    save.add_argument("--info", required=True, help="知识库用途，后续供 Agent 选择知识库")
    for command in ("info", "sync"):
        sub = commands.add_parser(command, help="查看元数据" if command == "info" else "从 CURRENT 导入或修复索引元数据")
        sub.add_argument("--kb", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            init_db()
            result = dict(database=str(DATA_DIR / "assistant.db"))
        elif args.command == "list":
            result = list_knowledgebases()
        elif args.command == "save":
            kb_paths(args.kb)
            result = save_knowledgebase(args.kb, args.info)
        elif args.command == "sync":
            uploads, root = kb_paths(args.kb)
            result = dict(generation=sync_index(args.kb, root, uploads))
        else:
            result = knowledgebase_details(args.kb)
        # 保留中文输出，统一以 JSON 展示元数据，方便终端检查。
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except SQLAlchemyError:
        # 禁止将 SQL 参数、可能含原文的 JSON 或异常细节打印到终端。
        print("数据库操作失败，请检查 data 目录权限或数据库文件。")
        return 1
    except (ValueError, OSError) as error:
        print(f"操作失败：{error}")
        return 1


if __name__ == "__main__":
    # 把 main 的返回值作为进程退出码，便于脚本判断是否成功。
    raise SystemExit(main())
