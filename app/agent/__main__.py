"""终端入口：python -m app.agent --kb 教育资料。"""
import argparse
import json
import sys

from app.agent.service import AgentService


def main():
    parser = argparse.ArgumentParser(description="NanoClaw 风格私人助理 Agent")
    parser.add_argument("--kb", action="append", required=True, help="本次允许查询的知识库，可重复指定")
    parser.add_argument("--session", default="default", help="会话标识，相同标识恢复最近对话")
    parser.add_argument("--question", help="单次提问；省略时进入交互")
    parser.add_argument("--mode", choices=("hybrid", "vector", "bm25"), default="hybrid")
    parser.add_argument("--rerank", action="store_true", help="知识检索启用远程精排")
    parser.add_argument("--owner", default="local", help="长期记忆所有者标识；同一标识跨会话共享")
    parser.add_argument("--context-tokens", type=int, default=16000, help="按所选模型设置上下文窗口")
    parser.add_argument("--output-reserve", type=int, default=2000, help="输出与协议开销预留 token")
    args = parser.parse_args()
    try:
        service = AgentService(args.kb, mode=args.mode, rerank=args.rerank, owner_id=args.owner,
                               context_tokens=args.context_tokens, output_reserve=args.output_reserve)
        if args.question is not None:
            result = service.chat(args.question, session_id=args.session)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        print("输入问题开始；/exit 退出；/memory 查看记忆；/remember 名称 内容 保存；/forget 名称 删除。")
        while True:
            try:
                question = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            if question == "/exit":
                return 0
            if not question:
                continue
            try:
                if question == "/memory":
                    print(json.dumps(service.memory.list(), ensure_ascii=False, indent=2))
                    continue
                if question.startswith("/remember "):
                    parts = question.split(maxsplit=2)
                    if len(parts) != 3:
                        raise ValueError("用法：/remember 名称 内容")
                    service.memory.remember(parts[1], parts[2])
                    print("已保存长期记忆。")
                    continue
                if question.startswith("/forget "):
                    print("已删除。" if service.memory.forget(question[8:].strip()) else "未找到该记忆。")
                    continue
                print(service.chat(question, session_id=args.session).answer)
            except (ValueError, RuntimeError, OSError) as error:
                print(f"本轮失败：{error}", file=sys.stderr)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
