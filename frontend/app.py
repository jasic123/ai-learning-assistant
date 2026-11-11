"""Gradio 前端；聊天业务复用 FastAPI 路由使用的 Agent 服务函数。"""
from uuid import uuid4
import json
from html import escape
import sqlite3
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

import gradio as gr
from fastapi import HTTPException

from app.api.chat import ChatRequest, iter_chat, run_chat
from app.api.knowledgebase import knowledgebases
from app.api.learning import (ReviewRequest, StartQuizRequest, StartReviewRequest, SubmitQuizRequest,
                              ChatFeedbackRequest, chat_feedback, learning_history, list_mistakes, review_mistake,
                              ImportMistakeRequest, import_mistake, ocr_image,
                              start_quiz, start_review, submit_quiz, due_reviews)
from app.learning.recognition import structure_question_text as structure_question_fields
from app.api.questions import (GenerateQuestionsRequest, ReviewQuestionRequest,
                                generate_questions, review_question)
from app.api.mindmaps import GenerateMindMapRequest, SaveMindMapRequest, generate_mindmap, save_mindmap
from app.agent.session import SessionManager
from app import config
from app.questions.store import QuestionStore
from app.config import model_settings
from app.llm.client import ModelClient


CSS = """
body { background: #f4f7f5; }
.gradio-container { max-width: 1160px !important; }
.app-header { border-bottom: 1px solid #dce5e0; padding: 14px 0 18px; }
.app-title { color: #153b33; }
.side-controls { border-right: 1px solid #dce5e0; padding-right: 22px; }
.chat-panel { padding-left: 18px; }
footer { display: none !important; }
.mindmap-canvas { overflow: auto; min-height: 260px; padding: 28px; background: #fbfcfb; border: 1px solid #dce5e0; border-radius: 6px; }
.mindmap-tree, .mindmap-tree ul { display: flex; list-style: none; margin: 0; padding: 0; }
.mindmap-tree { min-height: 200px; align-items: center; }
.mindmap-tree ul { position: relative; flex-direction: column; justify-content: center; padding-left: 42px; }
.mindmap-tree ul:before { content: ''; position: absolute; left: 20px; top: 50%; width: 22px; border-top: 2px solid #9bb8ab; }
.mindmap-tree li { position: relative; display: flex; align-items: center; padding: 8px 0 8px 24px; }
.mindmap-tree li:before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; border-left: 2px solid #9bb8ab; }
.mindmap-tree li:after { content: ''; position: absolute; left: 0; top: 50%; width: 24px; border-top: 2px solid #9bb8ab; }
.mindmap-tree li:first-child:before { top: 50%; }
.mindmap-tree li:last-child:before { bottom: 50%; }
.mindmap-tree > li { padding-left: 0; }
.mindmap-tree > li:before, .mindmap-tree > li:after { display: none; }
.mindmap-node { display: block; min-width: 110px; max-width: 260px; padding: 10px 14px; background: white; border: 1px solid #b9cec4; border-radius: 6px; color: #24483b; text-align: center; overflow-wrap: anywhere; cursor: grab; box-shadow: 0 1px 2px #173b3310; }
.mindmap-node::-webkit-details-marker { display: none; }
.mindmap-node:before { content: '⌄ '; color: #668c79; }
.mindmap-node:not(:has(+ ul)):before { content: ''; }
.mindmap-node.root { background: #e4f1eb; border-color: #5c927a; font-weight: 700; }
.mindmap-empty { min-height: 240px; display: grid; place-items: center; color: #6b7c74; }
"""


def chat(message, history, session_id, selected, mode, rerank, study_mode="general"):
    """处理单条用户消息；异常只显示安全业务提示，不中断 UI 事件队列。"""
    if not message or not message.strip():
        return history, ""
    history = list(history or [])
    if not selected:
        return history + [{"role": "assistant", "content": "请先选择一个已入库的知识库。"}], ""
    try:
        result = run_chat(ChatRequest(message=message, session_id=session_id,
                                      knowledgebases=selected, mode=mode, rerank=rerank,
                                      owner_id="local", study_mode=study_mode))
        answer = result["answer"]
        if result["sources"]:
            references = "\n\n**资料来源**\n" + "\n".join(
                f"- [{source['citation']}] {source['source']} · 第 {source['page']} 页"
                for source in result["sources"]
            )
            answer += references
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": answer}], ""
    except Exception as error:
        detail = (error.detail if isinstance(error, HTTPException)
                  else str(error) if isinstance(error, (ValueError, RuntimeError))
                  else "请求失败，请检查 API 配置和服务状态。")
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": f"请求失败：{detail}"}], ""


def chat_stream(message, history, session_id, selected, mode, rerank, study_mode="general"):
    """逐段更新 Gradio 对话；Agent 的工具调用由同一服务流程处理。"""
    if not message or not message.strip():
        yield history or [], ""
        return
    messages = list(history or [])
    if not selected:
        yield messages + [{"role": "user", "content": message},
                          {"role": "assistant", "content": "请求失败：请先选择一个已入库的知识库。"}], ""
        return
    messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": ""}])
    yield messages, ""
    answer = ""
    try:
        payload = ChatRequest(message=message, session_id=session_id,
                              knowledgebases=selected, mode=mode, rerank=rerank,
                              owner_id="local", study_mode=study_mode)
        for event in iter_chat(payload):
            if event["type"] == "token":
                answer += event["content"]
                messages[-1]["content"] = answer
                yield messages, ""
            elif event["type"] == "done":
                result = event["result"]
                if result["sources"]:
                    answer += "\n\n**资料来源**\n" + "\n".join(
                        f"- [{source['citation']}] {source['source']} · 第 {source['page']} 页"
                        for source in result["sources"]
                    )
                messages[-1]["content"] = answer
                yield messages, ""
                return
            elif event["type"] == "error":
                messages[-1]["content"] = f"请求失败：{event['detail']}"
                yield messages, ""
                return
    except Exception:
        messages[-1]["content"] = "请求失败，请检查 API 配置和服务状态。"
        yield messages, ""


def refresh_knowledgebases():
    options = knowledgebases()
    choices = [item["name"] for item in options]
    return gr.Dropdown(choices=choices, value=choices[:1])


def refresh_sessions():
    choices = []
    for item in SessionManager().list_sessions():
        label = f"{item['preview'] or '新会话'} · {item['updated_at'][:16]} · {item['turn_count']} 轮"
        choices.append((label, item["session_id"]))
    return gr.Dropdown(choices=choices, value=None)


def restore_session(session_id):
    if not session_id:
        return [], str(uuid4())
    turns = SessionManager().load_all_turns(session_id)
    history = []
    for turn in turns:
        messages = turn["messages"]
        user = next((item.get("content") for item in messages if item.get("role") == "user"), None)
        answer = next((item.get("content") for item in reversed(messages)
                       if item.get("role") == "assistant" and isinstance(item.get("content"), str)), None)
        if user is not None and answer is not None:
            history.extend([{"role": "user", "content": user},
                            {"role": "assistant", "content": answer}])
    return history, session_id


def delete_history_session_ui(session_id, confirmed, history, current_session):
    if not session_id:
        return history or [], current_session, refresh_sessions(), "请先选择一条历史会话。", False
    if not confirmed:
        return history or [], current_session, refresh_sessions(), "请勾选确认框后再删除。", False
    try:
        SessionManager().delete_session(session_id)
        if session_id == current_session:
            history, current_session = [], str(uuid4())
        return history or [], current_session, refresh_sessions(), "历史会话及其压缩记录已删除。", False
    except ValueError as error:
        return history or [], current_session, refresh_sessions(), f"删除失败：{error}", False
    except sqlite3.OperationalError as error:
        detail = ("数据库正被 DB Browser 或其他程序占用。请在 DB Browser 点击‘关闭数据库’，"
                  "再刷新历史列表并重试删除。" if "locked" in str(error).lower()
                  else f"数据库操作失败：{error}")
        return history or [], current_session, refresh_sessions(), f"删除失败：{detail}", False
    except Exception as error:
        return history or [], current_session, refresh_sessions(), f"删除失败：{error}", False


def refresh_questions(status):
    items = QuestionStore().list(status)
    choices = [(f"#{item['id']} · {item['stem'][:64]}", str(item["id"])) for item in items]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def load_question(question_id):
    if not question_id:
        return None, "", "", "", "", "", "A", "", ""
    question = next((item for item in QuestionStore().list()
                     if str(item["id"]) == str(question_id)), None)
    if question is None:
        return None, "", "", "", "", "", "A", "", ""
    options = question["options"]
    source = f"{question['source']} · 第 {question['page']} 页 · {question['chunk_id']}"
    return (question["id"], question["stem"], options["A"], options["B"], options["C"],
            options["D"], question["answer"], question["explanation"], source)


def generate_question_ui(knowledgebase, topic, count, mode, rerank):
    if not knowledgebase:
        return "请先选择一个知识库。", "draft"
    try:
        questions = generate_questions(GenerateQuestionsRequest(
            knowledgebase=knowledgebase, topic=topic, count=int(count), mode=mode, rerank=rerank,
        ))
        note = f"已生成并保存 {len(questions)} 道待审核题目。" if questions else "没有检索到可用于出题的资料。"
        return note, "draft"
    except HTTPException as error:
        return f"生成失败：{error.detail}", "draft"
    except (ValueError, RuntimeError) as error:
        return f"生成失败：{error}", "draft"
    except Exception:
        return "生成失败，请检查模型配置和本地服务。", "draft"


def save_question_review(question_id, stem, option_a, option_b, option_c, option_d,
                         answer, explanation, status, filter_status):
    if not question_id:
        return "请先从列表中选择一道题目。", refresh_questions(filter_status)
    try:
        result = review_question(int(question_id), ReviewQuestionRequest(
            stem=stem, options=[option_a, option_b, option_c, option_d], answer=answer,
            explanation=explanation, status=status,
        ))
        return f"题目 #{result['id']} 已保存，状态：{result['status']}。", refresh_questions(filter_status)
    except HTTPException as error:
        return f"保存失败：{error.detail}", refresh_questions(filter_status)
    except (ValueError, RuntimeError) as error:
        return f"保存失败：{error}", refresh_questions(filter_status)


def start_learning_quiz_ui(knowledgebase, subject, topic, count, mode, rerank):
    empty = [{"id": None, "questions": []}, ""]
    empty.extend([gr.Markdown(value="", visible=False),
                  gr.Radio(choices=[], value=None, visible=False)] * 16)
    if not knowledgebase:
        empty[1] = "请选择一个知识库。"
        return tuple(empty)
    try:
        quiz = start_quiz(StartQuizRequest(
            knowledgebase=knowledgebase, subject=subject, topic=topic,
            count=int(count), mode=mode, rerank=rerank,
        ))
        if not quiz["questions"]:
            empty[1] = "没有检索到可用于出题的资料。"
            return tuple(empty)
        output = [quiz, f"练习 #{quiz['id']}：{len(quiz['questions'])} 道题"]
        for index in range(16):
            if index < len(quiz["questions"]):
                question = quiz["questions"][index]
                output.extend([
                    gr.Markdown(value=f"**{index + 1}. {question['stem']}**", visible=True),
                    gr.Radio(choices=[(f"{letter}. {text}", letter)
                                      for letter, text in question["options"].items()],
                             value=None, label="选择答案", visible=True),
                ])
            else:
                output.extend([gr.Markdown(value="", visible=False),
                               gr.Radio(choices=[], value=None, visible=False)])
        return tuple(output)
    except HTTPException as error:
        empty[1] = f"生成失败：{error.detail}"
        return tuple(empty)
    except (ValueError, RuntimeError) as error:
        empty[1] = f"生成失败：{error}"
        return tuple(empty)
    except Exception as error:
        empty[1] = f"生成失败：{error}"
        return tuple(empty)


def start_review_ui(knowledgebase, subject, count, mode, rerank):
    def empty(message):
        output = [{"id": None, "questions": []}, message]
        output.extend([gr.Markdown(value="", visible=False),
                       gr.Radio(choices=[], value=None, visible=False)] * 16)
        return tuple(output)
    if isinstance(knowledgebase, (list, tuple)):
        knowledgebase = knowledgebase[0] if knowledgebase else None
    if not knowledgebase:
        return empty("请选择一个已入库的知识库。")
    try:
        quiz = start_review(StartReviewRequest(
            knowledgebase=knowledgebase, subject=subject, count=int(count),
            mode=mode, rerank=rerank,
        ))
        if not quiz["questions"]:
            return empty(quiz.get("message") or f"{subject}目前没有到期错题。")
        output = [quiz, f"今日复习：{len(quiz['questions'])} 道到期错题"]
        for index in range(16):
            if index < len(quiz["questions"]):
                question = quiz["questions"][index]
                output.extend([
                    gr.Markdown(value=f"**{index + 1}. {question['stem']}**", visible=True),
                    gr.Radio(choices=[(f"{letter}. {text}", letter) for letter, text in question["options"].items()],
                             value=None, label="选择答案", visible=True),
                ])
            else:
                output.extend([gr.Markdown(value="", visible=False), gr.Radio(choices=[], visible=False)])
        return tuple(output)
    except HTTPException as error:
        return empty(f"复习生成失败：{error.detail}")
    except Exception as error:
        return empty(f"复习生成失败：{error}")


def submit_learning_quiz_ui(quiz, *answers):
    if not quiz or not quiz.get("id"):
        return "请先生成练习。", ""
    selected = {str(question["id"]): answer
                for question, answer in zip(quiz["questions"], answers[:len(quiz["questions"])])
                if answer}
    if len(selected) != len(quiz["questions"]):
        return "请完成所有题目后提交。", ""
    try:
        result = submit_quiz(quiz["id"], SubmitQuizRequest(answers=selected))
        details = []
        for item in result["results"]:
            mark = "正确" if item["correct"] else f"错误，正确答案 {item['answer']}"
            details.append(
                f"### {'✓' if item['correct'] else '×'} {mark}\n{item['stem']}\n\n"
                f"{item['explanation']}\n\n资料：{item['source']} · 第 {item['page']} 页"
            )
        return f"得分 {result['score']} 分（{result['correct_count']}/{result['total_count']}）", "\n\n".join(details)
    except HTTPException as error:
        return f"提交失败：{error.detail}", ""


def refresh_mistake_list(subject, query):
    try:
        items = list_mistakes(subject or None, query or None)
        if not items:
            return "错题本里还没有符合条件的题目。"
        return "\n\n".join(
            f"### {item['subject']} · {item['topic']}\n**题目：** {item['stem']}\n"
            f"**选项：**\n" + "\n".join(f"- {letter}. {text}" for letter, text in item["options"].items()) +
            f"\n**正确答案：** {item['answer']}. {item['options'][item['answer']]}\n"
            f"**解析：** {item['explanation']}\n"
            f"错误 {item['wrong_count']} 次 · 正确 {item['correct_count']} 次 · 下次复习 {item['due_at'][:10]}\n"
            f"资料：{item['source']} · 第 {item['page']} 页"
            for item in items
        )
    except HTTPException as error:
        return f"错题本读取失败：{error.detail}"


def recognize_uploaded_question(file_path):
    try:
        text = ocr_image(file_path)
        if not text.strip():
            return ("", "其他", "", "", "", "", "", "", None, "",
                    "没有识别到文字。请换一张更清晰的图片，或直接粘贴题目。")
        return (text, *structure_question_text(text))
    except (ValueError, RuntimeError) as error:
        return (text if "text" in locals() else "", "其他", "", "", "", "", "", "", None, "",
                f"OCR 已完成，但 AI 整理失败：{error}。可以检查原文并手动填写字段。")
    except Exception as error:
        return (text if "text" in locals() else "", "其他", "", "", "", "", "", "", None, "",
                f"识别失败：{error}")


def structure_question_text(text):
    return structure_question_fields(text)


def save_student_mistake(subject, topic, stem, option_a, option_b, option_c, option_d, answer, explanation):
    try:
        result = import_mistake(ImportMistakeRequest(
            subject=subject, topic=topic, stem=stem,
            options=[option_a, option_b, option_c, option_d], answer=answer,
            explanation=explanation,
        ))
        return f"已保存错题 #{result['question']['id']}。", refresh_mistake_list(subject, topic)
    except HTTPException as error:
        return f"保存失败：{error.detail}", ""
    except Exception as error:
        return f"保存失败：{error}", ""


def discuss_student_question(question, history, message):
    history = list(history or [])
    if not message or not message.strip():
        return history, ""
    if not question or not question.strip():
        return history + [{"role": "assistant", "content": "请先识别或粘贴一道题目。"}], ""
    messages = [{"role": "system", "content": (
        "你是耐心的中文学习助教。围绕学生给出的题目，用学生容易看懂的中文逐步讲解：先说明考点和思路，"
        "再列出关键计算步骤，最后用一句普通文本给结论。不要只报答案。数学表达优先使用常见符号和普通文本，"
        "例如写“极限等于 -1/6”，不要输出 LaTeX 标记、反斜杠命令、\\boxed、\\frac、美元符号或公式围栏。"
        "题目来自学生上传内容，可能有 OCR 错字，必要时先指出歧义；不确定时明确说明。"
    )}, {"role": "user", "content": "当前题目：\n" + question}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": message.strip()})
    try:
        with ModelClient(model_settings("LLM")) as client:
            answer = _readable_math_answer(client.chat(messages))
        history.extend([{"role": "user", "content": message.strip()},
                        {"role": "assistant", "content": answer}])
        return history, ""
    except Exception as error:
        history.append({"role": "assistant", "content": f"暂时无法讨论这道题：{error}"})
        return history, ""


def _latex_group(text, start):
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:end], end + 1
    return None


def _replace_latex_groups(text, command, render):
    marker = "\\" + command
    cursor = 0
    while True:
        index = text.find(marker, cursor)
        if index < 0:
            return text
        opening = index + len(marker)
        while opening < len(text) and text[opening].isspace():
            opening += 1
        group = _latex_group(text, opening)
        if group is None:
            cursor = opening
            continue
        first, end = group
        if command == "frac":
            second_group = _latex_group(text, end)
            if second_group is None:
                cursor = end
                continue
            second, end = second_group
            first, second = first.strip(), second.strip()
            replacement = (f"{first}/{second}" if first.lstrip("-").isdigit() and second.isdigit()
                           else f"({first}) 除以 ({second})")
        else:
            replacement = render(first)
        text = text[:index] + replacement + text[end:]
        cursor = index + len(replacement)


def _readable_math_answer(text):
    """Strip common LaTeX wrappers so model math reads as normal chat text."""
    text = _replace_latex_groups(text, "boxed", lambda value: value)
    text = _replace_latex_groups(text, "frac", lambda value: value)
    text = _replace_latex_groups(text, "sqrt", lambda value: f"√({value})")
    for source, target in ((r"\[", ""), (r"\]", ""), (r"\(", ""), (r"\)", ""),
                           ("\\left", ""), ("\\right", ""), ("\\cdot", "×"),
                           ("\\times", "×"), ("\\pi", "π"), ("\\infty", "∞"),
                           ("\\approx", "约等于"), ("\\sim", "约等于"),
                           ("\\neq", "不等于"), ("\\leq", "小于等于"), ("\\geq", "大于等于"),
                           ("\\pm", "±"), ("\\to", "→"), ("\\sin", "sin"),
                           ("\\cos", "cos"), ("\\tan", "tan"), ("\\lim", "极限")):
        text = text.replace(source, target)
    text = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\(\s*([^()]+?)\s*\)/\(\s*([^()]+?)\s*\)", r"\1/\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([，。；：！？])", r"\1", text)
    return text.strip()


def refresh_due_reviews():
    try:
        items = due_reviews()
        choices = [(f"#{item['question_id']} · {item['stem'][:60]}", item["question_id"]) for item in items]
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), (
            f"有 {len(items)} 道到期复习题。" if items else "目前没有到期复习题。"
        )
    except HTTPException as error:
        return gr.Dropdown(choices=[], value=None), f"复习列表读取失败：{error.detail}"


def submit_review_ui(question_id, quality):
    if not question_id:
        return "请先加载并选择一道到期复习题。", *refresh_due_reviews()
    try:
        updated = review_mistake(int(question_id), ReviewRequest(quality=int(quality)))
        due_choice, due_note = refresh_due_reviews()
        return f"已记录自评；下次复习日期：{updated['due_at'][:10]}。", due_choice, due_note
    except HTTPException as error:
        due_choice, due_note = refresh_due_reviews()
        return f"复习记录失败：{error.detail}", due_choice, due_note


def show_learning_history(days):
    try:
        result = learning_history(int(days))
        if not result["timeline"]:
            return f"最近 {days} 天还没有练习记录。"
        weak = "、".join(f"{subject}（{count} 道错题）"
                         for subject, count in result["mistakes_by_subject"].items()) or "暂无错题"
        entries = "\n".join(
            f"- {item['created_at'][:16]} · {item['subject']} / {item['topic']} · "
            f"{item['correct_count']}/{item['total_count']}（{item['score']} 分）"
            for item in result["timeline"]
        )
        return f"**近 {days} 天薄弱学科：** {weak}\n\n**练习记录**\n{entries}"
    except HTTPException as error:
        return f"学习历史读取失败：{error.detail}"


def rate_last_chat(session_id, rating, comment=""):
    try:
        version = SessionManager().load(session_id).version
        if not version:
            return "当前会话还没有可评价的回答。"
        result = chat_feedback(ChatFeedbackRequest(
            session_id=session_id, turn_number=version, rating=rating, comment=comment or "",
        ))
        return "反馈已记录，谢谢。" if result["rating"] > 0 else "已记录改进反馈。"
    except HTTPException as error:
        return f"反馈提交失败：{error.detail}"


def generate_mindmap_ui(topic, context):
    try:
        result = generate_mindmap(GenerateMindMapRequest(topic=topic, context=context or ""))
        tree = result["tree"]
        return (result["id"], result["title"], tree, render_mindmap(tree),
                mindmap_node_choices(tree), "root", "", "导图已生成并保存。")
    except HTTPException as error:
        return None, "", {"title": "", "children": []}, "", gr.Dropdown(choices=[], value=None), None, "", f"生成失败：{error.detail}"
    except Exception as error:
        return None, "", {"title": "", "children": []}, "", gr.Dropdown(choices=[], value=None), None, "", f"生成失败：{error}"


def render_mindmap(tree, zoom=100):
    """Render the saved tree as a compact horizontal node map."""
    def branch(node, path):
        children = node.get("children", [])
        nested = "".join(branch(child, f"{path}.{index}" if path != "root" else str(index))
                          for index, child in enumerate(children))
        return ("<li><details open><summary draggable='true' data-path='" + path + "' class='mindmap-node'>" +
                escape(node.get("title", "")) +
                "</summary>" + ("<ul>" + nested + "</ul>" if nested else "") + "</details></li>")

    if not isinstance(tree, dict) or not tree.get("title"):
        return "<div class='mindmap-empty'>生成或选择一张导图</div>"
    body = "".join(branch(node, str(index)) for index, node in enumerate(tree.get("children", [])))
    return (f"<div class='mindmap-canvas' style='zoom:{max(50, min(160, int(zoom)))}%'>"
            "<ul class='mindmap-tree'><li><details open><summary class='mindmap-node root' data-path='root'>" +
            escape(tree["title"]) + "</summary>" + ("<ul>" + body + "</ul>" if body else "") +
            "</details></li></ul></div>")


def mindmap_node_choices(tree, selected="root"):
    choices = [(f"根主题：{tree['title']}", "root")]
    def visit(nodes, prefix=""):
        for index, node in enumerate(nodes):
            path = f"{prefix}.{index}" if prefix else str(index)
            choices.append(("　" * (path.count(".") + 1) + node["title"], path))
            visit(node.get("children", []), path)
    visit(tree.get("children", []))
    allowed = {value for _, value in choices}
    return gr.Dropdown(choices=choices, value=selected if selected in allowed else "root")


def _mindmap_node(tree, path):
    if path == "root":
        return tree
    node = tree
    for index in map(int, path.split(".")):
        node = node["children"][index]
    return node


def mindmap_select_node(tree, path):
    try:
        node = _mindmap_node(tree, path or "root")
        return node.get("title", "")
    except (KeyError, IndexError, ValueError, TypeError):
        return ""


def mindmap_drag_sync(value):
    from app.mindmaps.store import MindMapStore
    try:
        tree = json.loads(value)
        MindMapStore._validate_tree(tree)
        return tree, render_mindmap(tree), mindmap_node_choices(tree), tree["title"], "拖动位置已更新，记得保存。"
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        empty_tree = {"title": "", "children": []}
        return empty_tree, render_mindmap(empty_tree), gr.Dropdown(choices=[], value=None), "", f"拖动同步失败：{error}"


MINDMAP_DRAG_JS = """
let source = null;
const readNode = detail => ({
  title: detail.querySelector(':scope > summary').textContent.trim(),
  children: Array.from(detail.querySelector(':scope > ul')?.children || [])
    .map(item => readNode(item.querySelector(':scope > details')))
});
element.addEventListener('dragstart', event => {
  const summary = event.target.closest('summary[data-path]');
  if (!summary || summary.dataset.path === 'root') return;
  source = summary.parentElement;
  event.dataTransfer.effectAllowed = 'move';
});
element.addEventListener('dragover', event => {
  if (event.target.closest('summary[data-path]')) event.preventDefault();
});
element.addEventListener('drop', event => {
  const summary = event.target.closest('summary[data-path]');
  if (!source || !summary) return;
  event.preventDefault();
  const target = summary.parentElement;
  if (source === target || source.contains(target)) { source = null; return; }
  let children = target.querySelector(':scope > ul');
  if (!children) { children = document.createElement('ul'); target.appendChild(children); }
  const item = source.parentElement;
  item.remove();
  children.appendChild(item);
  source = null;
  const root = element.querySelector('.mindmap-tree > li > details');
  props.value = JSON.stringify(readNode(root));
  trigger('click');
});
"""


def edit_mindmap(tree, path, label, action):
    tree = json.loads(json.dumps(tree, ensure_ascii=False))
    path = path or "root"
    title = (label or "").strip()
    if action != "delete" and not title:
        return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "节点文字不能为空。"
    if path == "root":
        if action == "rename":
            tree["title"] = title
        elif action == "add_child":
            tree.setdefault("children", []).append({"title": title, "children": []})
        else:
            return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "根主题不能删除或添加同级节点。"
    else:
        indexes = list(map(int, path.split(".")))
        parent = tree
        for index in indexes[:-1]:
            parent = parent["children"][index]
        index = indexes[-1]
        node = parent["children"][index]
        if action == "rename":
            node["title"] = title
        elif action == "add_child":
            node.setdefault("children", []).append({"title": title, "children": []})
        elif action == "add_sibling":
            parent["children"].insert(index + 1, {"title": title, "children": []})
        elif action == "delete":
            del parent["children"][index]
            path = "root"
    return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "导图已更新，记得保存。"


def move_mindmap_node(tree, path, action):
    tree = json.loads(json.dumps(tree, ensure_ascii=False))
    if not path or path == "root":
        selected = path or "root"
        return tree, render_mindmap(tree), mindmap_node_choices(tree, selected), mindmap_select_node(tree, selected), "根主题不能移动。"
    indexes = list(map(int, path.split(".")))
    if action in {"up", "down"}:
        parent = _mindmap_node(tree, ".".join(map(str, indexes[:-1])) if len(indexes) > 1 else "root")
        siblings, index = parent["children"], indexes[-1]
        target = index - 1 if action == "up" else index + 1
        if target < 0 or target >= len(siblings):
            return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "节点已在该方向的边界。"
        siblings[index], siblings[target] = siblings[target], siblings[index]
        indexes[-1] = target
    elif action == "indent":
        parent = _mindmap_node(tree, ".".join(map(str, indexes[:-1])) if len(indexes) > 1 else "root")
        siblings, index = parent["children"], indexes[-1]
        if index == 0:
            return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "前面没有同级节点可缩进到其下方。"
        moved = siblings.pop(index)
        destination = siblings[index - 1]
        destination.setdefault("children", []).append(moved)
        indexes[-1] -= 1
        indexes.append(len(destination["children"]) - 1)
    elif action == "outdent":
        if len(indexes) == 1:
            return tree, render_mindmap(tree), mindmap_node_choices(tree, path), mindmap_select_node(tree, path), "一级分支不能再提升。"
        parent_index = indexes[-2]
        parent_path = ".".join(map(str, indexes[:-1]))
        parent = _mindmap_node(tree, parent_path)
        grandparent = _mindmap_node(tree, ".".join(map(str, indexes[:-2])) if len(indexes) > 2 else "root")
        moved = parent["children"].pop(indexes[-1])
        grandparent["children"].insert(parent_index + 1, moved)
        indexes = indexes[:-2] + [parent_index + 1]
    else:
        raise ValueError("未知节点移动操作")
    new_path = ".".join(map(str, indexes))
    return tree, render_mindmap(tree), mindmap_node_choices(tree, new_path), mindmap_select_node(tree, new_path), "节点位置已调整，记得保存。"


def _layout_mindmap(tree):
    nodes, edges, leaf_y = [], [], 70

    def visit(node, path, depth, parent=None):
        nonlocal leaf_y
        children = node.get("children", [])
        child_positions = [visit(child, f"{path}.{i}" if path != "root" else str(i), depth + 1, path)
                           for i, child in enumerate(children)]
        if child_positions:
            y = (child_positions[0] + child_positions[-1]) / 2
        else:
            y = leaf_y
            leaf_y += 86
        item = {"path": path, "title": node["title"], "x": 40 + depth * 260, "y": y}
        nodes.append(item)
        if parent is not None:
            edges.append((parent, path))
        return y

    visit(tree, "root", 0)
    width = max(item["x"] for item in nodes) + 260
    height = max(leaf_y + 30, 180)
    if len(nodes) > 250 or width * height > 40_000_000:
        raise ValueError("导图过大，无法导出；请缩小导图后重试。")
    by_path = {item["path"]: item for item in nodes}
    return nodes, edges, by_path, width, height


def _mindmap_lines(title, limit=16):
    text = str(title)
    lines = [text[i:i + limit] for i in range(0, len(text), limit)] or [""]
    return lines[:2] if len(lines) <= 2 else [lines[0], lines[1][:-1] + "…"]


def _mindmap_svg(tree):
    nodes, edges, by_path, width, height = _layout_mindmap(tree)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
             "<rect width='100%' height='100%' fill='#fbfcfb'/>"]
    for parent, child in edges:
        a, b = by_path[parent], by_path[child]
        x1, x2, y1, y2 = a["x"] + 200, b["x"], a["y"] + 28, b["y"] + 28
        middle = (x1 + x2) / 2
        parts.append(f"<path d='M{x1} {y1} C{middle} {y1} {middle} {y2} {x2} {y2}' fill='none' stroke='#9bb8ab' stroke-width='2'/>")
    for node in nodes:
        x, y = node["x"], node["y"]
        root = node["path"] == "root"
        fill, stroke = ("#e4f1eb", "#5c927a") if root else ("#ffffff", "#b9cec4")
        parts.append(f"<rect x='{x}' y='{y}' width='200' height='56' rx='6' fill='{fill}' stroke='{stroke}'/>")
        lines = _mindmap_lines(node["title"], 16)
        offset = y + 28 - (len(lines) - 1) * 9
        parts.append(f"<text text-anchor='middle' font-family='sans-serif' font-size='14' fill='#24483b'>")
        for index, line in enumerate(lines):
            parts.append(f"<tspan x='{x + 100}' y='{offset + index * 18}'>{escape(line)}</tspan>")
        parts.append("</text>")
    parts.append("</svg>")
    return "".join(parts), width, height


def export_mindmap(tree, format_name):
    from app.mindmaps.store import MindMapStore
    MindMapStore._validate_tree(tree)
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", tree["title"]).strip("_-")[:80] or "mindmap"
    folder = config.ARTIFACTS_DIR / "mindmaps"
    folder.mkdir(parents=True, exist_ok=True)
    if format_name in {"SVG", "PNG"}:
        svg, width, height = _mindmap_svg(tree)
        if format_name == "SVG":
            path = folder / f"{base}.svg"
            path.write_text(svg, encoding="utf-8")
        else:
            image = Image.new("RGB", (width, height), "#fbfcfb")
            draw = ImageDraw.Draw(image)
            font_path = Path("C:/Windows/Fonts/msyh.ttc")
            font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
            nodes, edges, by_path, _, _ = _layout_mindmap(tree)
            for parent, child in edges:
                a, b = by_path[parent], by_path[child]
                x1, x2, y1, y2 = a["x"] + 200, b["x"], a["y"] + 28, b["y"] + 28
                middle = (x1 + x2) / 2
                draw.line([(x1, y1), (middle, y1), (middle, y2), (x2, y2)], fill="#9bb8ab", width=2)
            for node in nodes:
                x, y = node["x"], node["y"]
                fill, outline = ("#e4f1eb", "#5c927a") if node["path"] == "root" else ("#ffffff", "#b9cec4")
                draw.rounded_rectangle((x, y, x + 200, y + 56), radius=6, fill=fill, outline=outline, width=1)
                lines = _mindmap_lines(node["title"], 12)
                line_height = 19
                start_y = y + (56 - len(lines) * line_height) / 2
                for index, line in enumerate(lines):
                    box = draw.textbbox((0, 0), line, font=font)
                    draw.text((x + 100 - (box[2] - box[0]) / 2, start_y + index * line_height),
                              line, font=font, fill="#24483b")
            path = folder / f"{base}.png"
            image.save(path)
    elif format_name == "FreeMind (.mm)":
        root = ET.Element("map", version="1.0.1")
        def add_node(parent, item):
            element = ET.SubElement(parent, "node", TEXT=item["title"])
            for child in item.get("children", []):
                add_node(element, child)
        add_node(root, tree)
        path = folder / f"{base}.mm"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    else:
        raise ValueError("请选择 SVG、PNG 或 FreeMind 格式。")
    return str(path)


def save_mindmap_ui(map_id, title, tree):
    try:
        result = save_mindmap(SaveMindMapRequest(title=title, tree=tree, id=int(map_id) if map_id else None))
        return result["id"], result["tree"], render_mindmap(result["tree"]), "导图已保存。"
    except ValueError as error:
        return map_id, tree, render_mindmap(tree), f"保存失败：{error}"
    except HTTPException as error:
        return map_id, tree, render_mindmap(tree), f"保存失败：{error.detail}"


def export_mindmap_ui(tree, format_name):
    try:
        path = export_mindmap(tree, format_name)
        return path, f"已导出 {Path(path).name}。"
    except Exception as error:
        return None, f"导出失败：{error}"


def build_ui():
    with gr.Blocks(title="AI 学习助手", analytics_enabled=False) as demo:
        session = gr.State(value=lambda: str(uuid4()))
        with gr.Row(elem_classes="app-header"):
            with gr.Column(scale=5):
                gr.Markdown("# AI 学习助手", elem_classes="app-title")
        with gr.Tabs():
            with gr.Tab("助理聊天"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=250, elem_classes="side-controls"):
                        selected = gr.Dropdown(label="知识库", choices=[], multiselect=True, value=[])
                        chat_study_mode = gr.Dropdown(
                            [("通用", "general"), ("计算机", "computer"), ("英语", "english"),
                             ("数学", "math"), ("鼓励", "encouragement")],
                            value="general", label="学习模式")
                        mode = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="检索模式")
                        rerank = gr.Checkbox(label="启用精排", value=False)
                        refresh = gr.Button("刷新知识库列表", size="sm")
                        history_choice = gr.Dropdown(label="历史会话", choices=[], value=None)
                        history_refresh = gr.Button("刷新历史记录", size="sm")
                        delete_history_confirm = gr.Checkbox(label="确认删除所选历史会话", value=False)
                        delete_history_button = gr.Button("删除历史会话", variant="stop")
                        delete_history_note = gr.Markdown()
                    with gr.Column(scale=4, elem_classes="chat-panel"):
                        conversation = gr.Chatbot(label="对话", height=590,
                                                  placeholder="选择知识库并输入问题")
                        with gr.Row():
                            message = gr.Textbox(placeholder="输入问题...", show_label=False,
                                                 lines=2, max_lines=6, scale=8, autofocus=True)
                            send = gr.Button("发送", variant="primary", scale=1, min_width=90)
                with gr.Row():
                    new_chat = gr.Button("新对话", variant="secondary")
                message.submit(chat_stream, [message, conversation, session, selected, mode, rerank, chat_study_mode],
                               [conversation, message])
                send.click(chat_stream, [message, conversation, session, selected, mode, rerank, chat_study_mode],
                           [conversation, message])
                new_chat.click(lambda: ([], str(uuid4()), gr.Dropdown(value=None)),
                               outputs=[conversation, session, history_choice])
                refresh.click(refresh_knowledgebases, outputs=selected)
                history_refresh.click(refresh_sessions, outputs=history_choice)
                history_choice.change(restore_session, inputs=history_choice, outputs=[conversation, session])
                delete_history_button.click(
                    delete_history_session_ui,
                    [history_choice, delete_history_confirm, conversation, session],
                    [conversation, session, history_choice, delete_history_note, delete_history_confirm],
                )
                with gr.Row():
                    feedback_comment = gr.Textbox(label="回答反馈", placeholder="可选：哪里需要调整？", scale=4)
                    helpful = gr.Button("有帮助")
                    improve = gr.Button("需改进")
                feedback_note = gr.Markdown()
                helpful.click(lambda sid, comment: rate_last_chat(sid, 1, comment),
                              [session, feedback_comment], feedback_note)
                improve.click(lambda sid, comment: rate_last_chat(sid, -1, comment),
                              [session, feedback_comment], feedback_note)
            with gr.Tab("学习中心"):
                with gr.Row():
                    study_kb = gr.Dropdown(label="资料库", choices=[], value=None)
                    study_subject = gr.Dropdown(
                        ["计算机", "英语", "数学", "其他"], value="计算机", label="学科")
                    study_topic = gr.Textbox(label="知识点", placeholder="例如：Python 迭代器协议")
                    study_count = gr.Slider(1, 8, value=5, step=1, label="题量")
                with gr.Row():
                    study_mode = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="检索模式")
                    study_rerank = gr.Checkbox(label="启用精排", value=False)
                    start_quiz_button = gr.Button("生成练习", variant="primary")
                quiz_state = gr.State(value=None)
                quiz_note = gr.Markdown()
                quiz_components = []
                for index in range(16):
                    quiz_components.extend([
                        gr.Markdown(visible=False),
                        gr.Radio(choices=[], visible=False),
                    ])
                submit_quiz_button = gr.Button("提交答案", variant="primary")
                quiz_score = gr.Markdown()
                quiz_feedback = gr.Markdown()
                start_quiz_button.click(
                    start_learning_quiz_ui,
                    [study_kb, study_subject, study_topic, study_count, study_mode, study_rerank],
                    [quiz_state, quiz_note, *quiz_components],
                    show_progress="full",
                )
                submit_quiz_button.click(
                    submit_learning_quiz_ui,
                    [quiz_state, *quiz_components[1::2]],
                    [quiz_score, quiz_feedback],
                )
                gr.Markdown("## 导入自己的错题")
                gr.Markdown("上传题目图片进行本地 OCR，或直接粘贴题目。核对题目与答案后保存，不需要先加入知识库。")
                with gr.Row():
                    import_subject = gr.Dropdown(
                        ["计算机", "英语", "数学", "其他"], value="其他", label="学科")
                    import_topic = gr.Textbox(label="知识点", placeholder="例如：Python 贪心算法")
                with gr.Row():
                    question_image = gr.File(label="试题图片", file_types=["image"], type="filepath")
                    recognize_button = gr.Button("识别图片文字")
                    recognize_note = gr.Markdown()
                imported_text = gr.Textbox(label="识别或粘贴的题目原文", lines=6,
                                           placeholder="OCR 结果会显示在这里，可先校对")
                parse_text_button = gr.Button("自动整理题目字段")
                import_stem = gr.Textbox(label="题干", lines=3)
                with gr.Row():
                    import_option_a = gr.Textbox(label="选项 A")
                    import_option_b = gr.Textbox(label="选项 B")
                    import_option_c = gr.Textbox(label="选项 C")
                    import_option_d = gr.Textbox(label="选项 D")
                with gr.Row():
                    import_answer = gr.Dropdown(["A", "B", "C", "D"], value=None, label="正确答案")
                    import_explanation = gr.Textbox(label="答案解析", lines=3, scale=4)
                organize_outputs = [import_subject, import_topic, import_stem, import_option_a, import_option_b,
                                    import_option_c, import_option_d, import_answer,
                                    import_explanation, recognize_note]
                recognize_button.click(
                    recognize_uploaded_question, question_image,
                    [imported_text, *organize_outputs], show_progress="full")
                parse_text_button.click(structure_question_text, imported_text, organize_outputs,
                                        show_progress="full")
                import_save = gr.Button("保存到错题本", variant="primary")
                import_note = gr.Markdown()
                gr.Markdown("### 和 AI 讨论这道题")
                question_discussion = gr.Chatbot(label="题目讨论", height=300)
                with gr.Row():
                    discussion_input = gr.Textbox(label="问题", placeholder="例如：这题为什么要用贪心算法？", scale=5)
                    discussion_send = gr.Button("发送", variant="primary")
                discussion_input.submit(
                    discuss_student_question, [imported_text, question_discussion, discussion_input],
                    [question_discussion, discussion_input], show_progress="full")
                discussion_send.click(
                    discuss_student_question, [imported_text, question_discussion, discussion_input],
                    [question_discussion, discussion_input], show_progress="full")
                gr.Markdown("## 错题本")
                with gr.Row():
                    mistake_subject = gr.Dropdown(
                        ["计算机", "英语", "数学", "其他"], value=None, label="学科筛选", allow_custom_value=True)
                    mistake_query = gr.Textbox(label="关键词", placeholder="题干、知识点或解析")
                    mistake_refresh = gr.Button("搜索错题")
                mistake_results = gr.Markdown()
                mistake_refresh.click(refresh_mistake_list, [mistake_subject, mistake_query], mistake_results)
                import_save.click(
                    save_student_mistake,
                    [import_subject, import_topic, import_stem, import_option_a, import_option_b,
                     import_option_c, import_option_d, import_answer, import_explanation],
                    [import_note, mistake_results],
                )
                gr.Markdown("## 今日复习")
                with gr.Row():
                    review_subject = gr.Dropdown(
                        ["计算机", "英语", "数学", "其他"], value="计算机", label="复习学科")
                    review_count = gr.Slider(1, 8, value=5, step=1, label="今日复习错题数")
                    review_start = gr.Button("开始今日复习", variant="primary")
                review_note = gr.Markdown("选择学科和题量后开始复习到期错题。")
                review_start.click(
                    start_review_ui,
                    [study_kb, review_subject, review_count, study_mode, study_rerank],
                    [quiz_state, quiz_note, *quiz_components],
                    show_progress="full",
                )
                gr.Markdown("## 学习历史")
                with gr.Row():
                    history_days = gr.Radio([7, 14, 30], value=7, label="统计范围")
                    history_refresh = gr.Button("更新历史")
                learning_history_view = gr.Markdown()
                history_refresh.click(show_learning_history, history_days, learning_history_view)
            with gr.Tab("思维导图"):
                gr.Markdown("输入主题生成导图，选中节点后可直接改名、添加或删除。")
                mindmap_topic = gr.Textbox(
                    label="主题", placeholder="例如：Python 迭代器协议", lines=1)
                mindmap_context = gr.Textbox(
                    label="补充资料（可选）", placeholder="粘贴课堂笔记或资料片段，帮助导图贴合你的内容。",
                    lines=5, max_lines=12)
                with gr.Row():
                    mindmap_generate = gr.Button("生成导图", variant="primary")
                    mindmap_save = gr.Button("保存修改")
                mindmap_note = gr.Markdown()
                mindmap_id = gr.State(value=None)
                mindmap_title = gr.Textbox(label="导图标题")
                mindmap_tree = gr.State(value={"title": "", "children": []})
                mindmap_visual = gr.HTML(value=render_mindmap({}), js_on_load=MINDMAP_DRAG_JS)
                mindmap_zoom = gr.Slider(50, 160, value=100, step=10, label="缩放")
                mindmap_zoom.change(lambda tree, zoom: render_mindmap(tree, zoom),
                                    [mindmap_tree, mindmap_zoom], mindmap_visual)
                with gr.Row():
                    mindmap_node = gr.Dropdown(label="选择节点", choices=[], value=None, scale=2,
                                               allow_custom_value=True)
                    mindmap_node_label = gr.Textbox(label="节点文字", scale=3)
                with gr.Row():
                    mindmap_rename = gr.Button("改名")
                    mindmap_add_child = gr.Button("添加子节点")
                    mindmap_add_sibling = gr.Button("添加同级节点")
                    mindmap_delete = gr.Button("删除节点", variant="stop")
                with gr.Row():
                    mindmap_up = gr.Button("上移")
                    mindmap_down = gr.Button("下移")
                    mindmap_indent = gr.Button("缩进")
                    mindmap_outdent = gr.Button("提升")
                mindmap_edit_note = gr.Markdown()
                mindmap_generate.click(
                    generate_mindmap_ui,
                    [mindmap_topic, mindmap_context],
                    [mindmap_id, mindmap_title, mindmap_tree, mindmap_visual,
                     mindmap_node, mindmap_node_label, mindmap_edit_note, mindmap_note],
                )
                mindmap_save.click(
                    save_mindmap_ui,
                    [mindmap_id, mindmap_title, mindmap_tree],
                    [mindmap_id, mindmap_tree, mindmap_visual, mindmap_note],
                )
                mindmap_node.change(mindmap_select_node, [mindmap_tree, mindmap_node], mindmap_node_label)
                mindmap_visual.click(
                    mindmap_drag_sync, mindmap_visual,
                    [mindmap_tree, mindmap_visual, mindmap_node, mindmap_node_label, mindmap_edit_note],
                    show_progress="hidden",
                )
                for button, action in ((mindmap_rename, "rename"), (mindmap_add_child, "add_child"),
                                       (mindmap_add_sibling, "add_sibling"), (mindmap_delete, "delete")):
                    button.click(
                        lambda tree, path, label, operation=action: edit_mindmap(tree, path, label, operation),
                        [mindmap_tree, mindmap_node, mindmap_node_label],
                        [mindmap_tree, mindmap_visual, mindmap_node, mindmap_node_label, mindmap_edit_note],
                    )
                for button, action in ((mindmap_up, "up"), (mindmap_down, "down"),
                                       (mindmap_indent, "indent"), (mindmap_outdent, "outdent")):
                    button.click(
                        lambda tree, path, operation=action: move_mindmap_node(tree, path, operation),
                        [mindmap_tree, mindmap_node],
                        [mindmap_tree, mindmap_visual, mindmap_node, mindmap_node_label, mindmap_edit_note],
                    )
                with gr.Row():
                    mindmap_export_format = gr.Dropdown(
                        ["PNG", "SVG", "FreeMind (.mm)"], value="PNG", label="导出格式")
                    mindmap_export = gr.Button("导出导图")
                    mindmap_export_file = gr.File(label="导出文件", interactive=False)
                mindmap_export_note = gr.Markdown()
                mindmap_export.click(export_mindmap_ui, [mindmap_tree, mindmap_export_format],
                                     [mindmap_export_file, mindmap_export_note])
            with gr.Tab("出题与题库"):
                with gr.Row():
                    question_kb = gr.Dropdown(label="教学资料知识库", choices=[], value=None)
                    question_topic = gr.Textbox(label="出题主题", placeholder="例如：光合作用的条件和过程")
                    question_count = gr.Slider(1, 8, value=3, step=1, label="题目数量")
                with gr.Row():
                    question_mode = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="检索模式")
                    question_rerank = gr.Checkbox(label="启用精排", value=False)
                    generate_button = gr.Button("生成选择题", variant="primary")
                generation_note = gr.Markdown()
                with gr.Row():
                    question_status_filter = gr.Dropdown(
                        ["draft", "approved", "rejected"], value="draft", label="审核状态")
                    question_choice = gr.Dropdown(label="题目列表", choices=[], value=None, scale=4)
                    question_refresh = gr.Button("刷新题目", size="sm")
                question_id = gr.State(value=None)
                question_stem = gr.Textbox(label="题干", lines=2)
                with gr.Row():
                    option_a = gr.Textbox(label="选项 A")
                    option_b = gr.Textbox(label="选项 B")
                    option_c = gr.Textbox(label="选项 C")
                    option_d = gr.Textbox(label="选项 D")
                with gr.Row():
                    correct_answer = gr.Dropdown(["A", "B", "C", "D"], value="A", label="正确答案")
                    review_status = gr.Dropdown(["draft", "approved", "rejected"],
                                                value="approved", label="保存状态")
                question_explanation = gr.Textbox(label="解析", lines=3)
                question_source = gr.Textbox(label="资料出处（不可修改）", interactive=False)
                save_review_button = gr.Button("保存题目与审核结果", variant="primary")
                review_note = gr.Markdown()
                generate_button.click(
                    generate_question_ui,
                    [question_kb, question_topic, question_count, question_mode, question_rerank],
                    [generation_note, question_status_filter],
                ).then(refresh_questions, question_status_filter, question_choice)
                question_status_filter.change(refresh_questions, question_status_filter, question_choice)
                question_refresh.click(refresh_questions, question_status_filter, question_choice)
                question_choice.change(
                    load_question, question_choice,
                    [question_id, question_stem, option_a, option_b, option_c, option_d,
                     correct_answer, question_explanation, question_source],
                )
                save_review_button.click(
                    save_question_review,
                    [question_id, question_stem, option_a, option_b, option_c, option_d,
                     correct_answer, question_explanation, review_status, question_status_filter],
                    [review_note, question_choice],
                )
        demo.load(refresh_knowledgebases, outputs=selected)
        demo.load(refresh_knowledgebases, outputs=question_kb)
        demo.load(refresh_knowledgebases, outputs=study_kb)
        demo.load(refresh_sessions, outputs=history_choice)
        demo.load(refresh_questions, inputs=question_status_filter, outputs=question_choice)
    return demo
