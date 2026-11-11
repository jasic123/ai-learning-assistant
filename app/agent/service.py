"""应用入口：组装 Agent 组件，为终端、页面和 API 提供同一调用方式。"""
from queue import Queue
from threading import Thread

from app.agent.context import ContextBuilder
from app.agent.loop import AgentLoop
from app.agent.memory import MemoryStore
from app.agent.session import SessionManager, validate_session_id
from app.config import model_settings
from app.llm.client import ModelClient
from app.learning.store import LearningStore
from app.tools.files import WorkspaceFileTool
from app.tools.history import HistoryReadTool
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.memory import LongTermMemoryTool
from app.tools.registry import ToolRegistry
from app.tools.time import CurrentTimeTool
from app.tools.weather import WeatherTool
from app.tools.web import WebSearchTool
from app.tools.web_fetch import WebFetchTool


class AgentService:
    def __init__(self, knowledgebases, *, mode="hybrid", rerank=False,
                 data_dir=None, max_iterations=8, owner_id="local",
                 context_tokens=16000, output_reserve=2000, study_mode="general",
                 vector_weight=0.5, keyword_weight=0.5):
        if isinstance(knowledgebases, str):
            raise ValueError("knowledgebases 须为知识库名称列表")
        if type(max_iterations) is not int or not 1 <= max_iterations <= 32:
            raise ValueError("模型迭代上限须为 1～32")
        tool = KnowledgeSearchTool(knowledgebases, mode=mode, rerank=rerank,
                                   vector_weight=vector_weight, keyword_weight=keyword_weight)
        self.knowledge_tool = tool
        self.context = ContextBuilder(tool.knowledgebases, study_mode=study_mode)
        self.sessions = SessionManager(data_dir, owner_id=owner_id)
        self.memory = MemoryStore(self.sessions)
        self.data_dir = data_dir
        self.max_iterations = max_iterations
        if (type(context_tokens) is not int or type(output_reserve) is not int
            or context_tokens < 1024 or not 128 <= output_reserve <= context_tokens // 2):
            raise ValueError("上下文窗口/输出预留配置无效")
        self.context_tokens, self.output_reserve = context_tokens, output_reserve

    def chat(self, message: str, *, session_id="default", on_delta=None, on_status=None):
        validate_session_id(session_id)
        if not isinstance(message, str) or not message.strip() or len(message) > 8000:
            raise ValueError("用户问题须为 1～8000 字符")
        self.sessions.load(session_id)
        tools = ToolRegistry()
        tools.register(self.knowledge_tool)
        tools.register(HistoryReadTool(self.sessions, session_id))
        tools.register(CurrentTimeTool())
        tools.register(LongTermMemoryTool(self.memory))
        tools.register(WeatherTool())
        tools.register(WebSearchTool())
        tools.register(WebFetchTool())
        tools.register(WorkspaceFileTool(self.data_dir))
        memory = self.memory.list()
        learning_profile = LearningStore(self.sessions.data_dir, owner_id=self.sessions.owner_id).learning_profile()
        with ModelClient(model_settings("LLM")) as client:
            loop = AgentLoop(client, tools, self.context, self.sessions,
                             max_iterations=self.max_iterations, memory=memory,
                             context_tokens=self.context_tokens, output_reserve=self.output_reserve,
                             learning_profile=learning_profile)
            return loop.run(message, session_id=session_id, on_delta=on_delta, on_status=on_status)

    def stream_chat(self, message: str, *, session_id="default"):
        """在线程中运行同步 SDK 请求，并传出 token、完成或错误事件。"""
        events = Queue()

        def run():
            try:
                result = self.chat(message, session_id=session_id,
                                   on_delta=lambda text: events.put(("token", text)),
                                   on_status=lambda status: events.put(("status", status)))
                events.put(("done", result))
            except Exception as error:
                events.put(("error", error))

        Thread(target=run, daemon=True).start()
        while True:
            kind, value = events.get()
            if kind == "token":
                yield {"type": kind, "content": value}
            elif kind == "status":
                yield {"type": kind, **value}
            elif kind == "done":
                yield {"type": kind, "result": value.to_dict()}
                return
            else:
                yield {"type": kind, "error": value}
                return
