import { useCallback, useEffect, useRef, useState } from "react";
import { deleteSession, getKnowledgebases, getSession, getSessions, streamChat } from "./api.js";
import { readableText } from "./textFormat.js";

const studyModes = [
  ["general", "通用"],
  ["computer", "计算机"],
  ["english", "英语"],
  ["math", "数学"],
  ["encouragement", "鼓励"],
];

const retrievalModes = [
  ["hybrid", "混合检索"],
  ["vector", "语义检索"],
  ["bm25", "关键词检索"],
];

const timeFormatter = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
});

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() ?? `study-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function timeLabel(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : timeFormatter.format(date);
}

function restoredMessages(turns) {
  const result = [];
  for (const turn of turns) {
    const items = turn.messages ?? [];
    const user = items.find((item) => item.role === "user" && typeof item.content === "string");
    const answer = [...items].reverse().find((item) => item.role === "assistant" && typeof item.content === "string");
    if (user && answer) result.push({
      id: `turn-${turn.turn_number}-user`, role: "user", content: user.content,
    }, {
      id: `turn-${turn.turn_number}-assistant`, role: "assistant", content: answer.content,
      sources: [], status: "complete", statusText: "已完成",
    });
  }
  return result;
}

function StudyPage() {
  const [knowledgebases, setKnowledgebases] = useState([]);
  const [selectedKnowledgebases, setSelectedKnowledgebases] = useState([]);
  const [studyMode, setStudyMode] = useState("general");
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [rerank, setRerank] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [selectedSession, setSelectedSession] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [deletingHistory, setDeletingHistory] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);
  const controllerRef = useRef(null);

  const refreshSessions = useCallback(async () => {
    const items = await getSessions();
    setSessions(items);
    return items;
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([getKnowledgebases(controller.signal), getSessions(controller.signal)])
      .then(([libraries, history]) => {
        setKnowledgebases(libraries);
        setSelectedKnowledgebases(libraries.length ? [libraries[0].name] : []);
        setSessions(history);
      })
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingInitial(false);
      });
    return () => {
      controller.abort();
      controllerRef.current?.abort();
    };
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages]);

  function toggleKnowledgebase(name) {
    setSelectedKnowledgebases((current) => current.includes(name)
      ? current.filter((item) => item !== name)
      : [...current, name]);
  }

  function startNewConversation() {
    controllerRef.current?.abort();
    setSending(false);
    setMessages([]);
    setInput("");
    setError("");
    setSelectedSession("");
    setSessionId(newSessionId());
  }

  async function restoreSelectedSession(value) {
    setSelectedSession(value);
    if (!value) return;
    setLoadingHistory(true);
    setError("");
    try {
      const detail = await getSession(value);
      setMessages(restoredMessages(detail.turns));
      setSessionId(value);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingHistory(false);
    }
  }

  async function deleteSelectedSession() {
    if (!selectedSession || deletingHistory) return;
    const selected = sessions.find((item) => item.session_id === selectedSession);
    const label = (selected?.preview || "这段历史会话").replace(/\s+/g, " ").slice(0, 48);
    if (!window.confirm(`确定删除“${label}”吗？删除后无法恢复。`)) return;
    setDeletingHistory(true);
    setError("");
    try {
      await deleteSession(selectedSession);
      const deletedId = selectedSession;
      const remaining = await refreshSessions();
      if (sessionId === deletedId) startNewConversation();
      else setSelectedSession("");
      setError("");
      if (!remaining.length) setSelectedSession("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setDeletingHistory(false);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question || sending) return;
    if (!selectedKnowledgebases.length) {
      setError("请先选择至少一个资料库。");
      return;
    }

    const next = [
      ...messages,
      { id: `user-${Date.now()}`, role: "user", content: question },
      { id: `assistant-${Date.now()}`, role: "assistant", content: "", sources: [],
        status: "thinking", statusText: "正在思考…" },
    ];
    const assistantIndex = next.length - 1;
    setMessages(next);
    setInput("");
    setError("");
    setSending(true);
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const result = await streamChat({
        message: question,
        session_id: sessionId,
        knowledgebases: selectedKnowledgebases,
        mode: retrievalMode,
        rerank,
        study_mode: studyMode,
      }, (event) => {
        if (event.type === "token") {
          next[assistantIndex] = { ...next[assistantIndex], content: next[assistantIndex].content + event.content };
          setMessages([...next]);
        } else if (event.type === "status") {
          next[assistantIndex] = { ...next[assistantIndex], status: event.state, statusText: event.message };
          setMessages([...next]);
        } else if (event.type === "done") {
          next[assistantIndex] = {
            ...next[assistantIndex],
            content: event.result.answer ?? next[assistantIndex].content,
            sources: event.result.sources ?? [],
            status: "complete", statusText: "已完成",
          };
          setMessages([...next]);
        }
      }, controller.signal);
      next[assistantIndex] = {
        ...next[assistantIndex],
        content: result.answer ?? next[assistantIndex].content,
        sources: result.sources ?? [],
        status: "complete", statusText: "已完成",
      };
      setMessages([...next]);
      try {
        const history = await refreshSessions();
        setSelectedSession(history.some((item) => item.session_id === sessionId) ? sessionId : "");
      } catch {
        // The answer is already saved; a failed history refresh should not replace it with an error.
      }
    } catch (requestError) {
      if (requestError.name !== "AbortError") {
        setError(requestError.message);
        next[assistantIndex] = { ...next[assistantIndex], content: "这次没有完成，请检查服务后重试。",
          status: "failed", statusText: "未完成" };
        setMessages([...next]);
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      setSending(false);
    }
  }

  return (
    <section className="page-frame study-page" aria-labelledby="page-title">
      <header className="page-heading study-heading">
        <div>
          <p className="eyebrow">基于个人资料的学习对话</p>
          <h1 id="page-title">AI 学习</h1>
        </div>
        <div className="study-history-controls">
          <label className="sr-only" htmlFor="session-history">历史会话</label>
          <select id="session-history" value={selectedSession} onChange={(event) => restoreSelectedSession(event.target.value)}
                  disabled={loadingHistory || !sessions.length}>
            <option value="">{loadingHistory ? "正在恢复…" : sessions.length ? "历史会话" : "暂无历史会话"}</option>
            {sessions.map((item) => (
              <option key={item.session_id} value={item.session_id}>
                {(item.preview || "新会话").replace(/\s+/g, " ").slice(0, 48)} · {item.turn_count} 轮 · {timeLabel(item.updated_at)}
              </option>
            ))}
          </select>
          <button className="quiet-button delete-history-button" type="button"
                  onClick={deleteSelectedSession} disabled={!selectedSession || loadingHistory || deletingHistory || sending}
                  title={selectedSession ? "删除选中的历史会话" : "先选择一段历史会话"}>
            {deletingHistory ? "删除中…" : "删除历史"}
          </button>
          <button className="quiet-button" type="button" onClick={startNewConversation} disabled={sending}>新对话</button>
          <button className="quiet-button history-refresh" type="button"
                  onClick={() => refreshSessions().catch((requestError) => setError(requestError.message))}>
            刷新历史
          </button>
        </div>
      </header>

      {error && (
        <div className="dashboard-error" role="alert">
          <span>{error}</span>
          <button className="text-button" type="button" onClick={() => setError("")}>关闭</button>
        </div>
      )}

      <div className="study-workspace">
        <aside className="study-settings" aria-label="对话设置">
          <div className="settings-section">
            <div className="section-heading compact-heading">
              <div><p className="section-kicker">资料范围</p><h2>知识库</h2></div>
              <button className="icon-text-button" type="button"
                      onClick={() => getKnowledgebases().then(setKnowledgebases).catch((e) => setError(e.message))}>
                刷新
              </button>
            </div>
            {loadingInitial ? <p className="subtle-empty">正在读取资料库…</p> : knowledgebases.length ? (
              <div className="knowledgebase-list">
                {knowledgebases.map((item) => (
                  <label className="knowledgebase-option" key={item.name}>
                    <input type="checkbox" checked={selectedKnowledgebases.includes(item.name)}
                           disabled={sending} onChange={() => toggleKnowledgebase(item.name)} />
                    <span className="knowledgebase-copy">
                      <strong>{item.name}</strong>
                      <small>{item.chunk_count.toLocaleString("zh-CN")} 个资料片段</small>
                    </span>
                  </label>
                ))}
              </div>
            ) : <p className="subtle-empty">还没有已入库的资料。</p>}
          </div>

          <label className="study-setting-field">
            <span>学习模式</span>
            <select value={studyMode} onChange={(event) => setStudyMode(event.target.value)} disabled={sending}>
              {studyModes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>

          <fieldset className="study-setting-field study-mode-fieldset">
            <legend>检索方式</legend>
            <div className="study-retrieval-options">
              {retrievalModes.map(([value, label]) => (
                <label key={value}>
                  <input type="radio" name="retrieval-mode" value={value} checked={retrievalMode === value}
                         disabled={sending} onChange={() => setRetrievalMode(value)} />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <label className="rerank-toggle study-rerank">
            <input type="checkbox" checked={rerank} disabled={sending}
                   onChange={(event) => setRerank(event.target.checked)} />
            <span>启用精排</span>
          </label>
        </aside>

        <div className="study-chat-column">
          <div className="study-conversation" aria-live="polite" aria-label="对话内容">
            {messages.length ? messages.map((message) => (
              <article className={`chat-message ${message.role}`} key={message.id}>
                <div className="message-role">{message.role === "user" ? "你" : "学习助手"}</div>
                {message.role === "assistant" && (message.statusText || sending) && (
                  <div className={`message-status ${message.status || "thinking"}`} aria-live="polite">
                    {message.statusText || "正在思考…"}
                  </div>
                )}
                <div className="message-content">
                  {message.content ? readableText(message.content) : (sending && message.role === "assistant" ? <span className="typing-indicator">正在思考…</span> : "")}
                </div>
                {message.role === "assistant" && message.sources?.length > 0 && (
                  <div className="message-sources">
                    <span>资料来源</span>
                    {message.sources.map((source) => (
                      <div className="source-reference" key={source.citation}>
                        <strong>[{source.citation}]</strong>
                        <span>{source.source || source.file}{source.page ? ` · 第 ${source.page} 页` : ""}</span>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            )) : (
              <div className="study-welcome">
                <p className="section-kicker">开始学习</p>
                <h2>今天想弄懂什么？</h2>
                <p>选择资料库并提出问题，回答会尽量附上教材来源。</p>
                <div className="suggested-questions">
                  {["用简单例子解释这个知识点", "帮我比较两个容易混淆的概念", "根据资料给我一个自测问题"].map((text) => (
                    <button key={text} type="button" onClick={() => setInput(text)}>{text}<span aria-hidden="true">↗</span></button>
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form className="study-composer" onSubmit={sendMessage}>
            <label className="sr-only" htmlFor="study-message">输入学习问题</label>
            <textarea id="study-message" value={input} maxLength={8000} rows={2}
                      placeholder="输入问题…（Shift + Enter 换行）" disabled={sending || loadingInitial}
                      onChange={(event) => setInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          event.currentTarget.form.requestSubmit();
                        }
                      }} />
            <div className="composer-footer">
              <span>{selectedKnowledgebases.length ? `已选 ${selectedKnowledgebases.length} 个资料库` : "请先选择知识库"}</span>
              <button className="primary-button" type="submit"
                      disabled={sending || !input.trim() || !selectedKnowledgebases.length}>
                {sending ? "回答中…" : "发送"}<span aria-hidden="true">↑</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}

export default StudyPage;
