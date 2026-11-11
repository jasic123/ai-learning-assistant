async function getJson(path, signal) {
  const response = await fetch(path, { signal, credentials: "same-origin", headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

export function getDueReviews(signal) {
  return getJson("/api/learning/reviews/due?limit=100", signal);
}

export function deleteAccountSession(signal) {
  return logout(signal);
}

export function getKnowledgebases(signal) {
  return getJson("/api/knowledgebases", signal);
}

export function getTeacherKnowledgebases(signal) {
  return getJson("/api/knowledgebases/manage", signal);
}

export function getTeacherKnowledgebase(name, signal) {
  return getJson(`/api/knowledgebases/${encodeURIComponent(name)}`, signal);
}

export async function createTeacherKnowledgebase(payload, signal) {
  return postJson("/api/knowledgebases", payload, "创建资料库失败", signal);
}

export async function uploadTeacherPdf(name, file, signal) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/api/knowledgebases/${encodeURIComponent(name)}/files`, {
    method: "POST", signal, credentials: "same-origin", body,
  });
  if (!response.ok) {
    let detail = `上传 PDF 失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function buildTeacherKnowledgebase(name, signal) {
  const response = await fetch(`/api/knowledgebases/${encodeURIComponent(name)}/build`, {
    method: "POST", signal, credentials: "same-origin", headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = `构建索引失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export function getTeacherQuestions(status, signal) {
  return getJson(`/api/questions?status=${encodeURIComponent(status)}`, signal);
}

export async function generateTeacherQuestions(payload, signal) {
  const response = await fetch("/api/questions/generate", {
    method: "POST", signal, credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `生成题目失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function reviewTeacherQuestion(id, payload, signal) {
  const response = await fetch(`/api/questions/${id}`, {
    method: "PUT", signal, credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `保存审核结果失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export function getTeacherQuestionAnalytics(signal) {
  return getJson("/api/questions/analytics", signal);
}

export function getSessions(signal) {
  return getJson("/api/sessions?limit=100", signal);
}

export function getCurrentUser(signal) {
  return getJson("/api/auth/me", signal);
}

export async function authenticate(mode, credentials, signal) {
  return postJson(`/api/auth/${mode}`, credentials, `登录失败`, signal);
}

export async function logout(signal) {
  return postJson("/api/auth/logout", {}, "退出登录失败", signal);
}

async function postJson(path, payload, fallback, signal) {
  const response = await fetch(path, {
    method: "POST", signal, credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `${fallback}（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export function getSession(sessionId, signal) {
  return getJson(`/api/sessions/${encodeURIComponent(sessionId)}`, signal);
}

export async function deleteSession(sessionId, signal) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE", signal, credentials: "same-origin", headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = `删除历史会话失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export function getMindmaps(signal) {
  return getJson("/api/mindmaps", signal);
}

export async function generateMindmap(payload, signal) {
  const response = await fetch("/api/mindmaps/generate", {
    method: "POST",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `生成导图失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function saveMindmap(payload, signal) {
  const response = await fetch("/api/mindmaps", {
    method: "PUT",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `保存导图失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function streamChat(payload, onEvent, signal) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `对话请求失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const eventText of events) {
      const data = eventText.split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data || data === "[DONE]") continue;
      const event = JSON.parse(data);
      onEvent(event);
      if (event.type === "error") throw new Error(event.detail || "对话请求失败");
      if (event.type === "done") return event.result;
    }
    if (done) break;
  }
  throw new Error("对话连接提前结束，请重试。");
}

export function getLearningHistory(signal) {
  return getJson("/api/learning/history?days=7", signal);
}

export async function startPractice(payload, signal) {
  const response = await fetch("/api/learning/quizzes", {
    method: "POST",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `生成练习失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function replacePracticeQuestion(quizId, payload, signal) {
  const response = await fetch(`/api/learning/quizzes/${quizId}/replace`, {
    method: "POST", signal, credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `替换题目失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export function getMistakes({ subject, query } = {}, signal) {
  const params = new URLSearchParams();
  if (subject) params.set("subject", subject);
  if (query?.trim()) params.set("query", query.trim());
  const suffix = params.size ? `?${params}` : "";
  return getJson(`/api/learning/mistakes${suffix}`, signal);
}

export async function recognizeMistakeImage(file, signal) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/learning/mistakes/ocr", {
    method: "POST", signal, credentials: "same-origin", headers: { Accept: "application/json" }, body,
  });
  if (!response.ok) {
    let detail = `图片识别失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function structureMistakeText(text, onEvent, signal) {
  const response = await fetch("/api/learning/mistakes/structure/stream", {
    method: "POST", signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    let detail = `题目整理失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const eventText of events) {
      const data = eventText.split(/\r?\n/).find((line) => line.startsWith("data:"))?.slice(5).trimStart();
      if (!data) continue;
      const event = JSON.parse(data);
      onEvent?.(event);
      if (event.type === "error") throw new Error(event.detail || "题目整理失败");
      if (event.type === "done") return event.result;
    }
    if (done) break;
  }
  throw new Error("整理连接提前结束，请重试。");
}

export async function importMistake(payload, signal) {
  const response = await fetch("/api/learning/mistakes/import", {
    method: "POST", signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `保存错题失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  return response.json();
}

export async function startReview({ subject, count }, signal) {
  const response = await fetch("/api/learning/reviews/start", {
    method: "POST",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ subject, count }),
  });
  if (!response.ok) {
    let detail = `开始复习失败（${response.status}）`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function submitQuiz(quizId, answers, signal) {
  const response = await fetch(`/api/learning/quizzes/${quizId}/submit`, {
    method: "POST",
    signal,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ answers }),
  });
  if (!response.ok) {
    let detail = `提交答案失败（${response.status}）`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

export function saveAttemptErrorReasons(attemptId, reasons, signal) {
  return postJson(`/api/learning/attempts/${attemptId}/error-reasons`, { reasons }, "保存错因失败", signal);
}

export function getLearningProfile(signal) {
  return getJson("/api/learning/profile", signal);
}
