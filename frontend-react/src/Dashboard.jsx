import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getDueReviews, getLearningHistory } from "./api.js";

const shanghaiDate = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  month: "numeric",
  day: "numeric",
});

const shanghaiDateTime = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  month: "numeric",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function formatDate(value, formatter) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : formatter.format(date);
}

function Dashboard() {
  const [data, setData] = useState({ due: null, history: null });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async (signal) => {
    setLoading(true);
    setError("");
    try {
      const [due, history] = await Promise.all([
        getDueReviews(signal),
        getLearningHistory(signal),
      ]);
      setData({ due, history });
    } catch (requestError) {
      if (requestError.name !== "AbortError") setError(requestError.message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const stats = useMemo(() => {
    const timeline = data.history?.timeline ?? [];
    const total = timeline.reduce((sum, item) => sum + item.total_count, 0);
    const correct = timeline.reduce((sum, item) => sum + item.correct_count, 0);
    return {
      attempts: timeline.length,
      accuracy: total ? Math.round((correct / total) * 100) : null,
      questions: total,
    };
  }, [data.history]);

  const due = data.due ?? [];
  const dueCount = due.length === 100 ? "100+" : due.length;
  const recent = (data.history?.timeline ?? []).slice(0, 5);
  const weak = Object.entries(data.history?.mistakes_by_subject ?? {}).slice(0, 4);
  const daily = data.history?.daily ?? [];
  const maxDaily = Math.max(1, ...daily.map((item) => item.attempts));

  return (
    <section className="page-frame dashboard" aria-labelledby="page-title">
      <header className="page-heading dashboard-heading">
        <div>
          <p className="eyebrow">学习概览 · 最近 7 天</p>
          <h1 id="page-title">今日学习</h1>
        </div>
        <button className="quiet-button" type="button" onClick={() => load()} disabled={loading}>
          {loading ? "更新中…" : "刷新数据"}
        </button>
      </header>

      {error && (
        <div className="dashboard-error" role="alert">
          <span>{error}</span>
          <button className="text-button" type="button" onClick={() => load()}>重试</button>
        </div>
      )}

      <section className="due-banner" aria-label="今日复习">
        <div>
          <p className="section-kicker">到期复习</p>
          <div className="due-count">{loading && data.due === null ? "—" : dueCount}</div>
          <p className="due-caption">道题等待复习</p>
        </div>
        <Link className="primary-button" to="/mistakes">开始复习 <span aria-hidden="true">→</span></Link>
      </section>

      <section className="stats-strip" aria-label="近七天学习统计">
        <div className="stat-item">
          <span className="stat-label">练习次数</span>
          <strong>{loading && data.history === null ? "—" : stats.attempts}</strong>
          <span className="stat-note">最近 7 天</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">平均正确率</span>
          <strong>{stats.accuracy === null ? "—" : `${stats.accuracy}%`}</strong>
          <span className="stat-note">{stats.questions ? `${stats.questions} 道题` : "完成练习后显示"}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">当前薄弱学科</span>
          <strong className="stat-subject">{weak[0]?.[0] ?? "—"}</strong>
          <span className="stat-note">{weak[0] ? `${weak[0][1]} 道错题` : "暂无错题记录"}</span>
        </div>
      </section>

      <div className="dashboard-columns">
        <section className="dashboard-section" aria-labelledby="recent-title">
          <div className="section-heading">
            <div>
              <p className="section-kicker">练习轨迹</p>
              <h2 id="recent-title">最近练习</h2>
            </div>
            <Link className="text-link" to="/practice">查看全部 <span aria-hidden="true">→</span></Link>
          </div>
          {loading && !data.history ? (
            <p className="empty-state">正在读取学习记录…</p>
          ) : recent.length ? (
            <ul className="activity-list">
              {recent.map((item) => (
                <li className="activity-row" key={item.attempt_id}>
                  <div className="activity-main">
                    <strong>{item.topic || "未命名练习"}</strong>
                    <span>{item.subject} · {formatDate(item.created_at, shanghaiDateTime)}</span>
                  </div>
                  <div className="activity-result">
                    <strong>{item.score} 分</strong>
                    <span>{item.correct_count}/{item.total_count} 正确</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="empty-state">近 7 天还没有练习记录。</p>
          )}
        </section>

        <section className="dashboard-section rhythm-section" aria-labelledby="rhythm-title">
          <div className="section-heading">
            <div>
              <p className="section-kicker">持续学习</p>
              <h2 id="rhythm-title">每日练习</h2>
            </div>
          </div>
          {daily.length ? (
            <div className="activity-bars" aria-label="每天练习次数">
              {daily.slice(0, 7).reverse().map((day) => (
                <div className="activity-day" key={day.date}>
                  <span className="bar-count">{day.attempts}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ height: `${Math.max(8, (day.attempts / maxDaily) * 100)}%` }} />
                  </div>
                  <span className="bar-date">{formatDate(day.date, shanghaiDate)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-state">完成一次练习后，这里会显示每日学习节奏。</p>
          )}
          <div className="weak-list">
            <h3>需要巩固</h3>
            {weak.length ? weak.map(([subject, count]) => (
              <div className="weak-row" key={subject}>
                <span>{subject}</span><strong>{count} 道错题</strong>
              </div>
            )) : <p className="subtle-empty">错题会在这里按学科汇总。</p>}
          </div>
        </section>
      </div>
    </section>
  );
}

export default Dashboard;
