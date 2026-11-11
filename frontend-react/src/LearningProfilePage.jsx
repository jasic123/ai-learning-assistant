import { useEffect, useState } from "react";
import { getLearningProfile } from "./api.js";
import { errorReasonLabels } from "./learningProfile.js";

function LearningProfilePage() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getLearningProfile(controller.signal)
      .then(setProfile)
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const topics = profile?.topics ?? [];
  const weak = topics.filter((item) => item.status === "needs_review");
  const early = topics.filter((item) => item.status === "sample_limited" && item.incorrect > 0);

  return (
    <section className="page-frame learning-profile-page" aria-labelledby="page-title">
      <header className="page-heading dashboard-heading">
        <div>
          <p className="eyebrow">基于长期作答记录</p>
          <h1 id="page-title">学习画像</h1>
        </div>
        <span className="profile-data-note">只总结记录，不推断性格</span>
      </header>

      {error && <div className="dashboard-error" role="alert">{error}</div>}
      {loading ? <p className="empty-state">正在读取学习记录…</p> : profile && (
        <>
          <section className="profile-stats" aria-label="学习记录概况">
            <div><span>累计作答</span><strong>{profile.total_questions}</strong></div>
            <div><span>练习次数</span><strong>{profile.total_attempts}</strong></div>
            <div><span>总体正确率</span><strong>{profile.overall_accuracy == null ? "—" : `${profile.overall_accuracy}%`}</strong></div>
          </section>

          <section className="profile-section" aria-labelledby="weak-topics-title">
            <div className="section-heading">
              <div><p className="section-kicker">多次作答后形成的信号</p><h2 id="weak-topics-title">需要巩固的知识点</h2></div>
              <span className="inline-count">{weak.length}</span>
            </div>
            {weak.length ? (
              <div className="profile-topic-list">
                {weak.map((item) => (
                  <article className="profile-topic-row" key={`${item.subject}:${item.topic}`}>
                    <div className="profile-topic-main">
                      <strong>{item.topic}</strong>
                      <span>{item.subject} · {item.incorrect}/{item.attempts} 次答错 · 记录 {item.attempts} 题</span>
                    </div>
                    <div className="profile-topic-score">
                      <strong>{item.accuracy}%</strong>
                      <span>正确率</span>
                    </div>
                  </article>
                ))}
              </div>
            ) : <p className="subtle-empty">目前没有达到关注门槛的知识点；每个知识点至少积累 3 次作答后才会评估。</p>}
          </section>

          {early.length > 0 && (
            <section className="profile-section" aria-labelledby="early-topics-title">
              <div className="section-heading">
                <div><p className="section-kicker">样本还少，暂不下结论</p><h2 id="early-topics-title">正在积累记录</h2></div>
              </div>
              <div className="profile-topic-list">
                {early.map((item) => (
                  <article className="profile-topic-row" key={`${item.subject}:${item.topic}`}>
                    <div className="profile-topic-main">
                      <strong>{item.topic}</strong><span>{item.subject} · {item.incorrect} 次答错 · 当前 {item.attempts} 题</span>
                    </div>
                    <span className="profile-evidence">样本不足</span>
                  </article>
                ))}
              </div>
            </section>
          )}

          <section className="profile-section" aria-labelledby="error-patterns-title">
            <div className="section-heading">
              <div><p className="section-kicker">答错后由学生自选</p><h2 id="error-patterns-title">自我报告的错因</h2></div>
              <span className="profile-data-note">共 {profile.reason_sample_count} 条记录</span>
            </div>
            {profile.error_patterns.length ? (
              <div className="profile-reason-list">
                {profile.error_patterns.map((item) => (
                  <div className="profile-reason-row" key={item.reason}>
                    <span>{errorReasonLabels[item.reason] || item.reason}</span>
                    <strong>{item.count} 次</strong>
                    {item.evidence === "sample_limited" && <small>样本不足</small>}
                  </div>
                ))}
              </div>
            ) : <p className="subtle-empty">答错后可以自选错因；留空也可以，画像不会替你猜原因。</p>}
          </section>

          <p className="profile-disclaimer">{profile.note}</p>
        </>
      )}
    </section>
  );
}

export default LearningProfilePage;
