import { useCallback, useEffect, useState } from "react";
import {
  buildTeacherKnowledgebase, createTeacherKnowledgebase, generateTeacherQuestions,
  getTeacherKnowledgebase, getTeacherKnowledgebases, getTeacherQuestionAnalytics,
  getTeacherQuestions, reviewTeacherQuestion, uploadTeacherPdf,
} from "./api.js";

const subjects = ["计算机", "英语", "数学", "其他"];

function TeacherPage() {
  const [libraries, setLibraries] = useState([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState("计算机");
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(5);
  const [drafts, setDrafts] = useState([]);
  const [approved, setApproved] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [draftSubject, setDraftSubject] = useState("");
  const [draftQuery, setDraftQuery] = useState("");
  const [draftLimit, setDraftLimit] = useState(5);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const filteredDrafts = drafts.filter((item) => {
    const query = draftQuery.trim().toLocaleLowerCase();
    const text = `${item.topic || ""} ${item.stem} ${item.source}`.toLocaleLowerCase();
    return (!draftSubject || item.subject === draftSubject) && (!query || text.includes(query));
  });

  const refresh = useCallback(async (libraryName = selected) => {
    const [all, pending, published, stats] = await Promise.all([
      getTeacherKnowledgebases(), getTeacherQuestions("draft"),
      getTeacherQuestions("approved"), getTeacherQuestionAnalytics(),
    ]);
    setLibraries(all);
    setDrafts(pending);
    setApproved(published);
    setAnalytics(Object.fromEntries(stats.map((item) => [item.question_id, item])));
    const next = libraryName || all[0]?.name || "";
    if (next) {
      setSelected(next);
      setDetail(await getTeacherKnowledgebase(next));
    } else {
      setSelected("");
      setDetail(null);
    }
  }, [selected]);

  useEffect(() => {
    refresh("").catch((requestError) => setError(requestError.message));
  }, []);

  async function run(action, success) {
    setWorking(true);
    setError("");
    setNotice("");
    try {
      await action();
      if (success) setNotice(success);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function createLibrary(event) {
    event.preventDefault();
    await run(async () => {
      const result = await createTeacherKnowledgebase({ name: name.trim() });
      setName("");
      await refresh(result.name);
    }, "资料库已创建。");
  }

  async function uploadFile(event) {
    event.preventDefault();
    if (!selected || !file) return;
    const form = event.currentTarget;
    await run(async () => {
      await uploadTeacherPdf(selected, file);
      setFile(null);
      form.reset();
      await refresh(selected);
    }, "PDF 已上传，请构建索引后供学生检索。");
  }

  async function buildIndex() {
    if (!selected) return;
    await run(async () => {
      await buildTeacherKnowledgebase(selected);
      await refresh(selected);
    }, "索引构建完成，资料库已发布给学生使用。");
  }

  async function generate(event) {
    event.preventDefault();
    if (!selected || !topic.trim()) return;
    await run(async () => {
      const created = await generateTeacherQuestions({
        knowledgebase: selected, subject, topic: topic.trim(), count: Number(count),
        source: "library", mode: "hybrid",
      });
      if (!created.length) throw new Error("当前索引没有召回到相关资料，请换知识点或检查资料内容。");
      await refresh(selected);
    }, "题目草稿已生成，审核通过后会进入学生题库。");
  }

  function editDraft(id, field, value) {
    setDrafts((items) => items.map((item) => item.id !== id ? item : ({ ...item,
      [field]: field === "options" ? { ...item.options, ...value } : value,
    })));
  }

  async function decide(question, status) {
    await run(async () => {
      await reviewTeacherQuestion(question.id, {
        stem: question.stem, options: "ABCD".split("").map((letter) => question.options[letter]),
        answer: question.answer, explanation: question.explanation, subject: question.subject || subject,
        topic: question.topic || topic, status,
      });
      await refresh(selected);
    }, status === "approved" ? "题目已批准，学生可以在练习中使用。" : "题目已驳回。");
  }

  return (
    <section className="page-frame teacher-page" aria-labelledby="page-title">
      <header className="page-heading teacher-heading">
        <div><p className="eyebrow">教师工作台 · 单机教学流程</p><h1 id="page-title">教学准备</h1></div>
        <button className="quiet-button" type="button" onClick={() => refresh().catch((e) => setError(e.message))} disabled={working}>刷新</button>
      </header>
      {error && <div className="dashboard-error" role="alert"><span>{error}</span><button className="text-button" onClick={() => setError("")}>关闭</button></div>}
      {notice && <div className="teacher-notice" role="status">{notice}</div>}

      <section className="teacher-section" aria-labelledby="library-title">
        <div className="teacher-section-heading"><div><p className="section-kicker">01 / 资料</p><h2 id="library-title">资料库入库</h2></div>
          <span>{libraries.length} 个资料库</span></div>
        <form className="teacher-create-library" onSubmit={createLibrary}>
          <label className="practice-field"><span>新资料库名称</span><input value={name} onChange={(e) => setName(e.target.value)} maxLength={128} placeholder="例如：高等数学上册" /></label>
          <button className="quiet-button" disabled={working || !name.trim()}>创建资料库</button>
        </form>
        <div className="teacher-library-workflow">
          <label className="practice-field"><span>当前资料库</span><select value={selected} onChange={(e) => { setSelected(e.target.value); getTeacherKnowledgebase(e.target.value).then(setDetail).catch((err) => setError(err.message)); }}>
            <option value="">选择资料库</option>{libraries.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
          </select></label>
          <form className="teacher-upload" onSubmit={uploadFile}>
            <label className="practice-field"><span>上传 PDF（最大 50 MB）</span><input type="file" accept="application/pdf,.pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} disabled={!selected} /></label>
            <button className="quiet-button" disabled={working || !selected || !file}>上传</button>
          </form>
          <button className="primary-button" type="button" onClick={buildIndex} disabled={working || !selected || !detail?.files?.some((item) => item.status !== "indexed")}>
            {working ? "处理中…" : "构建并发布索引"}
          </button>
        </div>
        {detail && <div className="teacher-library-details">
          <span>当前版本：{detail.knowledgebase.current_generation ? detail.knowledgebase.current_generation.slice(0, 10) : "尚未发布"}</span>
          <span>{detail.knowledgebase.chunk_count || 0} 个切片</span>
          <span>{detail.files.length} 份 PDF</span>
          {detail.files.map((item) => <span className={`teacher-file-status ${item.status}`} key={item.source}>{item.source} · {item.status === "indexed" ? `${item.chunk_count} 切片` : item.status}</span>)}
        </div>}
      </section>

      <section className="teacher-section" aria-labelledby="draft-title">
        <div className="teacher-section-heading"><div><p className="section-kicker">02 / 命题</p><h2 id="draft-title">生成并审核题目</h2></div><span>{drafts.length} 道待审核 · {approved.length} 道已批准</span></div>
        <form className="teacher-generate" onSubmit={generate}>
          <label className="practice-field"><span>学科</span><select value={subject} onChange={(e) => setSubject(e.target.value)}>{subjects.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label className="practice-field"><span>知识点</span><input value={topic} onChange={(e) => setTopic(e.target.value)} maxLength={2000} placeholder="输入本次命题范围" /></label>
          <label className="practice-field"><span>题量</span><input type="number" min="1" max="8" value={count} onChange={(e) => setCount(e.target.value)} /></label>
          <button className="primary-button" disabled={working || !selected || !detail?.knowledgebase?.current_generation || !topic.trim()}>生成草稿</button>
        </form>
        {!detail?.knowledgebase?.current_generation && <p className="teacher-empty">先上传教材 PDF 并成功构建索引，才能根据资料出题。</p>}
        <div className="teacher-draft-controls">
          <label className="practice-field"><span>按学科筛选</span><select value={draftSubject} onChange={(e) => { setDraftSubject(e.target.value); setDraftLimit(5); }}><option value="">全部学科</option>{subjects.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label className="practice-field"><span>搜索知识点或题干</span><input value={draftQuery} onChange={(e) => { setDraftQuery(e.target.value); setDraftLimit(5); }} placeholder="输入关键词" /></label>
          <span className="teacher-draft-count">显示 {Math.min(draftLimit, filteredDrafts.length)} / {filteredDrafts.length}</span>
        </div>
        <div className="teacher-question-list">
          {filteredDrafts.slice(0, draftLimit).map((question) => <article className="teacher-question" key={question.id}>
            <div className="teacher-question-meta"><strong>草稿 #{question.id}</strong><span>{question.source}{question.page ? ` · 第 ${question.page} 页` : ""}</span></div>
            <label className="practice-field"><span>题干</span><textarea rows="2" value={question.stem} onChange={(e) => editDraft(question.id, "stem", e.target.value)} /></label>
            <div className="teacher-options">{"ABCD".split("").map((letter) => <label className="practice-field" key={letter}><span>选项 {letter}</span><input value={question.options[letter]} onChange={(e) => editDraft(question.id, "options", { [letter]: e.target.value })} /></label>)}</div>
            <div className="teacher-review-fields"><label className="practice-field"><span>正确答案</span><select value={question.answer} onChange={(e) => editDraft(question.id, "answer", e.target.value)}>{"ABCD".split("").map((letter) => <option key={letter}>{letter}</option>)}</select></label>
              <label className="practice-field"><span>解析</span><textarea rows="2" value={question.explanation} onChange={(e) => editDraft(question.id, "explanation", e.target.value)} /></label></div>
            <div className="teacher-question-actions"><button className="quiet-button" type="button" disabled={working} onClick={() => decide(question, "rejected")}>驳回</button><button className="primary-button" type="button" disabled={working} onClick={() => decide(question, "approved")}>保存并批准</button></div>
          </article>)}
          {!filteredDrafts.length && <p className="teacher-empty">没有符合筛选条件的待审核题目。</p>}
        </div>
        {filteredDrafts.length > draftLimit && <button className="quiet-button teacher-load-more" type="button" onClick={() => setDraftLimit((limit) => limit + 5)}>再显示 5 道</button>}
      </section>

      <section className="teacher-section" aria-labelledby="feedback-title">
        <div className="teacher-section-heading"><div><p className="section-kicker">03 / 学习反馈</p><h2 id="feedback-title">已批准题目使用情况</h2></div></div>
        {approved.length ? <div className="teacher-approved-list">{approved.map((question) => { const stats = analytics[question.id] || {}; return <article key={question.id}>
          <div><strong>{question.topic || question.stem}</strong><span>{question.subject || "未分类"} · {question.stem}</span></div>
          <small>{stats.attempts || 0} 次作答 · {stats.learner_count || 0} 位学习者 · {stats.accuracy == null ? "暂无正确率" : `${stats.accuracy}% 正确`}</small>
        </article>; })}</div> : <p className="teacher-empty">批准后的题目会进入学生练习题库，并在这里显示匿名汇总。</p>}
      </section>
      <p className="teacher-scope-note">当前为单机教学闭环：题库在本地共享，练习统计只展示匿名汇总；班级、作业分发和教师账号权限尚未实现。</p>
    </section>
  );
}

export default TeacherPage;
