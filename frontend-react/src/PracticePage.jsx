import { useEffect, useMemo, useState } from "react";
import { getKnowledgebases, replacePracticeQuestion, saveAttemptErrorReasons, startPractice, submitQuiz } from "./api.js";
import { errorReasonOptions } from "./learningProfile.js";

const subjects = ["计算机", "英语", "数学", "其他"];
const retrievalModes = [
  ["hybrid", "混合"],
  ["vector", "语义"],
  ["bm25", "关键词"],
];

function PracticePage() {
  const activeKey = "ai-learning-active-practice";
  const [knowledgebases, setKnowledgebases] = useState([]);
  const [knowledgebase, setKnowledgebase] = useState("");
  const [subject, setSubject] = useState("计算机");
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(5);
  const [mode, setMode] = useState("hybrid");
  const [rerank, setRerank] = useState(false);
  const [vectorWeight, setVectorWeight] = useState("0.5");
  const [keywordWeight, setKeywordWeight] = useState("0.5");
  const [source, setSource] = useState("auto");
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [errorReasons, setErrorReasons] = useState({});
  const [reasonsSaved, setReasonsSaved] = useState(false);
  const [loadingLibraries, setLoadingLibraries] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(activeKey) || "null");
      if (!saved?.quiz?.id || !Array.isArray(saved.quiz.questions)) return;
      setQuiz(saved.quiz);
      setAnswers(saved.answers && typeof saved.answers === "object" ? saved.answers : {});
      if (saved.subject) setSubject(saved.subject);
      if (saved.topic) setTopic(saved.topic);
      if (saved.source) setSource(saved.source);
      if (saved.knowledgebase) setKnowledgebase(saved.knowledgebase);
    } catch {
      sessionStorage.removeItem(activeKey);
    }
  }, []);

  useEffect(() => {
    if (!quiz) return;
    sessionStorage.setItem(activeKey, JSON.stringify({
      quiz, answers, subject, topic, source, knowledgebase,
    }));
  }, [quiz, answers, subject, topic, source, knowledgebase]);

  useEffect(() => {
    const controller = new AbortController();
    getKnowledgebases(controller.signal)
      .then((items) => {
        setKnowledgebases(items);
        setKnowledgebase((current) => current || items[0]?.name || "");
      })
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingLibraries(false);
      });
    return () => controller.abort();
  }, []);

  const totalWeight = Number(vectorWeight) + Number(keywordWeight);
  const weightsValid = [vectorWeight, keywordWeight].every((value) => {
    const number = Number(value);
    return value !== "" && Number.isFinite(number) && number >= 0 && number <= 1;
  });
  const weightError = mode !== "hybrid" ? ""
    : !weightsValid ? "权重需填写 0 到 1 之间的数字。"
      : Math.abs(totalWeight - 1) > 0.000001 ? "两路权重之和需要等于 1。" : "";

  const sourceLabel = useMemo(() => {
    if (source === "bank") return "已审核题库";
    if (source === "web") return "联网搜索";
    if (!knowledgebase) return "";
    return knowledgebases.find((item) => item.name === knowledgebase)?.name ?? knowledgebase;
  }, [knowledgebase, knowledgebases]);
  const gradedResult = result?.data;

  async function generate(event) {
    event.preventDefault();
    if (!knowledgebase && source === "library") {
      setError("请先选择一个已入库的资料库。");
      return;
    }
    if (!topic.trim()) {
      setError("请输入要练习的知识点。");
      return;
    }
    if (weightError) {
      setError(weightError);
      return;
    }
    setWorking(true);
    setError("");
    setResult(null);
    setErrorReasons({});
    setReasonsSaved(false);
    setAnswers({});
    try {
      const nextQuiz = await startPractice({
        knowledgebase,
        subject,
        topic: topic.trim(),
        count: Number(count),
        mode,
        rerank,
        vector_weight: Number(vectorWeight),
        keyword_weight: Number(keywordWeight),
        source,
      });
      if (!nextQuiz.questions?.length) {
        setQuiz(null);
        setError("没有检索到足够的资料来生成练习，请调整知识点或检查资料库内容。");
      } else {
        setQuiz(nextQuiz);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function replaceQuestion(question) {
    if (!quiz) return;
    setWorking(true);
    setError("");
    try {
      const result = await replacePracticeQuestion(quiz.id, {
        question_id: question.id, knowledgebase, subject, topic: topic.trim(), source,
      });
      setQuiz((current) => {
        const questions = [...current.questions];
        questions[questions.findIndex((item) => item.id === question.id)] = result.question;
        return { ...current, questions };
      });
      setAnswers((current) => {
        const next = { ...current };
        delete next[question.id];
        return next;
      });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!quiz) return;
    if (Object.keys(answers).length !== quiz.questions.length) {
      setError("请回答所有题目后提交。");
      return;
    }
    setWorking(true);
    setError("");
    try {
      const graded = await submitQuiz(quiz.id, answers);
      setResult({ data: graded, subject: quiz.subject, topic: quiz.topic });
      setErrorReasons({});
      setReasonsSaved(false);
      setQuiz(null);
      sessionStorage.removeItem(activeKey);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function saveReasons() {
    if (!result?.data.attempt_id || !Object.keys(errorReasons).length) return;
    setWorking(true);
    setError("");
    try {
      await saveAttemptErrorReasons(result.data.attempt_id, errorReasons);
      setReasonsSaved(true);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  return (
    <section className="page-frame practice-page" aria-labelledby="page-title">
      <header className="page-heading">
        <p className="eyebrow">按知识点练习</p>
        <h1 id="page-title">练习</h1>
      </header>

      {error && (
        <div className="dashboard-error" role="alert">
          <span>{error}</span>
          <button className="text-button" type="button" onClick={() => setError("")}>关闭</button>
        </div>
      )}

      <form className="practice-form" onSubmit={generate}>
        <div className="practice-form-heading">
          <div>
            <p className="section-kicker">出题范围</p>
            <h2>生成一组选择题</h2>
          </div>
          <span className="form-source">{sourceLabel || (loadingLibraries ? "读取资料库…" : "未选择资料库")}</span>
        </div>

        <div className="practice-fields">
          <label className="practice-field field-library">
            <span>资料库</span>
            <select value={knowledgebase} onChange={(event) => setKnowledgebase(event.target.value)}
                    disabled={loadingLibraries || !knowledgebases.length || source === "web" || source === "bank"}>
              <option value="">选择资料库</option>
              {knowledgebases.map((item) => (
                <option key={item.name} value={item.name}>{item.name}</option>
              ))}
            </select>
          </label>
          <label className="practice-field field-subject">
            <span>学科</span>
            <select value={subject} onChange={(event) => setSubject(event.target.value)}>
              {subjects.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="practice-field field-count">
            <span>题量</span>
            <input type="number" min="1" max="8" step="1" value={count}
                   onChange={(event) => setCount(event.target.value)} />
          </label>
          <label className="practice-field field-topic">
            <span>知识点</span>
            <input value={topic} onChange={(event) => setTopic(event.target.value)}
                   maxLength={2000} placeholder="例如：Python 迭代器协议" />
          </label>
          <label className="practice-field field-source">
            <span>出题来源</span>
            <select value={source} onChange={(event) => setSource(event.target.value)}>
              <option value="auto">自动补充</option>
              <option value="library">资料库</option>
              <option value="bank">已审核题库</option>
              <option value="web">联网搜索</option>
            </select>
          </label>
        </div>

        <div className="practice-options-row">
          <fieldset className="mode-fieldset">
            <legend>检索方式 {source === "web" ? "（联网搜索不使用资料库检索）" : source === "bank" ? "（已审核题库不使用资料库检索）" : ""}</legend>
            <div className="mode-segment">
              {retrievalModes.map(([value, label]) => (
                <button key={value} type="button" className={mode === value ? "selected" : ""}
                        aria-pressed={mode === value} disabled={source === "web" || source === "bank"}
                        onClick={() => setMode(value)}>
                  {label}
                </button>
              ))}
            </div>
          </fieldset>
          {mode === "hybrid" && source !== "web" && source !== "bank" && (
            <div className="weight-controls">
              <label className="practice-field weight-field">
                <span>语义权重</span>
                <input type="number" min="0" max="1" step="0.05" value={vectorWeight}
                       onChange={(event) => setVectorWeight(event.target.value)} />
              </label>
              <label className="practice-field weight-field">
                <span>关键词权重</span>
                <input type="number" min="0" max="1" step="0.05" value={keywordWeight}
                       onChange={(event) => setKeywordWeight(event.target.value)} />
              </label>
              <span className={`weight-sum${weightError ? " invalid" : ""}`}>
                合计 {Number.isFinite(totalWeight) ? totalWeight.toFixed(2) : "—"}
              </span>
            </div>
          )}
          <label className={`rerank-toggle${source === "web" || source === "bank" ? " disabled" : ""}`}>
            <input type="checkbox" checked={rerank} disabled={source === "web" || source === "bank"}
                   onChange={(event) => setRerank(event.target.checked)} />
            <span>启用精排</span>
          </label>
          <button className="primary-button generate-practice" type="submit"
                  disabled={working || ((source === "library" || source === "auto") && loadingLibraries)
                    || ((source === "library" || source === "auto") && !knowledgebases.length) || Boolean(weightError)}>
            {working ? "处理中…" : "生成练习"}<span aria-hidden="true">→</span>
          </button>
        </div>
      </form>

      {quiz && (
        <form className="practice-quiz" onSubmit={submit}>
          <div className="section-heading">
            <div>
              <p className="section-kicker">{quiz.subject} · {quiz.topic}</p>
              <h2>本组题目 <span className="inline-count">{quiz.questions.length}</span></h2>
            </div>
            <button className="text-button" type="button" onClick={() => {
              setQuiz(null); setAnswers({}); sessionStorage.removeItem(activeKey);
            }}>
              放弃本组
            </button>
          </div>
            {quiz.questions.map((question, index) => (
            <fieldset className="practice-question" key={question.id}>
              <legend><span>{index + 1}</span>{question.stem}
                <button className="question-replace" type="button" onClick={() => replaceQuestion(question)} disabled={working}
                        title="不满意，换一道同知识点题目">换一道</button>
              </legend>
              <div className="practice-answer-options">
                {Object.entries(question.options).map(([letter, text]) => (
                  <label className={`practice-answer-option${answers[question.id] === letter ? " selected" : ""}`}
                         key={letter}>
                    <input type="radio" name={`practice-${question.id}`} value={letter}
                           checked={answers[question.id] === letter}
                           onChange={() => setAnswers((current) => ({ ...current, [question.id]: letter }))} />
                    <span className="option-letter">{letter}</span>
                    <span>{text}</span>
                  </label>
                ))}
              </div>
              <small className="question-source">{question.source}{question.page ? ` · 第 ${question.page} 页` : ""}</small>
            </fieldset>
          ))}
          <button className="primary-button submit-practice" type="submit" disabled={working}>
            {working ? "判题中…" : "提交答案"}
          </button>
        </form>
      )}

      {result && (
        <section className="practice-result" aria-live="polite">
          <div className="result-heading">
            <div>
              <p className="section-kicker">{result.subject} · {result.topic}</p>
              <h2>{gradedResult.correct_count}/{gradedResult.total_count} 正确 · {gradedResult.score} 分</h2>
            </div>
            <span className={`result-status${gradedResult.correct_count === gradedResult.total_count ? " good" : " needs-work"}`}>
              {gradedResult.correct_count === gradedResult.total_count ? "全部答对" : "错题已加入错题本"}
            </span>
          </div>
          {gradedResult.results.map((item, index) => (
            <article className="result-item" key={item.question_id}>
              <h3>{index + 1}. {item.correct ? "答对" : `答错 · 正确答案 ${item.answer}`}</h3>
              <p>{item.stem}</p>
              <div className="result-explanation">{item.explanation}</div>
              {!item.correct && (
                <label className="error-reason-field">
                  <span>这次主要原因（可选）</span>
                  <select value={errorReasons[item.question_id] || ""}
                          onChange={(event) => {
                            setErrorReasons((current) => {
                              const next = { ...current };
                              if (event.target.value) next[item.question_id] = event.target.value;
                              else delete next[item.question_id];
                              return next;
                            });
                            setReasonsSaved(false);
                          }} disabled={reasonsSaved}>
                    <option value="">暂不记录</option>
                    {errorReasonOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </label>
              )}
              <small>{item.source}{item.page ? ` · 第 ${item.page} 页` : ""}</small>
            </article>
          ))}
          {gradedResult.results.some((item) => !item.correct) && (
            <div className="error-reason-actions">
              <p>{reasonsSaved ? "错因已记录，可在学习画像中查看。" : "自选错因会用于总结学习记录，不会用一次答错给你贴标签。"}</p>
              <button className="quiet-button" type="button" onClick={saveReasons}
                      disabled={working || reasonsSaved || !Object.keys(errorReasons).length}>
                {reasonsSaved ? "已保存" : "保存错因"}
              </button>
            </div>
          )}
        </section>
      )}
    </section>
  );
}

export default PracticePage;
