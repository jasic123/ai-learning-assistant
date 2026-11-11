import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getDueReviews, getMistakes, importMistake, recognizeMistakeImage,
  saveAttemptErrorReasons, startReview, structureMistakeText, submitQuiz,
} from "./api.js";
import { errorReasonOptions } from "./learningProfile.js";

function MistakesPage() {
  const [subject, setSubject] = useState("");
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [allMistakes, setAllMistakes] = useState([]);
  const [mistakes, setMistakes] = useState([]);
  const [due, setDue] = useState([]);
  const [count, setCount] = useState(5);
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [errorReasons, setErrorReasons] = useState({});
  const [reasonsSaved, setReasonsSaved] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [ocrText, setOcrText] = useState("");
  const [importSubject, setImportSubject] = useState("其他");
  const [importTopic, setImportTopic] = useState("");
  const [importStem, setImportStem] = useState("");
  const [importOptions, setImportOptions] = useState({ A: "", B: "", C: "", D: "" });
  const [importAnswer, setImportAnswer] = useState("");
  const [importExplanation, setImportExplanation] = useState("");
  const [importWorking, setImportWorking] = useState(false);
  const [importNote, setImportNote] = useState("");
  const [structureOutput, setStructureOutput] = useState("");
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");

  const subjects = useMemo(
    () => [...new Set(allMistakes.map((item) => item.subject).filter(Boolean))].sort(),
    [allMistakes],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [allItems, items, dueItems] = await Promise.all([
        getMistakes({}, undefined),
        getMistakes({ subject, query }),
        getDueReviews(),
      ]);
      setAllMistakes(allItems);
      setMistakes(items);
      setDue(dueItems);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [subject, query]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!imageFile) {
      setImagePreview("");
      return undefined;
    }
    const url = URL.createObjectURL(imageFile);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  const dueForSubject = useMemo(
    () => subject ? due.filter((item) => item.subject === subject) : due,
    [due, subject],
  );
  const maxReviewCount = dueForSubject.length;
  const requestedCount = Number(count);
  const countError = count === "" || !Number.isInteger(requestedCount) || requestedCount < 1
    ? "请输入大于 0 的整数。"
    : requestedCount > 100
        ? "单次最多复习 100 道题。"
        : requestedCount > maxReviewCount
          ? `当前学科只有 ${maxReviewCount} 道到期题，请减少题量。`
          : "";

  async function beginReview() {
    if (!subject) {
      setError("请选择复习学科。");
      return;
    }
    if (!maxReviewCount) {
      setError(`${subject}目前没有到期错题。`);
      return;
    }
    if (countError) {
      setError(countError);
      return;
    }
    setWorking(true);
    setError("");
    setResult(null);
    setErrorReasons({});
    setReasonsSaved(false);
    try {
      const nextQuiz = await startReview({ subject, count: Number(count) });
      if (!nextQuiz.questions?.length) {
        setQuiz(null);
        setAnswers({});
        setError(nextQuiz.message || `${subject}目前没有到期错题。`);
      } else {
        setQuiz(nextQuiz);
        setAnswers({});
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function finishReview() {
    if (!quiz) return;
    if (Object.keys(answers).length !== quiz.questions.length) {
      setError("请先回答本组所有题目。");
      return;
    }
    setWorking(true);
    setError("");
    try {
      const submitted = await submitQuiz(quiz.id, answers);
      setResult(submitted);
      setErrorReasons({});
      setReasonsSaved(false);
      setQuiz(null);
      setAnswers({});
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function saveReasons() {
    if (!result?.attempt_id || !Object.keys(errorReasons).length) return;
    setWorking(true);
    setError("");
    try {
      await saveAttemptErrorReasons(result.attempt_id, errorReasons);
      setReasonsSaved(true);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  function applySearch(event) {
    event.preventDefault();
    setQuery(queryInput.trim());
  }

  function applyStructuredFields(data) {
    setImportSubject(data.subject || "其他");
    setImportTopic(data.topic || "");
    setImportStem(data.stem || "");
    setImportOptions({ A: data.options?.A || "", B: data.options?.B || "",
      C: data.options?.C || "", D: data.options?.D || "" });
    setImportAnswer(data.answer || "");
    setImportExplanation(data.explanation || "");
    setImportNote(data.message || "字段已整理，请核对后保存。");
  }

  async function recognizeAndStructure() {
    if (!imageFile && !ocrText.trim()) {
      setImportNote("请选择试题图片，或粘贴题目原文。");
      return;
    }
    setImportWorking(true);
    setImportNote("");
    try {
      let sourceText = ocrText;
      if (imageFile) {
        const recognized = await recognizeMistakeImage(imageFile);
        sourceText = recognized.text || "";
        setOcrText(sourceText);
      }
      if (!sourceText.trim()) {
        setImportNote("没有识别到文字，请换一张更清晰的图片，或直接粘贴题目原文。");
        return;
      }
      setStructureOutput("");
      setImportNote(imageFile ? "图片 OCR 已完成，AI 正在整理字段…" : "AI 正在整理字段…");
      const fields = await structureMistakeText(sourceText, (event) => {
        if (event.type === "token") setStructureOutput((current) => current + event.content);
      });
      applyStructuredFields(fields);
    } catch (requestError) {
      setImportNote(requestError.message);
    } finally {
      setImportWorking(false);
    }
  }

  async function saveImportedMistake(event) {
    event.preventDefault();
    if (!importSubject || !importTopic.trim() || !importStem.trim()
      || Object.values(importOptions).some((option) => !option.trim())
      || !importAnswer || !importExplanation.trim()) {
      setImportNote("请补齐学科、知识点、完整题干、四个选项、正确答案和解析。");
      return;
    }
    setImportWorking(true);
    setImportNote("");
    try {
      const saved = await importMistake({
        subject: importSubject,
        topic: importTopic.trim(),
        stem: importStem.trim(),
        options: [importOptions.A.trim(), importOptions.B.trim(), importOptions.C.trim(), importOptions.D.trim()],
        answer: importAnswer,
        explanation: importExplanation.trim(),
        ocr_text: ocrText,
      });
      setImportNote(`已保存错题 #${saved.question.id}。`);
      setImageFile(null);
      setOcrText("");
      setImportTopic("");
      setImportStem("");
      setImportOptions({ A: "", B: "", C: "", D: "" });
      setImportAnswer("");
      setImportExplanation("");
      await refresh();
    } catch (requestError) {
      setImportNote(requestError.message);
    } finally {
      setImportWorking(false);
    }
  }

  return (
    <section className="page-frame mistakes-page" aria-labelledby="page-title">
      <header className="page-heading dashboard-heading">
        <div>
          <p className="eyebrow">主动回忆 · 间隔复习</p>
          <h1 id="page-title">错题与复习</h1>
        </div>
        <button className="quiet-button" type="button" onClick={refresh} disabled={loading}>
          {loading ? "更新中…" : "刷新"}
        </button>
      </header>

      {error && (
        <div className="dashboard-error" role="alert">
          <span>{error}</span>
          <button className="text-button" type="button" onClick={() => setError("")}>关闭</button>
        </div>
      )}

      <section className="mistake-import" aria-labelledby="mistake-import-title">
        <div className="section-heading mistake-import-heading">
          <div>
            <p className="section-kicker">个人题库</p>
            <h2 id="mistake-import-title">上传试题</h2>
          </div>
          <p>图片 OCR 识别后自动整理字段，核对后保存到错题本。</p>
        </div>
        <div className="mistake-import-grid">
          <div className="mistake-import-source">
            <label className="practice-field">
              <span>试题图片</span>
              <input type="file" accept="image/png,image/jpeg,image/webp,image/bmp"
                     onChange={(event) => setImageFile(event.target.files?.[0] ?? null)} disabled={importWorking} />
            </label>
            {imagePreview && <img className="mistake-image-preview" src={imagePreview} alt="待识别的试题图片预览" />}
            <label className="practice-field">
              <span>识别原文</span>
              <textarea value={ocrText} onChange={(event) => setOcrText(event.target.value)} rows={7}
                        maxLength={30000} placeholder="上传图片后点击识别，也可以直接粘贴题目原文。" />
            </label>
            <button className="quiet-button" type="button" onClick={recognizeAndStructure} disabled={importWorking}>
              {importWorking ? "识别整理中…" : imageFile ? "识别并整理字段" : "自动整理字段"}
            </button>
            {importWorking && <p className="ocr-progress" role="status">{imageFile && !ocrText ? "本地 OCR 正在识别图片…" : "AI 整理中，正在接收模型输出…"}</p>}
            {structureOutput && <details className="structure-stream"><summary>查看实时整理内容</summary><pre>{structureOutput}</pre></details>}
          </div>
          <form className="mistake-import-fields" onSubmit={saveImportedMistake}>
            <div className="mistake-import-meta">
              <label className="practice-field">
                <span>学科</span>
                <select value={importSubject} onChange={(event) => setImportSubject(event.target.value)} disabled={importWorking}>
                  {["计算机", "英语", "数学", "其他"].map((item) => <option key={item}>{item}</option>)}
                </select>
              </label>
              <label className="practice-field">
                <span>知识点</span>
                <input value={importTopic} onChange={(event) => setImportTopic(event.target.value)}
                       maxLength={2000} placeholder="例如：贪心算法" disabled={importWorking} />
              </label>
            </div>
            <label className="practice-field">
              <span>题干</span>
              <textarea value={importStem} onChange={(event) => setImportStem(event.target.value)}
                        rows={3} maxLength={4000} placeholder="完整题干，包含公式、条件和数据" disabled={importWorking} />
            </label>
            <div className="mistake-import-options">
              {Object.keys(importOptions).map((letter) => (
                <label className="practice-field" key={letter}>
                  <span>选项 {letter}</span>
                  <input value={importOptions[letter]} onChange={(event) => setImportOptions((current) => ({ ...current, [letter]: event.target.value }))}
                         maxLength={2000} disabled={importWorking} />
                </label>
              ))}
            </div>
            <div className="mistake-import-meta">
              <label className="practice-field">
                <span>正确答案</span>
                <select value={importAnswer} onChange={(event) => setImportAnswer(event.target.value)} disabled={importWorking}>
                  <option value="">选择答案</option>
                  {["A", "B", "C", "D"].map((item) => <option key={item}>{item}</option>)}
                </select>
              </label>
              <label className="practice-field">
                <span>答案解析</span>
                <textarea value={importExplanation} onChange={(event) => setImportExplanation(event.target.value)}
                          rows={3} maxLength={8000} placeholder="核对或补充解析" disabled={importWorking} />
              </label>
            </div>
            <div className="mistake-import-footer">
              <p className="import-note" role="status">{importNote}</p>
              <button className="primary-button" type="submit" disabled={importWorking}>
                {importWorking ? "保存中…" : "保存到错题本"}
              </button>
            </div>
          </form>
        </div>
      </section>

      <section className="review-panel" aria-labelledby="review-title">
        <div className="review-panel-copy">
          <p className="section-kicker">今日复习</p>
          <h2 id="review-title">{dueForSubject.length} 道到期题</h2>
          <p>按学科随机抽题；当前有 {maxReviewCount} 道到期题。</p>
        </div>
        <div className="review-controls">
          <label className="control-label">
            <span>学科</span>
            <select value={subject} onChange={(event) => setSubject(event.target.value)}>
              <option value="">选择学科</option>
              {subjects.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className="control-label count-control">
            <span>题量</span>
            <input
              type="number"
              min="1"
              max="100"
              step="1"
              value={count}
              onChange={(event) => setCount(event.target.value)}
              disabled={!maxReviewCount}
              aria-invalid={Boolean(countError)}
              aria-describedby={countError ? "review-count-error" : undefined}
            />
            <span className={`field-error${countError ? "" : " field-error-empty"}`}
                  id="review-count-error" aria-hidden={!countError}>
              {countError || " "}
            </span>
          </label>
          <button className="primary-button" type="button" onClick={beginReview}
                  disabled={working || loading || !subject || !maxReviewCount || Boolean(countError)}>
            {working ? "处理中…" : "开始复习"}<span aria-hidden="true">→</span>
          </button>
        </div>
      </section>

      {quiz && (
        <section className="quiz-panel" aria-labelledby="quiz-title">
          <div className="section-heading">
            <div>
              <p className="section-kicker">{quiz.subject} · {quiz.questions.length} 道</p>
              <h2 id="quiz-title">复习作答</h2>
            </div>
            <button className="text-button" type="button" onClick={() => setQuiz(null)}>退出本组</button>
          </div>
          <div className="review-questions">
            {quiz.questions.map((question, index) => (
              <fieldset className="review-question" key={`${question.id}-${index}`}>
                <legend><span>{index + 1}</span>{question.stem}</legend>
                <div className="review-options">
                  {Object.entries(question.options).map(([letter, text]) => (
                    <label className={`review-option${answers[question.id] === letter ? " selected" : ""}`} key={letter}>
                      <input
                        type="radio"
                        name={`review-${question.id}`}
                        value={letter}
                        checked={answers[question.id] === letter}
                        onChange={() => setAnswers((current) => ({ ...current, [question.id]: letter }))}
                      />
                      <span className="option-letter">{letter}</span>
                      <span>{text}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
          </div>
          <button className="primary-button submit-review" type="button" onClick={finishReview} disabled={working}>
            {working ? "判题中…" : "提交本组答案"}
          </button>
        </section>
      )}

      {result && (
        <section className="review-result" aria-live="polite">
          <div className="result-heading">
            <div>
              <p className="section-kicker">本次复习结果</p>
              <h2>{result.correct_count}/{result.total_count} 正确 · {result.score} 分</h2>
            </div>
            <span className={result.correct_count === result.total_count ? "result-status good" : "result-status needs-work"}>
              {result.correct_count === result.total_count ? "已掌握" : "继续巩固"}
            </span>
          </div>
          {result.results.map((item) => (
            <article className="result-item" key={item.question_id}>
              <h3>{item.correct ? "答对" : `答错 · 正确答案 ${item.answer}`}</h3>
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
          {result.results.some((item) => !item.correct) && (
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

      <section className="mistake-list-section" aria-labelledby="mistakes-title">
        <div className="section-heading mistake-list-heading">
          <div>
            <p className="section-kicker">个人题库</p>
            <h2 id="mistakes-title">错题本 <span className="inline-count">{mistakes.length}</span></h2>
          </div>
          <form className="mistake-search" onSubmit={applySearch}>
            <label className="sr-only" htmlFor="mistake-search-input">搜索错题关键词</label>
            <input
              id="mistake-search-input"
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              placeholder="搜索题目、知识点或解析"
            />
            <button className="quiet-button" type="submit">搜索</button>
          </form>
        </div>

        {loading ? <p className="empty-state">正在读取错题…</p> : mistakes.length ? (
          <div className="mistake-list">
            {mistakes.map((item) => (
              <article className="mistake-item" key={item.question_id}>
                <div className="mistake-meta">
                  <span>{item.subject}</span>
                  <span>{item.topic}</span>
                  <span>错 {item.wrong_count} 次</span>
                </div>
                <h3>{item.stem}</h3>
                <div className="mistake-options">
                  {Object.entries(item.options).map(([letter, text]) => (
                    <div className={item.answer === letter ? "correct-option" : ""} key={letter}>
                      <span>{letter}.</span> {text}
                    </div>
                  ))}
                </div>
                <p className="answer-line">正确答案：{item.answer}. {item.options[item.answer]}</p>
                <p className="explanation-line">{item.explanation}</p>
                <div className="mistake-footer">
                  <span>{item.source}{item.page ? ` · 第 ${item.page} 页` : ""}</span>
                  <span>下次复习 {new Date(item.due_at).toLocaleDateString("zh-CN", { timeZone: "Asia/Shanghai" })}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-state">没有找到符合条件的错题。</p>
        )}
      </section>
    </section>
  );
}

export default MistakesPage;
