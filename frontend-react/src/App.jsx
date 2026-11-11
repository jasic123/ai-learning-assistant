import { NavLink, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";
import Dashboard from "./Dashboard.jsx";
import MistakesPage from "./MistakesPage.jsx";
import PracticePage from "./PracticePage.jsx";
import StudyPage from "./StudyPage.jsx";
import MindMapPage from "./MindMapPage.jsx";
import LearningProfilePage from "./LearningProfilePage.jsx";
import TeacherPage from "./TeacherPage.jsx";
import { authenticate, getCurrentUser, logout } from "./api.js";

const navigation = [
  { to: "/", label: "今日学习", end: true },
  { to: "/study", label: "AI 学习" },
  { to: "/practice", label: "练习" },
  { to: "/mistakes", label: "错题与复习" },
  { to: "/profile", label: "学习画像" },
  { to: "/organize", label: "知识整理" },
  { to: "/teacher", label: "教学准备" },
];

const pages = {
  "/": "今日学习",
  "/study": "AI 学习",
  "/practice": "练习",
  "/mistakes": "错题与复习",
  "/profile": "学习画像",
  "/organize": "知识整理",
  "/teacher": "教学准备",
};

function PageFrame({ title }) {
  return (
    <section className="page-frame" aria-labelledby="page-title">
      <div className="page-heading">
        <p className="eyebrow">AI 学习助手</p>
        <h1 id="page-title">{title}</h1>
      </div>
      <div className="page-surface" />
    </section>
  );
}

function LoginPage({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    setWorking(true);
    setError("");
    try {
      const result = await authenticate(mode, { username, password });
      onLogin(result.user);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  return (
    <main className="auth-page">
      <form className="auth-form" onSubmit={submit}>
        <div className="brand-mark">学</div>
        <p className="eyebrow">AI 学习助手</p>
        <h1>{mode === "login" ? "登录学习空间" : "创建学习账号"}</h1>
        <p className="auth-subtitle">会话、错题和学习记录将保存在你的账号下。</p>
        {error && <div className="dashboard-error" role="alert">{error}</div>}
        <label className="practice-field"><span>用户名</span><input autoComplete="username" required minLength={3} maxLength={32} value={username} onChange={(event) => setUsername(event.target.value)} placeholder="3～32 位字母、数字或下划线" /></label>
        <label className="practice-field"><span>密码</span><input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={8} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="至少 8 位" /></label>
        <button className="primary-button auth-submit" disabled={working}>{working ? "请稍候…" : mode === "login" ? "登录" : "注册并登录"}</button>
        <button className="quiet-button auth-switch" type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
          {mode === "login" ? "首次使用？创建账号" : "已有账号？返回登录"}
        </button>
      </form>
    </main>
  );
}

function App() {
  const [auth, setAuth] = useState({ loading: true, user: null, required: true });

  useEffect(() => {
    const controller = new AbortController();
    getCurrentUser(controller.signal)
      .then((result) => setAuth({ loading: false, user: result.user, required: result.auth_required }))
      .catch((error) => {
        if (error.name !== "AbortError") setAuth({ loading: false, user: null, required: true });
      });
    return () => controller.abort();
  }, []);

  if (auth.loading) return <main className="auth-page"><p>正在检查登录状态…</p></main>;
  if (auth.required && !auth.user) return <LoginPage onLogin={(user) => setAuth({ loading: false, user, required: true })} />;
  if (!auth.user) return <LoginPage onLogin={(user) => setAuth({ loading: false, user, required: false })} />;

  async function signOut() {
    await logout().catch(() => {});
    setAuth({ loading: false, user: null, required: true });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="学习助手首页">
          <span className="brand-mark">学</span>
          <span>学习助手</span>
        </a>
        <nav className="primary-nav" aria-label="主导航">
          {navigation.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className="status-dot" />
          <span className="signed-in-user">{auth.user.username}</span>
          <button className="sign-out-button" type="button" onClick={signOut} title="退出登录" aria-label="退出登录">退出</button>
        </div>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/study" element={<StudyPage />} />
          <Route path="/mistakes" element={<MistakesPage />} />
          <Route path="/profile" element={<LearningProfilePage />} />
          <Route path="/practice" element={<PracticePage />} />
          <Route path="/organize" element={<MindMapPage />} />
          <Route path="/teacher" element={<TeacherPage />} />
          {Object.entries(pages).filter(([path]) => path !== "/" && path !== "/mistakes" && path !== "/practice" && path !== "/study" && path !== "/organize" && path !== "/profile").map(([path, title]) => (
            <Route key={path} path={path} element={<PageFrame title={title} />} />
          ))}
          <Route path="*" element={<PageFrame title="页面不存在" />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
