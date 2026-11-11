import { useEffect, useMemo, useState } from "react";
import { generateMindmap, getMindmaps, saveMindmap } from "./api.js";

const emptyTree = () => ({ title: "新知识主题", children: [] });

function normalizeTree(node, fallbackTitle = "未命名主题") {
  return {
    title: typeof node?.title === "string" && node.title.trim() ? node.title : fallbackTitle,
    children: Array.isArray(node?.children)
      ? node.children.map((child) => normalizeTree(child))
      : [],
  };
}

function nodeAt(tree, path) {
  return path ? path.split(".").reduce((node, index) => node.children?.[Number(index)], tree) : tree;
}

function updateAt(tree, path, update) {
  if (!path) return update(tree);
  const indexes = path.split(".").map(Number);
  const next = structuredClone(tree);
  let node = next;
  for (const index of indexes) node = node.children[index];
  update(node);
  return next;
}

function MapBranch({ node, path, selected, onSelect }) {
  const children = node.children ?? [];
  return (
    <li className="map-branch">
      <button className={`map-node${path === "" ? " map-node-root" : ""}${selected === path ? " selected" : ""}`}
              type="button" onClick={() => onSelect(path)} aria-pressed={selected === path}>
        {node.title}
      </button>
      {children.length > 0 && (
        <ul className="map-children">
          {children.map((child, index) => (
            <MapBranch key={`${path}.${index}`} node={child}
                       path={path ? `${path}.${index}` : String(index)}
                       selected={selected} onSelect={onSelect} />
          ))}
        </ul>
      )}
    </li>
  );
}

function MindMapPage() {
  const [maps, setMaps] = useState([]);
  const [mapId, setMapId] = useState(null);
  const [topic, setTopic] = useState("");
  const [context, setContext] = useState("");
  const [title, setTitle] = useState("");
  const [tree, setTree] = useState(emptyTree);
  const [selectedPath, setSelectedPath] = useState("");
  const [nodeTitle, setNodeTitle] = useState("");
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getMindmaps(controller.signal)
      .then(setMaps)
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const selectedNode = useMemo(() => nodeAt(tree, selectedPath), [tree, selectedPath]);

  useEffect(() => {
    setNodeTitle(selectedNode?.title ?? "");
  }, [selectedNode]);

  function showMap(map) {
    setMapId(map.id);
    setTitle(map.title);
    setTopic(map.title);
    setTree(normalizeTree(map.tree, map.title));
    setSelectedPath("");
    setError("");
    setNotice("");
  }

  function newMap() {
    const next = emptyTree();
    setMapId(null);
    setTitle(next.title);
    setTree(next);
    setSelectedPath("");
    setError("");
    setNotice("");
  }

  async function generate(event) {
    event.preventDefault();
    if (!topic.trim()) {
      setError("先输入一个要整理的主题。");
      return;
    }
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await generateMindmap({ topic: topic.trim(), context: context.trim() });
      showMap(result);
      setMaps((current) => [result, ...current.filter((item) => item.id !== result.id)]);
      setNotice("导图已生成并保存。");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  async function save() {
    if (!title.trim()) {
      setError("导图标题不能为空。");
      return;
    }
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await saveMindmap({ id: mapId, title: title.trim(), tree });
      showMap(result);
      setMaps((current) => [result, ...current.filter((item) => item.id !== result.id)]);
      setNotice("修改已保存。");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setWorking(false);
    }
  }

  function applyNodeEdit(action) {
    const label = nodeTitle.trim();
    if (action !== "delete" && !label) {
      setError("节点名称不能为空。");
      return;
    }
    setError("");
    setNotice("");
    if (action === "rename") {
      setTree((current) => updateAt(current, selectedPath, (node) => { node.title = label; }));
      setNotice("节点已改名，记得保存。");
    } else if (action === "add_child") {
      if ((selectedNode.children?.length ?? 0) >= 30) {
        setError("单个节点最多添加 30 个子节点。");
        return;
      }
      const child = { title: label, children: [] };
      setTree((current) => updateAt(current, selectedPath, (node) => {
        node.children = [...(node.children ?? []), child];
      }));
      const nextPath = selectedPath ? `${selectedPath}.${selectedNode.children?.length ?? 0}` : "0";
      setSelectedPath(nextPath);
      setNotice("子节点已添加，记得保存。");
    } else if (action === "add_sibling") {
      if (!selectedPath) {
        setError("根主题不能添加同级节点。");
        return;
      }
      const parts = selectedPath.split(".");
      const index = Number(parts.pop());
      const parentPath = parts.join(".");
      const parent = nodeAt(tree, parentPath);
      if (parent.children.length >= 30) {
        setError("单个节点最多添加 30 个子节点。");
        return;
      }
      setTree((current) => updateAt(current, parentPath, (node) => {
        node.children.splice(index + 1, 0, { title: label, children: [] });
      }));
      setSelectedPath([...parts, index + 1].join("."));
      setNotice("同级节点已添加，记得保存。");
    } else {
      if (!selectedPath) {
        setError("根主题不能删除。");
        return;
      }
      const parts = selectedPath.split(".");
      const index = Number(parts.pop());
      const parentPath = parts.join(".");
      setTree((current) => updateAt(current, parentPath, (node) => { node.children.splice(index, 1); }));
      setSelectedPath(parentPath);
      setNotice("节点已删除，记得保存。");
    }
  }

  return (
    <section className="page-frame organize-page" aria-labelledby="page-title">
      <header className="page-heading organize-heading">
        <div>
          <p className="eyebrow">整理主题与学习资料</p>
          <h1 id="page-title">知识整理</h1>
        </div>
        <div className="organize-actions">
          <label className="map-picker">
            <span className="sr-only">已保存导图</span>
            <select value={mapId ?? ""} onChange={(event) => {
              const map = maps.find((item) => String(item.id) === event.target.value);
              if (map) showMap(map);
            }} disabled={loading || !maps.length}>
              <option value="">{loading ? "正在读取…" : maps.length ? "打开已保存导图" : "暂无已保存导图"}</option>
              {maps.map((map) => <option key={map.id} value={map.id}>{map.title}</option>)}
            </select>
          </label>
          <button className="quiet-button" type="button" onClick={newMap} disabled={working}>新建</button>
          <button className="primary-button organize-save" type="button" onClick={save} disabled={working}>保存</button>
        </div>
      </header>

      {error && <div className="dashboard-error" role="alert"><span>{error}</span><button className="text-button" type="button" onClick={() => setError("")}>关闭</button></div>}
      {notice && <p className="organize-notice" role="status">{notice}</p>}

      <div className="organize-workspace">
        <aside className="organize-sidebar">
          <form className="organize-generate" onSubmit={generate}>
            <p className="section-kicker">AI 生成</p>
            <label className="organize-field">
              <span>整理主题</span>
              <input value={topic} onChange={(event) => setTopic(event.target.value)} maxLength={2000}
                     placeholder="例如：Python 迭代器" />
            </label>
            <label className="organize-field">
              <span>补充资料 <small>可选</small></span>
              <textarea value={context} onChange={(event) => setContext(event.target.value)} maxLength={12000}
                        rows={5} placeholder="粘贴课堂笔记或想整理的内容" />
            </label>
            <button className="primary-button organize-generate-button" type="submit" disabled={working}>
              {working ? "处理中…" : "生成思维导图"}
            </button>
          </form>

          <div className="organize-editor">
            <div className="section-heading"><div><p className="section-kicker">节点编辑</p><h2>编辑选中节点</h2></div></div>
            <label className="organize-field">
              <span>导图标题</span>
              <input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={200} />
            </label>
            <label className="organize-field">
              <span>节点名称</span>
              <input value={nodeTitle} onChange={(event) => setNodeTitle(event.target.value)} maxLength={500}
                     disabled={!selectedNode} />
            </label>
            <div className="node-edit-actions">
              <button className="quiet-button" type="button" onClick={() => applyNodeEdit("rename")} disabled={!selectedNode || working}>改名</button>
              <button className="quiet-button" type="button" onClick={() => applyNodeEdit("add_child")} disabled={!selectedNode || working}>加子节点</button>
              <button className="quiet-button" type="button" onClick={() => applyNodeEdit("add_sibling")} disabled={!selectedNode || !selectedPath || working}>加同级</button>
              <button className="danger-text-button" type="button" onClick={() => applyNodeEdit("delete")} disabled={!selectedNode || !selectedPath || working}>删除</button>
            </div>
            <p className="organize-help">选择画布中的节点后，可在这里修改结构。编辑后记得保存。</p>
          </div>
        </aside>

        <section className="organize-canvas" aria-label="思维导图画布">
          <div className="canvas-toolbar">
            <div><span className="canvas-title">{title || "未命名导图"}</span><span className="canvas-caption">{mapId ? "已保存导图" : "未保存"}</span></div>
            <span className="canvas-node-count">{countNodes(tree)} 个节点</span>
          </div>
          {tree.title ? (
            <div className="map-viewport">
              <ul className="map-tree"><MapBranch node={tree} path="" selected={selectedPath} onSelect={setSelectedPath} /></ul>
            </div>
          ) : <div className="map-empty">输入主题，生成一张知识导图</div>}
        </section>
      </div>
    </section>
  );
}

function countNodes(node) {
  return 1 + (node.children ?? []).reduce((total, child) => total + countNodes(child), 0);
}

export default MindMapPage;
