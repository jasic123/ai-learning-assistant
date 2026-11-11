import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from app.mindmaps.store import MindMapStore
from app.api.mindmaps import _parse_tree_response
from frontend.app import (edit_mindmap, export_mindmap, mindmap_node_choices,
                          mindmap_drag_sync, move_mindmap_node, render_mindmap)


class MindMapStoreTests(unittest.TestCase):
    def test_model_json_parser_accepts_code_fence_and_surrounding_text(self):
        tree = {"title": "主题", "children": [{"title": "分支"}]}
        self.assertEqual(_parse_tree_response("```json\n" + json.dumps(tree) + "\n```"), tree)
        self.assertEqual(_parse_tree_response("结果如下：\n" + json.dumps(tree) + "\n完成"), tree)
    def test_save_edit_and_validate_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MindMapStore(Path(directory))
            tree = {"title": "迭代器", "children": [{"title": "协议", "children": [{"title": "__iter__"}]}]}
            saved = store.save("Python 迭代器", tree)
            self.assertEqual(saved["tree"]["children"][0]["title"], "协议")
            edited = store.save("Python 迭代器", {"title": "迭代器", "children": [{"title": "__next__"}]}, saved["id"])
            self.assertEqual(edited["id"], saved["id"])
            self.assertEqual(len(store.list()), 1)
            with self.assertRaises(ValueError):
                store.save("坏导图", {"title": "", "children": []})

    def test_visual_tree_editor_renames_adds_and_deletes_nodes(self):
        tree = {"title": "迭代器", "children": [{"title": "协议", "children": [{"title": "__iter__"}]}]}
        updated, html, choices, label, note = edit_mindmap(tree, "0", "迭代协议", "rename")
        self.assertEqual(updated["children"][0]["title"], "迭代协议")
        self.assertIn("迭代协议", html)
        self.assertEqual(label, "迭代协议")
        updated, _, _, _, _ = edit_mindmap(updated, "0", "__next__", "add_sibling")
        self.assertEqual(updated["children"][1]["title"], "__next__")
        updated, _, _, _, _ = edit_mindmap(updated, "0.0", "", "delete")
        self.assertEqual(updated["children"][0]["children"], [])
        self.assertEqual(mindmap_node_choices(updated).value, "root")

    def test_visual_tree_escapes_user_text(self):
        rendered = render_mindmap({"title": "<script>", "children": []})
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_reorder_controls_move_and_reparent_nodes(self):
        tree = {"title": "根", "children": [
            {"title": "甲", "children": []}, {"title": "乙", "children": []}]}
        tree, _, selected_component, _, _ = move_mindmap_node(tree, "1", "up")
        self.assertEqual([item["title"] for item in tree["children"]], ["乙", "甲"])
        self.assertEqual(selected_component.value, "0")
        tree, _, _, _, _ = move_mindmap_node(tree, "0", "down")
        tree, _, selected_component, _, _ = move_mindmap_node(tree, "1", "indent")
        self.assertEqual(tree["children"][0]["children"][0]["title"], "乙")
        self.assertEqual(selected_component.value, "0.0")
        tree, _, _, _, _ = move_mindmap_node(tree, "0.0", "outdent")
        self.assertEqual([item["title"] for item in tree["children"]], ["甲", "乙"])

    def test_exports_svg_png_and_freemind(self):
        tree = {"title": "Python", "children": [{"title": "迭代器", "children": [{"title": "协议"}]}]}
        with tempfile.TemporaryDirectory() as directory, patch("app.config.ARTIFACTS_DIR", Path(directory)):
            svg = Path(export_mindmap(tree, "SVG"))
            png = Path(export_mindmap(tree, "PNG"))
            mm = Path(export_mindmap(tree, "FreeMind (.mm)"))
            self.assertEqual(ET.parse(svg).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(ET.parse(mm).getroot().find("node").attrib["TEXT"], "Python")

    def test_drag_sync_accepts_updated_tree_and_refreshes_node_selector(self):
        tree = {"title": "主题", "children": [{"title": "父节点", "children": [{"title": "拖入节点"}]}]}
        state, html, selector, label, note = mindmap_drag_sync(json.dumps(tree, ensure_ascii=False))
        self.assertEqual(state["children"][0]["children"][0]["title"], "拖入节点")
        self.assertIn("data-path='0.0'", html)
        self.assertEqual(selector.value, "root")
        self.assertEqual(label, "主题")


if __name__ == "__main__":
    unittest.main()
