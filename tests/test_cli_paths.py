"""核验 RAG 包入口与非项目工作目录启动脚本，避免迁移后再次出现 app 导入或相对路径错误。"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import validate_rag


class CliPathTests(unittest.TestCase):
    def test_rag_and_script_entry_points_show_help(self):
        root = Path(__file__).resolve().parents[1]
        commands = [
            (["-m", "app.rag.cli", "--help"], root),
            (["-m", "app.rag", "--help"], root),
            (["-m", "app.rag.db", "--help"], root),
            ([str(root / "scripts/validate_rag.py"), "--help"], root / "scripts"),
        ]
        for arguments, directory in commands:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, *arguments], cwd=directory,
                                        env={**os.environ, "PYTHONUTF8": "1"},
                                        capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_validation_pdf_path_is_based_on_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "data/uploads/教育资料/Python参考手册(第4版).pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"fixture")
            uploads = root / "uploads"
            previous = Path.cwd()
            try:
                os.chdir(root.parent)
                with patch.object(validate_rag, "PROJECT_ROOT", root), \
                     patch.object(validate_rag, "ARTIFACTS_DIR", root / "artifacts"), \
                     patch.object(validate_rag, "model_settings"), \
                     patch.object(validate_rag, "ModelClient"), \
                     patch.object(validate_rag, "kb_paths", return_value=(uploads, root / "indexes")), \
                     patch.object(validate_rag, "build_index", side_effect=RuntimeError("offline-stop")) as build, \
                     patch.object(sys, "argv", ["validate_rag.py", "--rebuild"]), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(validate_rag.main(), 1)
                    build.assert_called_once()
                    self.assertEqual((uploads / pdf.name).read_bytes(), pdf.read_bytes())
                    report = json.loads((root / "artifacts/rag-validation/hybrid.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["error"], "offline-stop")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
