import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.full_asr_experiment import save_json, transcript_md
from tools.report_full_asr_experiment import delta, report


class FullExperimentTest(unittest.TestCase):
    def test_failed_serialization_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            save_json(path, {"saved": [1, 2]})
            with self.assertRaises(TypeError):
                save_json(path, {"saved": object()})
            self.assertEqual(json.loads(path.read_text()), {"saved": [1, 2]})

    def test_markdown_retains_uncertain_and_unsafe_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.md"
            transcript_md(path, "test", [{"start": 0, "end": 30, "text": "推測", "unsafe": True,
                                          "marked": '<span style="color:red">推測</span>'}])
            text = path.read_text(encoding="utf-8")
            self.assertIn("⟦推測⟧", text)
            self.assertIn("要確認", text)

    def test_diff_escapes_source_text_before_adding_markup(self):
        text = delta("<script>a</script>", "<script>b</script>")
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("<del>a</del><ins>b</ins>", text)

    def test_incomplete_report_is_rejected(self):
        with patch("tools.report_full_asr_experiment.read_json", side_effect=[
                {}, {"complete": True}, {"complete": False}, {"complete": True}, {"clips": []}]):
            with self.assertRaisesRegex(ValueError, "未完了"):
                report(Path("unused"))

    def test_complete_report_compares_all_windows_and_preserves_warnings(self):
        source = dict(source="synthetic.m4a", duration=30, preprocess_sec=1,
                      created_at="2026-10-08", sha256="test", preprocessing="test")
        w = dict(complete=True, wall_sec=2, missing_sec=3, refilled_sec=4,
                 segments=[dict(text="確認")])
        q = dict(complete=True, wall_sec=3, peak_allocated_mib=1, peak_reserved_mib=2,
                 results=[dict(id="clip-001", text="確認", start=0, end=30,
                               elapsed_sec=1, token_limit_reached=False, unsafe=False)])
        g = dict(complete=True, wall_sec=4, model="test", results=[dict(id="clip-001", text="確認。")])
        clips = dict(clips=[dict(id="clip-001", start=0, end=30, whisper="<記録>", path="clip-001.wav")])
        with tempfile.TemporaryDirectory() as tmp, \
                patch("tools.report_full_asr_experiment.read_json", side_effect=[source, w, q, g, clips]):
            stats = report(Path(tmp))
            self.assertEqual(stats["total_compute_sec"], 10)
            self.assertEqual(stats["gemma_changed_windows"], 1)
            self.assertEqual(stats["gemma_nonpunctuation_changed_windows"], 0)
            page = (Path(tmp) / "comparison.html").read_text(encoding="utf-8")
            self.assertIn("&lt;記録&gt;", page)
            self.assertIn("clips/clip-001.wav", page)
            self.assertTrue((Path(tmp) / "実験レポート.md").exists())


if __name__ == "__main__":
    unittest.main()
