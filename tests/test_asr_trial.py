import tempfile
import unittest
from pathlib import Path
import subprocess
import os
from unittest.mock import Mock, patch

from tools.try_qwen_recovery import select_clips, whisper_text, render_report, run_worker


class AsrTrialTest(unittest.TestCase):
    def test_timeout_stops_owned_worker_before_returning(self):
        proc = Mock(pid=12345)
        proc.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 0]
        proc.poll.return_value = None
        with patch("tools.try_qwen_recovery.subprocess.Popen", return_value=proc), \
                patch("tools.try_qwen_recovery.subprocess.run") as kill_tree:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_worker("python", "qwen", "request.json", "out.json", 1)
        if os.name == "nt":
            self.assertEqual(kill_tree.call_args.args[0], ["taskkill", "/PID", "12345", "/T", "/F"])
        else:
            proc.kill.assert_called_once()

    def test_long_missing_span_is_split_and_bounded(self):
        clips = select_clips([(1, 95)], [], 100, limit=3)
        self.assertEqual([(c["start"], c["end"]) for c in clips], [(0, 30), (30, 60), (60, 90)])

    def test_padding_is_clipped_to_audio_and_empty_spans_are_ignored(self):
        clips = select_clips([(0, 3), (3, 3), (float("nan"), 9), (8, 10)], [], 10)
        self.assertEqual([(c["start"], c["end"]) for c in clips], [(0, 5), (6, 10)])

    def test_overlapping_low_confidence_and_missing_regions_merge(self):
        segments = [{"start": 12, "end": 15, "text": "不確かな語", "words": [{"prob": 0.2}] * 3}]
        clips = select_clips([(10, 13)], segments, 30)
        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (8, 17))
        self.assertIn("文字のない声", clips[0]["reason"])
        self.assertIn("確信度", clips[0]["reason"])

    def test_comparison_uses_words_inside_the_window(self):
        segments = [{"start": 0, "end": 10, "text": "前本体後", "words": [
            {"start": 0, "end": 1, "word": "前"},
            {"start": 4, "end": 6, "word": "本体"},
            {"start": 9, "end": 10, "word": "後"}]}]
        self.assertEqual(whisper_text(segments, 3, 7), "本体")

    def test_html_escapes_candidate_text_and_labels_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            clips = [{"id": "clip-01", "start": 1, "end": 3, "reason": "比較用",
                      "path": str(folder / "clip-01.wav"), "whisper": "<script>alert(1)</script>"}]
            result = {"peak_allocated_mib": 100, "results": [
                {"id": "clip-01", "text": "<候補>", "token_limit_reached": True}]}
            render_report(folder, clips, result, {}, {"qwen_inference_sec": 1})
            report = (folder / "comparison.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", report)
            self.assertIn("&lt;候補&gt;", report)
            self.assertIn("生成上限", report)
            self.assertIn("未確認", report)


if __name__ == "__main__":
    unittest.main()
