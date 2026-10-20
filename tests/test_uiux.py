"""画面の案内が保存状態と一致すること、確認すべき箇所へ移動できる構造を検証。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from core import document_status, run_control
from core import utterances as U


class UIUXTest(unittest.TestCase):
    def test_document_guidance_and_action_follow_freshness_and_unsaved_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            job = dict(job_dir=folder, source="test.wav", meeting_date="", memo="",
                       utterances=[dict(start=0, end=1, speaker="", text="確認した本文。")])
            self.assertIn("作成できます", app.document_hint(None, "議事録.md"))
            self.assertEqual(app.document_button(job, "議事録.md"), "議事録を作成")
            document_status.save(Path(folder), "議事録.md", "# 議事録", document_status.inputs(job))
            self.assertIn("保存済みの本文", app.document_hint(job, "議事録.md"))
            self.assertEqual(app.document_button(job, "議事録.md"), "議事録を作り直す")
            job["utterances"][0]["text"] = "修正した本文。"
            self.assertIn("本文が更新", app.document_hint(job, "議事録.md"))
            self.assertIn("更新前の文書: 議事録", app._job_notice(job))
            self.assertEqual(app._read_md(job, "議事録.md"), "# 議事録")
            job["_dirty"] = {"text"}
            self.assertIn("未保存", app.document_hint(job, "議事録.md"))
            job["incomplete"] = True
            self.assertIn("全編が完了", app.document_hint(job, "議事録.md"))

    def test_review_markers_include_ai_omissions_and_keep_editor_numbers(self):
        utts = [
            dict(start=0, end=6, speaker="話者1", text="一文目です。二文目です。",
                 marked="一文目です。二文目です。", words=[]),
            dict(start=7, end=9, speaker="話者1", text="聞き直す部分です。",
                 marked=U.RED_OPEN + "聞き直す" + U.RED_CLOSE + "部分です。", words=[]),
            dict(start=10, end=13, speaker="話者1", text="AI回答がない行です。",
                 marked="AI回答がない行です。", proofread_pending=True, words=[]),
        ]
        rows = U.sentence_rows(utts)
        self.assertEqual([r["utterance_index"] for r in rows], [1, 1, 2, 3])
        markup = U.render_html(utts, "/audio.m4a")
        self.assertIn("要確認 2箇所", markup)
        self.assertEqual(markup.count('data-review="true"'), 2)
        self.assertIn("次の要確認へ", markup)
        self.assertIn('aria-live="polite"', markup)
        self.assertIn("AI校正の再確認", markup)
        self.assertIn('aria-label="発言3 要確認"', markup)
        clean = U.render_html(utts[:1])
        self.assertNotIn("次の要確認へ", clean)

    def test_pause_button_tracks_only_own_running_session(self):
        own = SimpleNamespace(session_hash="uiux-own")
        other = SimpleNamespace(session_hash="uiux-other")
        self.assertFalse(app.pause_button_state(own)["interactive"])
        with run_control.session(own.session_hash):
            self.assertTrue(app.pause_button_state(own)["interactive"])
            self.assertFalse(app.pause_button_state(other)["interactive"])
            run_control.request_pause(own.session_hash)
            update = app.pause_button_state(own)
            self.assertFalse(update["interactive"])
            self.assertEqual(update["value"], "保存して停止中…")
        self.assertFalse(app.pause_button_state(own)["interactive"])

    def test_empty_search_has_recovery_guidance_without_fake_result_row(self):
        with patch.object(app.search_mod, "search_outputs", return_value=[]):
            rows, notice = app.do_search("見つからない言葉")
            self.assertEqual(rows, [])
            self.assertIn("別の表記", notice)
            self.assertIn("入力", app.do_search("  ")[1])


if __name__ == "__main__":
    unittest.main()
