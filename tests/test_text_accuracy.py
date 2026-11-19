"""同音異義語の文脈、校正での事実変更、議事録の担当者取り違えの回帰テスト。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from core import config, document_status, glossary, minutes, text_fidelity
from core import utterances as U


def row(text, speaker="話者A"):
    return dict(start=0, end=3, speaker=speaker, text=text, marked=text, words=[])


class ContextCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(glossary, "LEARNED_PATH", Path(self.temp.name) / "learned_words.yaml")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_single_kanji_edit_is_remembered_with_context_not_global_replacement(self):
        self.assertEqual(glossary.learn_from_diff("顧客へ家電します。", "顧客へ架電します。"), 1)
        examples = glossary.load_context_examples()
        self.assertEqual(examples[0]["before"], "顧客へ家電します。")
        hints = glossary.context_hints("担当顧客へ家電して説明する。")
        self.assertIn("架電", hints)
        self.assertEqual(glossary.apply_corrections("家電製品を購入する。"), "家電製品を購入する。")
        self.assertEqual(glossary.learn_from_diff("顧客へ家電します。", "顧客へ架電します。"), 0)

    def test_hint_table_round_trip_and_validation_does_not_replace_file(self):
        app.save_context_examples_table([["家電", "架電", "顧客に電話をかける話"],
                                         ["カーサログ", "監査ログ", "操作やアクセスの記録"]])
        self.assertEqual(len(app.load_context_examples_table()), 2)

        self.assertIn("監査ログ", glossary.context_hints("カーサログを残します。"))
        with self.assertRaises(app.gr.Error):
            app.save_context_examples_table([["欠けた入力", "", ""]])
        self.assertEqual(len(app.load_context_examples_table()), 2)

    def test_many_long_examples_do_not_overflow_hint_budget(self):
        glossary.save_context_examples([
            dict(before="顧客への家電" + "あ" * 280 + str(i),
                 after="顧客への架電" + "あ" * 280 + str(i), context="説明" * 100)
            for i in range(20)])
        hints = glossary.context_hints("顧客への家電について")
        self.assertIn("架電", hints)
        self.assertLessEqual(len(hints), 2000)
        self.assertEqual(glossary.context_hints("顧客への家電について", max_chars=10), "")
    def test_proofreading_uses_context_and_marks_edits_even_without_model_marks(self):
        app.save_context_examples_table([["家電", "架電", "顧客に電話をかける話"],
                                         ["カーサログ", "監査ログ", "操作記録"]])
        rows = [row("顧客へ家電します。"), row("家電製品を購入します。"), row("カーサログを保存します。")]
        answer = "[1] 顧客へ架電します。\n[2] 家電製品を購入します。\n[3] 監査ログを保存します。"
        with patch.object(minutes.people, "load_people", return_value=[]), \
                patch.object(minutes, "_context_block", return_value=""), \
                patch.object(minutes.llm, "generate", return_value=answer) as generate:
            self.assertEqual(minutes.proofread_utterances({}, rows), (3, 2))
        self.assertIn("顧客に電話をかける話", generate.call_args.args[1])
        self.assertEqual(rows[1]["text"], "家電製品を購入します。")
        self.assertNotIn(U.RED_OPEN, rows[1]["marked"])
        self.assertIn(U.RED_OPEN, rows[0]["marked"])
        self.assertIn(U.RED_OPEN, rows[2]["marked"])
        self.assertEqual(U.strip_marks(rows[2]["marked"]), rows[2]["text"])

    def test_fact_changes_remain_pending(self):
        people = [{"display_name": n} for n in ("青山", "白井", "黒田")]
        source = ["青山と白井が対応します。", "開始は10時です。", "実施しない方針です。"]
        rows = [row(t) for t in source]
        with patch.object(minutes.people, "load_people", return_value=people), \
                patch.object(minutes, "_context_block", return_value=""), \
                patch.object(minutes.llm, "generate", return_value="[1] 白井と黒田が対応します。\n[2] 開始は11時です。\n[3] 実施する方針です。"):
            self.assertEqual(minutes.proofread_utterances({}, rows), (0, 0))
        self.assertEqual([r["text"] for r in rows], source)
        self.assertTrue(all(r["proofread_pending"] for r in rows))

    def test_alias_normalization_and_existing_red_are_preserved(self):
        names = text_fidelity.name_map([{"display_name": "青山", "aliases": ["あおやま"]}], ["あおやま"])
        self.assertFalse(text_fidelity.correction_changes_facts("あおやまが対応", "青山が対応", names))
        marked = text_fidelity.mark_corrections("家電と不明", "架電と" + U.RED_OPEN + "不明" + U.RED_CLOSE)
        self.assertEqual(U.strip_marks(marked), "架電と不明")
        self.assertIn(U.RED_OPEN + "不明" + U.RED_CLOSE, marked)
        self.assertEqual(text_fidelity.mark_corrections("これは" + U.RED_OPEN + "不明" + U.RED_CLOSE, "これは不明"),
                         "これは" + U.RED_OPEN + "不明" + U.RED_CLOSE)


class AttributionTest(unittest.TestCase):
    transcript = "[00:00:10] 黒田: 問い合わせには青山と白井が対応します。\n[00:00:20] 黒田: 私は会場の準備をします。"
    wrong = "【アクションアイテム】\n| タスク | 担当者 | 期限 |\n|---|---|---|\n| 問い合わせに対応（根拠 00:00:10） | 白井と黒田 | 未定 |"
    right = "| 問い合わせに対応（根拠 00:00:10） | 青山と白井 | 未定 |"

    def verify(self, reply, transcript=None, md=None):
        with patch.object(minutes.people, "load_people", return_value=[{"display_name": n} for n in ("青山", "白井", "黒田")]), \
                patch.object(minutes.llm, "generate", return_value=reply):
            return minutes.verify_document(copy.deepcopy(config.DEFAULTS), transcript or self.transcript, md or self.wrong)

    def quote(self, text="問い合わせには青山と白井が対応します。", stamp="00:00:10"):
        return '\n[1] 根拠: ' + json.dumps([dict(time=stamp, quote=text)], ensure_ascii=False)

    def test_false_ok_cannot_use_reporter_as_assignee(self):
        result, report = self.verify("[1] OK" + self.quote())
        self.assertIn("白井と黒田", result)
        self.assertIn(minutes.REVIEW_MARK, result)
        self.assertEqual(report["pending"], 1)

    def test_supported_fix_preserves_both_assignees_and_records_quote(self):
        result, report = self.verify("[1] 修正: " + self.right + self.quote())
        self.assertIn(self.right, result)
        self.assertNotIn(minutes.REVIEW_MARK, result)
        self.assertEqual(report["fixed"], 1)
        self.assertEqual(report["items"][0]["quotes"][0]["time"], "00:00:10")

    def test_invented_quote_or_wrong_timestamp_cannot_validate_assignees(self):
        for support in (self.quote("問い合わせには白井と黒田が対応します。"), self.quote(stamp="00:00:20"), ""):
            with self.subTest(support=support):
                result, report = self.verify("[1] OK" + support)
                self.assertIn(minutes.REVIEW_MARK, result)
                self.assertEqual(report["fixed"], 0)

    def test_missing_duplicate_and_unapplicable_verdicts_stay_visible(self):
        for reply in ("", "[1] OK\n[1] 確認できず", "[1] 修正: | 不正な列 |"):
            result, report = self.verify(reply)
            self.assertIn(minutes.REVIEW_MARK, result)
            self.assertEqual(report["pending"], 1)
        result, _ = self.verify("[1] 修正: 青山と白井が対応", md="【要点サマリー】\n白井と黒田が対応する。")
        self.assertIn("白井と黒田が対応する。" + minutes.REVIEW_MARK, result)

    def test_first_person_can_support_speaker_but_nearby_mentions_do_not(self):
        md = self.wrong.replace("白井と黒田", "黒田")
        transcript = "[00:00:10] 黒田: 問い合わせには私が対応します。"
        result, _ = self.verify("[1] OK" + self.quote("問い合わせには私が対応します。"), transcript, md)
        self.assertNotIn(minutes.REVIEW_MARK, result)

    def test_same_person_different_task_and_missing_coassignee_are_not_validated(self):
        result, _ = self.verify("[1] OK" + self.quote("私は会場の準備をします。", "00:00:20"),
                                md=self.wrong.replace("白井と黒田", "黒田"))
        self.assertIn(minutes.REVIEW_MARK, result)
        result, _ = self.verify("[1] OK" + self.quote(), md=self.wrong.replace("白井と黒田", "白井"))
        self.assertIn(minutes.REVIEW_MARK, result)

    def test_verification_is_persisted_with_document_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = dict(transcript=self.transcript, source="synthetic", meeting_date="")
            document_status.save(Path(directory), "議事録.md", self.right, source, {"enabled": True, "pending": 1})
            record = json.loads((Path(directory) / document_status.INDEX).read_text(encoding="utf-8"))["議事録.md"]
            self.assertEqual(record["verification"]["pending"], 1)
            self.assertEqual(record["content_hash"], document_status.digest(self.right))


class ReadabilityTest(unittest.TestCase):
    def test_yomiyasu_is_used_during_generation_without_extra_rewrite(self):
        with patch.object(minutes, "_context_block", return_value=""), \
                patch.object(minutes.llm, "generate", return_value="【決定事項】\n- 未定") as generate:
            minutes.generate_minutes(copy.deepcopy(config.DEFAULTS), "[00:00:10] 話者: まだ未定です。")
        generate.assert_called_once()
        self.assertIn("複数の担当者は組み合わせを保つ", generate.call_args.args[1])
        self.assertEqual(minutes._readability_rules({"llm": {"yomiyasu": False}}), "")


if __name__ == "__main__":
    unittest.main()
