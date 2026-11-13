"""2026-11-13 の強化分: 文書の照合・一括置換・1件ずつの修正・声紋による話者の見直し・LLM入力の分割。
個人データ・実音声・実LLMは使わない。"""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import app
from core import config, diarization, glossary, llm, minutes, output, pipeline
from core import utterances as U


def utterance(text="開始は10時です。", start=0, speaker="話者A", **extra):
    u = dict(start=start, end=start + 3, text=text, marked=text, speaker=speaker, words=[])
    u.update(extra)
    return u


# =====================================================================
# LLM入力の分割（上限を超えて先頭が切られない）
# =====================================================================

class ContextBudgetTest(unittest.TestCase):
    def setUp(self):
        self.settings = copy.deepcopy(config.DEFAULTS)
        self.settings["llm"]["ollama_max_ctx"] = 16384

    def test_budget_counts_system_and_output_reserve(self):
        # (16384 - 6144 - 1000*0.8 - 64) / 0.8 = 11720 字
        budget = llm.input_budget_chars(self.settings, "x" * 1000)
        self.assertEqual(budget, 11720)
        self.assertTrue(llm.fits_context(self.settings, "x" * 1000, "y" * 11720))
        self.assertFalse(llm.fits_context(self.settings, "x" * 1000, "y" * 11721))
        self.assertEqual(llm.input_budget_chars(self.settings, "x" * 1000, 4096), 14280)

    def test_num_ctx_never_exceeds_limit(self):
        self.assertEqual(llm._num_ctx(self.settings, "", "x" * 100), 16384)
        self.assertEqual(llm._num_ctx(self.settings, "", "x" * 100000), 16384)
        self.settings["llm"]["ollama_max_ctx"] = 32768
        self.assertEqual(llm._num_ctx(self.settings, "", "x" * 100), 16384)
        self.assertEqual(llm._num_ctx(self.settings, "", "x" * 30000), 32768)

    def test_long_transcript_is_split_into_chunks_that_fit(self):
        self.settings["llm"]["ollama_max_ctx"] = 8192
        # 700字の長い発言でも、直前の区間の参考行を足した入力が上限に収まること
        transcript = "\n".join(f"[00:{i // 60:02d}:{i % 60:02d}] 話者A: " + "あ" * 700 for i in range(60))
        seen = []

        def fake_generate(settings, system, user, **kw):
            seen.append((system, user))
            self.assertTrue(llm.fits_context(settings, system, user, minutes.NOTES_OUTPUT_RESERVE),
                            "区間メモの入力が上限を超えている")
            self.assertEqual(kw.get("output_reserve"), minutes.NOTES_OUTPUT_RESERVE)
            return "- [00:00:01] 話者A: メモ"

        with patch.object(minutes.llm, "generate", side_effect=fake_generate), \
                patch.object(minutes, "_context_block", return_value=""):
            minutes._notes_cache.clear()
            shrunk = minutes._shrink_transcript(self.settings, "system", transcript)
        self.assertGreater(len(seen), 1)
        self.assertIn("第1区間のメモ", shrunk)
        self.assertIn("参考: 直前の区間の末尾", seen[1][1])
        self.assertLessEqual(len(seen[1][1].split("（ここから第2区間）")[0]), minutes.NOTES_OVERLAP_CHARS + 120)
        minutes._notes_cache.clear()

    def test_notes_are_shrunk_again_when_still_too_long(self):
        self.settings["llm"]["ollama_max_ctx"] = 8192
        transcript = "\n".join(f"[00:{i // 60:02d}:{i % 60:02d}] 話者A: " + "い" * 300 for i in range(80))
        calls = []

        def fake_generate(settings, system, user, **kw):
            calls.append(user)
            return "- メモ " + "う" * 2500  # 1区間のメモが長く、まとめても上限を超える

        with patch.object(minutes.llm, "generate", side_effect=fake_generate), \
                patch.object(minutes, "_context_block", return_value=""):
            minutes._notes_cache.clear()
            shrunk = minutes._shrink_transcript(self.settings, "system", transcript)
        self.assertTrue(any("（2段目）" in u or "第1区間のメモ" in u for u in calls))
        self.assertTrue(llm.fits_context(self.settings, "system", shrunk) or len(calls) > 4)
        minutes._notes_cache.clear()

    def test_generate_passes_memo_and_low_temperature(self):
        calls = []
        with patch.object(minutes.llm, "generate", side_effect=lambda *a, **k: calls.append((a, k)) or "【会議概要】\n日時: 不明\n"), \
                patch.object(minutes.people, "people_summary_md", return_value="（人物マスタ未登録）"), \
                patch.object(minutes.glossary, "load_glossary", return_value=[]):
            minutes.generate_minutes(self.settings, "[00:00:00] 話者A: こんにちは", memo="議題: 予算")
        self.assertIn("予算", calls[0][0][1])
        self.assertEqual(calls[0][1].get("temperature"), minutes.FACTUAL_TEMPERATURE)


# =====================================================================
# 文書の照合
# =====================================================================

MINUTES_MD = """【会議概要】
日時: 不明
参加者: 坪内、岡田

【要点サマリー】
予算は100万円で合意した。

【議題別の詳細】
### 議題1: 予算
- 報告者・担当: 坪内
- 坪内から予算は200万円との説明があった（根拠 00:01:10）
- 期限は未確定である（根拠 00:02:00）
- 来年度の体制も決定した（根拠 00:09:00）

【決定事項】
- 予算を100万円とする（根拠 00:01:10）

【アクションアイテム】
| タスク内容 | 担当者 | 期限 |
|---|---|---|
| 見積を送る（根拠 00:03:00） | 坪内 | 未定 |

【AIの意見】
- 期限が曖昧である
"""

TRANSCRIPT = "\n".join([
    "[00:01:05] 坪内: 予算は100万円です。",
    "[00:01:20] 岡田: 100万円で進めましょう。",
    "[00:01:50] 岡田: 期限はまだ決めていません。",
    "[00:03:00] 岡田: 見積は私が送ります。",
])


class VerifyDocumentTest(unittest.TestCase):
    def setUp(self):
        self.settings = copy.deepcopy(config.DEFAULTS)

    def test_items_exclude_headings_opinions_and_table_header(self):
        items = [line for _, line in minutes.verifiable_items(MINUTES_MD)]
        self.assertIn("予算は100万円で合意した。", items)
        self.assertIn("- 坪内から予算は200万円との説明があった（根拠 00:01:10）", items)
        self.assertIn("| 見積を送る（根拠 00:03:00） | 坪内 | 未定 |", items)
        self.assertNotIn("| タスク内容 | 担当者 | 期限 |", items)
        self.assertNotIn("- 期限が曖昧である", items)
        self.assertFalse(any("日時:" in line for line in items))
        self.assertFalse(any(line.startswith("### 議題") for line in items))

    def test_evidence_uses_timestamp_window_then_keywords(self):
        lines = minutes._transcript_lines(TRANSCRIPT)
        ev, how = minutes._evidence(lines, "坪内から予算は200万円との説明があった（根拠 00:01:10）")
        self.assertEqual(how, "time")
        self.assertIn("予算は100万円です", ev)
        self.assertNotIn("見積は私が送ります", ev)
        ev, how = minutes._evidence(lines, "見積を送る")
        self.assertEqual(how, "keyword")
        self.assertIn("見積は私が送ります", ev)
        self.assertEqual(minutes._evidence(lines, "ぜんぜん関係ない話"), ("", ""))

    def test_verify_fixes_marks_and_keeps_structure(self):
        def fake_generate(settings, system, user, **kw):
            out = []
            for line in user.splitlines():
                if line.startswith("[") and "下書き:" in line:
                    n = int(line[1:line.index("]")])
                    body = line.split("下書き:", 1)[1].strip()
                    if "200万円" in body:
                        out.append(f"[{n}] 修正: {body.replace('200万円', '100万円')}")
                    elif "体制" in body:
                        out.append(f"[{n}] 確認できず")
                    elif body.startswith("|"):
                        out.append(f"[{n}] 修正: | 見積を送る（根拠 00:03:00） | 岡田 | 未定 |")
                    else:
                        out.append(f"[{n}] OK")
            return "\n".join(out)

        with patch.object(minutes.llm, "generate", side_effect=fake_generate), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, report = minutes.verify_document(self.settings, TRANSCRIPT, MINUTES_MD)
        self.assertIn("- 坪内から予算は100万円との説明があった（根拠 00:01:10）", fixed)
        self.assertIn("| 見積を送る（根拠 00:03:00） | 岡田 | 未定 |", fixed)
        self.assertIn("- 来年度の体制も決定した（根拠 00:09:00）" + minutes.UNVERIFIED_MARK, fixed)
        self.assertIn("【AIの意見】\n- 期限が曖昧である", fixed)  # 対象外はそのまま
        self.assertEqual(report["fixed"], 2)
        self.assertEqual(report["unverified"], 1)
        self.assertGreaterEqual(report["checked"], 5)
        self.assertIn("修正", minutes.verify_summary(report))

    def test_broken_replies_leave_document_untouched(self):
        with patch.object(minutes.llm, "generate", return_value="わかりません"), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, report = minutes.verify_document(self.settings, TRANSCRIPT, MINUTES_MD)
        self.assertEqual(fixed, MINUTES_MD)
        self.assertEqual(report["fixed"], 0)
        self.assertEqual(report["failed"], 0)
        # 表の列数が変わる・極端に短い修正は採用しない（項目5が表の行）
        bad = "[5] 修正: | 見積 |\n[2] 修正: x"
        with patch.object(minutes.llm, "generate", return_value=bad), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, _ = minutes.verify_document(self.settings, TRANSCRIPT, MINUTES_MD)
        self.assertEqual(fixed, MINUTES_MD)

    def test_fix_guards_reject_misnumbered_and_meaning_changing_replies(self):
        items = minutes.verifiable_items(MINUTES_MD)
        body = lambda i: minutes._split_prefix(items[i][1])[1]
        decision = next(b for _, l in items for b in [minutes._split_prefix(l)[1]] if b.startswith("予算を100万円とする"))
        task = next(b for _, l in items for b in [minutes._split_prefix(l)[1]] if b.startswith("|"))
        # 別の項目の内容をそのまま返した（番号の取り違え）→ 不採用
        self.assertFalse(minutes._acceptable_fix(decision, "坪内から予算は100万円との説明があった（根拠 00:01:10）",
                                                 ["坪内から予算は100万円との説明があった（根拠 00:01:10）"]))
        # 根拠の時刻が消えた → 不採用。残っていて内容が近ければ採用
        self.assertFalse(minutes._acceptable_fix(decision, "予算は決まらなかった"))
        self.assertTrue(minutes._acceptable_fix(decision, "予算を100万円とする方向が示された（根拠 00:01:10）"))
        # 元と似ていない書き換えは不採用
        self.assertFalse(minutes._acceptable_fix(decision, "来期の体制について岡田が報告した（根拠 00:01:10）"))
        self.assertTrue(task.startswith("|") and body(0))

    def test_paragraphs_and_keyword_evidence_are_never_rewritten(self):
        def fake_generate(settings, system, user, **kw):
            out = []
            for line in user.splitlines():
                if line.startswith("[") and "下書き:" in line:
                    n = int(line[1:line.index("]")])
                    b = line.split("下書き:", 1)[1].strip()
                    if "万円" in b:  # 段落・語の重なりの項目にも「修正」を返してくる状況
                        out.append(f"[{n}] 修正: {b.replace('200万円', '100万円').replace('100万円', '100万円で合意')}")
                    else:
                        out.append(f"[{n}] OK")
            return "\n".join(out)

        md = MINUTES_MD.replace("- 予算を100万円とする（根拠 00:01:10）", "- 予算を100万円とする")  # 時刻なし→語の重なり
        with patch.object(minutes.llm, "generate", side_effect=fake_generate), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, report = minutes.verify_document(self.settings, TRANSCRIPT, md)
        self.assertIn("予算は100万円で合意した。\n", fixed)  # 要点サマリーの段落は書き換えない
        self.assertIn("- 予算を100万円とする\n", fixed)   # 語の重なりだけの根拠では書き換えない
        self.assertIn("- 坪内から予算は100万円で合意との説明があった（根拠 00:01:10）", fixed)  # 時刻つきは直す
        self.assertEqual(report["fixed"], 1)

    def test_memo_line_is_not_evidence_and_risk_items_are_not_marked(self):
        lines = minutes._transcript_lines("（会議メモ: 予算は200万円の予定）\n" + TRANSCRIPT)
        self.assertFalse(any("会議メモ" in line for _, line in lines))
        self.assertEqual(len(lines), 4)
        md = "# ⚠️ リスク・注意点\n- 聞き逃されている論点がある（根拠 00:01:10）\n\n# 🔍 詳細サマリー\n- 架空の項目（根拠 00:01:10）\n"
        with patch.object(minutes.llm, "generate", return_value="[1] 確認できず\n[2] 確認できず"), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, report = minutes.verify_document(self.settings, TRANSCRIPT, md)
        self.assertIn("- 聞き逃されている論点がある（根拠 00:01:10）\n", fixed)
        self.assertIn("- 架空の項目（根拠 00:01:10）" + minutes.UNVERIFIED_MARK, fixed)
        self.assertEqual(report["unverified"], 1)

    def test_llm_error_midway_keeps_earlier_fixes(self):
        calls = []

        def fake_generate(settings, system, user, **kw):
            calls.append(user)
            if len(calls) > 2:  # 2回目（項目3・4）までは答え、3回目で落ちる
                raise minutes.llm.LLMError("落ちた")
            out = []
            for line in user.splitlines():
                if line.startswith("[") and "下書き:" in line:
                    n = int(line[1:line.index("]")])
                    b = line.split("下書き:", 1)[1].strip()
                    out.append(f"[{n}] 修正: {b.replace('200万円', '100万円')}" if "200万円" in b else f"[{n}] OK")
            return "\n".join(out)

        with patch.object(minutes, "VERIFY_BATCH_ITEMS", 2), \
                patch.object(minutes.llm, "generate", side_effect=fake_generate), \
                patch.object(minutes, "_context_block", return_value=""):
            fixed, report = minutes.verify_document(self.settings, TRANSCRIPT, MINUTES_MD)
        self.assertIn("坪内から予算は100万円との説明があった", fixed)
        self.assertEqual(report["error"], "落ちた")
        self.assertGreater(report["failed"], 0)
        self.assertIn("途中でAIの応答が止まった", minutes.verify_summary(report))


# =====================================================================
# 一括置換・要確認の移動・学習
# =====================================================================

class ReplaceAndReviewTest(unittest.TestCase):
    def test_replace_text_updates_marked_and_flags(self):
        utts = [utterance("預かり菌を確認した。", marked='<span style="color:red">預かり菌</span>を確認した。'),
                utterance("関係ない。", start=5), utterance("預かり菌と預かり菌。", start=10, proofread_pending=True)]
        self.assertEqual(U.count_text(utts, "預かり菌"), (2, 3))
        self.assertEqual(U.replace_text(utts, "預かり菌", "預かり金"), 2)
        self.assertEqual(utts[0]["text"], "預かり金を確認した。")
        self.assertEqual(utts[0]["marked"], '<span style="color:red">預かり金</span>を確認した。')
        self.assertNotIn("human", utts[0])  # 置き換えた発言もAI校正の対象のまま
        self.assertTrue(utts[2]["proofread_pending"])
        self.assertEqual(U.replace_text(utts, "", "x"), 0)

    def test_replace_across_red_boundary_drops_marks_for_that_utterance(self):
        u = utterance("預かり菌を確認", marked='預かり<span style="color:red">菌を</span>確認')
        U.replace_text([u], "預かり菌", "預かり金")
        self.assertEqual(u["marked"], "預かり金を確認")

    def test_next_review_index_wraps(self):
        utts = [utterance(), utterance(marked='<span style="color:red">x</span>', start=3),
                utterance(start=6), utterance(start=9, proofread_pending=True)]
        self.assertEqual(U.next_review_index(utts, 0), 1)
        self.assertEqual(U.next_review_index(utts, 1), 3)
        self.assertEqual(U.next_review_index(utts, 3), 1)
        self.assertIsNone(U.next_review_index([utterance()], 0))

    def test_learn_pair_promotes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(glossary, "LEARNED_PATH", Path(tmp) / "l.yaml"):
            self.assertTrue(glossary.learn_pair("預かり菌", "預かり金", promote=True))
            self.assertEqual(glossary.apply_corrections("預かり菌です"), "預かり金です")
            self.assertFalse(glossary.learn_pair("。", "、", promote=True))
            self.assertTrue(glossary.learn_pair("清風", "青風"))
            self.assertEqual(glossary.apply_corrections("清風"), "清風")  # 1回では適用しない


# =====================================================================
# 画面の1件ずつ修正・一括置換（保存・学習・履歴）
# =====================================================================

class ItemEditorTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.folder = self.root / "meeting"
        self.folder.mkdir()
        self.stack.enter_context(patch.object(output, "OUTPUT_DIR", self.root))
        self.stack.enter_context(patch.object(config, "load_settings", return_value=copy.deepcopy(config.DEFAULTS)))
        self.stack.enter_context(patch.object(app.diarization, "available", return_value=False))
        self.stack.enter_context(patch.object(app.people, "load_people", return_value=[]))
        self.stack.enter_context(patch.object(app.glossary, "apply_corrections", side_effect=lambda t: t))
        self.learn = self.stack.enter_context(patch.object(app.glossary, "learn_from_diff", return_value=1))
        self.pair = self.stack.enter_context(patch.object(app.glossary, "learn_pair", return_value=True))
        job = dict(job_dir=str(self.folder), source="test.wav", duration=30,
                   utterances=[utterance("預かり菌を確認した。", marked='<span style="color:red">預かり菌</span>を確認した。'),
                               utterance("資料を送ります。", start=10, speaker="話者B"),
                               utterance("預かり菌の件です。", start=20)])
        pipeline._save_job(job)
        self.job = app._load_job("meeting")

    def test_open_step_and_review_navigation(self):
        views = app.open_item(self.job, 2)
        self.assertEqual(len(views), app.ITEM_VIEWS)
        self.assertEqual(views[0]["value"], 2)
        self.assertIn("発言 2 / 3", views[1])
        self.assertIn("話者B", views[4]["choices"])
        self.assertEqual(views[5]["value"], "資料を送ります。")
        self.assertEqual(views[6], {"job_dir": str(self.folder), "index": 1, "text": "資料を送ります。", "speaker": "話者B"})
        self.assertEqual(app.step_item(self.job, 3, 1)[0]["value"], 3)   # 末尾で止まる
        self.assertEqual(app.step_item(self.job, 1, -1)[0]["value"], 1)
        self.assertEqual(app.step_item(self.job, 9, 1, views[6], "資料を送ります。", "話者B")[0]["value"], 3)  # 番号欄より開いている発言
        self.assertEqual(app.next_review_item(self.job, 1)[0]["value"], 1)  # 要確認は1だけ → 一周
        with self.assertRaises(app.gr.Error):
            app.open_item(self.job, 9)
        self.assertEqual(app.pick_item(self.job, "3|123")[0]["selected"], "edit")
        self.assertEqual(app.pick_item(self.job, "3|123")[1]["value"], 3)

    def test_navigation_blocks_when_editor_text_is_unsaved(self):
        opened = app.open_item(self.job, 2)[6]
        for move in (lambda: app.step_item(self.job, 2, 1, opened, "書きかけ", "話者B"),
                     lambda: app.next_review_item(self.job, 2, opened, "資料を送ります。", "坪内"),
                     lambda: app.open_item(self.job, 1, opened, "書きかけ", "話者B"),
                     lambda: app.pick_item(self.job, "1|1", opened, "書きかけ", "話者B")):
            with self.assertRaises(app.gr.Error):
                move()
        self.assertEqual(app.step_item(self.job, 2, 1, opened, "資料を送ります。 ", "話者B")[0]["value"], 3)

    def test_save_item_targets_opened_utterance_not_number_box(self):
        opened = app.open_item(self.job, 1)[6]
        app.save_item(self.job, opened, "話者A", "直した本文。", False)  # 番号欄が何であれ、開いている発言1を保存
        saved = app._load_job("meeting")
        self.assertEqual(saved["utterances"][0]["text"], "直した本文。")
        self.assertEqual(saved["utterances"][1]["text"], "資料を送ります。")
        with self.assertRaises(app.gr.Error):  # 開いた後に本文が変わった（古い画面）→ 保存しない
            app.save_item(saved, opened, "話者A", "さらに変更", False)
        with self.assertRaises(app.gr.Error):  # 別の会議の記録 → 保存しない
            app.save_item(saved, dict(opened, job_dir="other"), "話者A", "x", False)
        with self.assertRaises(app.gr.Error):  # 開いていない
            app.save_item(saved, None, "話者A", "x", False)
        self.assertEqual(app._load_job("meeting")["utterances"][0]["text"], "直した本文。")

    def test_item_saves_coalesce_history_backups(self):
        app._item_backup_at.clear()
        opened = app.open_item(self.job, 1)[6]
        first = app.save_item(self.job, opened, "話者A", "1回目。", False)[1]
        second = app.save_item(first, first and app.open_item(first, 1)[6], "話者A", "2回目。", False)[1]
        self.assertEqual(len(app.job_history.choices(second)), 1)  # 10分以内の連続保存は履歴1件
        with patch.dict(app._item_backup_at, {str(self.folder): 0.0}):
            third = app.save_item(second, app.open_item(second, 1)[6], "話者A", "3回目。", False)[1]
        self.assertEqual(len(app.job_history.choices(third)), 2)
        app._item_backup_at.clear()

    def test_save_item_persists_learns_and_advances(self):
        app._item_backup_at.clear()
        opened = app.open_item(self.job, 1)[6]
        result = app.save_item(self.job, opened, "坪内", "預かり金を確認した。", True)
        self.assertIn("発言 1 を保存", result[0])
        saved = app._load_job("meeting")
        self.assertEqual(saved["utterances"][0]["text"], "預かり金を確認した。")
        self.assertEqual(saved["utterances"][0]["speaker"], "坪内")
        self.assertTrue(saved["utterances"][0]["human"])
        self.assertNotIn("color:red", saved["utterances"][0]["marked"])
        self.learn.assert_called_once()
        self.assertEqual(len(result), 2 + app.VIEW_COUNT)
        self.assertEqual(result[-1], "")  # 末尾は注意表示（警告なし）
        item_no = result[2 + 9]  # 分割用一覧の次が修正欄の番号
        self.assertEqual(item_no["value"], 2)
        self.assertTrue(result[2 + 8]["choices"])  # 分割用の話者一覧
        self.assertEqual(result[2 + 8]["choices"], ["坪内", "話者B", "話者A"])

    def test_save_item_rejects_unsaved_table_edits_and_empty_text(self):
        opened = app.open_item(self.job, 1)[6]
        dirty = app.mark_unsaved(self.job, "text")
        with self.assertRaises(app.gr.Error):
            app.save_item(dirty, opened, "話者A", "x", False)
        with self.assertRaises(app.gr.Error):
            app.save_item(self.job, opened, "話者A", "   ", False)
        self.assertEqual(app._load_job("meeting")["utterances"][0]["text"], "預かり菌を確認した。")

    def test_delete_item_keeps_history(self):
        result = app.delete_item(self.job, app.open_item(self.job, 2)[6])
        self.assertEqual(len(app._load_job("meeting")["utterances"]), 2)
        self.assertEqual(result[2 + 9]["value"], 2)
        self.assertTrue(app.job_history.choices(result[1]))

    def test_bulk_replace_saves_and_learns(self):
        self.assertIn("2 件の発言に 2 回", app.preview_replace(self.job, "預かり菌", "預かり金"))
        self.assertIn("ありません", app.preview_replace(self.job, "存在しない", ""))
        result = app.apply_replace(self.job, "預かり菌", "預かり金", True, 3)
        self.assertIn("2 件の発言", result[0])
        saved = app._load_job("meeting")
        self.assertEqual([u["text"] for u in saved["utterances"]],
                         ["預かり金を確認した。", "資料を送ります。", "預かり金の件です。"])
        self.pair.assert_called_once_with("預かり菌", "預かり金")
        self.assertEqual(result[2 + 9]["value"], 3)
        with self.assertRaises(app.gr.Error):
            app.apply_replace(self.job, "a", "a", False)

    def test_speaker_rows_include_voiceprint_candidates(self):
        job = dict(self.job, cluster_embeddings={"話者A": [1.0, 0.0], "話者B": [0.0, 1.0]})
        with patch.object(app.diarization, "load_voiceprint_groups",
                          return_value={"岡田": [np.array([0.9, 0.1], dtype="float32")],
                                        "石川": [np.array([0.1, 0.9], dtype="float32")]}):
            rows = app._speaker_rows(job)
        self.assertEqual(rows[0][0], "話者A")
        self.assertTrue(rows[0][2].startswith("岡田"))
        self.assertTrue(rows[1][2].startswith("石川"))

    def test_make_minutes_verifies_and_survives_verification_failure(self):
        with patch.object(app, "_llm_ready", return_value=""), \
                patch.object(app.minutes, "generate_minutes", return_value="- 項目（根拠 00:00:01）"), \
                patch.object(app.minutes, "verify_document", return_value=("- 直した項目（根拠 00:00:01）",
                                                                           {"checked": 1, "fixed": 1, "unverified": 0, "skipped": 0})):
            result = app.make_minutes(self.job, True, progress=Mock())
        self.assertIn("1項目を根拠に合わせて修正", result[0])
        self.assertEqual((self.folder / "議事録.md").read_text(encoding="utf-8"), "- 直した項目（根拠 00:00:01）")
        with patch.object(app, "_llm_ready", return_value=""), \
                patch.object(app.minutes, "generate_minutes", return_value="- 下書き"), \
                patch.object(app.minutes, "verify_document", side_effect=app.llm.LLMError("落ちた")):
            result = app.make_minutes(self.job, True, progress=Mock())
        self.assertIn("照合は完了しませんでした", result[0])
        self.assertEqual((self.folder / "議事録.md").read_text(encoding="utf-8"), "- 下書き")
        with patch.object(app, "_llm_ready", return_value=""), \
                patch.object(app.minutes, "generate_minutes", return_value="- 下書き2"), \
                patch.object(app.minutes, "verify_document") as verify:
            app.make_minutes(self.job, False, progress=Mock())
        verify.assert_not_called()

    def test_proofread_scope_review_only_targets_flagged_utterances(self):
        seen = []

        def fake(settings, utts, progress_cb=None, skip=None, on_chunk=None, memo=""):
            seen.extend(i for i, u in enumerate(utts) if not skip(u))
            return 0, 0

        with patch.object(app, "_llm_ready", return_value=""), patch.object(app.minutes, "proofread_utterances", side_effect=fake):
            app.proofread_job(self.job, False, "review", progress=Mock())
        self.assertEqual(seen, [0])
        seen.clear()
        with patch.object(app, "_llm_ready", return_value=""), patch.object(app.minutes, "proofread_utterances", side_effect=fake):
            app.proofread_job(self.job, False, "all", progress=Mock())
        self.assertEqual(seen, [0, 1, 2])


# =====================================================================
# 声紋による話者の見直し・分割・候補
# =====================================================================

def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


class VoiceprintRefineTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b = _unit([1, 0, 0]), _unit([0, 1, 0])
        self.groups = {"岡田": [self.a], "石川": [self.b]}

    def _segments(self):
        words = lambda s: [dict(start=s + k, end=s + k + 1, word="あ", prob=.9) for k in range(4)]
        return [dict(start=0, end=4, text="", words=words(0)), dict(start=10, end=14, text="", words=words(10)),
                dict(start=20, end=24, text="", words=words(20)), dict(start=30, end=32, text="", words=words(30)[:2])]

    def test_runs_relabel_only_with_clear_margin_and_enough_total(self):
        segments = self._segments()
        word_speakers = {0: ["岡田"] * 4, 1: ["岡田"] * 4, 2: ["岡田"] * 4, 3: ["岡田"] * 2}
        # 区間1・2は石川の声、区間0は岡田、区間3は短いので対象外
        embeds = [self.a, self.b, self.b]
        with patch.object(diarization, "load_voiceprint_groups", return_value=self.groups), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack(embeds)):
            changed, matched = diarization._refine_with_voiceprints(
                np.zeros(16000 * 40, dtype="float32"), segments, word_speakers, 0.55)
        self.assertEqual(changed, 2)
        self.assertEqual(word_speakers[1], ["石川"] * 4)
        self.assertEqual(word_speakers[0], ["岡田"] * 4)
        self.assertEqual(word_speakers[3], ["岡田"] * 2)
        self.assertAlmostEqual(matched["石川"], 1.0, places=2)

    def test_own_cluster_margin_blocks_relabel_and_failures_are_contained(self):
        segments = self._segments()
        word_speakers = {0: ["話者A"] * 4, 1: ["話者A"] * 4, 2: ["話者A"] * 4, 3: ["話者A"] * 2}
        with patch.object(diarization, "load_voiceprint_groups", return_value=self.groups), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack([self.a, self.a, self.a])):
            # 話者Aのクラスタ代表が岡田の声そのもの → 岡田本人かどうか区別できないので付け替えない
            changed, _ = diarization._refine_with_voiceprints(np.zeros(16000 * 40, dtype="float32"), segments,
                                                              word_speakers, 0.55, {"話者A": self.a})
            self.assertEqual(changed, 0)
            # 代表が別の声なら付け替える
            changed, _ = diarization._refine_with_voiceprints(np.zeros(16000 * 40, dtype="float32"), segments,
                                                              word_speakers, 0.55, {"話者A": _unit([0, 0, 1])})
            self.assertEqual(changed, 3)
        with patch.object(diarization, "load_voiceprint_groups", side_effect=RuntimeError("壊れた")):
            self.assertEqual(diarization._refine_with_voiceprints(np.zeros(10, dtype="float32"), segments, word_speakers, 0.55), (0, {}))

    def test_single_run_is_not_enough(self):
        segments = self._segments()
        word_speakers = {0: ["岡田"] * 4, 1: ["岡田"] * 4, 2: ["岡田"] * 4, 3: ["岡田"] * 2}
        with patch.object(diarization, "load_voiceprint_groups", return_value=self.groups), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack([self.a, self.b, self.a])):
            changed, _ = diarization._refine_with_voiceprints(np.zeros(16000 * 40, dtype="float32"), segments, word_speakers, 0.55)
        self.assertEqual(changed, 0)
        self.assertEqual(word_speakers[1], ["岡田"] * 4)

    def test_unnamed_speaker_requires_clear_second_margin(self):
        segments = self._segments()
        word_speakers = {0: ["話者A"] * 4, 1: ["話者A"] * 4, 2: ["話者A"] * 4, 3: ["話者A"] * 2}
        close = _unit([0.72, 0.69, 0])  # 2人の中間の声 → 付け替えない
        with patch.object(diarization, "load_voiceprint_groups", return_value=self.groups), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack([close, close, close])):
            changed, _ = diarization._refine_with_voiceprints(np.zeros(16000 * 40, dtype="float32"), segments, word_speakers, 0.55)
        self.assertEqual(changed, 0)
        with patch.object(diarization, "load_voiceprint_groups", return_value={"岡田": [self.a]}), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack([self.a] * 3)):
            self.assertEqual(diarization._refine_with_voiceprints(np.zeros(16000 * 40, dtype="float32"),
                                                                  segments, word_speakers, 0.55)[0], 0)  # 登録1人では使わない

    def test_split_speaker_two_voices(self):
        utts = [utterance(start=i * 5, speaker="話者A" if i < 8 else "話者B") for i in range(10)]
        voices = [self.a if i % 2 == 0 else self.b for i in range(8)]
        with patch.object(diarization, "_load_wav", return_value=np.zeros(16000 * 60, dtype="float32")), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack(voices)), \
                patch.object(diarization, "load_voiceprint_groups", return_value=self.groups):
            result = diarization.split_speaker("x.wav", utts, "話者A", 2, match_similarity=0.55)
        self.assertEqual(set(result["labels"]), set(range(8)))
        self.assertEqual(set(result["names"]), {"岡田", "石川"})
        self.assertEqual(result["counts"]["岡田"], 4)
        self.assertLess(result["separation"], 0.2)
        with patch.object(diarization, "_load_wav", return_value=np.zeros(16000 * 60, dtype="float32")), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack(voices)), \
                patch.object(diarization, "load_voiceprint_groups", return_value={}):
            result = diarization.split_speaker("x.wav", utts, "話者A", 2)
        self.assertEqual(set(result["names"]), {"話者A", "話者A（2）"})
        self.assertFalse(diarization.is_named("話者A（2）"))
        self.assertFalse(diarization.is_named("岡田（2）"))
        self.assertTrue(diarization.is_named("岡田"))

    def test_split_keeps_registered_name_on_matching_group_and_avoids_suffix_collision(self):
        # 「岡田」の8発言のうち、岡田本人の声は3件・残り5件は未登録の人。多い方ではなく本人の組に「岡田」を残す
        utts = [utterance(start=i * 5, speaker="岡田") for i in range(8)] + [utterance(start=50, speaker="岡田（2）")]
        voices = [self.a if i < 3 else _unit([0, 0, 1]) for i in range(8)]
        with patch.object(diarization, "_load_wav", return_value=np.zeros(16000 * 60, dtype="float32")), \
                patch.object(diarization, "_embed_chunks", return_value=np.stack(voices)), \
                patch.object(diarization, "load_voiceprint_groups", return_value={"岡田": [self.a], "石川": [self.b]}):
            result = diarization.split_speaker("x.wav", utts, "岡田", 2, match_similarity=0.55)
        self.assertEqual(result["counts"]["岡田"], 3)
        self.assertEqual(set(result["names"]), {"岡田", "岡田（3）"})  # 「岡田（2）」は既にあるので使わない
        self.assertEqual(result["counts"]["岡田（3）"], 5)
        with patch.object(diarization, "_load_wav", return_value=np.zeros(16000 * 60, dtype="float32")), \
                self.assertRaises(ValueError):
            diarization.split_speaker("x.wav", utts[:3], "話者A", 2)

    def test_cluster_candidates(self):
        with patch.object(diarization, "load_voiceprint_groups", return_value=self.groups):
            cand = diarization.cluster_candidates({"話者A": [0.9, 0.1, 0]}, top=2)
        self.assertEqual(cand["話者A"][0][0], "岡田")
        self.assertEqual(len(cand["話者A"]), 2)
        with patch.object(diarization, "load_voiceprint_groups", return_value={}):
            self.assertEqual(diarization.cluster_candidates({"話者A": [1, 0]}), {})

    def test_split_entry_runs_in_worker_and_saves(self):
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            folder = root / "meeting"
            folder.mkdir()
            stack.enter_context(patch.object(output, "OUTPUT_DIR", root))
            stack.enter_context(patch.object(config, "load_settings", return_value=copy.deepcopy(config.DEFAULTS)))
            stack.enter_context(patch.object(app.diarization, "available", return_value=True))
            stack.enter_context(patch.object(app.llm, "unload_ollama"))
            stack.enter_context(patch.object(app.people, "load_people", return_value=[]))
            job = dict(job_dir=str(folder), source="test.wav", duration=30,
                       utterances=[utterance(start=i * 5) for i in range(4)])
            pipeline._save_job(job)
            (folder / "audio.m4a").write_bytes(b"x")
            job = app._load_job("meeting")
            fake = {"labels": {0: "話者A", 1: "話者A（2）", 2: "話者A", 3: "話者A（2）"},
                    "names": ["話者A", "話者A（2）"], "counts": {"話者A": 2, "話者A（2）": 2},
                    "candidates": {}, "separation": 0.9}
            run = stack.enter_context(patch.object(app.worker, "run", return_value=fake))
            result = app.split_speaker_job(job, "話者A", 2, progress=Mock())
            self.assertEqual(run.call_args.args[0], "core.pipeline:split_speaker")
            self.assertIn("2 人に分けました", result[0])
            self.assertIn("似ています", result[0])
            saved = app._load_job("meeting")
            self.assertEqual([u["speaker"] for u in saved["utterances"]], ["話者A", "話者A（2）", "話者A", "話者A（2）"])
            self.assertTrue(app.job_history.choices(saved))


if __name__ == "__main__":
    unittest.main()
