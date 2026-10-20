"""途中停止・履歴再読込・議事録の鮮度を合成データで確認。音声/GPU/実名辞書は使わない。"""
import copy
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import app
from core import checkpoints, config, document_status as docs, output, pipeline


def segment(text="開始は10時です。", start=20, end=23):
    return {"start": start, "end": end, "text": text, "words": []}


def job_at(folder):
    return {"job_dir": str(folder), "source": "test.wav", "duration": 100,
            "meeting_date": "2026-10-14", "memo": "", "meta_line": "",
            "utterances": [{"start": 20, "end": 23, "text": "開始は10時です。",
                            "marked": "開始は10時です。", "speaker": "話者1", "words": []}]}


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.folder = self.root / "test"
        self.folder.mkdir()
        self.stack.enter_context(patch.object(output, "OUTPUT_DIR", self.root))
        settings = config._deep_merge(config.DEFAULTS, {"diarization": {"auto_learn": False}})
        self.stack.enter_context(patch.object(config, "load_settings", return_value=settings))
        self.settings = settings

    def writer(self):
        return checkpoints.Writer(self.folder, "test.wav", 100, "2026-10-14", "")

    def test_atomic_save_keeps_previous_file_after_replace_or_serialization_failure(self):
        output.save_json(self.folder, "data.json", {"old": True})
        with patch.object(output.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                output.save_json(self.folder, "data.json", {"new": True})
        with self.assertRaises(TypeError):
            output.save_json(self.folder, "data.json", object())
        self.assertEqual(output.load_json(self.folder, "data.json"), {"old": True})
        self.assertEqual(list(self.folder.glob("*.tmp")), [])

    def test_checkpoint_is_throttled_but_phase_end_and_interruption_flush_latest(self):
        writer = self.writer()
        with patch.object(checkpoints.time, "monotonic", side_effect=[0, 5, 11, 12, 13]):
            for i in (1, 2, 3):
                writer({"phase": "main", "segments": [segment(str(i))], "processed_sec": 23 + i})
                expected = "1" if i <= 2 else "3"
                self.assertEqual(output.load_json(self.folder, checkpoints.FILENAME)["segments"][0]["text"], expected)
            writer({"phase": "main_complete", "segments": [segment("main complete")], "processed_sec": 100})
            self.assertEqual(output.load_json(self.folder, checkpoints.FILENAME)["phase"], "main_complete")
            writer({"phase": "recovery", "segments": [segment("recovered")]})
            writer.interrupted()
        saved = output.load_json(self.folder, checkpoints.FILENAME)
        self.assertTrue(saved["interrupted"])
        self.assertEqual(saved["processed_sec"], 100)
        self.assertEqual(saved["segments"][0]["text"], "recovered")

    def test_partial_reopens_with_timestamps_notice_and_download_without_becoming_complete(self):
        self.writer()({"phase": "main", "segments": [segment()], "processed_sec": 23})
        self.assertEqual(output.list_editable_jobs(), ["test"])
        loaded = app._load_job("test")
        self.assertTrue(loaded["incomplete"])
        self.assertEqual(loaded["utterances"][0]["start"], 20)
        view = app.open_job("test")
        self.assertEqual(len(view), 2 + app.VIEW_COUNT)
        self.assertIn("全編未完了", view[-1])
        download = Path(view[5]["value"])
        self.assertIn("開始は10時", download.read_text(encoding="utf-8"))
        self.assertIn("全編未完了", download.read_text(encoding="utf-8"))
        self.assertFalse(view[6]["interactive"])  # 議事録
        self.assertFalse(view[9]["interactive"])  # 修正表
        self.assertIn("開始は10時", view[2])
        self.assertFalse((self.folder / "utterances.json").exists())
        # 表示用の出力が、ワーカーの新しい途中結果を巻き戻さないこと。
        self.writer()({"phase": "main", "segments": [segment("新しい発言")], "processed_sec": 50})
        app._download_transcript(loaded)
        self.assertIn("新しい発言", (self.folder / checkpoints.TRANSCRIPT).read_text(encoding="utf-8"))

    def test_partial_blocks_edit_learning_and_document_generation(self):
        self.writer()({"phase": "main", "segments": [segment()], "processed_sec": 23})
        loaded = app._load_job("test")
        with patch.object(app, "_learn_voices") as learn, patch.object(app.worker, "run") as worker:
            for call in [lambda: app.save_utterance_edits([], loaded),
                         lambda: app.assign_speakers([], True, loaded),
                         lambda: app.reflow_job(loaded),
                         lambda: app.rediarize_job(loaded, 0),
                         lambda: app.proofread_job(loaded, False),
                         lambda: app.make_minutes(loaded)]:
                self.assertIn("閲覧・ダウンロード用", call()[0])
            learn.assert_not_called()
            worker.assert_not_called()
        with self.assertRaises(ValueError):
            pipeline._save_job(loaded)

    def pipeline_setup(self, mock_transcribe, diarize=False):
        wav = self.root / "temp.wav"
        wav.touch()
        for obj, name, value in [
            (output, "create_job_dir", self.folder), (pipeline.audio, "prepare_audio", str(wav)),
            (pipeline.audio, "get_duration", 100), (pipeline.audio, "make_preview_audio", None),
            (pipeline.audio, "enhance_for_asr", str(wav)), (pipeline.people, "load_people", []),
            (pipeline.people, "initial_prompt_terms", []), (pipeline.glossary, "build_initial_prompt", ""),
            (pipeline.glossary, "build_hotwords", "")]:
            self.stack.enter_context(patch.object(obj, name, return_value=value))
        self.stack.enter_context(patch.object(pipeline, "_finalize_text"))
        self.stack.enter_context(patch.object(pipeline.transcriber, "transcribe", side_effect=mock_transcribe))
        return dict(model="auto", language="ja", noise=False, diarize=diarize, fillers=False, num_speakers=0)

    def test_asr_failure_saves_latest_even_inside_throttle_window(self):
        def fail(*args, checkpoint_cb, **kwargs):
            checkpoint_cb({"phase": "main", "segments": [segment("前の発言")], "processed_sec": 23})
            checkpoint_cb({"phase": "main", "segments": [segment("最後の発言")], "processed_sec": 24})
            raise RuntimeError("simulated interruption")
        opts = self.pipeline_setup(fail)
        result = pipeline.transcribe_files(["test.wav"], opts, self.settings)
        self.assertIn("simulated interruption", result[0]["error"])
        loaded = app._load_job("test")
        self.assertIn("最後の発言", loaded["utterances"][0]["text"])
        self.assertTrue(loaded["checkpoint"]["interrupted"])
        self.assertFalse((self.folder / "utterances.json").exists())
        self.assertFalse((self.root / "temp.wav").exists())

    def test_diarization_failure_leaves_full_asr_available(self):
        opts = self.pipeline_setup(lambda *a, **k: {"segments": [segment()]}, diarize=True)
        with patch.object(pipeline.diarization, "available", return_value=True), \
                patch.object(pipeline.diarization, "diarize", side_effect=RuntimeError("diarization failed")) as diarize:
            result = pipeline.transcribe_files(["test.wav"], opts, self.settings)
        diarize.assert_called_once()
        self.assertIn("diarization failed", result[0]["error"])
        loaded = app._load_job("test")
        self.assertTrue(loaded["incomplete"])
        self.assertEqual(loaded["checkpoint"]["phase"], "postprocess")
        self.assertEqual(loaded["checkpoint"]["processed_sec"], 100)
        self.assertIn("開始は10時", loaded["utterances"][0]["text"])

    def test_complete_result_takes_precedence_over_checkpoint(self):
        opts = self.pipeline_setup(lambda *a, **k: {"segments": [segment()]})
        result = pipeline.transcribe_files(["test.wav"], opts, self.settings)
        self.assertIn("job", result[0])
        self.assertTrue((self.folder / checkpoints.FILENAME).exists())
        loaded = app._load_job("test")
        self.assertFalse(loaded.get("incomplete"))
        self.assertEqual(app._job_views(loaded)[-1], "")
        self.assertEqual(output.list_editable_jobs(), ["test"])

    def test_checkpoint_survives_process_exit_without_cleanup(self):
        code = """
import os, sys
from pathlib import Path
from core.checkpoints import Writer
writer = Writer(Path(sys.argv[1]), 'test.wav', 100, '', '')
writer({'phase': 'main', 'segments': [{'start': 20, 'end': 23, 'text': 'saved'}], 'processed_sec': 23})
os._exit(17)
"""
        proc = subprocess.run([sys.executable, "-c", code, str(self.folder)],
                              capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 17, proc.stderr)
        loaded = app._load_job("test")
        self.assertTrue(loaded["incomplete"])
        self.assertEqual(loaded["utterances"][0]["text"], "saved")

    def test_recovery_failure_keeps_main_pass_and_completed_recovery_windows(self):
        def fail(*args, checkpoint_cb, **kwargs):
            checkpoint_cb({"phase": "main_complete", "segments": [segment()], "processed_sec": 100})
            checkpoint_cb({"phase": "recovery", "segments": [segment(), segment("補完した発言", 50, 53)]})
            raise RuntimeError("recovery failed")
        opts = self.pipeline_setup(fail)
        pipeline.transcribe_files(["test.wav"], opts, self.settings)
        loaded = app._load_job("test")
        self.assertEqual(loaded["checkpoint"]["phase"], "recovery")
        self.assertEqual([u["start"] for u in loaded["utterances"]], [20, 50])
        self.assertIn("補完した発言", loaded["utterances"][1]["text"])

    def test_failed_worker_refreshes_history_including_checkpoints(self):
        self.writer()({"phase": "main", "segments": [segment()], "processed_sec": 23})
        with patch.object(app.llm, "unload_ollama"), \
                patch.object(app.worker, "run", side_effect=RuntimeError("worker killed")):
            result = app.run_batch(["test.wav"], "auto", "accurate", "ja", 0,
                                   False, False, False, False, "", "", progress=Mock())
        self.assertEqual(len(result), app.VIEW_COUNT + 3)
        self.assertEqual(result[-1]["choices"], ["test"])

    def test_document_fingerprint_tracks_inputs_not_editing_flags(self):
        job = job_at(self.folder)
        source = docs.inputs(job)
        docs.save(self.folder, "議事録.md", "開始10時", source)
        self.assertEqual(docs.status(job, "議事録.md"), "current")
        for field, value in [("speaker", "話者2"), ("marked", "開始11時"), ("start", 30)]:
            changed = copy.deepcopy(job)
            changed["utterances"][0][field] = value
            self.assertEqual(docs.status(changed, "議事録.md"), "stale")
        for field in ("memo", "meeting_date", "source"):
            changed = dict(job, **{field: "changed"})
            self.assertEqual(docs.status(changed, "議事録.md"), "stale")
        job["utterances"][0]["human"] = True
        job["utterances"][0]["proofread"] = True
        self.assertEqual(docs.status(job, "議事録.md"), "current")

    def test_edit_reopen_warn_regenerate_cycle_and_independent_insights(self):
        job = job_at(self.folder)
        pipeline._save_job(job)
        for name in docs.DOCUMENTS:
            docs.save(self.folder, name, "開始は10時", docs.inputs(job))
        with patch.object(app.glossary, "learn_from_diff", return_value=0):
            result = app.save_utterance_edits([[1, "00:00:20", "話者1", "開始は11時です。"]], job)
        self.assertIn("更新前の文書:", result[-1])
        self.assertIn("議事録", result[-1])
        self.assertIn("開始は11時", result[2])
        self.assertIn("要約と考察", result[-1])
        loaded = app._load_job("test")
        self.assertEqual(docs.status(loaded, "議事録.md"), "stale")
        self.assertIn("更新前", app._job_views(loaded)[4]["label"])
        self.assertEqual((self.folder / "議事録.md").read_text(encoding="utf-8"), "開始は10時")
        with patch.object(app, "_llm_ready", return_value=""), \
                patch.object(app.minutes, "generate_minutes", return_value="開始は11時"):
            result = app.make_minutes(loaded, progress=Mock())
        self.assertEqual(docs.status(app._load_job("test"), "議事録.md"), "current")
        self.assertNotIn("議事録", result[-1])
        self.assertIn("更新前の文書: 要約と考察", result[-1])

    def test_failed_generation_preserves_old_document_and_warning(self):
        job = job_at(self.folder)
        docs.save(self.folder, "議事録.md", "以前の議事録", docs.inputs(job))
        job["memo"] = "変更"
        before = (self.folder / docs.INDEX).read_bytes()
        with patch.object(app, "_llm_ready", return_value=""), \
                patch.object(app.minutes, "generate_minutes", side_effect=app.llm.LLMError("failed")):
            result = app.make_minutes(job, progress=Mock())
        self.assertIn("作れませんでした", result[0])
        self.assertEqual((self.folder / docs.INDEX).read_bytes(), before)
        self.assertEqual((self.folder / "議事録.md").read_text(encoding="utf-8"), "以前の議事録")
        self.assertEqual(docs.status(job, "議事録.md"), "stale")

    def test_legacy_or_externally_edited_document_is_unknown(self):
        job = job_at(self.folder)
        output.write_text(self.folder, "議事録.md", "古い議事録")
        self.assertEqual(docs.status(job, "議事録.md"), "unknown")
        docs.save(self.folder, "議事録.md", "新しい議事録", docs.inputs(job))
        output.write_text(self.folder, "議事録.md", "外部で編集")
        self.assertEqual(docs.status(job, "議事録.md"), "unknown")

    def test_windows_line_endings_and_invalid_metadata_are_handled(self):
        job = job_at(self.folder)
        docs.save(self.folder, "議事録.md", "一行目\r\n二行目\r\n", docs.inputs(job))
        self.assertEqual(docs.status(job, "議事録.md"), "current")
        for data in ([], {"議事録.md": None}):
            output.save_json(self.folder, docs.INDEX, data)
            self.assertEqual(docs.status(job, "議事録.md"), "unknown")

    def test_generation_uses_input_snapshot_and_partial_metadata_write_is_unknown(self):
        job = job_at(self.folder)
        source = docs.inputs(job)
        job["memo"] = "生成中に変更"
        docs.save(self.folder, "議事録.md", "古い入力で生成", source)
        self.assertEqual(docs.status(job, "議事録.md"), "stale")
        with patch.object(output, "save_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                docs.save(self.folder, "議事録.md", "新しい内容", docs.inputs(job))
        self.assertEqual(docs.status(job, "議事録.md"), "unknown")

    def test_export_warns_without_changing_original_document(self):
        job = job_at(self.folder)
        docs.save(self.folder, "議事録.md", "開始10時", docs.inputs(job))
        job["memo"] = "変更"
        pipeline._save_job(job)
        with patch.object(app.export_mod, "md_to_docx"):
            msg, files = app.do_export("test", "議事録.md", True, False)
        self.assertIn("更新前", msg)
        self.assertTrue(files)
        self.assertEqual((self.folder / "議事録.md").read_text(encoding="utf-8"), "開始10時")


if __name__ == "__main__":
    unittest.main()
