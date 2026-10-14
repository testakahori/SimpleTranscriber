import copy
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from core import checkpoints as cp, output, pipeline, run_control as control, transcriber as t, worker
from pause_fixture import isolated, segment


class PauseTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.folder, self.wav, self.opts, self.settings = self.stack.enter_context(isolated(self.root))
        self.flag = str(self.root / "pause.request")

    def saved(self):
        return output.load_json(self.folder, cp.FILENAME)

    def run_model(self, model, **opts):
        with patch.object(t, "get_model", return_value=model):
            return pipeline._transcribe_one("original.wav", {**self.opts, **opts}, self.settings, Mock(), self.flag)

    def test_session_request_is_owned_idempotent_and_not_reused(self):
        with control.session("first") as first, control.session("other") as other:
            self.assertFalse(control.request_pause("unknown"))
            self.assertTrue(control.request_pause("first"))
            self.assertTrue(control.request_pause("first"))
            self.assertTrue(control.requested(first))
            self.assertFalse(control.requested(other))
            with self.assertRaises(ValueError):
                with control.session("first"):
                    pass
            self.assertTrue(control.request_pause("first"))
        self.assertFalse(Path(first).exists())
        self.assertFalse(control.request_pause("first"))
        with control.session("first") as new:
            self.assertNotEqual(new, first)
            self.assertFalse(control.requested(new))

    def test_pause_flushes_latest_main_result_and_resume_finishes(self):
        def generate():
            yield segment()
            Path(self.flag).touch()
            yield segment("最後の発言。", 5, 6)
            self.fail("pause must stop the iterator")
        model = Mock(transcribe=Mock(return_value=(generate(), SimpleNamespace(language="ja"))))
        with patch.object(cp.time, "monotonic", return_value=1):
            paused = self.run_model(model)
        self.assertTrue(paused["incomplete"])
        self.assertTrue(self.saved()["paused"])
        self.assertFalse(self.saved()["interrupted"])
        self.assertEqual(self.saved()["processed_sec"], 6)
        self.assertEqual(self.saved()["segments"][-1]["text"], "最後の発言。")
        self.assertIn("一時停止しました", cp.notice(paused))
        self.assertNotIn("処理エラー", cp.notice(paused))
        self.assertFalse((self.folder / "utterances.json").exists())
        cp.audio_paths(self.folder, self.saved())
        with cp.job_lock(self.folder):
            pass
        # 次の実行には前回の要求ファイルを渡さない。境界以降だけを認識して完成する。
        def finish(path, **kwargs):
            import soundfile as sf
            self.assertEqual(sf.info(path).duration, 16)
            return iter([segment("再開した発言。", 1, 2), segment("続き。", 8, 9)]), SimpleNamespace(language="ja")
        model.transcribe.side_effect = finish
        with patch.object(t, "get_model", return_value=model):
            done = pipeline.resume_job(str(self.folder), [], {}, self.settings)
        self.assertFalse(done.get("incomplete"))
        self.assertEqual([u["start"] for u in done["utterances"]], [0, 5, 12])
        self.assertFalse(self.saved()["paused"])
        self.assertFalse((self.folder / cp.RESUME_AUDIO[0]).exists())

    def test_pause_during_preparation_saves_audio_before_stopping(self):
        Path(self.flag).touch()
        with patch.object(t, "transcribe") as transcribe:
            paused = pipeline._transcribe_one("original.wav", self.opts, self.settings, Mock(), self.flag)
        transcribe.assert_not_called()
        self.assertEqual(self.saved()["phase"], "prepared")
        self.assertTrue(paused["checkpoint"]["paused"])
        cp.validate(self.saved())
        cp.audio_paths(self.folder, self.saved())

    def test_pause_after_model_loading_does_not_start_recognition(self):
        def load(name):
            Path(self.flag).touch()
            return Mock()
        with patch.object(t, "get_model", side_effect=load):
            paused = pipeline._transcribe_one("original.wav", self.opts, self.settings, Mock(), self.flag)
        self.assertEqual(paused["checkpoint"]["phase"], "prepared")
        self.assertTrue(paused["checkpoint"]["paused"])

    def test_pause_during_diarization_preserves_raw_asr_and_resumes_without_whisper(self):
        model = Mock(transcribe=Mock(return_value=(iter([segment()]), SimpleNamespace(language="ja"))))
        def diarize(*args, **kwargs):
            Path(self.flag).touch()
            return {}, {}, {}
        with patch.object(pipeline.diarization, "available", return_value=True), \
                patch.object(pipeline.diarization, "diarize", side_effect=diarize):
            self.run_model(model, diarize=True)
        self.assertEqual(self.saved()["phase"], "postprocess")
        self.assertEqual(self.saved()["segments"][0]["text"], "保存する発言。")
        with patch.object(t, "transcribe") as transcribe, patch.object(pipeline.diarization, "available", return_value=True), \
                patch.object(pipeline.diarization, "diarize", return_value=({}, {}, {})) as diarize:
            pipeline.resume_job(str(self.folder), [], {}, self.settings)
        transcribe.assert_not_called()
        diarize.assert_called_once()

    def test_pause_save_failure_is_an_error_and_keeps_resume_audio(self):
        real_save = output.save_json
        def save(folder, filename, data):
            if filename == cp.FILENAME and data.get("paused"):
                raise OSError("disk full")
            return real_save(folder, filename, data)
        Path(self.flag).touch()
        with patch.object(output, "save_json", side_effect=save), self.assertRaisesRegex(OSError, "disk full"):
            pipeline._transcribe_one("original.wav", self.opts, self.settings, Mock(), self.flag)
        self.assertFalse(self.saved()["paused"])
        cp.audio_paths(self.folder, self.saved())
        self.assertFalse((self.folder / "utterances.json").exists())

    def test_transient_windows_sharing_error_retries_without_losing_previous_data(self):
        output.write_text(self.folder, "test.md", "old")
        locked = PermissionError("file in use")
        locked.winerror = 32
        real_replace = output.os.replace
        attempts = 0
        def replace(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                self.assertEqual((self.folder / "test.md").read_text(), "old")
                raise locked
            return real_replace(source, destination)
        with patch.object(output.os, "replace", side_effect=replace), patch.object(output.time, "sleep") as sleep:
            output.write_text(self.folder, "test.md", "new")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual((self.folder / "test.md").read_text(), "new")
        with patch.object(output.os, "replace", side_effect=locked), patch.object(output.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                output.write_text(self.folder, "test.md", "must not replace")
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual((self.folder / "test.md").read_text(), "new")

    def test_completed_file_is_not_marked_paused_if_request_arrives_during_final_save(self):
        real_save = pipeline._save_job
        def save(job):
            Path(self.flag).touch()
            real_save(job)
        model = Mock(transcribe=Mock(return_value=(iter([segment()]), SimpleNamespace(language="ja"))))
        with patch.object(pipeline, "_save_job", side_effect=save):
            done = self.run_model(model)
        self.assertFalse(done.get("incomplete"))
        self.assertTrue((self.folder / "utterances.json").exists())
        self.assertFalse(self.saved()["paused"])

    def test_batch_stops_before_next_file_and_reports_unstarted_names(self):
        def generate():
            Path(self.flag).touch()
            yield segment()
        model = Mock(transcribe=Mock(return_value=(generate(), SimpleNamespace(language="ja"))))
        with patch.object(t, "get_model", return_value=model), patch.object(pipeline, "_transcribe_one", wraps=pipeline._transcribe_one) as run:
            results = pipeline.transcribe_files(["original.wav", "next.wav"], self.opts, self.settings, pause_file=self.flag)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(results[0]["paused"]["incomplete"])
        self.assertEqual(results[0]["remaining"], ["next.wav"])
        with patch.object(pipeline, "_transcribe_one") as run:
            early = pipeline.transcribe_files(["next.wav"], self.opts, self.settings, pause_file=self.flag)
        run.assert_not_called()
        self.assertIsNone(early[0]["paused"])

    def test_actual_worker_receives_pause_and_exits_releasing_job_lock(self):
        module_dir = str(Path(__file__).parent)
        with patch.dict(os.environ, {"PYTHONPATH": module_dir}), control.session("worker-test") as flag:
            def progress(f, desc):
                if desc == "ready-to-pause":
                    self.assertTrue(control.request_pause("worker-test"))
            result = worker.run("pause_fixture:run", str(self.root), progress=progress, pause_file=flag)
        self.assertNotEqual(result["pid"], os.getpid())
        self.assertTrue(result["job"]["checkpoint"]["paused"])
        self.assertEqual(self.saved()["segments"][-1]["text"], "最後に保存する発言。")
        with cp.job_lock(self.folder):
            pass

    def test_ui_receipt_and_paused_result_are_not_reported_as_success_or_failure(self):
        request = SimpleNamespace(session_hash="ui")
        with control.session("ui") as flag:
            self.assertIn("受け付けました", app.pause_transcription(request))
            self.assertTrue(control.requested(flag))
        self.assertIn("ありません", app.pause_transcription(request))
        Path(self.flag).touch()
        paused = pipeline._transcribe_one("original.wav", self.opts, self.settings, Mock(), self.flag)
        with patch.object(app.worker, "run", return_value=[{"paused": paused, "remaining": ["next.wav"]}]), \
                patch.object(app.llm, "unload_ollama"), patch.object(app, "_write_meta") as meta:
            result = app.run_batch(["original.wav", "next.wav"], "auto", "accurate", "ja", 0,
                                   False, False, False, False, "", "", progress=Mock(), request=request)
        meta.assert_not_called()
        self.assertIn("一時停止しました", result[0])
        self.assertIn("next.wav", result[0])
        self.assertTrue(result[1]["incomplete"])
        self.assertNotIn("❌", result[0])
        self.assertFalse(control.request_pause("ui"))
        with patch.object(app.worker, "run", return_value=paused), patch.object(app.llm, "unload_ollama"), \
                patch.object(app, "_write_meta") as meta:
            result = app.resume_transcription(paused, [], "auto", "accurate", "ja", 0,
                                              False, False, False, False, "", "", progress=Mock(), request=request)
        meta.assert_not_called()
        self.assertIn("一時停止しました", result[0])


class RecoveryPauseTest(unittest.TestCase):
    def test_pause_commits_completed_window_and_skips_remaining_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flag = str(root / "pause.request")
            import numpy as np
            import soundfile as sf
            wav = root / "test.wav"
            sf.write(wav, np.zeros(20 * 16000), 16000)
            saved = []
            def generate():
                Path(flag).touch()
                yield segment("補完した発言。", .5, 1)
            model = Mock(transcribe=Mock(return_value=(generate(), None)))
            with patch.object(t, "_missed_spans", return_value=[(5, 7), (10, 12)]):
                with self.assertRaises(control.PauseRequested):
                    t._fill_gaps(model, str(wav), [], 20, {},
                                 checkpoint_cb=lambda data: saved.append(copy.deepcopy(data)),
                                 pause_cb=lambda: control.check(flag))
            self.assertEqual(model.transcribe.call_count, 1)
            self.assertEqual(saved[-1]["recovery"]["next_index"], 1)
            self.assertEqual(saved[-1]["segments"][0]["start"], 5)


if __name__ == "__main__":
    unittest.main()
