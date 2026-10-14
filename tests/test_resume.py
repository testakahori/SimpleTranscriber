"""保存→プロセス終了→再開を合成音声で検証。実録音・実名辞書・GPUは使わない。"""
import copy
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import app
from core import audio, checkpoints as cp, config, output, pipeline, transcriber as t


def segment(text="保存済み。", start=1., end=3.):
    return dict(text=text, start=start, end=end, words=[dict(word=text, start=start, end=end, prob=.9)])


def recognized(text="続きです。", start=0., end=2.):
    return SimpleNamespace(text=text, start=start, end=end, no_speech_prob=.01, avg_logprob=-.2,
                           compression_ratio=1.1,
                           words=[SimpleNamespace(word=text, start=start, end=end, probability=.9)])


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.folder = self.root / "job"
        self.folder.mkdir()
        self.wav = self.root / "original.wav"
        sf.write(self.wav, np.zeros(20 * 16000), 16000, subtype="PCM_16")
        self.settings = config._deep_merge(config.DEFAULTS, {"diarization": {"auto_learn": False}})
        self.opts = pipeline._run_options(dict(model="tiny", noise=False, eq=False, fillers=False), self.settings)
        self.stack.enter_context(patch.object(output, "OUTPUT_DIR", self.root))
        for obj, key, value in [(config, "load_settings", self.settings),
                                (pipeline.people, "load_people", []), (pipeline.people, "initial_prompt_terms", []),
                                (pipeline.glossary, "build_initial_prompt", "保存したヒント"),
                                (pipeline.glossary, "build_hotwords", "保存した用語"),
                                (audio, "make_preview_audio", None)]:
            self.stack.enter_context(patch.object(obj, key, return_value=value))
        self.stack.enter_context(patch.object(pipeline, "_finalize_text"))

    def saved(self, phase="main", legacy=False):
        writer = cp.Writer(self.folder, self.wav.name, 20, "", "メモ")
        if not legacy:
            writer.base.update(resume_audio=cp.keep_audio(self.folder, str(self.wav), str(self.wav)),
                               run_options=self.opts, hints=dict(initial_prompt="保存したヒント", hotwords="用語"))
        writer(dict(phase=phase, segments=[segment()], processed_sec=3 if phase == "main" else 20))
        return output.load_json(self.folder, cp.FILENAME)

    def test_saved_audio_and_settings_used_after_original_removed(self):
        self.saved()
        self.wav.unlink()
        model = Mock(return_value=dict(segments=[segment(), segment("続き。", 10, 12)], model="tiny", speed="accurate"))
        with patch.object(t, "transcribe", model), patch.object(audio, "prepare_audio") as prepare, \
                patch.object(audio, "enhance_for_asr") as enhance:
            job = pipeline.resume_job(str(self.folder), [], dict(model="large-v3", noise=True), self.settings)
        self.assertEqual(model.call_args.kwargs["model_name"], "tiny")
        self.assertEqual(model.call_args.kwargs["initial_prompt"], "保存したヒント")
        self.assertEqual(model.call_args.kwargs["resume"]["phase"], "main")
        prepare.assert_not_called()
        enhance.assert_not_called()
        self.assertEqual(Path(job["job_dir"]), self.folder)
        self.assertEqual(job["memo"], "メモ")
        self.assertTrue((self.folder / "utterances.json").exists())
        self.assertFalse(any((self.folder / n).exists() for n in cp.RESUME_AUDIO))
        self.assertFalse(app._load_job("job").get("incomplete"))
        with self.assertRaisesRegex(ValueError, "完了済み"):
            pipeline.resume_job(str(self.folder), [], {}, self.settings)

    def test_audio_hash_failure_does_not_change_saved_data(self):
        self.saved()
        before = (self.folder / cp.FILENAME).read_bytes()
        (self.folder / cp.RESUME_AUDIO[0]).write_bytes(b"changed")
        with patch.object(t, "transcribe") as model, self.assertRaisesRegex(ValueError, "変更されています"):
            pipeline.resume_job(str(self.folder), [], {}, self.settings)
        model.assert_not_called()
        self.assertEqual((self.folder / cp.FILENAME).read_bytes(), before)

    def test_separate_recognition_audio_is_retained_and_checked(self):
        enhanced = self.root / "enhanced.wav"
        sf.write(enhanced, np.full(20 * 16000, .01), 16000)
        records = cp.keep_audio(self.folder, str(self.wav), str(enhanced))
        source_path, asr_path = cp.audio_paths(self.folder, dict(resume_audio=records))
        self.assertNotEqual(source_path, asr_path)
        self.assertEqual(Path(source_path).read_bytes(), self.wav.read_bytes())
        self.assertEqual(Path(asr_path).read_bytes(), enhanced.read_bytes())
        Path(asr_path).unlink()
        with self.assertRaisesRegex(ValueError, "見つからない"):
            cp.audio_paths(self.folder, dict(resume_audio=records))

    def test_postprocess_failure_resumes_without_whisper_and_keeps_raw_segments(self):
        data = self.saved("postprocess")
        data["asr_result"] = dict(model="tiny", speed="accurate", elapsed=9, dropped=2)
        output.save_json(self.folder, cp.FILENAME, data)
        def change(segments):
            segments[0]["text"] = "整形した文章。"
        with patch.object(t, "transcribe") as model, patch.object(pipeline.punctuate, "punctuate_segments", side_effect=change), \
                patch.object(pipeline, "_save_job", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                pipeline.resume_job(str(self.folder), [], {}, self.settings)
        model.assert_not_called()
        saved = output.load_json(self.folder, cp.FILENAME)
        self.assertEqual(saved["segments"][0]["text"], "保存済み。")
        self.assertTrue(saved["interrupted"])
        self.assertTrue((self.folder / cp.RESUME_AUDIO[0]).exists())
        with patch.object(t, "transcribe") as model:
            job = pipeline.resume_job(str(self.folder), [], {}, self.settings)
        model.assert_not_called()
        self.assertEqual(job["stats"]["elapsed_sec"], 9)
        self.assertFalse(output.load_json(self.folder, cp.FILENAME)["interrupted"])

    def test_initial_model_failure_is_resumable_and_does_not_save_tokens(self):
        settings = copy.deepcopy(self.settings)
        settings["diarization"]["hf_token"] = "TEST_SECRET_DO_NOT_SAVE"
        with patch.object(output, "create_job_dir", return_value=self.folder), \
                patch.object(audio, "prepare_audio", return_value=str(self.wav)), \
                patch.object(t, "transcribe", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                pipeline._transcribe_one("original.wav", self.opts, settings, Mock())
        data = output.load_json(self.folder, cp.FILENAME)
        self.assertEqual(data["phase"], "prepared")
        self.assertEqual(data["segments"], [])
        self.assertEqual(data["schema_version"], 2)
        self.assertNotIn("TEST_SECRET", (self.folder / cp.FILENAME).read_text(encoding="utf-8"))
        cp.validate(data)
        self.assertTrue(Path(cp.audio_paths(self.folder, data)[0]).exists())

    def test_legacy_requires_matching_original_then_migrates(self):
        self.saved(legacy=True)
        before = (self.folder / cp.FILENAME).read_bytes()
        for paths in ([], ["wrong.wav"], [str(self.wav)] * 2):
            with self.assertRaisesRegex(ValueError, "同じ名前"):
                pipeline.resume_job(str(self.folder), paths, self.opts, self.settings)
        wrong_length = self.root / "converted.wav"
        sf.write(wrong_length, np.zeros(10 * 16000), 16000)
        with patch.object(audio, "prepare_audio", return_value=str(wrong_length)), \
                self.assertRaisesRegex(ValueError, "長さ"):
            pipeline.resume_job(str(self.folder), [str(self.wav)], self.opts, self.settings)
        self.assertEqual((self.folder / cp.FILENAME).read_bytes(), before)
        with patch.object(audio, "prepare_audio", return_value=str(self.wav)), \
                patch.object(t, "transcribe", side_effect=RuntimeError("later failure")):
            with self.assertRaises(RuntimeError):
                pipeline.resume_job(str(self.folder), [str(self.wav)], self.opts, self.settings)
        data = output.load_json(self.folder, cp.FILENAME)
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["segments"][0]["text"], "保存済み。")
        self.assertTrue(Path(cp.audio_paths(self.folder, data)[0]).exists())

    def test_resume_rewinds_boundary_and_keeps_absolute_word_times_in_all_modes(self):
        saved = self.saved()
        saved.update(processed_sec=10, segments=[segment("前半。", 1, 3), segment("境界の旧文。", 7, 10)],
                     asr_info=dict(model="tiny", language="ja"), elapsed=12)
        for mode in ("accurate", "balanced", "fast"):
            with self.subTest(mode=mode), ExitStack() as stack:
                model = Mock()
                tails = []
                def recognize(path, **kwargs):
                    tails.append(path)
                    self.assertAlmostEqual(sf.info(path).duration, 13)
                    self.assertTrue(kwargs["vad_filter"])
                    self.assertNotIn("clip_timestamps", kwargs)
                    return iter([recognized("境界の新文。"), recognized("後半。", 5, 5.4)]), SimpleNamespace(language="ja")
                model.transcribe.side_effect = recognize
                stack.enter_context(patch.object(t, "detect_device", return_value=("cpu", "int8")))
                get = stack.enter_context(patch.object(t, "get_model", return_value=model))
                stack.enter_context(patch.object(t, "_fill_gaps", return_value=dict(dropped=0, refilled_sec=0, missing_sec=0)))
                if mode == "fast":
                    stack.enter_context(patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(BatchedInferencePipeline=Mock(return_value=model))}))
                result = t.transcribe(str(self.wav), speed=mode, resume=saved, duration=20)
                get.assert_called_once_with("tiny")
                self.assertEqual([s["text"] for s in result["segments"]], ["前半。", "境界の新文。", "後半。"])
                self.assertEqual([s["start"] for s in result["segments"]], [1, 7, 12])
                self.assertEqual(result["segments"][2]["words"][0]["start"], 12)
                self.assertGreaterEqual(result["elapsed"], 12)
                self.assertFalse(Path(tails[0]).exists())
        self.assertEqual(saved["segments"][-1]["text"], "境界の旧文。")

    def test_audio_tail_preserves_samples_and_cleans_up_on_error(self):
        samples = np.arange(20 * 16000, dtype=np.int16)
        sf.write(self.wav, samples, 16000, subtype="PCM_16")
        with self.assertRaises(RuntimeError):
            with audio.asr_tail(str(self.wav), 7.125) as (tail, offset):
                np.testing.assert_array_equal(sf.read(tail, dtype="int16")[0], samples[int(offset * 16000):])
                self.assertEqual(offset, 7.125)
                raise RuntimeError()
        self.assertFalse(Path(tail).exists())

    def test_recovery_resumes_after_completed_empty_window(self):
        model = Mock()
        model.transcribe.side_effect = [(iter([]), None), RuntimeError("stopped")]
        snapshots = []
        with patch.object(t, "_missed_spans", return_value=[(5, 7), (10, 12)]):
            with self.assertRaises(RuntimeError):
                t._fill_gaps(model, str(self.wav), [segment()], 20, {}, checkpoint_cb=lambda x: snapshots.append(copy.deepcopy(x)))
        saved = snapshots[-1]
        self.assertEqual(saved["recovery"]["next_index"], 1)
        model.reset_mock(side_effect=True)
        model.transcribe.return_value = (iter([recognized(start=.5, end=2.)]), None)
        with patch.object(t, "_missed_spans", return_value=[]):
            stats = t._fill_gaps(model, str(self.wav), saved["segments"], 20, {}, resume_state=saved["recovery"])
        self.assertEqual(model.transcribe.call_count, 1)
        self.assertEqual(saved["segments"][-1]["start"], 10)
        self.assertEqual(stats["refilled_sec"], 2)

    def test_recovery_phase_skips_main_pass(self):
        saved = self.saved("recovery")
        saved["recovery"] = dict(spans=[[5, 7]], next_index=1, stats=dict(dropped=0, refilled_sec=2, missing_sec=0))
        with patch.object(t, "get_model") as model, patch.object(t, "detect_device", return_value=("cpu", "int8")), \
                patch.object(t, "_missed_spans", return_value=[]):
            result = t.transcribe(str(self.wav), resume=saved, duration=20)
        model.return_value.transcribe.assert_not_called()
        self.assertEqual(result["refilled_sec"], 2)

    def test_process_crash_releases_lock_and_saved_result_resumes(self):
        self.saved("postprocess")
        code = "from pathlib import Path; import os,sys; from core.checkpoints import job_lock; lock=job_lock(Path(sys.argv[1])); lock.__enter__(); os._exit(17)"
        with cp.job_lock(self.folder):
            blocked = subprocess.run([sys.executable, "-c", code, str(self.folder)], capture_output=True, timeout=30)
            self.assertNotEqual(blocked.returncode, 17)
            with self.assertRaisesRegex(ValueError, "実行中"):
                pipeline.resume_job(str(self.folder), [], {}, self.settings)
        crashed = subprocess.run([sys.executable, "-c", code, str(self.folder)], capture_output=True, timeout=30)
        self.assertEqual(crashed.returncode, 17, crashed.stderr)
        with patch.object(t, "transcribe") as model:
            job = pipeline.resume_job(str(self.folder), [], {}, self.settings)
        model.assert_not_called()
        self.assertTrue(job["utterances"])

    def test_invalid_progress_is_rejected_before_recognition(self):
        saved = self.saved()
        for changes in (dict(processed_sec=float("nan")), dict(processed_sec=21), dict(phase="unknown"),
                        dict(phase="postprocess"), dict(phase="prepared"),
                        dict(segments=[segment(end=30)]), dict(recovery=dict(spans=[[4, 5]], next_index=.5))):
            with self.subTest(changes=changes):
                output.save_json(self.folder, cp.FILENAME, {**saved, **changes})
                with patch.object(t, "transcribe") as model, self.assertRaisesRegex(ValueError, "進捗データ"):
                    pipeline.resume_job(str(self.folder), [], {}, self.settings)
                model.assert_not_called()

    def test_vad_fallback_plan_stays_inside_audio(self):
        with patch.dict(sys.modules, {"faster_whisper.vad": None}):
            spans = t._missed_spans(str(self.wav), [], 20)
        self.assertEqual(spans, [(0, 20)])

    def test_ui_resume_refreshes_result_after_failure_and_success(self):
        self.saved()
        state = app._load_job("job")
        args = (state, [], "auto", "accurate", "ja", 0, False, False, False, False, "", "")
        with patch.object(app.llm, "unload_ollama"), patch.object(app.worker, "run", side_effect=RuntimeError("worker stopped")):
            failed = app.resume_transcription(*args, progress=Mock())
        self.assertEqual(len(failed), 3 + app.VIEW_COUNT)
        self.assertTrue(failed[1]["incomplete"])
        with patch.object(app.llm, "unload_ollama"), \
                patch.object(t, "transcribe", return_value=dict(segments=[segment()], model="tiny")), \
                patch.object(app.worker, "run", side_effect=lambda target, *args, **kw: pipeline.resume_job(*args)) as worker:
            done = app.resume_transcription(*args, progress=Mock())
        self.assertEqual(worker.call_args.args[0], "core.pipeline:resume_job")
        self.assertFalse(done[1].get("incomplete"))
        self.assertEqual(done[-2], "")
        self.assertTrue(done[9]["interactive"])
        self.assertEqual(output.load_json(self.folder, "meta.json")["whisper_model"], "tiny")


if __name__ == "__main__":
    unittest.main()
