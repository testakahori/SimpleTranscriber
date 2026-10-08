"""認識方式と、抜けの再認識の退行テスト（モデルの取得・GPUは不要）。"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import config, transcriber as t


def segment(text="確認しました。", start=0.0, end=2.0):
    return SimpleNamespace(
        text=text, start=start, end=end, no_speech_prob=0.01,
        avg_logprob=-0.2, compression_ratio=1.1,
        words=[SimpleNamespace(start=start, end=end, word=text, probability=0.9)],
    )


class TranscriptionModesTest(unittest.TestCase):
    def test_checkpoint_preserves_main_pass_before_recovery_failure(self):
        model = Mock()
        model.transcribe.return_value = (iter([segment()]), SimpleNamespace(duration=10, language="ja"))
        saved = []
        with patch.object(t, "detect_device", return_value=("cpu", "int8")), \
                patch.object(t, "get_model", return_value=model), \
                patch.object(t, "_fill_gaps", side_effect=RuntimeError("recovery failed")):
            with self.assertRaisesRegex(RuntimeError, "recovery failed"):
                t.transcribe("unused.wav", checkpoint_cb=lambda value: saved.append(dict(value)))
        self.assertEqual(saved[-1]["phase"], "main_complete")
        self.assertEqual(saved[-1]["segments"][0]["text"], "確認しました。")

    def test_checkpoint_write_errors_are_not_silenced(self):
        model = Mock()
        model.transcribe.return_value = (iter([segment()]), SimpleNamespace(duration=10, language="ja"))
        with patch.object(t, "detect_device", return_value=("cpu", "int8")), \
                patch.object(t, "get_model", return_value=model):
            with self.assertRaisesRegex(OSError, "disk full"):
                t.transcribe("unused.wav", checkpoint_cb=Mock(side_effect=OSError("disk full")))

    def run_transcription(self, **options):
        model = Mock()
        model.transcribe.return_value = (iter([segment()]), SimpleNamespace(duration=10, language="ja"))
        with patch.object(t, "detect_device", return_value=("cpu", "int8")), \
                patch.object(t, "get_model", return_value=model), \
                patch.object(t, "_fill_gaps", return_value={"dropped": 0, "refilled_sec": 0, "missing_sec": 0}) as fill:
            result = t.transcribe("unused.wav", initial_prompt="テスト用語", hotwords="テスト用語", **options)
        options = model.transcribe.call_args.kwargs if model.transcribe.called else {}
        return result, options, fill.call_args.args[-1]

    def test_default_is_stable_without_repeated_vocabulary(self):
        result, options, retry = self.run_transcription()
        self.assertFalse(options["condition_on_previous_text"])
        self.assertNotIn("hotwords", options)
        self.assertIn("テスト用語", options["initial_prompt"])
        # 主処理で温度の再試行を減らすと、取れていた発言まで落ちたため維持する。
        self.assertNotIn("temperature", options)
        self.assertEqual(options["beam_size"], 5)
        self.assertEqual(result["decoding"], "stable")
        self.assertEqual(result["segments"][0]["text"], "確認しました。")
        self.assertTrue(retry["word_timestamps"])

    def test_context_reproduces_previous_recognition_options(self):
        result, options, _ = self.run_transcription(decoding="context")
        self.assertTrue(options["condition_on_previous_text"])
        self.assertEqual(options["hotwords"], "テスト用語")
        self.assertIn(t.PUNCT_PROMPT, options["initial_prompt"])
        self.assertEqual(result["decoding"], "context")

    def test_experiment_can_disable_internal_silence_skip_without_reducing_fallbacks(self):
        result, options, retry = self.run_transcription(silence_guard=False)
        self.assertIsNone(options["hallucination_silence_threshold"])
        self.assertTrue(options["vad_filter"])
        self.assertTrue(options["word_timestamps"])
        self.assertNotIn("temperature", options)
        self.assertFalse(result["silence_guard"])
        self.assertIsNone(retry["hallucination_silence_threshold"])

    def test_balanced_keeps_its_beam_width(self):
        _, options, _ = self.run_transcription(speed="balanced")
        self.assertEqual((options["beam_size"], options["best_of"]), (1, 1))
        self.assertFalse(options["condition_on_previous_text"])

    def test_fast_mode_does_not_receive_prompt_or_sequential_options(self):
        batched = Mock()
        batched.transcribe.return_value = (iter([segment()]), SimpleNamespace(duration=10, language="ja"))
        fake_module = SimpleNamespace(BatchedInferencePipeline=Mock(return_value=batched))
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result, _, retry = self.run_transcription(speed="fast", decoding="context")
        options = batched.transcribe.call_args.kwargs
        for key in ("hotwords", "initial_prompt", "condition_on_previous_text", "best_of"):
            self.assertNotIn(key, options)
        self.assertEqual(result["decoding"], "batched")
        self.assertNotIn("batch_size", retry)
        self.assertNotIn("chunk_length", retry)

    def test_invalid_mode_fails_before_model_loading(self):
        with patch.object(t, "get_model") as get_model:
            with self.assertRaises(ValueError):
                t.transcribe("unused.wav", decoding="typo")
            get_model.assert_not_called()

    def test_existing_settings_inherit_stable_default(self):
        merged = config._deep_merge(config.DEFAULTS, {"whisper": {"model": "large-v3"}})
        self.assertEqual(merged["whisper"]["decoding"], "stable")
        merged = config._deep_merge(config.DEFAULTS, {"whisper": {"decoding": "context"}})
        self.assertEqual(merged["whisper"]["decoding"], "context")
        self.assertEqual(config.DEFAULTS["whisper"]["decoding"], "stable")


class GapRecoveryTest(unittest.TestCase):
    def test_recovery_resets_hints_and_keeps_absolute_word_times(self):
        # 本文の認識方式によらず、抜けだけは毎回ヒントも文脈も外して再認識する。
        model = Mock()
        model.transcribe.return_value = (
            iter([segment("既存", 0.0, 0.2), segment("追加", 0.5, 1.5), segment("境界外", 4.8, 5.0)]), None)
        fake_sf = SimpleNamespace(info=lambda _: SimpleNamespace(samplerate=16000),
                                  read=lambda *a, **k: (SimpleNamespace(ndim=1), 16000))
        segments = [{"start": 1.0, "end": 2.0, "text": "既存", "words": []}]
        checkpoint = Mock()
        with patch.dict(sys.modules, {"soundfile": fake_sf}), \
                patch.object(t, "_missed_spans", side_effect=[[(10.0, 14.0)], []]):
            result = t._fill_gaps(model, "unused.wav", segments, 20,
                                  dict(initial_prompt="用語", hotwords="用語", condition_on_previous_text=True),
                                  checkpoint_cb=checkpoint)
        options = model.transcribe.call_args.kwargs
        self.assertFalse(options["condition_on_previous_text"])
        self.assertNotIn("initial_prompt", options)
        self.assertNotIn("hotwords", options)
        self.assertEqual(options["temperature"], list(t.GAP_TEMPERATURES))
        self.assertLessEqual(options["max_new_tokens"], 430)
        self.assertEqual([s["text"] for s in segments], ["既存", "追加"])
        self.assertEqual(segments[1]["words"][0]["start"], 10.0)
        self.assertEqual(result["missing_sec"], 0)
        saved = checkpoint.call_args.args[0]
        self.assertEqual(saved["phase"], "recovery")
        self.assertEqual(saved["completed_spans"], 1)
        self.assertEqual([s["text"] for s in saved["segments"]], ["既存", "追加"])


if __name__ == "__main__":
    unittest.main()
