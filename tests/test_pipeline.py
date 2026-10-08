"""画面の選択が認識処理・保存情報まで届くことを確認する。ユーザーデータは使わない。"""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from core import pipeline


class DecodingPipelineTest(unittest.TestCase):
    def test_selected_and_default_options_are_passed_and_recorded(self):
        cases = [
            ("context", "stable", False, True, "accurate", "context", False),
            (None, "context", True, False, "balanced", "context", True),
            (None, None, None, None, "accurate", "stable", True),
            (None, None, None, False, "accurate", "stable", False),
            (None, None, False, True, "fast", "batched", None),
            (None, None, True, False, "fast", "batched", None),
        ]
        for selected, configured, selected_guard, configured_guard, speed, expected, actual_guard in cases:
            with self.subTest(speed=speed, selected_guard=selected_guard, configured_guard=configured_guard), \
                    tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                wav = root / "audio.wav"
                wav.touch()
                result = {"segments": [], "speed": speed, "decoding": expected, "silence_guard": actual_guard,
                          "model": "large-v3", "device": "cpu", "elapsed": 1}
                transcribe = stack.enter_context(patch.object(pipeline.transcriber, "transcribe", return_value=result))
                for obj, name, value in [
                    (pipeline.output, "create_job_dir", root),
                    (pipeline.audio, "prepare_audio", str(wav)),
                    (pipeline.audio, "get_duration", 30),
                    (pipeline.audio, "make_preview_audio", None),
                    (pipeline.audio, "enhance_for_asr", str(wav)),
                    (pipeline.people, "load_people", []),
                    (pipeline.people, "initial_prompt_terms", []),
                    (pipeline.glossary, "build_initial_prompt", ""),
                    (pipeline.glossary, "build_hotwords", ""),
                ]:
                    stack.enter_context(patch.object(obj, name, return_value=value))
                stack.enter_context(patch.object(pipeline, "_finalize_text"))
                saved = stack.enter_context(patch.object(pipeline, "_save_job"))
                opts = dict(model="auto", language="ja", noise=False, diarize=False, fillers=False, speed=speed)
                if selected:
                    opts["decoding"] = selected
                if selected_guard is not None:
                    opts["silence_guard"] = selected_guard
                settings = {"whisper": {}}
                if configured:
                    settings["whisper"]["decoding"] = configured
                if configured_guard is not None:
                    settings["whisper"]["silence_guard"] = configured_guard
                job = pipeline._transcribe_one("test.wav", opts, settings, Mock())
                self.assertEqual(transcribe.call_args.kwargs["decoding"], selected or configured or "stable")
                requested_guard = selected_guard if selected_guard is not None else (
                    configured_guard if configured_guard is not None else True)
                self.assertIs(transcribe.call_args.kwargs["silence_guard"], requested_guard)
                self.assertEqual(job["stats"]["asr_decoding"], expected)
                self.assertIs(job["stats"]["asr_silence_guard"], actual_guard)
                label = "対象外（高速モード）" if speed == "fast" else ("ON" if actual_guard else "OFF")
                self.assertIn(f"内部スキップ: {label}", job["meta_line"])
                if speed != "fast":
                    self.assertIn("認識方式", job["meta_line"])
                saved.assert_called_once_with(job)
                self.assertFalse(wav.exists())


if __name__ == "__main__":
    unittest.main()
