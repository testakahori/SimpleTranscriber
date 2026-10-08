"""画面の選択が認識処理・保存情報まで届くことを確認する。ユーザーデータは使わない。"""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from core import pipeline


class DecodingPipelineTest(unittest.TestCase):
    def test_selected_and_default_modes_are_passed_and_recorded(self):
        for selected, configured, expected in [("context", "stable", "context"),
                                               (None, "context", "context"),
                                               (None, None, "stable")]:
            with self.subTest(selected=selected, configured=configured), \
                    tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                wav = root / "audio.wav"
                wav.touch()
                result = {"segments": [], "speed": "accurate", "decoding": expected,
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
                opts = dict(model="auto", language="ja", noise=False, diarize=False, fillers=False)
                if selected:
                    opts["decoding"] = selected
                settings = {"whisper": {"decoding": configured}} if configured else {}
                job = pipeline._transcribe_one("test.wav", opts, settings, Mock())
                self.assertEqual(transcribe.call_args.kwargs["decoding"], expected)
                self.assertEqual(job["stats"]["asr_decoding"], expected)
                self.assertIn("認識方式", job["meta_line"])
                saved.assert_called_once_with(job)
                self.assertFalse(wav.exists())


if __name__ == "__main__":
    unittest.main()
