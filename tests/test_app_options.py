"""画面の切替がワーカーへ届くことを、音声処理や外部接続なしで確認する。"""
import inspect
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import app
from core import config


class SilenceGuardUITest(unittest.TestCase):
    def test_checkbox_wiring_and_fast_mode(self):
        settings = config._deep_merge(config.DEFAULTS, {"whisper": {"silence_guard": False, "speed": "fast"}})
        with ExitStack() as stack:
            for obj, name, value in [
                (app.config, "load_settings", settings),
                (app.diarization, "available", False),
                (app.output, "list_editable_jobs", []),
                (app.people, "to_table", []),
                (app.glossary, "load_glossary", []),
                (app, "_people_names", []),
                (app, "_voiceprint_rows", []),
                (app, "load_corrections_table", []),
                (app, "ollama_models", []),
            ]:
                stack.enter_context(patch.object(obj, name, return_value=value))
            stack.enter_context(patch.object(app, "_probe_thread", Mock(is_alive=Mock(return_value=True))))
            stack.enter_context(patch.object(app.llm, "unload_ollama"))
            worker = stack.enter_context(patch.object(app.worker, "run", return_value=[]))
            demo = app.build_ui()
            try:
                start = next(fn for fn in demo.fns.values() if fn.fn is app.run_batch)
                resume = next(fn for fn in demo.fns.values() if fn.fn is app.resume_transcription)
                pause = next(fn for fn in demo.fns.values() if fn.fn is app.pause_transcription)
                self.assertFalse(pause.queue)  # 認識の完了待ちの列に停止操作を並べない。
                self.assertEqual(pause.inputs, [])  # Gradioが画面のsession_hashを注入する。
                self.assertEqual(pause.outputs[0].elem_id, "st-pause-status")
                self.assertEqual(resume.inputs[1:], start.inputs)
                self.assertEqual(resume.outputs, start.outputs)
                self.assertEqual(resume.concurrency_id, start.concurrency_id)
                self.assertEqual(start.concurrency_id, "asr")
                transcript = start.outputs[2]
                self.assertIsInstance(transcript, app.gr.Markdown)
                self.assertFalse(transcript.sanitize_html)
                self.assertEqual(transcript.latex_delimiters, [])
                self.assertEqual(start.outputs[-2].elem_id, "st-job-notice")
                bound = inspect.signature(app.run_batch).bind(*[c.value for c in start.inputs])
                self.assertIs(bound.arguments["silence_guard"], False)
                guard = next(c for c in start.inputs if c.label == "無音付近の誤認識を除外（内部スキップ）")
                self.assertFalse(guard.interactive)
                speed_change = next(fn for fn in demo.fns.values() if fn.outputs == [guard])
                for speed in ("accurate", "balanced", "fast"):
                    update = speed_change.fn(speed)
                    self.assertEqual(update["interactive"], speed != "fast")
                    self.assertNotIn("value", update)  # 速さを戻してもユーザーのON/OFFは保つ
                for enabled in (False, True):
                    bound.arguments.update(files=["test.wav"], speed="accurate", silence_guard=enabled,
                                           progress=Mock())
                    start.fn(**bound.arguments)
                    self.assertEqual(worker.call_args.args[0], "core.pipeline:transcribe_files")
                    self.assertIs(worker.call_args.args[2]["silence_guard"], enabled)
            finally:
                demo.close()


if __name__ == "__main__":
    unittest.main()
