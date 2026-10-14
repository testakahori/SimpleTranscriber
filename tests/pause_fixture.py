"""一時停止の検証専用ワーカー。音声は無音、認識結果は固定。ユーザーデータは読まない。"""
import os
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from core import audio, config, output, pipeline, run_control, transcriber as t


def segment(text="保存する発言。", start=0, end=2):
    return SimpleNamespace(text=text, start=start, end=end, no_speech_prob=.01, avg_logprob=-.2,
                           compression_ratio=1.1,
                           words=[SimpleNamespace(word=text, start=start, end=end, probability=.9)])


@contextmanager
def isolated(root):
    root = Path(root)
    folder = root / "job"
    folder.mkdir(exist_ok=True)
    wav = root / "converted.wav"
    sf.write(wav, np.zeros(20 * 16000), 16000, subtype="PCM_16")
    settings = config._deep_merge(config.DEFAULTS, {"diarization": {"auto_learn": False}})
    opts = pipeline._run_options(dict(model="tiny", eq=False, noise=False, fillers=False), settings)
    with ExitStack() as stack:
        stack.enter_context(patch.object(output, "OUTPUT_DIR", root))
        for obj, key, value in [(config, "load_settings", settings), (output, "create_job_dir", folder),
                                (audio, "prepare_audio", str(wav)), (audio, "make_preview_audio", None),
                                (pipeline.people, "load_people", []), (pipeline.people, "initial_prompt_terms", []),
                                (pipeline.glossary, "build_initial_prompt", ""), (pipeline.glossary, "build_hotwords", ""),
                                (pipeline, "_finalize_text", None), (t, "detect_device", ("cpu", "int8")),
                                (t, "_fill_gaps", dict(dropped=0, refilled_sec=0, missing_sec=0))]:
            stack.enter_context(patch.object(obj, key, return_value=value))
        yield folder, wav, opts, settings


def run(root, progress=None, pause_file=None):
    with isolated(root) as (folder, wav, opts, settings):
        def generate():
            yield segment()
            progress(.25, "ready-to-pause")
            deadline = time.monotonic() + 15
            while not run_control.requested(pause_file):
                if time.monotonic() > deadline:
                    raise TimeoutError("pause request did not reach child")
                time.sleep(.02)
            yield segment("最後に保存する発言。", 5, 6)
            raise AssertionError("recognition continued after pause")
        model = Mock(transcribe=Mock(return_value=(generate(), SimpleNamespace(language="ja", duration=20))))
        with patch.object(t, "get_model", return_value=model):
            job = pipeline._transcribe_one("original.wav", opts, settings, progress, pause_file)
        return {"job": job, "pid": os.getpid()}
