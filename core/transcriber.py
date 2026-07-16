"""faster-whisper による文字起こし（GPU自動検出・単語タイムスタンプ付き）"""
import logging
import time

_models = {}  # (name, device) -> WhisperModel

MODEL_CHOICES = ["auto", "large-v3", "medium", "small"]


def detect_device() -> tuple[str, str]:
    """(device, compute_type) を返す。CUDAが使えれば float16、無ければ CPU int8。"""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def resolve_model_name(selected: str, device: str) -> str:
    if selected and selected != "auto":
        return selected
    return "large-v3" if device == "cuda" else "small"


def get_model(name: str):
    from faster_whisper import WhisperModel

    device, compute_type = detect_device()
    key = (name, device)
    if key in _models:
        return _models[key]

    try:
        logging.info(f"モデル '{name}' をロード中 (device={device}, compute={compute_type})...")
        model = WhisperModel(name, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            logging.warning(f"GPUロードに失敗したためCPUにフォールバックします: {e}")
            model = WhisperModel(name, device="cpu", compute_type="int8")
            key = (name, "cpu")
        else:
            raise
    _models[key] = model
    logging.info("モデルのロード完了")
    return model


def transcribe(wav_path: str, model_name: str = "auto", language: str = "ja",
               initial_prompt: str = "", duration: float = 0.0,
               progress_cb=None) -> dict:
    """文字起こしを実行し、単語レベル確信度付きのセグメント一覧を返す。

    Returns:
        {
          "language": str, "duration": float, "elapsed": float,
          "segments": [
            {"start": float, "end": float, "text": str,
             "words": [{"start": float, "end": float, "word": str, "prob": float}]}
          ]
        }
    """
    device, _ = detect_device()
    name = resolve_model_name(model_name, device)
    model = get_model(name)

    kwargs = dict(beam_size=5, vad_filter=True, word_timestamps=True)
    if language and language != "auto":
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt[:800]

    start_time = time.time()
    segments_iter, info = model.transcribe(wav_path, **kwargs)

    total = duration or getattr(info, "duration", 0.0) or 0.0
    segments = []
    for seg in segments_iter:
        words = []
        for w in (seg.words or []):
            words.append({
                "start": float(w.start), "end": float(w.end),
                "word": w.word, "prob": float(w.probability),
            })
        segments.append({
            "start": float(seg.start), "end": float(seg.end),
            "text": seg.text.strip(), "words": words,
        })
        logging.info(f"進捗: [{seg.start:.1f}s -> {seg.end:.1f}s]")
        if progress_cb and total > 0:
            try:
                progress_cb(min(seg.end / total, 1.0))
            except Exception:
                pass

    elapsed = time.time() - start_time
    logging.info(f"文字起こし完了 ({elapsed:.1f}秒, model={name}, device={device})")
    return {
        "language": getattr(info, "language", language),
        "duration": total,
        "elapsed": elapsed,
        "model": name,
        "device": device,
        "segments": segments,
    }
