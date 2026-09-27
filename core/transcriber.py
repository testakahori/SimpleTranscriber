"""faster-whisper による文字起こし（GPU自動検出・単語タイムスタンプ付き）"""
import gc
import logging
import re
import time

_models = {}  # (name, device) -> WhisperModel

# 表示名 → faster-whisper に渡すモデルID
MODEL_CHOICES = ["auto", "large-v3", "large-v3-turbo", "kotoba-whisper-v2.0", "medium", "small"]
MODEL_IDS = {
    "kotoba-whisper-v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

# Whisperが無音・BGM区間でよく捏造する定型文（日本語YouTube字幕由来）
HALLUCINATION_PATTERNS = [
    r"ご視聴ありがとうございました",
    r"チャンネル登録",
    r"高評価",
    r"字幕(は|を|作成|提供)",
    r"最後まで(ご覧|見て)",
    r"次回もお楽しみに",
    r"^(ん|う|あ)+[。、]?$",
]
_HALLUCINATION_RE = re.compile("|".join(HALLUCINATION_PATTERNS))


PUNCT_PROMPT = "では、会議を始めます。本日の議題は、来月のスケジュールについてです。よろしくお願いします。"


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

    model_id = MODEL_IDS.get(name, name)
    try:
        logging.info(f"モデル '{name}' をロード中 (device={device}, compute={compute_type})...")
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            logging.warning(f"GPUロードに失敗したためCPUにフォールバックします: {e}")
            model = WhisperModel(model_id, device="cpu", compute_type="int8")
            key = (name, "cpu")
        else:
            raise
    _models[key] = model
    logging.info("モデルのロード完了")
    return model


def unload() -> None:
    """Whisperモデルを解放してVRAMを空ける（後段のLLMにGPUを譲るため）"""
    _models.clear()
    gc.collect()


def _is_hallucination(seg) -> bool:
    text = seg.text.strip()
    if not text:
        return True
    # 無音判定が強く、かつ自信の低いセグメント
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return True
    # 同じ文字列の繰り返しループ
    if seg.compression_ratio > 2.6:
        return True
    # 定型ハルシネーション: 短い・無音寄り・自信が低い、が揃った時だけ除外
    # （会議で本当に「字幕を入れる」等と話した場合を消さないため）
    if _HALLUCINATION_RE.search(text) and len(text) <= 30 and \
            (seg.no_speech_prob > 0.3 or seg.avg_logprob < -0.7):
        return True
    return False


def transcribe(wav_path: str, model_name: str = "auto", language: str = "ja",
               initial_prompt: str = "", hotwords: str = "", duration: float = 0.0,
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

    kwargs = dict(
        beam_size=5,
        best_of=5,
        word_timestamps=True,
        # 前の発言を文脈として使う（精度と句読点が安定する）。
        # 同じ文のループは compression_ratio / 無音判定のフィルタで除外する
        condition_on_previous_text=True,
        # 無音が続く所で捏造された単語をスキップ
        hallucination_silence_threshold=2.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
    )
    if language and language != "auto":
        kwargs["language"] = language
    # 句読点付きのお手本を与えると、Whisperが「、」「。」を付けやすくなる
    kwargs["initial_prompt"] = (PUNCT_PROMPT + (initial_prompt or ""))[:800]
    if hotwords:
        kwargs["hotwords"] = hotwords[:500]

    start_time = time.time()
    segments_iter, info = model.transcribe(wav_path, **kwargs)

    total = duration or getattr(info, "duration", 0.0) or 0.0
    segments = []
    dropped = 0
    for seg in segments_iter:
        if progress_cb and total > 0:
            try:
                progress_cb(min(seg.end / total, 1.0))
            except Exception:
                pass
        if _is_hallucination(seg):
            dropped += 1
            logging.info(f"ハルシネーション疑いを除外: [{seg.start:.1f}s] {seg.text.strip()[:40]}")
            continue
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

    elapsed = time.time() - start_time
    logging.info(f"文字起こし完了 ({elapsed:.1f}秒, model={name}, device={device}, 除外={dropped})")
    return {
        "language": getattr(info, "language", language),
        "duration": total,
        "elapsed": elapsed,
        "model": name,
        "device": device,
        "dropped": dropped,
        "segments": segments,
    }
