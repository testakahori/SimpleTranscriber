"""faster-whisper による文字起こし（GPU自動検出・単語タイムスタンプ付き）"""
import gc
import logging
import re
import time

_models = {}  # (name, device) -> WhisperModel

# 表示名 → faster-whisper に渡すモデルID
MODEL_CHOICES = ["auto", "large-v3", "large-v3-turbo", "kotoba-whisper-v2.0", "medium", "small"]

# 速度モード（実会議4分×2の実測。RTX 5060 Ti / large-v3）
#   accurate: 1区間ずつ・ビーム幅5          … 基準（実時間の約2〜3倍速）
#   balanced: 1区間ずつ・ビーム幅1          … 約2.4倍速・取りこぼし約5%増
#   fast:     16区間まとめて処理(batched)   … 約8倍速・取りこぼし約15%増
SPEED_CHOICES = [("精度優先（遅い）", "accurate"), ("バランス（約2.4倍速）", "balanced"),
                 ("高速（約8倍速・取りこぼしと句読点が少し増減）", "fast")]
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


def _unstretch(w: dict) -> dict:
    """文字数に比べて長すぎる単語（直前の無音を取り込んでいる）の開始時刻を後ろへ寄せる。
    一括処理モードで起きやすく、放置すると「間」が消えて句読点・話者割り当てが狂う。"""
    n = max(len(w["word"].strip()), 1)
    if w["end"] - w["start"] > 0.3 * n + 0.8:
        w["start"] = w["end"] - (0.18 * n + 0.3)
    return w


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
               progress_cb=None, speed: str = "accurate") -> dict:
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
    engine = model
    if speed == "fast":
        # 16区間まとめてGPUに流す。文脈継続・お手本プロンプト・ホットワードは
        # 一括処理だと逆に文字を大きく落とすため使わない（句読点は punctuate.py で補う）
        from faster_whisper import BatchedInferencePipeline
        engine = BatchedInferencePipeline(model=model)
        for k in ("best_of", "condition_on_previous_text", "hallucination_silence_threshold",
                  "vad_parameters"):
            kwargs.pop(k, None)
        kwargs.update(batch_size=16, chunk_length=15)
    else:
        if speed == "balanced":
            kwargs.update(beam_size=1, best_of=1)
        # 句読点付きのお手本を与えると、Whisperが「、」「。」を付けやすくなる
        kwargs["initial_prompt"] = (PUNCT_PROMPT + (initial_prompt or ""))[:800]
        if hotwords:
            kwargs["hotwords"] = hotwords[:500]

    start_time = time.time()
    segments_iter, info = engine.transcribe(wav_path, **kwargs)

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
            words.append(_unstretch({
                "start": float(w.start), "end": float(w.end),
                "word": w.word, "prob": float(w.probability),
            }))
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
        "speed": speed,
        "device": device,
        "dropped": dropped,
        "segments": segments,
    }
