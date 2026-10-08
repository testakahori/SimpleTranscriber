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
DECODING_CHOICES = [("抜け・繰り返しを抑える（推奨）", "stable"),
                    ("文脈・用語ヒントを優先（従来方式）", "context")]
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


GAP_RETRY_SEC = 20.0  # 発話検出(VAD)が使えない時: 文字が1つも無い区間がこれ以上続いたらやり直す
MISS_MIN_SEC = 3.0    # 声があるのに文字が無い所がこれ以上あれば、その部分だけやり直す
COVER_PAD_SEC = 1.0   # 単語の前後これだけは文字があるとみなす（単語の時刻のずれ）
# やり直しは数秒の切れ端が数百か所になることがある。1か所ずつは軽く済ませる:
#   温度を変えた再試行は2段まで（既定の6段×ビーム5だと、繰り返しループした所で1か所数十秒かかった）
#   1窓で出す文字数の上限を切れ端の長さに合わせる（「岡田 岡田 岡田…」と延々続けさせない）
GAP_TEMPERATURES = (0.0, 0.4)
GAP_TOKENS_PER_SEC = 14  # 30秒窓で430（Whisperの上限448未満）
MAIN_PASS_SHARE = 0.9  # 進捗バーのうち1回目の文字起こしの割合（残りは抜けのやり直し）
_REPEAT_RE = re.compile(r"(\S{1,8}?)(?:\s*\1){3,}")  # 同じ語が4回以上続く（短い切れ端で起きる捏造）


def _segment_dict(seg, offset: float = 0.0) -> dict:
    words = [_unstretch({"start": float(w.start) + offset, "end": float(w.end) + offset,
                         "word": w.word, "prob": float(w.probability)})
             for w in (seg.words or [])]
    return {"start": float(seg.start) + offset, "end": float(seg.end) + offset,
            "text": seg.text.strip(), "words": words}


def _merge_spans(spans: list, join_sec: float = 0.0) -> list:
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= join_sec:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _missed_spans(wav_path: str, segments: list[dict], total: float) -> list[tuple[float, float]]:
    """声がある（VAD）のに文字が1つも付いていない部分 → [(開始, 終了)]。
    VADが使えない時は、文字が GAP_RETRY_SEC 以上続けて無い区間を返す。"""
    covered = []
    for s in segments:
        spans = [(w["start"], w["end"]) for w in s.get("words") or []] or [(s["start"], s["end"])]
        covered += [(a - COVER_PAD_SEC, b + COVER_PAD_SEC) for a, b in spans]
    covered = _merge_spans(covered)
    try:
        import soundfile as sf
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio_, rate = sf.read(wav_path, dtype="float32")
        if audio_.ndim > 1:
            audio_ = audio_.mean(axis=1)
        speech = [(t["start"] / rate, t["end"] / rate) for t in get_speech_timestamps(
            audio_, VadOptions(min_silence_duration_ms=500, speech_pad_ms=300), sampling_rate=rate)]
        del audio_
    except Exception as e:
        logging.warning(f"発話検出に失敗したため、文字の無い長い区間だけを確認します: {e}")
        edges = [-COVER_PAD_SEC] + [x for c in covered for x in c] + [total + COVER_PAD_SEC]
        return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= GAP_RETRY_SEC]

    # 声の区間から、文字がある所を引く
    missed = []
    for s, e in speech:
        t = s
        for cs, ce in covered:
            if ce <= t or cs >= e:
                continue
            if cs > t:
                missed.append((t, cs))
            t = max(t, ce)
        if t < e:
            missed.append((t, e))
    # 近い切れ端はまとめて1回でやり直す。声の合計が短い所（咳・相づち）は対象外
    out = []
    for a, b in _merge_spans(missed, join_sec=2.0):
        voiced = sum(min(b, me) - max(a, ms) for ms, me in missed if me > a and ms < b)
        if voiced >= MISS_MIN_SEC:
            out.append((max(0.0, a), min(total, b) if total else b))
    return out


def _fill_gaps(model, wav_path: str, segments: list[dict], total: float, kwargs: dict,
               progress_cb=None, checkpoint_cb=None) -> dict:
    """声があるのに文字が無い部分を、前の文脈を使わずにもう一度文字起こしして segments に足す。

    Whisperは「無音らしく自信も低い」30秒窓を黙って飛ばす。前の発言を文脈にしていると、
    文脈が崩れた所から窓を飛ばし続け、実会議（2時間）で38分ぶん丸ごと抜けていた
    （その区間だけを文字起こしすると普通に取れる）。短い抜けも同じ仕組みで起きる。
    Returns: {"dropped": 除外したハルシネーション数, "refilled_sec": 補った区間の長さ,
              "missing_sec": やり直しても文字にならなかった声の長さ}"""
    import soundfile as sf

    stats = {"dropped": 0, "refilled_sec": 0.0, "missing_sec": 0.0}
    spans = _missed_spans(wav_path, segments, total)
    if not spans:
        return stats
    rate = sf.info(wav_path).samplerate
    # 人名などのヒントは短い切れ端だと「その名前を繰り返す」捏造を招くので渡さない
    kwargs = {k: v for k, v in kwargs.items() if k not in ("initial_prompt", "hotwords")}
    kwargs.update(condition_on_previous_text=False, temperature=list(GAP_TEMPERATURES))
    added = []
    for i, (a, b) in enumerate(spans):
        if progress_cb:
            try:
                progress_cb(i / len(spans), f"聞き取れなかった所をやり直し中... {i + 1}/{len(spans)}か所")
            except Exception:
                pass
        lo = max(0.0, a - 0.5)  # 語頭が切れないよう少し広めに聞かせ、足すのは抜けの中の単語だけ
        audio_, _ = sf.read(wav_path, start=int(lo * rate), stop=int((b + 0.5) * rate), dtype="float32")
        if audio_.ndim > 1:
            audio_ = audio_.mean(axis=1)
        window = min(b + 0.5 - lo, 30.0)
        segs, _ = model.transcribe(audio_, max_new_tokens=int(window * GAP_TOKENS_PER_SEC) + 10, **kwargs)
        n, bad_run, prev = 0, 0, None
        for seg in segs:
            text = seg.text.strip()
            if _is_hallucination(seg) or _REPEAT_RE.search(text) or text == prev:
                stats["dropped"] += 1
                bad_run += 1
                if bad_run >= 2:  # 捏造のループに入った。この切れ端は諦めて次へ（続けても同じ文が出るだけ）
                    break
                continue
            bad_run, prev = 0, text
            d = _segment_dict(seg, lo)
            d["words"] = [w for w in d["words"] if a <= (w["start"] + w["end"]) / 2 <= b]
            if not d["words"]:
                continue
            d["start"], d["end"] = d["words"][0]["start"], d["words"][-1]["end"]
            d["text"] = "".join(w["word"] for w in d["words"]).strip()
            added.append(d)
            n += 1
        if n:
            stats["refilled_sec"] += b - a
            logging.warning(f"文字起こしの抜けを補いました: {a:.0f}s〜{b:.0f}s（{n}区間）")
        if checkpoint_cb:
            checkpoint_cb({"phase": "recovery", "segments": sorted(segments + added, key=lambda s: s["start"]),
                           "duration": total, "completed_spans": i + 1, "total_spans": len(spans)})
    segments.extend(added)
    segments.sort(key=lambda s: s["start"])
    remain = _missed_spans(wav_path, segments, total)
    stats["missing_sec"] = sum(b - a for a, b in remain)
    if remain:
        logging.warning("やり直しても文字にならなかった声: " +
                        "、".join(f"{a:.0f}s〜{b:.0f}s" for a, b in remain[:20]))
    stats["refilled_sec"] = round(stats["refilled_sec"], 1)
    stats["missing_sec"] = round(stats["missing_sec"], 1)
    return stats


def transcribe(wav_path: str, model_name: str = "auto", language: str = "ja",
               initial_prompt: str = "", hotwords: str = "", duration: float = 0.0,
               progress_cb=None, speed: str = "accurate", decoding: str = "stable",
               checkpoint_cb=None, silence_guard: bool = True) -> dict:
    """文字起こしを実行し、単語レベル確信度付きのセグメント一覧を返す。

    checkpoint_cb: 認識済みの中間結果を同期的に受け取る任意の保存関数。
        呼び出し側で保存頻度を制御する。保存失敗は握りつぶさず呼び出し元へ返す。
    silence_guard: faster-whisper内の無音に囲まれた誤認のスキップ。
        雑音で1秒刻みの再認識が続く場合は無効化できる。VAD・外側の除外は維持。
    Returns:
        {
          "language": str, "duration": float, "elapsed": float,
          "segments": [
            {"start": float, "end": float, "text": str,
             "words": [{"start": float, "end": float, "word": str, "prob": float}]}
          ]
        }
    """
    if decoding not in {value for _, value in DECODING_CHOICES}:
        raise ValueError(f"不明な認識方式: {decoding}")
    device, _ = detect_device()
    name = resolve_model_name(model_name, device)
    model = get_model(name)

    kwargs = dict(
        beam_size=5,
        best_of=5,
        word_timestamps=True,
        # 雑音で崩れた認識を次の窓へ引き継ぐと、繰り返しや長い抜けが続く。
        # stableでは窓ごとにリセットする。温度の再試行は既定の6段を維持する。
        condition_on_previous_text=decoding == "context",
        # 無音が続く所で捏造された単語をスキップ
        hallucination_silence_threshold=2.0 if silence_guard else None,
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
        # hotwordsは全窓に繰り返し入るので、聞き取れない所で人名一覧を捏造しやすい。
        # stableでも最初の窓のinitial_promptは残し、用語の手掛かりを与える。
        if hotwords and decoding == "context":
            kwargs["hotwords"] = hotwords[:500]

    start_time = time.time()
    segments_iter, info = engine.transcribe(wav_path, **kwargs)

    total = duration or getattr(info, "duration", 0.0) or 0.0
    segments = []
    dropped = 0
    for seg in segments_iter:
        if progress_cb and total > 0:
            try:
                progress_cb(MAIN_PASS_SHARE * min(seg.end / total, 1.0))
            except Exception:
                pass
        if _is_hallucination(seg):
            dropped += 1
            logging.info(f"ハルシネーション疑いを除外: [{seg.start:.1f}s] {seg.text.strip()[:40]}")
        else:
            segments.append(_segment_dict(seg))
        if checkpoint_cb:
            checkpoint_cb({"phase": "main", "segments": segments, "duration": total,
                           "processed_sec": float(seg.end), "dropped": dropped})
        logging.info(f"進捗: [{seg.start:.1f}s -> {seg.end:.1f}s]")

    if checkpoint_cb:
        checkpoint_cb({"phase": "main_complete", "segments": segments, "duration": total,
                       "processed_sec": total, "dropped": dropped})

    retry_kwargs = {k: v for k, v in kwargs.items() if k not in ("batch_size", "chunk_length")}
    fill_cb = None
    if progress_cb:
        def fill_cb(f, desc):
            progress_cb(MAIN_PASS_SHARE + (1 - MAIN_PASS_SHARE) * f, desc)
    fill = _fill_gaps(model, wav_path, segments, total, retry_kwargs,
                      progress_cb=fill_cb, checkpoint_cb=checkpoint_cb)
    dropped += fill["dropped"]

    elapsed = time.time() - start_time
    logging.info(f"文字起こし完了 ({elapsed:.1f}秒, model={name}, device={device}, 除外={dropped})")
    return {
        "language": getattr(info, "language", language),
        "duration": total,
        "elapsed": elapsed,
        "model": name,
        "speed": speed,
        "decoding": "batched" if speed == "fast" else decoding,
        "silence_guard": silence_guard if speed != "fast" else None,
        "device": device,
        "dropped": dropped,
        "refilled_sec": fill["refilled_sec"],
        "missing_sec": fill["missing_sec"],
        "segments": segments,
    }
