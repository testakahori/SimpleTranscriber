"""文字起こしの本体（音声変換 → Whisper → 話者分離 → 発言の構造化 → 保存）と声紋の学習。

GPUを使う処理は core.worker から別プロセスで呼ばれる（終わるとVRAM・RAMが全部OSに返る）。
このモジュールは gradio を読み込まない（別プロセスの起動を軽くするため）。
"""
import logging
import re
from pathlib import Path

from . import audio, config, diarization, glossary, output, people, punctuate, transcriber
from . import utterances as U


def _meeting_date_from_name(name: str) -> str:
    m = re.search(r"(20\d{2})[-_.年]?(\d{1,2})[-_.月]?(\d{1,2})", name)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}年{mo}月{d}日"
    return ""


def _save_job(job: dict) -> None:
    """発言データから transcript.md / srt / vtt / utterances.json を書き出す"""
    job_dir = Path(job["job_dir"])
    utts = job["utterances"]
    settings = config.load_settings()
    sub = settings.get("subtitle", {})

    md = U.render_md(utts, job["source"], job.get("duration", 0.0), job.get("meta_line", ""))
    output.write_text(job_dir, "transcript.md", md)
    output.write_text(job_dir, "文字起こし.md", _plain_md(job))
    cues = U.build_cues(utts, sub)
    output.write_text(job_dir, "transcript.srt", U.cues_to_srt(cues, sub.get("speaker_prefix", False)))
    output.write_text(job_dir, "transcript.vtt", U.cues_to_vtt(cues, sub.get("speaker_prefix", False)))
    output.save_json(job_dir, "utterances.json", {
        "source": job["source"],
        "duration": job.get("duration", 0.0),
        "meta_line": job.get("meta_line", ""),
        "meeting_date": job.get("meeting_date", ""),
        "memo": job.get("memo", ""),
        "utterances": utts,
    })
    if job.get("cluster_embeddings"):
        output.save_json(job_dir, "speaker_clusters.json", job["cluster_embeddings"])


def _plain_md(job: dict) -> str:
    return U.render_md(job["utterances"], job["source"], job.get("duration", 0.0),
                       job.get("meta_line", ""), plain=True)


def _finalize_text(utts: list[dict], people_list: list) -> None:
    """人名の別名・用語の直し・「ー」→「一」を発話テキストに反映する"""
    for u in utts:
        u["marked"] = U.fix_ichi(glossary.apply_corrections(people.apply_aliases(u["marked"], people_list)))
        u["text"] = U.strip_marks(u["marked"])


def _voice_tag(job_dir) -> str:
    return f"job:{Path(job_dir).name}"


def learn_voices(job: dict, progress=None) -> str:
    """確定した話者名と会議の音声から声紋を学習する（同じ会議の前回分は置き換え）。
    次の会議から、同じ声に自動で名前が付きやすくなる。結果の一言を返す。"""
    if not diarization.available() or not job.get("audio_path") \
            or not Path(job["audio_path"]).exists():
        return ""
    if not any(diarization.is_named(u.get("speaker", "")) for u in job.get("utterances", [])):
        return ""
    wav = None
    try:
        wav = audio.prepare_audio(job["audio_path"])
        learned = diarization.learn_from_utterances(wav, job["utterances"], _voice_tag(job["job_dir"]))
    except Exception as e:
        logging.warning(f"声紋の学習に失敗: {e}")
        return ""
    finally:
        if wav:
            Path(wav).unlink(missing_ok=True)
    if not learned:
        return ""
    return "🔊 この会議の声で声紋を学習しました（" + "、".join(learned) + "）。次回から自動で名前が付きます。"


def _transcribe_one(path: str, opts: dict, settings: dict, sub) -> dict:
    """フェーズ1: 音声変換 → 文字起こし → 話者分離 → 発話構築（GPU: Whisper / ECAPA）"""
    name = Path(path).name
    job_dir = output.create_job_dir(name)

    sub(0.01, "音声を変換中...")
    wav = audio.prepare_audio(path)
    asr_wav = wav
    try:
        duration = audio.get_duration(wav)
        preview = audio.make_preview_audio(path, str(job_dir / "audio.m4a"))
        if opts["noise"]:
            sub(0.03, "文字起こし用に音声を補正中（ノイズ除去）...")
        asr_wav = audio.enhance_for_asr(wav, eq=opts.get("eq", True), noise_reduction=opts["noise"])

        people_list = people.load_people()
        terms = people.initial_prompt_terms(people_list)
        sub(0.05, "文字起こし中...")
        result = transcriber.transcribe(
            asr_wav, model_name=opts["model"], language=opts["language"],
            initial_prompt=glossary.build_initial_prompt(terms),
            hotwords=glossary.build_hotwords(terms),
            duration=duration, speed=opts.get("speed", "accurate"),
            decoding=opts.get("decoding", settings.get("whisper", {}).get("decoding", "stable")),
            silence_guard=opts.get("silence_guard", settings.get("whisper", {}).get("silence_guard", True)),
            progress_cb=lambda f, desc=None: sub(0.05 + 0.75 * f, desc or f"文字起こし中... {int(f * 100)}%"),
        )
        punctuate.punctuate_segments(result["segments"])  # 「、」「。」を推定して補う

        word_speakers, cluster_embeddings, matched = {}, {}, {}
        if opts["diarize"] and diarization.available():
            sub(0.82, "話者を識別中（声紋照合）...")
            dia = settings.get("diarization", {})
            word_speakers, cluster_embeddings, matched = diarization.diarize(
                wav, result["segments"],
                num_speakers=int(opts["num_speakers"] or 0),
                max_speakers=int(dia.get("max_speakers", 12)), hf_token=dia.get("hf_token", ""),
                match_similarity=float(dia.get("match_similarity", 0.55)),
            )

        threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
        utts = U.build_utterances(result["segments"], word_speakers, threshold)
        fillers = 0
        if opts.get("fillers", True):
            utts, fillers = U.remove_fillers(utts, threshold)
        _finalize_text(utts, people_list)

        # 声紋DBとはっきり一致した人は、この会議の声もすぐ学習する（あいまいな一致は、
        # ②で確かめてから「③ 議事録を作る」の時に学習する）
        # 「はっきり」の判定は、会議ごとの塊ではなく声紋全体の代表との類似度で行う
        # （塊が多いほど偶然しきい値を超えやすく、間違いを自分で学習し続けるのを防ぐ）
        dia = settings.get("diarization", {})
        overall = diarization.load_voiceprints() if matched else {}
        thr = float(dia.get("match_similarity", 0.55))
        sure = {n for n in matched if n in overall and n in cluster_embeddings
                and float(overall[n] @ cluster_embeddings[n]) >= thr}
        if sure and dia.get("auto_learn", True):
            sub(0.97, "声紋を学習中...")
            try:
                diarization.learn_from_utterances(wav, utts, _voice_tag(job_dir), only=sure)
            except Exception as e:
                logging.warning(f"声紋の学習に失敗: {e}")
    finally:
        Path(wav).unlink(missing_ok=True)
        if asr_wav != wav:
            Path(asr_wav).unlink(missing_ok=True)

    speed_label = dict((v, k) for k, v in transcriber.SPEED_CHOICES).get(result.get("speed"), "")
    meta_line = f"モデル: {result.get('model')}（{result.get('device')}・{speed_label.split('（')[0]}）"
    decoding_label = dict((v, k) for k, v in transcriber.DECODING_CHOICES).get(result.get("decoding"))
    if decoding_label:
        meta_line += f" ／ 認識方式: {decoding_label}"
    silence_guard = result.get("silence_guard")
    if result.get("speed") == "fast":
        meta_line += " ／ 内部スキップ: 対象外（高速モード）"
    elif silence_guard is not None:
        meta_line += f" ／ 内部スキップ: {'ON' if silence_guard else 'OFF'}"
    job = {
        "job_dir": str(job_dir),
        "source": name,
        "duration": duration,
        "meta_line": meta_line,
        "meeting_date": opts.get("meeting_date") or _meeting_date_from_name(name),
        "memo": opts.get("memo", ""),
        "utterances": utts,
        "cluster_embeddings": {k: v.tolist() for k, v in cluster_embeddings.items()},
        "audio_path": preview,
        "errors": [],
        "matched": matched,
        "stats": {
            "whisper_model": result.get("model"),
            "asr_decoding": result.get("decoding"),
            "asr_silence_guard": silence_guard,
            "device": result.get("device"),
            "hallucinations_dropped": result.get("dropped", 0),
            "asr_refilled_sec": result.get("refilled_sec", 0.0),
            "asr_missing_sec": result.get("missing_sec", 0.0),
            "fillers_removed": fillers,
            "elapsed_sec": round(result.get("elapsed", 0), 1),
        },
    }
    _save_job(job)
    sub(1.0, "文字起こし完了")
    return job


# =====================================================================
# core.worker から別プロセスで呼ぶ入口（progress(割合, 説明) で進み具合を返す）
# =====================================================================

def transcribe_files(paths: list[str], opts: dict, settings: dict, progress=None) -> list[dict]:
    """①の本体。ファイルごとに {"job": 結果} か {"error": メッセージ} を返す。
    Whisperのモデルは1回だけ読み込んで全ファイルで使い回す。"""
    import traceback
    progress = progress or (lambda f, d="": None)
    n = len(paths)
    results = []
    for i, path in enumerate(paths):
        name = Path(path).name

        def sub(frac, desc, i=i, name=name):
            progress((i + frac) / n, f"[{i + 1}/{n}] {name}: {desc}")

        try:
            results.append({"job": _transcribe_one(path, opts, settings, sub)})
        except Exception as e:
            logging.error(traceback.format_exc())
            results.append({"error": f"{name}: {e}"})
    return results


def rediarize(audio_path: str, segments: list[dict], num_speakers: int, settings: dict,
              progress=None):
    """話者識別だけやり直す。Returns: (word_speakers, cluster_embeddings(list), matched)"""
    progress = progress or (lambda f, d="": None)
    progress(0.05, "音声を準備中...")
    wav = audio.prepare_audio(audio_path)
    try:
        progress(0.2, "話者を識別中（声紋照合）...")
        dia = settings.get("diarization", {})
        word_speakers, cluster_embeddings, matched = diarization.diarize(
            wav, segments, num_speakers=int(num_speakers or 0),
            max_speakers=int(dia.get("max_speakers", 12)), hf_token=dia.get("hf_token", ""),
            match_similarity=float(dia.get("match_similarity", 0.55)),
        )
    finally:
        Path(wav).unlink(missing_ok=True)
    return word_speakers, {k: v.tolist() for k, v in cluster_embeddings.items()}, matched


def enroll(src: str, name: str, progress=None) -> tuple[int, float]:
    wav = audio.prepare_audio(src)
    try:
        return diarization.enroll_from_audio(wav, name)
    finally:
        Path(wav).unlink(missing_ok=True)


def identify(src: str, progress=None) -> list[tuple[str, float]]:
    wav = audio.prepare_audio(src)
    try:
        return diarization.identify(wav)
    finally:
        Path(wav).unlink(missing_ok=True)
