"""文字起こしの本体（音声変換 → Whisper → 話者分離 → 発言の構造化 → 保存）と声紋の学習。

GPUを使う処理は core.worker から別プロセスで呼ばれる（終わるとVRAM・RAMが全部OSに返る）。
このモジュールは gradio を読み込まない（別プロセスの起動を軽くするため）。
"""
import copy
import json
import logging
import re
from pathlib import Path

from . import audio, checkpoints, config, diarization, glossary, output, people, punctuate, run_control, transcriber
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
    if job.get("incomplete"):
        raise ValueError(checkpoints.READ_ONLY_NOTICE)
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
    if job.get("cluster_embeddings"):
        output.save_json(job_dir, "speaker_clusters.json", job["cluster_embeddings"])
    # 完成データは最後に公開する。途中で失敗した場合、履歴はcheckpoint.jsonを開く。
    output.save_json(job_dir, "utterances.json", {
        "source": job["source"],
        "duration": job.get("duration", 0.0),
        "meta_line": job.get("meta_line", ""),
        "meeting_date": job.get("meeting_date", ""),
        "memo": job.get("memo", ""),
        "utterances": utts,
    })


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


def _run_options(opts, settings):
    """再開に必要な選択だけ保存。APIトークン等を含む設定全体は保存しない。"""
    values = dict(model="auto", speed="accurate", language="ja", num_speakers=0,
                  noise=False, eq=True, fillers=True, diarize=False, use_llm=False,
                  meeting_date="", memo="",
                  decoding=settings.get("whisper", {}).get("decoding", "stable"),
                  silence_guard=settings.get("whisper", {}).get("silence_guard", True))
    return {key: opts.get(key, default) for key, default in values.items()}


def _transcribe_one(path: str, opts: dict, settings: dict, sub, pause_file=None) -> dict:
    job_dir = output.create_job_dir(Path(path).name)
    with checkpoints.job_lock(job_dir):
        return _run_transcription(job_dir, path, _run_options(opts, settings), settings, sub, pause_file=pause_file)


def resume_job(job_dir: str, paths: list[str], opts: dict, settings: dict, progress=None, pause_file=None) -> dict:
    """別プロセスから、同じ保存先・同じ音声・同じ認識設定で続きを実行する。"""
    folder = Path(job_dir)
    with checkpoints.job_lock(folder):
        if (folder / "utterances.json").exists():
            raise ValueError("この文字起こしは完了済みです。履歴から開き直してください。")
        data = json.loads((folder / checkpoints.FILENAME).read_text(encoding="utf-8"))
        checkpoints.validate(data)
        if data.get("resume_audio"):
            if not data.get("run_options") or not isinstance(data.get("hints"), dict):
                raise ValueError("再開用の認識設定が見つかりません。元の保存フォルダを復元してください。")
            opts = dict(data["run_options"])
            settings = copy.deepcopy(settings)
            for section, values in data.get("processing_settings", {}).items():
                settings.setdefault(section, {}).update(values)
            path = None
        else:
            if len(paths) != 1 or Path(paths[0]).name != data["source"]:
                raise ValueError("旧形式の途中保存です。同じ名前の元音声を1件だけ選び、認識設定を当時の値に合わせてください。")
            path = paths[0]
            opts = _run_options(opts, settings)
        opts.update(meeting_date=data.get("meeting_date", ""), memo=data.get("memo", ""))
        return _run_transcription(folder, path, opts, settings, progress or (lambda f, d: None), data, pause_file)


def _run_transcription(job_dir, path, opts, settings, sub, resume=None, pause_file=None):
    name = resume["source"] if resume else Path(path).name
    temporary = set()
    checkpoint = None
    pause = lambda: run_control.check(pause_file)
    try:
        people_list = people.load_people()
        if resume and resume.get("resume_audio"):
            sub(0.01, "保存された音声を確認中...")
            wav, asr_wav = checkpoints.audio_paths(job_dir, resume)
            duration = resume["duration"]
            records, hints = resume["resume_audio"], resume["hints"]
        else:
            sub(0.01, "音声を変換中...")
            wav = audio.prepare_audio(path)
            temporary.add(wav)
            duration = audio.get_duration(wav)
            if resume and abs(duration - resume["duration"]) > 0.1:
                raise ValueError("選択した音声の長さが途中保存と一致しません。同じ元音声を選んでください。")
            if resume:
                duration = resume["duration"]
            sub(0.03, "文字起こし用の音声を準備中...")
            asr_wav = audio.enhance_for_asr(wav, eq=opts["eq"], noise_reduction=opts["noise"])
            temporary.add(asr_wav)
            records = checkpoints.keep_audio(job_dir, wav, asr_wav)
            terms = people.initial_prompt_terms(people_list)
            hints = dict(initial_prompt=glossary.build_initial_prompt(terms), hotwords=glossary.build_hotwords(terms))
        checkpoint = checkpoints.Writer(job_dir, name, duration,
                                         opts.get("meeting_date") or _meeting_date_from_name(name),
                                         opts.get("memo", ""), seed=resume)
        checkpoint.base.update(resume_audio=records, run_options=opts, hints=hints,
                               processing_settings={
                                   "postprocess": {"red_threshold": settings.get("postprocess", {}).get("red_threshold", 0.5)},
                                   "diarization": {key: settings.get("diarization", {}).get(key, default)
                                                   for key, default in (("max_speakers", 12), ("match_similarity", 0.55), ("auto_learn", True))},
                               })
        checkpoint({**(resume or dict(phase="prepared", segments=[], processed_sec=0)), "interrupted": False, "paused": False})
        pause()
        preview_path = job_dir / "audio.m4a"
        preview = str(preview_path) if preview_path.exists() else audio.make_preview_audio(path or wav, str(preview_path))
        pause()
        if resume and resume["phase"] == "postprocess":
            # 旧形式には認識統計がないため、分かる範囲だけ復元する。
            result = {**dict(model=opts["model"], speed=opts["speed"], decoding=opts["decoding"],
                            silence_guard=opts["silence_guard"], device="以前の処理", duration=duration),
                      **resume.get("asr_result", {}), "segments": copy.deepcopy(resume["segments"])}
        else:
            sub(0.05, "保存位置から文字起こしを再開中..." if resume else "文字起こし中...")
            result = transcriber.transcribe(
                asr_wav, model_name=opts["model"], language=opts["language"], **hints,
                duration=duration, speed=opts["speed"], decoding=opts["decoding"], silence_guard=opts["silence_guard"],
                progress_cb=lambda f, desc=None: sub(0.05 + 0.75 * f, desc or f"文字起こし中... {int(f * 100)}%"),
                checkpoint_cb=checkpoint, resume=resume, pause_cb=pause,
            )
        checkpoint({"phase": "postprocess", "segments": copy.deepcopy(result["segments"]),
                    "processed_sec": duration, "asr_result": {k: v for k, v in result.items() if k != "segments"}})
        job = _finish_transcription(job_dir, name, wav, preview, duration, result, opts, settings, people_list, sub, pause)
    except run_control.PauseRequested:
        checkpoint.pause()
        sub(1.0, "途中保存を完了し、一時停止しました")
        return checkpoints.as_job(job_dir, output.load_json(job_dir, checkpoints.FILENAME))
    except Exception:
        if checkpoint is not None:
            try:
                checkpoint.interrupted()
            except Exception:
                logging.exception("途中結果の最終保存に失敗しました。前回の保存は保持します")
        raise
    finally:
        for filename in temporary:
            Path(filename).unlink(missing_ok=True)
    # 完成データを公開できてから削除。保存に失敗した場合にも音声を残す。
    try:
        checkpoints.cleanup_audio(job_dir)
    except OSError:
        logging.warning("文字起こしは完了しましたが再開用音声の削除に失敗しました", exc_info=True)
    return job


def _finish_transcription(job_dir, name, wav, preview, duration, result, opts, settings, people_list, sub,
                          pause=lambda: None):
    pause()
    punctuate.punctuate_segments(result["segments"])  # 「、」「。」を推定して補う
    pause()

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
    pause()

    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
    utts = U.build_utterances(result["segments"], word_speakers, threshold)
    fillers = 0
    if opts.get("fillers", True):
        utts, fillers = U.remove_fillers(utts, threshold)
    _finalize_text(utts, people_list)
    pause()

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
        "run_options": opts,
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
    pause()
    _save_job(job)
    sub(1.0, "文字起こし完了")
    return job


# =====================================================================
# core.worker から別プロセスで呼ぶ入口（progress(割合, 説明) で進み具合を返す）
# =====================================================================

def transcribe_files(paths: list[str], opts: dict, settings: dict, progress=None, pause_file=None) -> list[dict]:
    """①の本体。ファイルごとに {"job": 結果} か {"error": メッセージ} を返す。
    Whisperのモデルは1回だけ読み込んで全ファイルで使い回す。"""
    import traceback
    progress = progress or (lambda f, d="": None)
    n = len(paths)
    results = []
    for i, path in enumerate(paths):
        if run_control.requested(pause_file):
            results.append({"paused": None, "remaining": [Path(p).name for p in paths[i:]]})
            break
        name = Path(path).name

        def sub(frac, desc, i=i, name=name):
            progress((i + frac) / n, f"[{i + 1}/{n}] {name}: {desc}")

        try:
            job = _transcribe_one(path, opts, settings, sub, pause_file)
            if job.get("incomplete"):
                results.append({"paused": job, "remaining": [Path(p).name for p in paths[i + 1:]]})
                break
            results.append({"job": job})
        except Exception as e:
            logging.error(traceback.format_exc())
            results.append({"error": f"{name}: {e}（保存済みの途中結果は「過去の結果を開く」で確認できます）"})
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
