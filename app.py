import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# OSの証明書ストアを使う（セキュリティソフトのSSL検査環境でもモデルDLが通るように）
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import datetime
import logging
import re
import traceback
from pathlib import Path

import gradio as gr
import httpx

from core import audio, config, diarization, glossary, llm, minutes, output, people, punctuate, transcriber
from core import export as export_mod
from core import search as search_mod
from core import utterances as U

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

config.ensure_dirs()

FILE_TYPES = [f".{e.lstrip('.')}" for e in audio.SUPPORTED_EXTS]
SPEAKER_COUNT_CHOICES = [("自動", 0)] + [(f"{n}人", n) for n in range(1, 11)]


def _rows(df) -> list[list]:
    """gr.Dataframe の値（pandas / list）→ list[list]"""
    if df is None:
        return []
    if hasattr(df, "values"):
        return df.values.tolist()
    return [list(r) for r in df]


def _cell(row, i, default=""):
    if len(row) <= i or row[i] is None:
        return default
    v = row[i]
    if isinstance(v, float) and v != v:  # NaN
        return default
    return str(v).strip()


def _meeting_date_from_name(name: str) -> str:
    m = re.search(r"(20\d{2})[-_.年]?(\d{1,2})[-_.月]?(\d{1,2})", name)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}年{mo}月{d}日"
    return ""


def _file_url(path: str | None) -> str:
    from urllib.parse import quote
    return f"/gradio_api/file={quote(Path(path).as_posix(), safe='/:')}" if path else ""


# =====================================================================
# ジョブ（1ファイル分の結果）の保存・再描画
# =====================================================================

def _save_job(job: dict) -> None:
    """発言データから transcript.md / srt / vtt / utterances.json を書き出す"""
    job_dir = Path(job["job_dir"])
    utts = job["utterances"]
    settings = config.load_settings()
    sub = settings.get("subtitle", {})

    md = U.render_md(utts, job["source"], job.get("duration", 0.0), job.get("meta_line", ""))
    output.write_text(job_dir, "transcript.md", md)
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


def _load_job(job_name: str) -> dict | None:
    job_dir = output.job_path(job_name)
    data = output.load_json(job_dir, "utterances.json")
    if not data:
        return None
    preview = job_dir / "audio.m4a"
    return {
        "job_dir": str(job_dir),
        "source": data.get("source", job_name),
        "duration": data.get("duration", 0.0),
        "meta_line": data.get("meta_line", ""),
        "meeting_date": data.get("meeting_date", ""),
        "memo": data.get("memo", ""),
        "utterances": data.get("utterances", []),
        "cluster_embeddings": output.load_json(job_dir, "speaker_clusters.json", {}) or {},
        "audio_path": str(preview) if preview.exists() else None,
    }


def _read_md(job: dict, name: str) -> str:
    path = Path(job["job_dir"]) / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _download(job: dict, name: str):
    path = Path(job["job_dir"]) / name
    if path.exists():
        return gr.update(value=str(path), interactive=True)
    return gr.update(value=None, interactive=False)


def _speaker_rows(job: dict) -> list[list]:
    labels = []
    for u in job["utterances"]:
        sp = u.get("speaker") or ""
        if sp and sp not in labels:
            labels.append(sp)
    return [[sp, ""] for sp in labels] or [["", ""]]


def _edit_rows(job: dict) -> list[list]:
    # 表に出した内容を覚えておき、保存時は「人が実際に変えたセル」だけを反映する
    job["_shown"] = [[u.get("speaker", ""), u["text"]] for u in job["utterances"]]
    return [[i + 1, U.format_hms(u["start"]), u.get("speaker", ""), u["text"]]
            for i, u in enumerate(job["utterances"])] or [[1, "", "", ""]]


def _cue_rows(cues: list[dict]) -> list[list]:
    return [[i + 1, U.format_srt_time(c["start"]), U.format_srt_time(c["end"]),
             c.get("speaker", ""), c["text"]] for i, c in enumerate(cues)] or [[1, "", "", "", ""]]


def _job_views(job: dict):
    """結果表示コンポーネント一式の更新値"""
    return (
        U.render_html(job["utterances"], _file_url(job.get("audio_path"))),
        _read_md(job, "議事録.md") or "_議事録はまだありません（LLM設定を確認して「議事録・要約を作り直す」を押してください）_",
        _read_md(job, "要約と考察.md") or "_要約と考察はまだありません_",
        _download(job, "transcript.md"),
        _download(job, "議事録.md"),
        _download(job, "transcript.srt"),
        _speaker_rows(job),
        _edit_rows(job),
    )


# =====================================================================
# メイン処理パイプライン
# =====================================================================

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
            progress_cb=lambda f: sub(0.05 + 0.75 * f, f"文字起こし中... {int(f * 100)}%"),
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
    finally:
        Path(wav).unlink(missing_ok=True)
        if asr_wav != wav:
            Path(asr_wav).unlink(missing_ok=True)

    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
    utts = U.build_utterances(result["segments"], word_speakers, threshold)
    fillers = 0
    if opts.get("fillers", True):
        utts, fillers = U.remove_fillers(utts, threshold)
    for u in utts:
        u["marked"] = glossary.apply_corrections(people.apply_aliases(u["marked"], people_list))
        u["text"] = U.strip_marks(u["marked"])

    speed_label = dict((v, k) for k, v in transcriber.SPEED_CHOICES).get(result.get("speed"), "")
    meta_line = f"モデル: {result.get('model')}（{result.get('device')}・{speed_label.split('（')[0]}）"
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
            "device": result.get("device"),
            "hallucinations_dropped": result.get("dropped", 0),
            "fillers_removed": fillers,
            "elapsed_sec": round(result.get("elapsed", 0), 1),
        },
    }
    _save_job(job)
    sub(1.0, "文字起こし完了")
    return job


def _llm_one(job: dict, opts: dict, settings: dict, sub) -> None:
    """フェーズ2: LLM校正 → 議事録 → 要約（GPU: Ollama）"""
    job_dir = Path(job["job_dir"])
    if opts["proofread"]:
        try:
            done, changed = minutes.proofread_utterances(
                settings, job["utterances"],
                progress_cb=lambda f: sub(0.4 * f, f"AI校正中... {int(f * 100)}%"))
            _save_job(job)
            job["proofread"] = f"{done}発言を校正（{changed}発言を修正）"
        except llm.LLMError as e:
            _save_job(job)  # 途中まで校正できた分も保存して表示と一致させる
            job["errors"].append(f"校正スキップ: {e}")

    transcript = minutes.transcript_for_llm(job["utterances"])
    if job.get("memo"):
        transcript = f"（会議メモ: {job['memo']}）\n\n" + transcript
    if opts["minutes"]:
        sub(0.45, "議事録を作成中...")
        try:
            md = minutes.generate_minutes(settings, transcript, job["source"], job.get("meeting_date", ""))
            output.write_text(job_dir, "議事録.md", md)
        except llm.LLMError as e:
            job["errors"].append(f"議事録スキップ: {e}")
    if opts["insights"]:
        sub(0.8, "要約と考察を作成中...")
        try:
            md = minutes.generate_insights(settings, transcript, job["source"], job.get("meeting_date", ""))
            output.write_text(job_dir, "要約と考察.md", md)
        except llm.LLMError as e:
            job["errors"].append(f"要約スキップ: {e}")
    notice = llm.pop_fallback_notice()
    if notice:
        job["errors"].append(notice)
    sub(1.0, "完了")


def _write_meta(job: dict, opts: dict, settings: dict) -> None:
    output.write_meta(Path(job["job_dir"]), {
        "source": job["source"],
        "duration_sec": round(job.get("duration", 0.0), 1),
        **job.get("stats", {}),
        "noise_reduction": opts["noise"],
        "asr_eq": opts.get("eq", True),
        "remove_fillers": opts.get("fillers", True),
        "llm_provider": llm.provider_label(settings) if opts["use_llm"] else "なし",
        "speakers": [s[0] for s in U.speaker_stats(job["utterances"])],
        "voiceprint_matched": job.get("matched", {}),
        "proofread": job.get("proofread", ""),
        "errors": job.get("errors", []),
    })


def run_batch(files, model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on, do_proofread,
              do_minutes, do_insights, meeting_date, memo, progress=gr.Progress()):
    if not files:
        return ("⚠️ ファイルを選択してください。", gr.update()) + \
            tuple(gr.update() for _ in range(8)) + (gr.update(),)

    settings = config.load_settings()
    use_llm = (do_proofread or do_minutes or do_insights) and settings["llm"]["provider"] != "none"
    opts = {
        "model": model_name, "speed": speed, "language": language, "num_speakers": num_speakers,
        "noise": noise, "eq": eq_on, "fillers": fillers_on, "diarize": diarize_on, "proofread": do_proofread,
        "minutes": do_minutes, "insights": do_insights, "use_llm": use_llm,
        "meeting_date": (meeting_date or "").strip(), "memo": (memo or "").strip(),
    }
    paths = [f if isinstance(f, str) else f.name for f in files]
    n = len(paths)
    # 進捗配分: 文字起こし 0〜(0.55 or 1.0)、LLM 残り
    t_share = 0.55 if use_llm else 1.0

    # 前回のLLM(gemma4:31b)がメモリに残っているとWhisperが読み込めないため先に解放
    llm.unload_ollama(settings)

    jobs, failures = [], []
    for i, path in enumerate(paths):
        name = Path(path).name

        def sub(frac, desc, i=i, name=name):
            progress(t_share * (i + frac) / n, desc=f"[{i + 1}/{n}] {name}: {desc}")

        try:
            jobs.append(_transcribe_one(path, opts, settings, sub))
        except Exception as e:
            logging.error(traceback.format_exc())
            failures.append(f"❌ {name}: {e}")

    if use_llm and jobs:
        # Whisper・声紋モデルをVRAMから降ろして、LLM(gemma4)にGPUを譲る
        transcriber.unload()
        diarization.unload()
        for i, job in enumerate(jobs):
            def sub(frac, desc, i=i, name=job["source"]):
                progress(t_share + (1 - t_share) * (i + frac) / len(jobs),
                         desc=f"[{i + 1}/{len(jobs)}] {name}: {desc}")
            try:
                _llm_one(job, opts, settings, sub)
            except Exception as e:
                logging.error(traceback.format_exc())
                job["errors"].append(f"LLM処理エラー: {e}")

    for job in jobs:
        _write_meta(job, opts, settings)

    lines = [f"### 処理結果（{len(jobs)}/{n} 件成功）"]
    for job in jobs:
        stats = U.speaker_stats(job["utterances"])
        speakers = "、".join(s[0] for s in stats) or "話者分離なし"
        lines.append(f"✅ **{job['source']}** → `output/{Path(job['job_dir']).name}/`  \n"
                     f"　発言 {len(job['utterances'])} 件 ／ 話者: {speakers}")
        if job.get("matched"):
            lines.append("　🔊 声紋で自動認識: " + "、".join(
                f"{k}（一致度{v:.2f}）" for k, v in job["matched"].items()))
        if job.get("proofread"):
            lines.append(f"　✍️ {job['proofread']}")
        dropped = job.get("stats", {}).get("hallucinations_dropped", 0)
        if dropped:
            lines.append(f"　🧹 無音・雑音区間の誤認識と思われる {dropped} 箇所を除外しました")
        for err in job["errors"]:
            lines.append(f"　⚠️ {err}")
    lines.extend(failures)
    summary = "\n\n".join(lines)

    if not jobs:
        return (summary, gr.update()) + tuple(gr.update() for _ in range(8)) + (gr.update(),)
    last = jobs[-1]
    views = _job_views(last)
    state = {k: v for k, v in last.items() if k not in ("errors", "matched", "stats")}
    jobs_list = output.list_editable_jobs()
    return (summary, state) + views + (
        gr.update(choices=jobs_list, value=Path(last["job_dir"]).name),)


def open_job(job_name):
    job = _load_job(job_name) if job_name else None
    if not job:
        return ("⚠️ 開けませんでした（古い形式の出力には対応していません）", gr.update()) + \
            tuple(gr.update() for _ in range(8))
    return (f"📂 `{job_name}` を開きました", job) + _job_views(job)


def rediarize_job(state, num_speakers, progress=gr.Progress()):
    """保存済みの文字起こし（単語タイミング）と再生用音声を使って、話者識別だけやり直す"""
    none = tuple(gr.update() for _ in range(8))
    if not state:
        return ("⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。", gr.update()) + none
    if not diarization.available():
        return ("❌ 話者分離ライブラリが未導入です。", gr.update()) + none
    if not state.get("audio_path") or not Path(state["audio_path"]).exists():
        return ("⚠️ この結果には再生用音声(audio.m4a)が無いため、やり直せません。", gr.update()) + none
    job = state
    settings = config.load_settings()
    llm.unload_ollama(settings)  # GPUを空ける
    progress(0.05, desc="音声を準備中...")
    segments = [{"start": u["start"], "end": u["end"], "text": u["text"],
                 "words": u.get("words") or []} for u in job["utterances"]]
    wav = audio.prepare_audio(job["audio_path"])
    try:
        progress(0.2, desc="話者を識別中（声紋照合）...")
        dia = settings.get("diarization", {})
        word_speakers, cluster_embeddings, matched = diarization.diarize(
            wav, segments, num_speakers=int(num_speakers or 0),
            max_speakers=int(dia.get("max_speakers", 12)), hf_token=dia.get("hf_token", ""),
            match_similarity=float(dia.get("match_similarity", 0.55)),
        )
    finally:
        Path(wav).unlink(missing_ok=True)
    if not word_speakers:
        return ("❌ 話者識別に失敗しました（ログを確認してください）", gr.update()) + none
    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
    people_list = people.load_people()
    utts = U.build_utterances(segments, word_speakers, threshold)
    for u in utts:
        u["marked"] = glossary.apply_corrections(people.apply_aliases(u["marked"], people_list))
        u["text"] = U.strip_marks(u["marked"])
    job["utterances"] = utts
    job["cluster_embeddings"] = {k: v.tolist() for k, v in cluster_embeddings.items()}
    _save_job(job)
    stats = U.speaker_stats(utts)
    msg = (f"✅ 話者識別をやり直しました: {len(stats)}名 ／ 発言 {len(utts)} 件"
           + (f"\n\n🔊 声紋で自動認識: " + "、".join(f"{k}（{v:.2f}）" for k, v in matched.items())
              if matched else "")
           + "\n\n※ 文字は音声認識の結果から作り直しました（AI校正・手修正は反映されていません）。"
             "議事録に反映するには「🔁 議事録・要約を作り直す」を押してください。")
    return (msg, job) + _job_views(job)


def refresh_history():
    return gr.update(choices=output.list_editable_jobs())


def regenerate_llm(state, do_proofread, progress=gr.Progress()):
    none = tuple(gr.update() for _ in range(8))
    if not state:
        return ("⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。", gr.update()) + none
    settings = config.load_settings()
    if settings["llm"]["provider"] == "none":
        return ("⚠️ 設定タブでLLMを選択してください。", gr.update()) + none
    transcriber.unload()
    diarization.unload()
    job = dict(state, errors=[])
    opts = {"proofread": do_proofread, "minutes": True, "insights": True}
    try:
        _llm_one(job, opts, settings, lambda f, d: progress(f, desc=d))
    except Exception as e:
        logging.error(traceback.format_exc())
        job["errors"].append(f"LLM処理エラー: {e}")
    msg = "✅ 議事録・要約を作り直しました"
    if job.get("proofread"):
        msg += f"（{job['proofread']}）"
    if job["errors"]:
        msg += "\n\n" + "\n\n".join(f"⚠️ {e}" for e in job["errors"])
    job.pop("errors", None)
    return (msg, job) + _job_views(job)


# =====================================================================
# 話者の確定・発言の修正（修正は学習される）
# =====================================================================

def assign_speakers(rows, save_voice, state):
    if not state:
        return ("⚠️ 先に文字起こしを実行してください。", state) + tuple(gr.update() for _ in range(8))
    job = state
    embeddings = job.get("cluster_embeddings", {}) or {}
    people_list = people.load_people()
    known_names = {p["display_name"] for p in people_list}
    mapping = {}
    for row in _rows(rows):
        label, new_name = _cell(row, 0), _cell(row, 1)
        if label and new_name and label != new_name:
            err = diarization.validate_name(new_name)
            if err:
                return (f"⚠️ {new_name}: {err}", state) + tuple(gr.update() for _ in range(8))
            mapping[label] = new_name
    if not mapping:
        return ("⚠️ 「名前」列に入力してください。", state) + tuple(gr.update() for _ in range(8))

    # 一括で置き換える（入れ替え A⇔B や連鎖 A→B→C でも話者が混ざらない）
    for u in job["utterances"]:
        u["speaker"] = mapping.get(u.get("speaker"), u.get("speaker"))
    grouped, saved = {}, []
    for label, vec in embeddings.items():
        grouped.setdefault(mapping.get(label, label), []).append(vec)
    for name, vecs in grouped.items():
        if save_voice and diarization.available() and name in mapping.values():
            diarization.save_voiceprint(name, vecs)
            saved.append(name)
    embeddings = {n: (v[0] if len(v) == 1 else [sum(x) / len(v) for x in zip(*v)])
                  for n, v in grouped.items()}
    done = []
    for label, new_name in mapping.items():
        if new_name not in known_names:
            people_list.append({"display_name": new_name, "aliases": []})
            known_names.add(new_name)
        done.append(f"{label}→{new_name}")
    people.save_people(people_list)
    job["cluster_embeddings"] = embeddings
    _save_job(job)
    msg = f"✅ 話者を確定しました: {'、'.join(done)}"
    if saved:
        msg += f"\n\n🔊 声紋を保存しました（{'、'.join(saved)}）。次回から自動で名前が付きます。"
    msg += "\n\n議事録に反映するには「🔁 議事録・要約を作り直す」を押してください。"
    return (msg, job) + _job_views(job)


def save_utterance_edits(rows, state):
    if not state:
        return ("⚠️ 先に文字起こしを実行してください。", state) + tuple(gr.update() for _ in range(8))
    job = state
    utts = job["utterances"]
    shown = job.get("_shown") or [[u.get("speaker", ""), u["text"]] for u in utts]
    seen, new_utts, learned, changed = set(), [], 0, 0
    for row in _rows(rows):
        try:
            idx = int(float(_cell(row, 0, "0"))) - 1
        except ValueError:
            continue
        if not (0 <= idx < len(utts)) or idx in seen:
            continue
        seen.add(idx)
        u = dict(utts[idx])
        old_sp, old_text = shown[idx] if idx < len(shown) else (u.get("speaker", ""), u["text"])
        speaker, text = _cell(row, 2), _cell(row, 3)
        if not text:
            changed += 1  # 空にした行は削除扱い
            continue
        if text != old_text:  # 表示していた内容から人が変えた場合のみ
            learned += glossary.learn_from_diff(old_text, text)
            u["text"] = text
            u["marked"] = text  # 人が確認した文は赤字を解除
            changed += 1
        if speaker != (old_sp or ""):
            u["speaker"] = speaker
            changed += 1
        new_utts.append(u)
    new_utts.sort(key=lambda u: u["start"])
    job["utterances"] = new_utts
    _save_job(job)
    msg = f"✅ 保存しました（{changed}箇所を変更）"
    if learned:
        msg += f"。{learned}件の言い間違い・誤認識を学習しました（同じ修正が{glossary.PROMOTE_COUNT}回で自動適用）"
    return (msg, job) + _job_views(job)


# =====================================================================
# SRT編集
# =====================================================================

def _sub_opts(max_chars, max_lines, max_dur, min_dur, pause, spk_prefix, drop_period) -> dict:
    return {
        "max_chars_line": int(max_chars), "max_lines": int(max_lines),
        "max_duration": float(max_dur), "min_duration": float(min_dur),
        "pause_split": float(pause), "speaker_prefix": bool(spk_prefix),
        "drop_period": bool(drop_period),
    }


def _rows_to_cues(rows) -> tuple[list[dict], list[str]]:
    cues, errors = [], []
    for i, row in enumerate(_rows(rows), 1):
        text = _cell(row, 4)
        if not text:
            continue
        try:
            start, end = U.parse_time(_cell(row, 1)), U.parse_time(_cell(row, 2))
        except ValueError as e:
            errors.append(f"行{i}: {e}")
            continue
        cues.append({"start": start, "end": end, "speaker": _cell(row, 3), "text": text})
    return cues, errors


def _warn_md(cues, opts, extra=None) -> str:
    warns = (extra or []) + U.validate_cues(cues, opts)
    if not warns:
        return f"✅ 問題なし（{len(cues)}枚）"
    head = f"⚠️ {len(warns)}件の注意（{len(cues)}枚）"
    return head + "\n\n" + "\n".join(f"- {w}" for w in warns[:30]) + \
        ("\n- …" if len(warns) > 30 else "")


def srt_regenerate(state, *sub_args):
    if not state:
        return gr.update(), "⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。"
    opts = _sub_opts(*sub_args)
    settings = config.load_settings()
    settings["subtitle"] = opts
    config.save_settings(settings)
    cues = U.build_cues(state["utterances"], opts)
    return _cue_rows(cues), _warn_md(cues, opts) + "（この分割設定は次回以降の既定値になります）"


def srt_import(file, *sub_args):
    if not file:
        return gr.update(), "⚠️ ファイルを選んでください"
    path = file if isinstance(file, str) else file.name
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return gr.update(), "❌ 文字コードを判別できませんでした"
    cues = U.parse_srt(text, _people_names())
    if not cues:
        return gr.update(), "❌ 字幕を読み取れませんでした（SRT/VTT形式か確認してください）"
    return _cue_rows(cues), f"📥 {len(cues)}枚の字幕を読み込みました\n\n" + \
        _warn_md(cues, _sub_opts(*sub_args))


def srt_autofix(rows, *sub_args):
    opts = _sub_opts(*sub_args)
    cues, errors = _rows_to_cues(rows)
    if not cues:
        return gr.update(), "⚠️ 字幕がありません"
    fixed = U.finalize_cues(cues, opts, add_tail=False)
    return _cue_rows(fixed), "🩺 自動修正しました（重なり解消・最小表示時間・改行位置）\n\n" + \
        _warn_md(fixed, opts, errors)


def srt_shift(rows, seconds, *sub_args):
    cues, errors = _rows_to_cues(rows)
    if not cues:
        return gr.update(), "⚠️ 字幕がありません"
    sec = float(seconds or 0)
    for c in cues:
        c["start"] = max(0.0, c["start"] + sec)
        c["end"] = max(c["start"] + 0.1, c["end"] + sec)
    return _cue_rows(cues), f"⏩ 全体を {sec:+.2f} 秒ずらしました\n\n" + \
        _warn_md(cues, _sub_opts(*sub_args), errors)


def srt_apply_learned(rows, *sub_args):
    cues, errors = _rows_to_cues(rows)
    people_list = people.load_people()
    n = 0
    for c in cues:
        new = glossary.apply_corrections(people.apply_aliases(c["text"], people_list))
        if new != c["text"]:
            c["text"] = new
            n += 1
    return _cue_rows(cues), f"📚 人物マスタ・学習済み修正を適用しました（{n}枚を修正）\n\n" + \
        _warn_md(cues, _sub_opts(*sub_args), errors)


def srt_save(rows, state, *sub_args):
    opts = _sub_opts(*sub_args)
    cues, errors = _rows_to_cues(rows)
    if not cues:
        return None, "⚠️ 字幕がありません"
    cues.sort(key=lambda c: c["start"])
    if state and state.get("job_dir"):
        out_dir = Path(state["job_dir"])
        stem = "transcript_edited"  # 自動生成(transcript.srt)とは別名 → 再生成で上書きされない
    else:
        out_dir = config.OUTPUT_DIR
        stem = "edited_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    srt = out_dir / f"{stem}.srt"
    vtt = out_dir / f"{stem}.vtt"
    srt.write_text(U.cues_to_srt(cues, opts["speaker_prefix"]), encoding="utf-8")
    vtt.write_text(U.cues_to_vtt(cues, opts["speaker_prefix"]), encoding="utf-8")
    return [str(srt), str(vtt)], "💾 保存しました\n\n" + _warn_md(cues, opts, errors)


def load_srt_for_state(state):
    if not state:
        return gr.update()
    settings = config.load_settings()
    return _cue_rows(U.build_cues(state["utterances"], settings.get("subtitle", {})))


# =====================================================================
# 人物・声紋
# =====================================================================

def _voiceprint_rows():
    rows = diarization.list_voiceprints()
    return [[n, c] for n, c in rows] or [["（未登録）", 0]]


def _people_names():
    names = [p["display_name"] for p in people.load_people()]
    for n, _ in diarization.list_voiceprints():
        if n not in names:
            names.append(n)
    return names


def enroll_voice(name, audio_path, file_obj):
    name = (name or "").strip()
    if not name:
        return "⚠️ 名前を入力してください。", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    err = diarization.validate_name(name)
    if err:
        return f"⚠️ {err}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    src = audio_path or (file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None))
    if not src:
        return "⚠️ 録音するか、音声ファイルをアップロードしてください。", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    if not diarization.available():
        return "❌ 話者分離ライブラリが未導入です。", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    try:
        wav = audio.prepare_audio(src)
        try:
            added, sec = diarization.enroll_from_audio(wav, name)
        finally:
            Path(wav).unlink(missing_ok=True)
    except Exception as e:
        logging.error(traceback.format_exc())
        return f"❌ 登録に失敗しました: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    people_list = people.load_people()
    if name not in {p["display_name"] for p in people_list}:
        people_list.append({"display_name": name, "aliases": []})
        people.save_people(people_list)
    total = dict(diarization.list_voiceprints()).get(name, added)
    tip = ""
    if sec < 10:
        tip = "\n\n💡 10秒以上話している音声を追加登録すると認識精度が上がります。"
    msg = (f"✅ **{name}** さんの声紋を登録しました（約{sec:.0f}秒分・{added}サンプル追加／合計{total}）。"
           f"次回の文字起こしから自動で名前が付きます。{tip}")
    return (msg, _voiceprint_rows(), gr.update(choices=_people_names()), people.to_table(),
            None, None)  # 登録に使った音声を消す（次の人に同じ音声を登録してしまう事故防止）


def delete_voice(name):
    if not name:
        return "⚠️ 削除する名前を選んでください。", gr.update(), gr.update()
    ok = diarization.delete_voiceprint(name)
    msg = f"🗑️ {name} さんの声紋を削除しました。" if ok else f"⚠️ {name} さんの声紋は見つかりません。"
    return msg, _voiceprint_rows(), gr.update(choices=_people_names(), value=None)


def test_voice(audio_path, file_obj):
    src = audio_path or (file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None))
    if not src:
        return "⚠️ 音声を録音またはアップロードしてください。"
    if not diarization.list_voiceprints():
        return "⚠️ 声紋がまだ登録されていません。"
    try:
        wav = audio.prepare_audio(src)
        try:
            scores = diarization.identify(wav)
        finally:
            Path(wav).unlink(missing_ok=True)
    except Exception as e:
        return f"❌ 判定に失敗しました: {e}"
    thr = config.load_settings()["diarization"].get("match_similarity", 0.55)
    lines = ["| 人物 | 一致度 | 判定 |", "|---|---|---|"]
    for n, s in scores:
        lines.append(f"| {n} | {s:.3f} | {'✅ 本人と判定' if s >= thr else '—'} |")
    return "\n".join(lines) + f"\n\n判定しきい値: {thr}（設定タブで変更可）"


def save_people_table(rows):
    people.save_people(people.from_table(_rows(rows)))
    return "✅ 人物マスタを保存しました。", gr.update(choices=_people_names())


# =====================================================================
# 検索 / 出力 / 辞書 / 設定
# =====================================================================

def do_search(query):
    rows = search_mod.search_outputs(query)
    return rows or [["", "", "", "該当なし"]]


def refresh_jobs():
    jobs = output.list_jobs()
    files = output.list_md_files(jobs[0]) if jobs else []
    return (gr.update(choices=jobs, value=jobs[0] if jobs else None),
            gr.update(choices=files, value=files[0] if files else None))


def on_job_change(job_name):
    files = output.list_md_files(job_name) if job_name else []
    return gr.update(choices=files, value=files[0] if files else None)


def do_export(job_name, md_name, to_docx, to_pdf):
    if not job_name or not md_name:
        return "⚠️ 出力フォルダとファイルを選択してください。", None
    if not (to_docx or to_pdf):
        return "⚠️ 出力形式（Word / PDF）を選択してください。", None
    md_path = output.job_path(job_name) / md_name
    if not md_path.exists():
        return f"⚠️ ファイルが見つかりません: {md_path}", None
    generated, errors = [], []
    if to_docx:
        try:
            out = md_path.with_suffix(".docx")
            export_mod.md_to_docx(str(md_path), str(out))
            generated.append(str(out))
        except Exception as e:
            errors.append(f"Word出力失敗: {e}")
    if to_pdf:
        try:
            out = md_path.with_suffix(".pdf")
            export_mod.md_to_pdf(str(md_path), str(out))
            generated.append(str(out))
        except Exception as e:
            errors.append(f"PDF出力失敗: {e}")
    msg = "✅ 出力しました: " + ", ".join(Path(g).name for g in generated) if generated else ""
    if errors:
        msg += ("\n" if msg else "") + "\n".join(f"❌ {e}" for e in errors)
    return msg, generated or None


def save_glossary_text(text):
    glossary.save_glossary([t for t in (text or "").split("\n") if t.strip()])
    return "✅ 用語辞書を保存しました。次回の文字起こしから認識ヒントに使われます。"


def save_corrections_table(rows):
    corrections = []
    for row in _rows(rows):
        wrong, right = _cell(row, 0), _cell(row, 1)
        if wrong and right:
            try:
                count = int(float(_cell(row, 2, "1")))
            except ValueError:
                count = 1
            corrections.append({"wrong": wrong, "right": right, "count": count})
    glossary.save_corrections(corrections)
    return "✅ 学習済み修正を保存しました。"


def load_corrections_table():
    rows = [[c.get("wrong", ""), c.get("right", ""), c.get("count", 1)]
            for c in glossary.load_corrections()]
    return rows or [["", "", 1]]


def ollama_models(url: str) -> list[str]:
    try:
        resp = httpx.get(f"{(url or 'http://localhost:11434').rstrip('/')}/api/tags", timeout=3)
        names = [m["name"] for m in resp.json().get("models", [])]
        return sorted(names, key=lambda n: (not n.startswith("gemma4"), n))
    except Exception:
        return []


def save_settings_ui(provider, ollama_url, ollama_model, ollama_think, anthropic_key,
                     anthropic_model, openai_key, openai_model, whisper_model, language,
                     red_threshold, cluster_sim, match_sim, hf_token):
    settings = config.load_settings()
    settings["llm"].update({
        "provider": provider,
        "ollama_url": ollama_url.strip(),
        "ollama_model": (ollama_model or "").strip(),
        "ollama_think": bool(ollama_think),
        "anthropic_api_key": anthropic_key.strip(),
        "anthropic_model": anthropic_model.strip(),
        "openai_api_key": openai_key.strip(),
        "openai_model": openai_model.strip(),
    })
    settings["whisper"].update({"model": whisper_model, "language": language})
    settings["postprocess"]["red_threshold"] = float(red_threshold)
    settings["diarization"]["hf_token"] = (hf_token or "").strip()
    settings["diarization"].update({"max_speakers": int(cluster_sim),
                                    "match_similarity": float(match_sim)})
    config.save_settings(settings)
    warn = ""
    if provider in ("anthropic", "openai"):
        warn = "\n⚠️ クラウドモードでは文字起こしテキストが外部APIに送信されます（音声自体は送信されません）。"
    return f"✅ 設定を保存しました。{warn}"


def test_llm():
    return llm.test_connection(config.load_settings())


# =====================================================================
# UI 構築
# =====================================================================

CSS = """
.gradio-container{max-width:1500px !important}
#st-summary p{margin:.2em 0}
.st-dl button{min-width:0}
"""


def build_ui():
    settings = config.load_settings()
    device, _ = transcriber.detect_device()
    dia_available = diarization.available()
    dia_dev = diarization.device_label() if dia_available else "-"
    device_label = "🟢 GPU" if device == "cuda" else "🟡 CPU"
    sub = settings.get("subtitle", {})

    with gr.Blocks(title="つよつよ文字起こし＆議事録ツール") as demo:
        gr.Markdown("# 🎙️ つよつよ文字起こし＆議事録ツール")
        gr.Markdown(
            f"文字起こし: {device_label} ／ 話者識別: "
            f"{'✅ ' + diarization.backend_label(settings['diarization'].get('hf_token', '')) + '・' + dia_dev.upper() if dia_available else '➖ 未導入'} ／ "
            f"LLM: {llm.provider_label(settings)} ／ 登録声紋: {len(diarization.list_voiceprints())}名"
        )

        job_state = gr.State(None)

        with gr.Tabs():
            # ======================== 文字起こし ========================
            with gr.Tab("🎙️ 文字起こし"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        files_input = gr.File(
                            label="音声・動画ファイル（複数まとめてドロップ可）",
                            file_count="multiple", file_types=FILE_TYPES, type="filepath",
                        )
                        with gr.Row():
                            num_spk_dd = gr.Dropdown(
                                SPEAKER_COUNT_CHOICES, value=0, label="話者の人数",
                                info="分かっていれば指定すると精度UP",
                            )
                            model_dd = gr.Dropdown(
                                transcriber.MODEL_CHOICES, value=settings["whisper"]["model"],
                                label="認識モデル", info="auto = GPUならlarge-v3（最高精度）",
                            )
                        speed_radio = gr.Radio(
                            transcriber.SPEED_CHOICES, value=settings["whisper"].get("speed", "accurate"),
                            label="文字起こしの速さ",
                            info="4時間の録音の目安: 精度優先 約1時間／バランス 約25分／高速 約8分")
                        meeting_date_box = gr.Textbox(
                            label="会議日時（任意）", placeholder="例: 2026年9月17日 10:00",
                            info="空欄ならファイル名から推定")
                        memo_box = gr.Textbox(
                            label="会議メモ（任意・議事録AIへのヒント）", lines=2,
                            placeholder="例: 清風学園との定例。参加者は坪内、岡田、平岡…")
                        with gr.Accordion("詳細オプション", open=False):
                            lang_dd = gr.Dropdown(
                                ["ja", "auto"], value=settings["whisper"]["language"],
                                label="言語", info="ja=日本語固定（推奨）")
                            eq_cb = gr.Checkbox(
                                value=settings["audio"]["asr_eq"],
                                label="音質補正（低音のうなり・高域ノイズをカット／推奨）")
                            noise_cb = gr.Checkbox(
                                value=settings["audio"]["noise_reduction"],
                                label="背景ノイズ除去（空調・雑音を弱めに除去／推奨）")
                            fillers_cb = gr.Checkbox(
                                value=settings["postprocess"]["remove_fillers"],
                                label="フィラー除去（「えー」「えーっと」「うーん」などを消す）")
                            diarize_cb = gr.Checkbox(
                                value=dia_available, interactive=dia_available,
                                label="話者識別・声紋照合" + ("" if dia_available else "（未導入）"))
                            proofread_cb = gr.Checkbox(
                                value=settings["postprocess"]["proofread"],
                                label="AI校正（誤変換を文脈で修正・推測箇所は赤字）")
                            minutes_cb = gr.Checkbox(value=True, label="議事録を作成")
                            insights_cb = gr.Checkbox(value=True, label="要約と考察を作成")
                        start_btn = gr.Button("🚀 文字起こし開始", variant="primary", size="lg")
                        result_md = gr.Markdown(elem_id="st-summary")
                        with gr.Row():
                            history_dd = gr.Dropdown(
                                label="📂 過去の結果を開く", choices=output.list_editable_jobs(),
                                scale=4)
                            history_refresh = gr.Button("🔄", scale=1, min_width=40)
                        rediarize_btn = gr.Button("👥 話者識別だけやり直す（上の「話者の人数」を使用）",
                                                  size="sm")

                    with gr.Column(scale=7):
                        with gr.Row(elem_classes="st-dl"):
                            dl_transcript = gr.DownloadButton("📥 文字起こし.md", interactive=False,
                                                              variant="primary")
                            dl_minutes = gr.DownloadButton("📥 議事録.md", interactive=False)
                            dl_srt = gr.DownloadButton("📥 字幕.srt", interactive=False)
                        with gr.Tabs():
                            with gr.Tab("📜 文字起こし"):
                                transcript_html = gr.HTML(
                                    '<div style="padding:40px;text-align:center;opacity:.6">'
                                    "ここに「時刻・話者・発言」が表示されます</div>")
                            with gr.Tab("📋 議事録"):
                                minutes_view = gr.Markdown()
                            with gr.Tab("💡 要約と考察"):
                                insights_view = gr.Markdown()

                        with gr.Accordion("👥 話者に名前を付ける（声紋を覚えて次回から自動認識）", open=True):
                            gr.Markdown("「名前」列に正式な名前を入力 → 確定。"
                                        "登録済み人物は自動で名前が付いています。")
                            speaker_table = gr.Dataframe(
                                headers=["今の表示", "名前"], datatype=["str", "str"],
                                column_count=(2, "fixed"), interactive=True,
                            )
                            with gr.Row():
                                save_voice_cb = gr.Checkbox(value=True, label="声紋として保存する")
                                assign_btn = gr.Button("✅ 話者を確定", variant="primary")
                            regen_btn = gr.Button("🔁 議事録・要約を作り直す（話者名の反映・AI校正）")
                            assign_status = gr.Markdown()

                        with gr.Accordion("✏️ 発言を修正（直した言葉は学習して次回から自動で直ります）",
                                          open=False):
                            gr.Markdown("話者・内容のセルを直接編集して保存。内容を空にするとその発言を削除します。")
                            edit_table = gr.Dataframe(
                                headers=["No", "時刻", "話者", "内容"],
                                datatype=["number", "str", "str", "str"],
                                column_count=(4, "fixed"), interactive=True, wrap=True,
                                max_height=500,
                            )
                            save_edit_btn = gr.Button("💾 修正を保存して学習", variant="primary")
                            edit_status = gr.Markdown()

            # ======================== SRT編集 ========================
            with gr.Tab("🎬 字幕(SRT)編集"):
                gr.Markdown("単語ごとのタイミングから、読みやすい長さ・区切りで字幕を自動生成します。"
                            "時刻は `HH:MM:SS,mmm` 形式で直接編集できます。")
                with gr.Accordion("字幕の分割ルール", open=True):
                    with gr.Row():
                        s_chars = gr.Slider(10, 42, value=sub.get("max_chars_line", 20), step=1,
                                            label="1行の最大文字数")
                        s_lines = gr.Slider(1, 3, value=sub.get("max_lines", 2), step=1,
                                            label="最大行数")
                        s_maxdur = gr.Slider(2, 12, value=sub.get("max_duration", 6.0), step=0.5,
                                             label="1枚の最大秒数")
                    with gr.Row():
                        s_mindur = gr.Slider(0.3, 3, value=sub.get("min_duration", 1.0), step=0.1,
                                             label="1枚の最小秒数")
                        s_pause = gr.Slider(0.2, 2, value=sub.get("pause_split", 0.6), step=0.1,
                                            label="この秒数の無音で切り替え")
                        with gr.Column():
                            s_spk = gr.Checkbox(value=sub.get("speaker_prefix", False),
                                                label="先頭に（話者名）を付ける")
                            s_period = gr.Checkbox(value=sub.get("drop_period", True),
                                                   label="句点「。」を消す（字幕の慣習）")
                    sub_inputs = [s_chars, s_lines, s_maxdur, s_mindur, s_pause, s_spk, s_period]
                with gr.Row():
                    srt_regen_btn = gr.Button("🔄 このルールで字幕を作り直す", variant="primary")
                    srt_import_file = gr.File(label="既存のSRT/VTTを読み込んで編集",
                                              file_types=[".srt", ".vtt"], type="filepath",
                                              height=90)
                srt_editor = gr.Dataframe(
                    headers=["番号", "開始", "終了", "話者", "テキスト"],
                    datatype=["number", "str", "str", "str", "str"],
                    column_count=(5, "fixed"), interactive=True, wrap=True, max_height=560,
                )
                with gr.Row():
                    shift_num = gr.Number(value=0.0, label="ずらす秒数（＋で遅く）", step=0.1,
                                          scale=1)
                    shift_btn = gr.Button("⏩ 全体の時刻をずらす", scale=1)
                    fix_btn = gr.Button("🩺 チェック＆自動修正", scale=1)
                    learned_btn = gr.Button("📚 人名・学習済み修正を適用", scale=1)
                srt_status = gr.Markdown()
                srt_save_btn = gr.Button("💾 SRT / VTT を保存", variant="primary")
                srt_files = gr.Files(label="保存された字幕ファイル")

            # ======================== 人物・声紋 ========================
            with gr.Tab("👤 人物・声紋登録"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("## 🎤 声紋を登録")
                        gr.Markdown(
                            "その人**だけ**が話している音声を録音またはアップロードしてください。"
                            "**10〜30秒**あると精度が上がります。同じ人に何回か登録すると、さらに精度が上がります。")
                        vp_name = gr.Dropdown(choices=_people_names(), allow_custom_value=True,
                                              label="名前（正式表記）", info="新しい名前も入力できます")
                        vp_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                            label="録音 または 音声ファイル")
                        vp_file = gr.File(label="動画・その他の形式はこちら（mp4, m4a, wma など）", elem_id="vp-file",
                                          file_types=FILE_TYPES, type="filepath", height=90)
                        with gr.Row():
                            vp_enroll_btn = gr.Button("🎤 声紋を登録", variant="primary")
                            vp_test_btn = gr.Button("🔍 誰の声か判定してみる")
                        vp_status = gr.Markdown()

                    with gr.Column(scale=1):
                        gr.Markdown("## 📇 登録済みの声紋")
                        vp_table = gr.Dataframe(headers=["名前", "サンプル数"], value=_voiceprint_rows(),
                                                interactive=False)
                        with gr.Row():
                            vp_del_name = gr.Dropdown(choices=_people_names(), label="削除する声紋",
                                                      scale=3)
                            vp_del_btn = gr.Button("🗑️ 削除", variant="stop", scale=1)
                        vp_del_status = gr.Markdown()

                gr.Markdown("## 📝 人物マスタ（呼び名 → 正式表記）")
                gr.Markdown("例: 正式表記「坪内」呼び名「ボンギン, ボンギンさん」→ 文字起こし・議事録で自動的に正式表記へ統一。"
                            "音声認識のヒントにも使われます。")
                people_table = gr.Dataframe(
                    headers=["正式表記", "呼び名（カンマ区切り）", "役職・所属（任意）"],
                    column_count=(3, "fixed"), interactive=True, value=people.to_table(),
                )
                people_save_btn = gr.Button("保存", variant="primary")
                people_status = gr.Markdown()

            # ======================== 検索 ========================
            with gr.Tab("🔍 検索"):
                gr.Markdown("過去の全議事録・文字起こし・要約を横断検索します（スペース区切りでAND検索）。")
                with gr.Row():
                    search_box = gr.Textbox(label="キーワード", placeholder="例: 見積 坪内", scale=4)
                    search_btn = gr.Button("検索", variant="primary", scale=1)
                search_results = gr.Dataframe(headers=["フォルダ", "ファイル", "行", "内容"],
                                              interactive=False, wrap=True)

            # ======================== Word/PDF ========================
            with gr.Tab("📄 Word/PDF出力"):
                gr.Markdown("生成済みのMarkdownを提出用の Word / PDF に変換します（赤字も維持されます）。")
                with gr.Row():
                    job_dd = gr.Dropdown(label="出力フォルダ", choices=[], scale=3)
                    md_dd = gr.Dropdown(label="ファイル", choices=[], scale=2)
                    refresh_btn = gr.Button("🔄 更新", scale=1)
                with gr.Row():
                    docx_cb = gr.Checkbox(value=True, label="Word (.docx)")
                    pdf_cb = gr.Checkbox(value=True, label="PDF")
                export_btn = gr.Button("変換", variant="primary")
                export_status = gr.Markdown()
                export_files = gr.Files(label="生成ファイル")

            # ======================== 用語辞書 ========================
            with gr.Tab("📚 用語辞書"):
                gr.Markdown("社名・製品名・専門用語を1行1語で登録すると、音声認識とAI校正の精度が上がります。")
                glossary_box = gr.Textbox(lines=10, label="用語（1行1語）",
                                          value="\n".join(glossary.load_glossary()))
                glossary_save_btn = gr.Button("保存", variant="primary")
                glossary_status = gr.Markdown()
                gr.Markdown("### 学習済み修正（発言の修正から自動で貯まります）")
                gr.Markdown(f"出現回数 {glossary.PROMOTE_COUNT} 回以上で次回から自動適用されます。")
                corrections_table = gr.Dataframe(headers=["誤", "正", "回数"],
                                                 column_count=(3, "fixed"), interactive=True,
                                                 value=load_corrections_table())
                corrections_save_btn = gr.Button("学習済み修正を保存")
                corrections_status = gr.Markdown()

            # ======================== 設定 ========================
            with gr.Tab("⚙️ 設定"):
                gr.Markdown("## LLM（校正・議事録・要約の頭脳）")
                provider_radio = gr.Radio(
                    choices=[("Ollama（ローカル・無料・完全オフライン）", "ollama"),
                             ("Claude API", "anthropic"), ("OpenAI API", "openai"),
                             ("使わない（文字起こしのみ）", "none")],
                    value=settings["llm"]["provider"], label="プロバイダ",
                )
                with gr.Group():
                    ollama_url_box = gr.Textbox(value=settings["llm"]["ollama_url"], label="Ollama URL")
                    ollama_list = ollama_models(settings["llm"]["ollama_url"])
                    ollama_model_box = gr.Dropdown(
                        choices=ollama_list or [settings["llm"]["ollama_model"]],
                        value=settings["llm"]["ollama_model"], allow_custom_value=True,
                        label="Ollamaモデル",
                        info="推奨: gemma4:31b（最高品質）。無い場合は `ollama pull gemma4:31b`",
                    )
                    ollama_think_cb = gr.Checkbox(
                        value=settings["llm"].get("ollama_think", False),
                        label="思考モード（議事録の質が上がるが時間は数倍）")
                with gr.Group():
                    anthropic_key_box = gr.Textbox(
                        value=settings["llm"]["anthropic_api_key"], label="Claude APIキー",
                        type="password", info="空欄なら環境変数 ANTHROPIC_API_KEY を使用")
                    anthropic_model_box = gr.Textbox(value=settings["llm"]["anthropic_model"],
                                                     label="Claudeモデル")
                with gr.Group():
                    openai_key_box = gr.Textbox(
                        value=settings["llm"]["openai_api_key"], label="OpenAI APIキー",
                        type="password", info="空欄なら環境変数 OPENAI_API_KEY を使用")
                    openai_model_box = gr.Textbox(value=settings["llm"]["openai_model"],
                                                  label="OpenAIモデル")

                gr.Markdown("## 文字起こし")
                whisper_model_dd = gr.Dropdown(transcriber.MODEL_CHOICES,
                                               value=settings["whisper"]["model"], label="既定モデル")
                lang_default_dd = gr.Dropdown(["ja", "auto"], value=settings["whisper"]["language"],
                                              label="既定の言語")
                red_threshold_slider = gr.Slider(
                    0.1, 0.9, value=settings["postprocess"]["red_threshold"], step=0.05,
                    label="赤字にする確信度のしきい値（低いほど赤字が減る）")

                gr.Markdown("## 話者識別")
                cluster_sim_slider = gr.Slider(
                    2, 20, value=settings["diarization"].get("max_speakers", 12), step=1,
                    label="自動判定する最大人数（人数は声の特徴から自動推定。分かっている時は文字起こし画面で人数を指定）")
                match_sim_slider = gr.Slider(
                    0.3, 0.9, value=settings["diarization"].get("match_similarity", 0.55),
                    step=0.01, label="声紋一致のしきい値（別人に名前が付くなら上げる／付かないなら下げる）")
                hf_token_box = gr.Textbox(
                    value=settings["diarization"].get("hf_token", ""), type="password",
                    label="Hugging Face トークン（pyannote話者分離を使う）",
                    info="pyannote/speaker-diarization-community-1 と pyannote/segmentation-3.0 の利用規約に同意したアカウントのトークン。空欄なら自前の声紋クラスタリング")

                with gr.Row():
                    settings_save_btn = gr.Button("設定を保存", variant="primary")
                    test_btn = gr.Button("LLM接続テスト")
                settings_status = gr.Markdown()

        # ======================== イベント ========================
        view_outputs = [transcript_html, minutes_view, insights_view,
                        dl_transcript, dl_minutes, dl_srt, speaker_table, edit_table]

        start_btn.click(
            fn=run_batch,
            inputs=[files_input, model_dd, speed_radio, lang_dd, num_spk_dd, noise_cb, eq_cb, fillers_cb, diarize_cb,
                    proofread_cb, minutes_cb, insights_cb, meeting_date_box, memo_box],
            outputs=[result_md, job_state] + view_outputs + [history_dd],
        ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor)

        history_dd.input(fn=open_job, inputs=history_dd,
                         outputs=[result_md, job_state] + view_outputs
                         ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor)
        history_refresh.click(fn=refresh_history, outputs=history_dd)
        rediarize_btn.click(fn=rediarize_job, inputs=[job_state, num_spk_dd],
                            outputs=[result_md, job_state] + view_outputs
                            ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor)

        assign_btn.click(fn=assign_speakers, inputs=[speaker_table, save_voice_cb, job_state],
                         outputs=[assign_status, job_state] + view_outputs
                         ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor
                                ).then(fn=_voiceprint_rows, outputs=vp_table)
        regen_btn.click(fn=regenerate_llm, inputs=[job_state, proofread_cb],
                        outputs=[assign_status, job_state] + view_outputs
                        ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor)
        save_edit_btn.click(fn=save_utterance_edits, inputs=[edit_table, job_state],
                            outputs=[edit_status, job_state] + view_outputs
                            ).then(fn=load_srt_for_state, inputs=job_state, outputs=srt_editor)

        srt_regen_btn.click(fn=srt_regenerate, inputs=[job_state] + sub_inputs,
                            outputs=[srt_editor, srt_status])
        srt_import_file.upload(fn=srt_import, inputs=[srt_import_file] + sub_inputs,
                               outputs=[srt_editor, srt_status])
        fix_btn.click(fn=srt_autofix, inputs=[srt_editor] + sub_inputs,
                      outputs=[srt_editor, srt_status])
        shift_btn.click(fn=srt_shift, inputs=[srt_editor, shift_num] + sub_inputs,
                        outputs=[srt_editor, srt_status])
        learned_btn.click(fn=srt_apply_learned, inputs=[srt_editor] + sub_inputs,
                          outputs=[srt_editor, srt_status])
        srt_save_btn.click(fn=srt_save, inputs=[srt_editor, job_state] + sub_inputs,
                           outputs=[srt_files, srt_status])

        vp_enroll_btn.click(fn=enroll_voice, inputs=[vp_name, vp_audio, vp_file],
                            outputs=[vp_status, vp_table, vp_del_name, people_table, vp_audio, vp_file])
        vp_test_btn.click(fn=test_voice, inputs=[vp_audio, vp_file], outputs=vp_status)
        vp_del_btn.click(fn=delete_voice, inputs=vp_del_name,
                         outputs=[vp_del_status, vp_table, vp_del_name])
        people_save_btn.click(fn=save_people_table, inputs=people_table,
                              outputs=[people_status, vp_name])

        search_btn.click(fn=do_search, inputs=search_box, outputs=search_results)
        search_box.submit(fn=do_search, inputs=search_box, outputs=search_results)
        refresh_btn.click(fn=refresh_jobs, outputs=[job_dd, md_dd])
        job_dd.change(fn=on_job_change, inputs=job_dd, outputs=md_dd)
        export_btn.click(fn=do_export, inputs=[job_dd, md_dd, docx_cb, pdf_cb],
                         outputs=[export_status, export_files])
        glossary_save_btn.click(fn=save_glossary_text, inputs=glossary_box, outputs=glossary_status)
        corrections_save_btn.click(fn=save_corrections_table, inputs=corrections_table,
                                   outputs=corrections_status)
        settings_save_btn.click(
            fn=save_settings_ui,
            inputs=[provider_radio, ollama_url_box, ollama_model_box, ollama_think_cb,
                    anthropic_key_box, anthropic_model_box, openai_key_box, openai_model_box,
                    whisper_model_dd, lang_default_dd, red_threshold_slider,
                    cluster_sim_slider, match_sim_slider, hf_token_box],
            outputs=settings_status,
        )
        test_btn.click(fn=test_llm, outputs=settings_status)

        demo.load(fn=refresh_jobs, outputs=[job_dd, md_dd])
        demo.load(fn=refresh_history, outputs=history_dd)

    return demo


demo = build_ui()

if __name__ == "__main__":
    print("==========================================================")
    print("Ready. Browser will open automatically.")
    print("==========================================================")
    demo.launch(server_name="127.0.0.1", share=False,
                inbrowser=os.environ.get("ST_NO_BROWSER") != "1",
                allowed_paths=[str(config.OUTPUT_DIR)], css=CSS)
