import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 完全ローカルで使うツールなので、gradio / Hugging Face への利用状況の送信を止める
# （起動のたびに外部へ通信していて、ネットワークが遅い時は起動も遅くなっていた）
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# OSの証明書ストアを使う（セキュリティソフトのSSL検査環境でもモデルDLが通るように）
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import datetime
import logging
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import httpx

from core import audio, checkpoints, config, diarization, document_status, glossary, llm, minutes, noise_profile, output, people, transcriber
from core import export as export_mod
from core import search as search_mod
from core import utterances as U
from core import run_control, worker
from core.pipeline import _finalize_text, _plain_md, _save_job, _voice_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

config.ensure_dirs()

FILE_TYPES = [f".{e.lstrip('.')}" for e in audio.SUPPORTED_EXTS]
SPEAKER_COUNT_CHOICES = [("自動", 0)] + [(f"{n}人", n) for n in range(1, 11)]
VIEW_COUNT = 9


def preview_noise_sample(source, start, end):
    """見本にする範囲だけを原音から切り出して確認する。"""
    import subprocess
    import tempfile
    if not source:
        raise gr.Error("元音声を選んでください")
    try:
        a, b = noise_profile.seconds(start), noise_profile.seconds(end)
        if not 0.5 <= b - a <= 30:
            raise ValueError("0.5〜30秒の範囲を選んでください")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run([audio.find_ffmpeg(), "-y", "-ss", str(a), "-i", source, "-t", str(b - a),
                        "-vn", "-ac", "1", "-ar", "16000", tmp.name],
                       capture_output=True, check=True)
        if audio.get_duration(tmp.name) < 0.5:
            raise ValueError("選択した区間が録音の外にあります")
        return tmp.name
    except (ValueError, subprocess.SubprocessError) as e:
        raise gr.Error(str(e)) from e


def apply_noise_sample(source, start, end, strength, progress=gr.Progress()):
    import uuid
    if not source:
        raise gr.Error("元音声を選んでください")
    folder = config.OUTPUT_DIR / (datetime.datetime.now().strftime("noise_profile_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    try:
        result = noise_profile.create_trial(source, start, end, folder, strength / 100,
                                            progress=lambda f: progress(f, desc="ノイズの見本を使って全体を補正中"))
    except (ValueError, audio.AudioError) as e:
        raise gr.Error(str(e)) from e
    return (f"保存しました: `{result['folder']}`。削った音に説明者の言葉があれば、強さを下げてください。",
            str(folder / "reduced.wav"), str(folder / "removed.wav"), str(folder / "reduced.wav"))


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




def _file_url(path: str | None) -> str:
    from urllib.parse import quote
    return f"/gradio_api/file={quote(Path(path).as_posix(), safe='/:')}" if path else ""


# =====================================================================
# ジョブ（1ファイル分の結果）の保存・再描画
# =====================================================================



def _load_job(job_name: str) -> dict | None:
    job_dir = output.job_path(job_name)
    data = output.load_json(job_dir, "utterances.json")
    if not data:
        partial = output.load_json(job_dir, checkpoints.FILENAME)
        return checkpoints.as_job(job_dir, partial) if partial else None
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
    if not path.exists():
        return ""
    warning = document_status.warning(job, name)
    return (warning + "\n\n---\n\n" if warning else "") + path.read_text(encoding="utf-8")






def _download_transcript(job: dict):
    """タグなしの文字起こし.md を渡す（以前のジョブで未作成なら今作る）"""
    job_dir = Path(job["job_dir"])
    if job.get("incomplete"):
        # ワーカーが更新する途中本文は上書きせず、表示中の時点の内容を渡す。
        output.write_text(job_dir, checkpoints.SNAPSHOT_TRANSCRIPT, checkpoints.markdown(job))
        return _download(job, checkpoints.SNAPSHOT_TRANSCRIPT)
    if not (job_dir / "文字起こし.md").exists() and job["utterances"]:
        output.write_text(job_dir, "文字起こし.md", _plain_md(job))
    return _download(job, "文字起こし.md")


def _download(job: dict, name: str):
    path = Path(job["job_dir"]) / name
    state = document_status.status(job, name)
    prefix = {"stale": "⚠️ 更新前: ", "unknown": "⚠️ 更新状況不明: "}.get(state, "📥 ")
    available = path.exists() and (not job.get("incomplete") or name == checkpoints.SNAPSHOT_TRANSCRIPT)
    return gr.update(label=prefix + name, value=str(path) if available else None, interactive=available)


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


def _job_views(job: dict):
    """結果表示コンポーネント一式の更新値"""
    return (
        U.render_html(job["utterances"], _file_url(job.get("audio_path"))),
        _read_md(job, "議事録.md") or "_議事録はまだありません（文字起こしを確認・修正してから「③ 📋 議事録を作る」を押してください）_",
        _read_md(job, "要約と考察.md") or "_要約と考察はまだありません（「💡 要約と考察を作る」で作成）_",
        _download_transcript(job),
        _download(job, "議事録.md"),
        _download(job, "要約と考察.md"),
        gr.update(value=_speaker_rows(job), interactive=not job.get("incomplete", False)),
        gr.update(value=_edit_rows(job), interactive=not job.get("incomplete", False)),
        "\n\n".join(filter(None, [checkpoints.notice(job)] +
                            [document_status.warning(job, name) for name in document_status.DOCUMENTS])),
    )


# =====================================================================
# メイン処理パイプライン
# =====================================================================







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


def _recognition_options(model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on,
                         meeting_date, memo, decoding, silence_guard):
    return {
        "model": model_name, "speed": speed, "language": language, "num_speakers": num_speakers,
        "decoding": decoding, "silence_guard": silence_guard,
        "noise": noise, "eq": eq_on, "fillers": fillers_on, "diarize": diarize_on, "use_llm": False,
        "meeting_date": (meeting_date or "").strip(), "memo": (memo or "").strip(),
    }


def pause_transcription(request: gr.Request):
    if run_control.request_pause(request.session_hash):
        return ("⏳ 一時停止を受け付けました。処理中の区間を終え、最新の途中結果を保存して止めます。"
                "音声準備・モデル読み込み・話者識別中は、その処理が戻るまでお待ちください。")
    return "この画面で実行中の文字起こしはありません。実行待ちの場合は処理開始後に押してください。"


def _run_asr(target, *args, settings, progress, request):
    with run_control.session(request.session_hash if request else None) as pause_file:
        # 前回のLLMがメモリに残っているとWhisperを読み込めないため、先に解放する。
        llm.unload_ollama(settings)
        return worker.run(target, *args, progress=progress, pause_file=pause_file)


def _paused_message(job):
    return (f"⏸ **{job['source']}** の途中結果と再開用音声を保存して一時停止しました。"
            "「▶ 途中保存から再開」で続行できます。アプリを閉じた場合も、履歴から開いて再開できます。")


def resume_transcription(state, files, model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on,
                         meeting_date, memo, decoding="stable", silence_guard=True, progress=gr.Progress(),
                         request: gr.Request = None):
    if not state or not state.get("incomplete"):
        return ("⚠️ 履歴から「途中保存」の結果を開いてください。", gr.update()) + \
            tuple(gr.update() for _ in range(VIEW_COUNT)) + (refresh_history(),)
    settings = config.load_settings()
    opts = _recognition_options(model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on,
                                meeting_date, memo, decoding, silence_guard)
    paths = [f if isinstance(f, str) else f.name for f in files or []]
    job_name = Path(state["job_dir"]).name
    try:
        progress(0, desc="途中保存を確認中...")
        job = _run_asr("core.pipeline:resume_job", state["job_dir"], paths, opts, settings,
                       settings=settings, progress=lambda f, d: progress(f, desc=d), request=request)
        if job.get("incomplete"):
            message = _paused_message(job)
        else:
            _write_meta(job, job["run_options"], settings)
            message = (f"✅ **{job['source']}** の続きを処理し、同じ保存先に完成版を保存しました。"
                       "次は ② 確認・修正で本文と話者名を確認してください。")
    except Exception as e:
        logging.error(traceback.format_exc())
        message = f"⚠️ 再開処理を完了できませんでした: {e}。保存済みの結果は保持しています。"
        job = _load_job(job_name) or state
    return (message, job) + _job_views(job) + (
        gr.update(choices=output.list_editable_jobs(), value=job_name),)


def run_batch(files, model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on,
              meeting_date, memo, decoding="stable", silence_guard=True, progress=gr.Progress(),
              request: gr.Request = None):
    """① 音声 → 文字起こし（話者識別まで）。AIの校正・議事録は②③で別に行う"""
    if not files:
        return ("⚠️ ファイルを選択してください。", gr.update()) + \
            tuple(gr.update() for _ in range(VIEW_COUNT)) + (gr.update(),)
    settings = config.load_settings()
    opts = _recognition_options(model_name, speed, language, num_speakers, noise, eq_on, fillers_on, diarize_on,
                                meeting_date, memo, decoding, silence_guard)
    paths = [f if isinstance(f, str) else f.name for f in files]
    n = len(paths)

    # Whisper・話者分離は別プロセスで動かす（終わるとVRAMが全部空き、③のLLMが遅くならない）
    jobs, failures = [], []
    paused_job, paused_batch, remaining = None, False, []
    progress(0.0, desc="文字起こしの準備中（モデルを読み込んでいます）...")
    try:
        results = _run_asr("core.pipeline:transcribe_files", paths, opts, settings,
                           settings=settings, progress=lambda f, d: progress(f, desc=d), request=request)
    except Exception as e:
        logging.error(traceback.format_exc())
        results = [{"error": str(e)}]
    for r in results:
        if "job" in r:
            jobs.append(r["job"])
        elif "paused" in r:
            paused_batch, paused_job, remaining = True, r["paused"], r["remaining"]
        else:
            failures.append(f"❌ {r['error']}")

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
        refilled = job.get("stats", {}).get("asr_refilled_sec", 0)
        missing = job.get("stats", {}).get("asr_missing_sec", 0)
        if refilled:
            lines.append(f"　🩹 文字起こしが抜けていた所（計{refilled:.0f}秒）を自動でやり直して補いました")
        if missing >= 10:
            lines.append(f"　⚠️ 声があるのに文字にならなかった所が計{missing:.0f}秒あります"
                         "（雑音・音楽・聞き取れない声の可能性。ログに時刻を出しています）")
        for err in job["errors"]:
            lines.append(f"　⚠️ {err}")
    lines.extend(failures)
    if paused_batch:
        lines.append(_paused_message(paused_job) if paused_job else "⏸ 次のファイルを始める前に一括処理を停止しました。")
        if remaining:
            lines.append("未着手のファイルは、あとで選び直して開始してください: " + "、".join(remaining))
    if failures:
        lines.append("認識済みの途中保存がある場合は「📂 過去の結果を開く」から確認できます。"
                     "アプリを再起動しても残ります。")
    if jobs and not paused_batch:
        lines.append("**次は ② 確認・修正**: 右の文字起こしを見て、話者名と内容を直してください"
                     "（「✍️ AI校正」で誤変換をAIに直させることもできます）。"
                     "直し終わったら「③ 📋 議事録を作る」。")
    summary = "\n\n".join(lines)

    if not jobs and not paused_job:
        return (summary, gr.update()) + tuple(gr.update() for _ in range(VIEW_COUNT)) + (refresh_history(),)
    last = paused_job or jobs[-1]
    views = _job_views(last)
    state = {k: v for k, v in last.items() if k not in ("errors", "matched", "stats")}
    jobs_list = output.list_editable_jobs()
    return (summary, state) + views + (
        gr.update(choices=jobs_list, value=Path(last["job_dir"]).name),)


def open_job(job_name):
    job = _load_job(job_name) if job_name else None
    if not job:
        return ("⚠️ 開けませんでした（古い形式の出力には対応していません）", gr.update()) + \
            tuple(gr.update() for _ in range(VIEW_COUNT))
    return (f"📂 `{job_name}` を開きました", job) + _job_views(job)


def rediarize_job(state, num_speakers, progress=gr.Progress()):
    """保存済みの文字起こし（単語タイミング）と再生用音声を使って、話者識別だけやり直す"""
    none = tuple(gr.update() for _ in range(VIEW_COUNT))
    if not state:
        return ("⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。", gr.update()) + none
    if state.get("incomplete"):
        return (checkpoints.READ_ONLY_NOTICE, state) + none
    if not diarization.available():
        return ("❌ 話者分離ライブラリが未導入です。", gr.update()) + none
    if not state.get("audio_path") or not Path(state["audio_path"]).exists():
        return ("⚠️ この結果には再生用音声(audio.m4a)が無いため、やり直せません。", gr.update()) + none
    job = state
    settings = config.load_settings()
    llm.unload_ollama(settings)  # GPUを空ける
    progress(0.02, desc="準備中...")
    segments = [{"start": u["start"], "end": u["end"], "text": u["text"],
                 "words": u.get("words") or []} for u in job["utterances"]]
    try:
        word_speakers, cluster_embeddings, matched = worker.run(
            "core.pipeline:rediarize", job["audio_path"], segments, int(num_speakers or 0), settings,
            progress=lambda f, d: progress(f, desc=d))
    except Exception as e:
        logging.error(traceback.format_exc())
        return (f"❌ 話者識別に失敗しました: {e}", gr.update()) + none
    if not word_speakers:
        return ("❌ 話者識別に失敗しました（ログを確認してください）", gr.update()) + none
    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
    people_list = people.load_people()
    utts = U.build_utterances(segments, word_speakers, threshold)
    _finalize_text(utts, people_list)
    job["utterances"] = utts
    job["cluster_embeddings"] = cluster_embeddings
    _save_job(job)
    stats = U.speaker_stats(utts)
    msg = (f"✅ 話者識別をやり直しました: {len(stats)}名 ／ 発言 {len(utts)} 件"
           + (f"\n\n🔊 声紋で自動認識: " + "、".join(f"{k}（{v:.2f}）" for k, v in matched.items())
              if matched else "")
           + "\n\n※ 文字は音声認識の結果から作り直しました（AI校正・手修正は反映されていません）。"
             "議事録に反映するには「③ 📋 議事録を作る」を押してください。")
    return (msg, job) + _job_views(job)


def reflow_job(state):
    """保存済みの結果を、今の句読点・話者まとめ規則で組み直す（GPU不要ですぐ終わる）"""
    none = tuple(gr.update() for _ in range(VIEW_COUNT))
    if not state:
        return ("⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。", gr.update()) + none
    if state.get("incomplete"):
        return (checkpoints.READ_ONLY_NOTICE, state) + none
    job = state
    people_list = people.load_people()
    edited = U.count_edited(job["utterances"], lambda t: U.fix_ichi(
        glossary.apply_corrections(people.apply_aliases(t, people_list))))
    if edited and not job.get("_reflow_confirm"):
        job["_reflow_confirm"] = True
        return (f"⚠️ この結果は {edited} 件の発言がAI校正または手修正されています。"
                "整え直すと文字は音声認識の結果に戻り、それらの修正は消えます。"
                "\n\nそれでもよければ、もう一度ボタンを押してください"
                "（おすすめ: 先に整え直してから「✍️ AI校正」をかける）。",
                job) + none
    job.pop("_reflow_confirm", None)
    settings = config.load_settings()
    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)
    before = len(U.sentence_rows(job["utterances"]))
    utts = U.reflow(job["utterances"], threshold)
    _finalize_text(utts, people_list)
    job["utterances"] = utts
    _save_job(job)
    msg = (f"✅ 文の区切り・句読点を整え直しました（{before} 行 → {len(U.sentence_rows(utts))} 行）"
           "\n\n※ 文字は音声認識の結果から作り直しました（AI校正・手修正は反映されていません）。"
           "議事録に反映するには「③ 📋 議事録を作る」を押してください。")
    return (msg, job) + _job_views(job)


def refresh_history():
    return gr.update(choices=output.list_editable_jobs())


def _learn_voices(job: dict) -> str:
    """声紋の学習（core.pipeline.learn_voices）を別プロセスで行う。失敗しても議事録作成は続ける"""
    if not diarization.available():
        return ""
    slim = {k: job.get(k) for k in ("job_dir", "audio_path", "utterances")}
    try:
        return worker.run("core.pipeline:learn_voices", slim)
    except Exception as e:
        logging.warning(f"声紋の学習に失敗: {e}")
        return ""


def _llm_ready(state):
    """②③の共通チェック。問題があればメッセージを返す"""
    if not state:
        return "⚠️ 先に文字起こしを実行するか、過去の結果を開いてください。"
    if state.get("incomplete"):
        return checkpoints.READ_ONLY_NOTICE
    # Whisper・声紋モデルをVRAMから降ろして、LLM(gemma4)にGPUを譲る
    transcriber.unload()
    diarization.unload()
    return ""


def _eta(started: float, frac: float) -> str:
    if frac <= 0.02 or frac >= 1:
        return ""
    rest = (time.time() - started) * (1 - frac) / frac
    if rest >= 3600:
        return f"（残り約{int(rest // 3600)}時間{int(rest % 3600 // 60)}分）"
    return f"（残り約{max(1, int(rest // 60))}分）"


def proofread_job(state, redo, progress=gr.Progress()):
    """② AI校正。校正済み・手修正済みの発言は飛ばすので、途中で止まっても続きから再開できる"""
    none = tuple(gr.update() for _ in range(VIEW_COUNT))
    err = _llm_ready(state)
    if err:
        return (err, gr.update()) + none
    job = state
    settings = config.load_settings()
    if redo:
        for u in job["utterances"]:
            u.pop("proofread", None)

    people_list = people.load_people()

    def norm(t):
        return U.fix_ichi(glossary.apply_corrections(people.apply_aliases(t, people_list)))

    def skip(u):
        # 校正済み・手修正済み（本文が音声認識の結果と違う＝以前の版で手で直した分も含む）は触らない
        return bool(u.get("proofread") or u.get("human") or U.count_edited([u], norm))

    todo = sum(1 for u in job["utterances"] if not skip(u))
    if not todo:
        return ("✅ すべての発言を校正済みです（最初からやり直す場合は「校正済みもやり直す」にチェック）",
                job) + _job_views(job)
    started = time.time()
    progress(0, desc=f"AI校正の準備中...（対象 {todo} 発言）")
    try:
        done, changed = minutes.proofread_utterances(
            settings, job["utterances"], skip=skip, on_chunk=lambda: _save_job(job),
            progress_cb=lambda f: progress(f, desc=f"AI校正中... {int(f * 100)}%{_eta(started, f)}"))
        msg = f"✅ AI校正が終わりました（{done}発言を確認、{changed}発言を修正）"
    except llm.LLMError as e:
        msg = f"⚠️ AI校正を途中で止めました: {e}\n\nここまでの結果は保存済みです。もう一度押すと続きから再開します。"
    _save_job(job)
    notice = llm.pop_fallback_notice()
    if notice:
        msg += f"\n\n⚠️ {notice}"
    msg += "\n\n直し終わったら「③ 📋 議事録を作る」を押してください。"
    return (msg, job) + _job_views(job)


def _make_doc(state, kind: str, progress):
    none = tuple(gr.update() for _ in range(VIEW_COUNT))
    if state and state.get("incomplete"):
        return (checkpoints.READ_ONLY_NOTICE, state) + none
    learned = ""
    if state and kind == "minutes" and config.load_settings()["diarization"].get("auto_learn", True):
        # ②で直した後の話者で学習する（LLMにGPUを譲る前に済ませる）
        progress(0.02, desc="直した話者で声紋を学習中...")
        learned = _learn_voices(state)
    err = _llm_ready(state)
    if err:
        return (err, gr.update()) + none
    job = state
    settings = config.load_settings()
    source = document_status.inputs(job)
    label, filename, fn = (("議事録", "議事録.md", minutes.generate_minutes) if kind == "minutes"
                           else ("要約と考察", "要約と考察.md", minutes.generate_insights))
    progress(0.1, desc=f"{label}を作成中...（長い会議は区間ごとのメモを作ってからまとめます）")
    try:
        md = fn(settings, source["transcript"], source["source"], source["meeting_date"])
        document_status.save(Path(job["job_dir"]), filename, md, source)
        msg = f"✅ {label}を作りました（上の「{label}」タブで確認・📥 からダウンロード）"
    except (llm.LLMError, OSError) as e:
        msg = f"⚠️ {label}を作れませんでした: {e}"
    notice = llm.pop_fallback_notice()
    if notice:
        msg += f"\n\n⚠️ {notice}"
    if learned:
        msg += f"\n\n{learned}"
    return (msg, job) + _job_views(job)


def make_minutes(state, progress=gr.Progress()):
    """③ 今の文字起こし（②で直した後）から、テンプレートに沿った議事録を作る"""
    return _make_doc(state, "minutes", progress)


def make_insights(state, progress=gr.Progress()):
    return _make_doc(state, "insights", progress)


# =====================================================================
# 話者の確定・発言の修正（修正は学習される）
# =====================================================================

def assign_speakers(rows, save_voice, state):
    if not state:
        return ("⚠️ 先に文字起こしを実行してください。", state) + tuple(gr.update() for _ in range(VIEW_COUNT))
    if state.get("incomplete"):
        return (checkpoints.READ_ONLY_NOTICE, state) + tuple(gr.update() for _ in range(VIEW_COUNT))
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
                return (f"⚠️ {new_name}: {err}", state) + tuple(gr.update() for _ in range(VIEW_COUNT))
            mapping[label] = new_name
    if not mapping:
        return ("⚠️ 「名前」列に入力してください。", state) + tuple(gr.update() for _ in range(VIEW_COUNT))

    # 一括で置き換える（入れ替え A⇔B や連鎖 A→B→C でも話者が混ざらない）
    for u in job["utterances"]:
        u["speaker"] = mapping.get(u.get("speaker"), u.get("speaker"))
    grouped = {}
    for label, vec in embeddings.items():
        grouped.setdefault(mapping.get(label, label), []).append(vec)
    if save_voice:
        learned = _learn_voices(job)
    else:  # ①の直後に自動で学習した分が付け間違いだった場合に残さない
        learned = ""
        if diarization.available():
            diarization.forget_job(_voice_tag(job["job_dir"]))
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
    if learned:
        msg += f"\n\n{learned}"
    msg += "\n\n議事録に反映するには「③ 📋 議事録を作る」を押してください。"
    return (msg, job) + _job_views(job)


def save_utterance_edits(rows, state):
    if not state:
        return ("⚠️ 先に文字起こしを実行してください。", state) + tuple(gr.update() for _ in range(VIEW_COUNT))
    if state.get("incomplete"):
        return (checkpoints.READ_ONLY_NOTICE, state) + tuple(gr.update() for _ in range(VIEW_COUNT))
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
            u["human"] = True  # AI校正で上書きしない
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
        added, sec = worker.run("core.pipeline:enroll", src, name)
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
        scores = worker.run("core.pipeline:identify", src)
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
    job = _load_job(job_name)
    if job:
        warning = document_status.warning(job, md_name)
        if warning:
            msg = warning + "\n\n" + msg
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


def save_settings_ui(ollama_url, ollama_model, ollama_think, whisper_model, language,
                     red_threshold, cluster_sim, match_sim, hf_token):
    settings = config.load_settings()
    settings["llm"].update({
        "ollama_url": ollama_url.strip(),
        "ollama_model": (ollama_model or "").strip(),
        "ollama_think": bool(ollama_think),
    })
    settings["whisper"].update({"model": whisper_model, "language": language})
    settings["postprocess"]["red_threshold"] = float(red_threshold)
    settings["diarization"]["hf_token"] = (hf_token or "").strip()
    settings["diarization"].update({"max_speakers": int(cluster_sim),
                                    "match_similarity": float(match_sim)})
    config.save_settings(settings)
    return "✅ 設定を保存しました。"


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


# GPUの有無の確認は ctranslate2 / torch の読み込みで数秒かかるので、起動を待たせないよう
# 裏で調べて、画面が開いてから状態の行に反映する
_devices: dict = {}


def _probe_devices() -> None:
    try:
        _devices["asr"] = transcriber.detect_device()[0]
        _devices["dia"] = diarization.device_label() if diarization.available() else "-"
    except Exception as e:
        logging.warning(f"GPUの確認に失敗: {e}")


_probe_thread = threading.Thread(target=_probe_devices, daemon=True)


def status_line() -> str:
    settings = config.load_settings()
    _probe_thread.join(timeout=30)
    asr = "🟢 GPU" if _devices.get("asr") == "cuda" else "🟡 CPU"
    if diarization.available():
        dia = ("✅ " + diarization.backend_label(settings["diarization"].get("hf_token", ""))
               + "・" + _devices.get("dia", "cpu").upper())
    else:
        dia = "➖ 未導入"
    return (f"文字起こし: {asr} ／ 話者識別: {dia} ／ "
            f"LLM: {llm.provider_label(settings)} ／ 登録声紋: {len(diarization.list_voiceprints())}名")


def build_ui():
    settings = config.load_settings()
    dia_available = diarization.available()  # 軽い（パッケージの有無だけ見る）
    if not _probe_thread.is_alive() and not _devices:
        _probe_thread.start()

    with gr.Blocks(title="つよつよ文字起こし＆議事録ツール") as demo:
        gr.Markdown("# 🎙️ つよつよ文字起こし＆議事録ツール")
        status_md = gr.Markdown("文字起こし: 確認中… ／ 話者識別: 確認中…")

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
                            decoding_radio = gr.Radio(
                                transcriber.DECODING_CHOICES,
                                value=settings["whisper"].get("decoding", "stable"),
                                label="認識方式",
                                info="推奨方式は前の誤認識を引きずりにくくします。人名・用語の表記が崩れる録音では"
                                     "従来方式も試せます。高速モードではこの選択は使いません。")
                            silence_guard_cb = gr.Checkbox(
                                value=settings["whisper"].get("silence_guard", True),
                                interactive=settings["whisper"].get("speed", "accurate") != "fast",
                                label="無音付近の誤認識を除外（内部スキップ）",
                                info="ONは従来の動作。雑音で処理が進まないときはOFFを試してください（全編実験の設定）。"
                                     "OFFでは誤った文が残ることがあります。高速モードには適用しません。")
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
                        start_btn = gr.Button("① 🚀 文字起こし開始", variant="primary", size="lg")
                        pause_btn = gr.Button("⏸ 一時停止（保存して止める）", size="sm")
                        pause_status = gr.Markdown(elem_id="st-pause-status")
                        gr.Markdown("<small>①は音声の文字起こしと話者識別だけを行います。"
                                    "一時停止は処理中の区間が終わってから保存して止めます。"
                                    "認識中は途中保存し、停止した場合は履歴から開いて続きを再開できます。"
                                    "AI校正（②）と議事録（③）は、確認・修正のあと右側のボタンで別に実行します。</small>")
                        result_md = gr.Markdown(elem_id="st-summary")
                        with gr.Row():
                            history_dd = gr.Dropdown(
                                label="📂 過去の結果を開く", choices=output.list_editable_jobs(),
                                scale=4)
                            history_refresh = gr.Button("🔄", scale=1, min_width=40)
                        resume_btn = gr.Button("▶ 途中保存から再開", size="sm")
                        gr.Markdown("<small>履歴から途中保存を開いて再開します。保存時の音声・設定を使うため、"
                                    "新しい形式の結果はファイルの再選択が不要です。境界付近だけ聞き直します。</small>")
                        rediarize_btn = gr.Button("👥 話者識別だけやり直す（上の「話者の人数」を使用）",
                                                  size="sm")
                        reflow_btn = gr.Button("✂️ 文の区切り・句読点を整え直す（数秒・音声解析なし）",
                                               size="sm")

                    with gr.Column(scale=7):
                        job_notice = gr.Markdown(elem_id="st-job-notice")
                        with gr.Row(elem_classes="st-dl"):
                            dl_transcript = gr.DownloadButton("📥 文字起こし.md", interactive=False,
                                                              variant="primary")
                            dl_minutes = gr.DownloadButton("📥 議事録.md", interactive=False)
                            dl_insights = gr.DownloadButton("📥 要約と考察.md", interactive=False)
                        with gr.Tabs():
                            with gr.Tab("📜 文字起こし"):
                                # Gradio 6.6のHTMLは連続更新時に古いDOMが残る。MarkdownのHTML表示を使う。
                                # 本文・話者・音声URLはU.render_htmlがエスケープする。時刻クリックは残す。
                                transcript_html = gr.Markdown(
                                    '<div style="padding:40px;text-align:center;opacity:.6">'
                                    "ここに「時刻・話者・発言」が表示されます</div>",
                                    sanitize_html=False, latex_delimiters=[])
                            with gr.Tab("📋 議事録"):
                                minutes_view = gr.Markdown()
                            with gr.Tab("💡 要約と考察"):
                                insights_view = gr.Markdown()

                        gr.Markdown("### ② 確認・修正")
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
                        with gr.Row():
                            proofread_btn = gr.Button("✍️ AI校正（誤変換を文脈で直す・途中からでも再開できる）",
                                                      scale=3)
                            redo_cb = gr.Checkbox(value=False, label="校正済みもやり直す", scale=1)
                        proofread_status = gr.Markdown()

                        gr.Markdown("### ③ 議事録")
                        gr.Markdown("<small>②で直した今の文字起こしから、`議事録テンプレート.md` の形で作ります。</small>")
                        with gr.Row():
                            minutes_btn = gr.Button("📋 議事録を作る", variant="primary", scale=2)
                            insights_btn = gr.Button("💡 要約と考察を作る", scale=1)
                        doc_status = gr.Markdown()

            # ======================== ノイズの見本で補正 ========================
            with gr.Tab("🔈 ノイズの見本で補正"):
                gr.Markdown("説明者が話しておらず、環境音だけが聞こえる範囲を選びます。"
                            "人のざわめきは声と重なるため、強く消すと発言まで削れることがあります。")
                noise_source = gr.Audio(label="元音声（再生して範囲を確認）", sources=["upload"], type="filepath")
                with gr.Row():
                    noise_start = gr.Textbox(label="見本の開始（秒数・分:秒）", value="0:00")
                    noise_end = gr.Textbox(label="見本の終了（0.5〜30秒の範囲）", value="0:05")
                noise_preview_btn = gr.Button("選んだ範囲だけを聞く")
                noise_sample_player = gr.Audio(label="ノイズに指定する範囲", interactive=False)
                noise_strength = gr.Slider(0, 80, value=35, step=5, label="除去の強さ（まずは弱めの35）")
                noise_apply_btn = gr.Button("この見本で音声全体のノイズを弱める", variant="primary")
                noise_status = gr.Markdown()
                with gr.Row():
                    noise_reduced = gr.Audio(label="除去後", interactive=False)
                    noise_removed = gr.Audio(label="削った音（ここに説明者の声がないか確認）", interactive=False)
                noise_download = gr.File(label="補正音声を保存", interactive=False)
                gr.Markdown("元音声は変更しません。補正音声で文字起こしする場合は、"
                            "文字起こしタブにこのファイルを指定し、「背景ノイズ除去」をOFFにして二重処理を避けてください。")

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
                gr.Markdown("Ollama 上の gemma4 だけで動きます（完全ローカル・文字起こしは外部に送りません）。")
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
                        dl_transcript, dl_minutes, dl_insights, speaker_table, edit_table, job_notice]

        noise_preview_btn.click(fn=preview_noise_sample, inputs=[noise_source, noise_start, noise_end],
                                outputs=noise_sample_player)
        noise_apply_btn.click(fn=apply_noise_sample, inputs=[noise_source, noise_start, noise_end, noise_strength],
                              outputs=[noise_status, noise_reduced, noise_removed, noise_download])

        pause_btn.click(fn=pause_transcription, outputs=pause_status, queue=False)
        start_btn.click(
            fn=run_batch,
            inputs=[files_input, model_dd, speed_radio, lang_dd, num_spk_dd, noise_cb, eq_cb, fillers_cb, diarize_cb,
                    meeting_date_box, memo_box, decoding_radio, silence_guard_cb],
            outputs=[result_md, job_state] + view_outputs + [history_dd],
            concurrency_id="asr",
        ).then(fn=lambda: "", outputs=pause_status, queue=False)

        resume_btn.click(
            fn=resume_transcription,
            inputs=[job_state, files_input, model_dd, speed_radio, lang_dd, num_spk_dd, noise_cb, eq_cb,
                    fillers_cb, diarize_cb, meeting_date_box, memo_box, decoding_radio, silence_guard_cb],
            outputs=[result_md, job_state] + view_outputs + [history_dd], concurrency_id="asr",
        ).then(fn=lambda: "", outputs=pause_status, queue=False)

        speed_radio.change(fn=lambda speed: gr.update(interactive=speed != "fast"),
                           inputs=speed_radio, outputs=silence_guard_cb, queue=False)

        history_dd.input(fn=open_job, inputs=history_dd,
                         outputs=[result_md, job_state] + view_outputs)
        history_refresh.click(fn=refresh_history, outputs=history_dd)
        rediarize_btn.click(fn=rediarize_job, inputs=[job_state, num_spk_dd],
                            outputs=[result_md, job_state] + view_outputs)
        reflow_btn.click(fn=reflow_job, inputs=job_state,
                         outputs=[result_md, job_state] + view_outputs)

        assign_btn.click(fn=assign_speakers, inputs=[speaker_table, save_voice_cb, job_state],
                         outputs=[assign_status, job_state] + view_outputs
                         ).then(fn=_voiceprint_rows, outputs=vp_table)
        proofread_btn.click(fn=proofread_job, inputs=[job_state, redo_cb],
                            outputs=[proofread_status, job_state] + view_outputs)
        minutes_btn.click(fn=make_minutes, inputs=job_state,
                          outputs=[doc_status, job_state] + view_outputs)
        insights_btn.click(fn=make_insights, inputs=job_state,
                           outputs=[doc_status, job_state] + view_outputs)
        save_edit_btn.click(fn=save_utterance_edits, inputs=[edit_table, job_state],
                            outputs=[edit_status, job_state] + view_outputs)

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
            inputs=[ollama_url_box, ollama_model_box, ollama_think_cb, whisper_model_dd, lang_default_dd, red_threshold_slider,
                    cluster_sim_slider, match_sim_slider, hf_token_box],
            outputs=settings_status,
        )
        test_btn.click(fn=test_llm, outputs=settings_status)

        demo.load(fn=status_line, outputs=status_md)
        demo.load(fn=refresh_jobs, outputs=[job_dd, md_dd])
        demo.load(fn=refresh_history, outputs=history_dd)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    print("==========================================================")
    print("Ready. Browser will open automatically.")
    print("==========================================================")
    demo.launch(server_name="127.0.0.1", share=False,
                inbrowser=os.environ.get("ST_NO_BROWSER") != "1",
                allowed_paths=[str(config.OUTPUT_DIR)], css=CSS)
