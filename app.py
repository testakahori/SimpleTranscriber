import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# OSの証明書ストアを使う（セキュリティソフトのSSL検査環境でもモデルDLが通るように）
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import logging
import traceback
from pathlib import Path

import gradio as gr

from core import audio, config, diarization, glossary, llm, minutes, output, people, postprocess
from core import export as export_mod
from core import search as search_mod
from core import transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

config.ensure_dirs()

FILE_TYPES = [f".{e.lstrip('.')}" for e in audio.SUPPORTED_EXTS]


# =====================================================================
# メイン処理パイプライン
# =====================================================================

def _process_one(path: str, opts: dict, settings: dict, progress, p_start: float, p_end: float) -> dict:
    """1ファイルを処理して job情報dict を返す"""
    name = Path(path).name
    span = p_end - p_start

    def sub(frac, desc):
        progress(p_start + span * frac, desc=f"{name}: {desc}")

    sub(0.02, "音声を変換中...")
    wav = audio.prepare_audio(path, noise_reduction=opts["noise"])
    duration = audio.get_duration(wav)

    try:
        people_list = people.load_people()
        initial_prompt = glossary.build_initial_prompt(people.initial_prompt_terms(people_list))

        sub(0.05, "文字起こし中...")
        result = transcriber.transcribe(
            wav, model_name=opts["model"], language=opts["language"],
            initial_prompt=initial_prompt, duration=duration,
            progress_cb=lambda f: sub(0.05 + 0.55 * f, "文字起こし中..."),
        )

        # 話者分離（オプショナル）
        speakers, cluster_embeddings = ({}, {})
        if opts["diarize"] and diarization.available():
            sub(0.62, "話者分離中...")
            dia = settings.get("diarization", {})
            speakers, cluster_embeddings = diarization.diarize(
                wav, result["segments"],
                match_threshold=dia.get("match_threshold", 0.75),
                cluster_threshold=dia.get("cluster_threshold", 0.68),
            )
    finally:
        Path(wav).unlink(missing_ok=True)

    threshold = settings.get("postprocess", {}).get("red_threshold", 0.5)

    # 人名置換・学習済み修正をテキストへ適用
    for seg in result["segments"]:
        seg["text"] = glossary.apply_corrections(people.apply_aliases(seg["text"], people_list))

    transcript_md = postprocess.build_transcript_md(result, threshold, speakers, source_name=name)
    transcript_md = glossary.apply_corrections(people.apply_aliases(transcript_md, people_list))

    job_dir = output.create_job_dir(name)
    files_written = []

    # LLM校正（赤字の文脈補完）
    proofread_error = None
    if opts["proofread"]:
        sub(0.70, "LLMで校正中（赤字補完）...")
        try:
            transcript_md = minutes.proofread_transcript(settings, transcript_md)
        except llm.LLMError as e:
            proofread_error = str(e)

    files_written.append(output.write_text(job_dir, "transcript.md", transcript_md))
    files_written.append(output.write_text(job_dir, "transcript.srt", postprocess.build_srt(result)))

    # 議事録
    errors = []
    if proofread_error:
        errors.append(f"校正スキップ: {proofread_error}")
    if opts["minutes"]:
        sub(0.80, "議事録を生成中...")
        try:
            md = minutes.generate_minutes(settings, transcript_md, name)
            files_written.append(output.write_text(job_dir, "議事録.md", md))
        except llm.LLMError as e:
            errors.append(f"議事録スキップ: {e}")

    # 要約と考察
    if opts["insights"]:
        sub(0.92, "要約と考察を生成中...")
        try:
            md = minutes.generate_insights(settings, transcript_md, name)
            files_written.append(output.write_text(job_dir, "要約と考察.md", md))
        except llm.LLMError as e:
            errors.append(f"要約スキップ: {e}")

    output.write_meta(job_dir, {
        "source": name,
        "duration_sec": round(duration, 1),
        "language": result.get("language"),
        "whisper_model": result.get("model"),
        "device": result.get("device"),
        "elapsed_sec": round(result.get("elapsed", 0), 1),
        "noise_reduction": opts["noise"],
        "low_confidence_ratio": postprocess.low_confidence_ratio(result, threshold),
        "llm_provider": llm.provider_label(settings),
        "speakers": sorted(set(speakers.values())) if speakers else [],
        "errors": errors,
    })

    sub(1.0, "完了")
    return {
        "name": name,
        "job_dir": str(job_dir),
        "transcript_md": transcript_md,
        "segments": result["segments"],
        "cluster_embeddings": cluster_embeddings,
        "errors": errors,
        "files": [f.name for f in files_written],
    }


def run_batch(files, model_name, language, noise, diarize_on, do_proofread,
              do_minutes, do_insights, progress=gr.Progress()):
    empty = (gr.update(), gr.update(), gr.update(), None, [], gr.update())
    if not files:
        return ("⚠️ ファイルを選択してください。",) + empty[1:]

    settings = config.load_settings()
    opts = {
        "model": model_name, "language": language, "noise": noise,
        "diarize": diarize_on, "proofread": do_proofread,
        "minutes": do_minutes, "insights": do_insights,
    }

    paths = [f if isinstance(f, str) else f.name for f in files]
    results, failures = [], []
    for i, path in enumerate(paths):
        p_start = i / len(paths)
        p_end = (i + 1) / len(paths)
        try:
            results.append(_process_one(path, opts, settings, progress, p_start, p_end))
        except Exception as e:
            logging.error(traceback.format_exc())
            failures.append(f"❌ {Path(path).name}: {e}")

    # 結果サマリー
    lines = [f"## 処理結果 ({len(results)}/{len(paths)} 件成功)"]
    for r in results:
        rel = Path(r["job_dir"]).name
        lines.append(f"✅ **{r['name']}** → `output/{rel}/` （{', '.join(r['files'])}）")
        for err in r["errors"]:
            lines.append(f"　⚠️ {err}")
    lines.extend(failures)
    summary = "\n\n".join(lines)

    if not results:
        return (summary,) + empty[1:]

    last = results[-1]
    preview_html = postprocess.md_to_display_html(last["transcript_md"])

    # 話者割り当てテーブル（未知話者のみ編集対象）
    speaker_rows = [[label, ""] for label in sorted(last["cluster_embeddings"].keys())]

    # SRT編集タブ用データ
    srt_rows = [
        [i + 1, postprocess.format_srt_time(s["start"]),
         postprocess.format_srt_time(s["end"]), s["text"]]
        for i, s in enumerate(last["segments"])
    ]

    state = {
        "job_dir": last["job_dir"],
        "transcript_md": last["transcript_md"],
        "cluster_embeddings": {k: v.tolist() if hasattr(v, "tolist") else v
                               for k, v in last["cluster_embeddings"].items()},
    }
    return (summary, preview_html, last["transcript_md"], state, speaker_rows, srt_rows)


# =====================================================================
# 編集の保存 → 修正学習
# =====================================================================

def save_edited_transcript(edited_md, state):
    if not state or not edited_md:
        return "⚠️ 保存対象がありません。先に文字起こしを実行してください。", gr.update(), state
    before = state.get("transcript_md", "")
    learned = glossary.learn_from_diff(before, edited_md)
    path = Path(state["job_dir"]) / "transcript.md"
    path.write_text(edited_md, encoding="utf-8")
    state["transcript_md"] = edited_md
    msg = f"✅ 保存しました: `{path}`"
    if learned:
        msg += f"（{learned}件の修正を学習しました。同じ修正が{glossary.PROMOTE_COUNT}回以上で自動適用されます）"
    return msg, postprocess.md_to_display_html(edited_md), state


def assign_speakers(rows, state):
    if not state:
        return "⚠️ 先に文字起こしを実行してください。", gr.update(), state
    import numpy as np
    embeddings = state.get("cluster_embeddings", {})
    md = state.get("transcript_md", "")
    assigned = 0
    for row in (rows or []):
        if len(row) < 2:
            continue
        label = str(row[0]).strip()
        new_name = str(row[1]).strip() if row[1] is not None else ""
        if not label or not new_name or label == new_name:
            continue
        if label in embeddings:
            diarization.save_voiceprint(new_name, np.asarray(embeddings[label], dtype="float32"))
        md = md.replace(f"] {label}:**", f"] {new_name}:**")
        assigned += 1
    if not assigned:
        return "⚠️ 割り当てが入力されていません（「割当する名前」列に記入してください）。", gr.update(), state
    path = Path(state["job_dir"]) / "transcript.md"
    path.write_text(md, encoding="utf-8")
    state["transcript_md"] = md
    return (f"✅ {assigned}名の話者を確定し、声紋を保存しました。次回から自動で認識されます。",
            postprocess.md_to_display_html(md), state)


# =====================================================================
# SRT編集
# =====================================================================

def save_srt_from_editor(editor_df, state):
    if editor_df is None or len(editor_df) == 0:
        return None
    rows = editor_df.values.tolist() if hasattr(editor_df, "values") else list(editor_df)
    srt_lines = []
    n = 0
    for row in rows:
        text = str(row[3]).strip() if len(row) > 3 and row[3] is not None else ""
        if not text:
            continue
        n += 1
        srt_lines.extend([str(n), f"{str(row[1]).strip()} --> {str(row[2]).strip()}", text, ""])
    if state and state.get("job_dir"):
        path = Path(state["job_dir"]) / "transcript.srt"
    else:
        path = config.OUTPUT_DIR / "edited.srt"
    path.write_text("\n".join(srt_lines), encoding="utf-8")
    return str(path)


# =====================================================================
# 検索 / 出力 / 人物 / 辞書 / 設定
# =====================================================================

def do_search(query):
    rows = search_mod.search_outputs(query)
    if not rows:
        return [["", "", "", "該当なし"]]
    return rows


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
    generated = []
    errors = []
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


def save_people_table(rows):
    data = rows.values.tolist() if hasattr(rows, "values") else list(rows or [])
    people.save_people(people.from_table(data))
    return "✅ 人物マスタを保存しました。"


def save_glossary_text(text):
    glossary.save_glossary([t for t in (text or "").split("\n") if t.strip()])
    return "✅ 用語辞書を保存しました。次回の文字起こしから認識ヒントに使われます。"


def save_corrections_table(rows):
    data = rows.values.tolist() if hasattr(rows, "values") else list(rows or [])
    corrections = []
    for row in data:
        if len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip():
            try:
                count = int(row[2]) if len(row) > 2 and row[2] is not None else 1
            except (ValueError, TypeError):
                count = 1
            corrections.append({"wrong": str(row[0]).strip(), "right": str(row[1]).strip(),
                                "count": count})
    glossary.save_corrections(corrections)
    return "✅ 学習済み修正を保存しました。"


def load_corrections_table():
    rows = [[c.get("wrong", ""), c.get("right", ""), c.get("count", 1)]
            for c in glossary.load_corrections()]
    return rows or [["", "", 1]]


def save_settings_ui(provider, ollama_url, ollama_model, anthropic_key, anthropic_model,
                     openai_key, openai_model, whisper_model, language, red_threshold):
    settings = config.load_settings()
    settings["llm"].update({
        "provider": provider,
        "ollama_url": ollama_url.strip(),
        "ollama_model": ollama_model.strip(),
        "anthropic_api_key": anthropic_key.strip(),
        "anthropic_model": anthropic_model.strip(),
        "openai_api_key": openai_key.strip(),
        "openai_model": openai_model.strip(),
    })
    settings["whisper"].update({"model": whisper_model, "language": language})
    settings["postprocess"]["red_threshold"] = float(red_threshold)
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

def build_ui():
    settings = config.load_settings()
    device, _ = transcriber.detect_device()
    device_label = "🟢 GPU (CUDA)" if device == "cuda" else "🟡 CPU"
    dia_available = diarization.available()

    with gr.Blocks(title="つよつよ文字起こし＆議事録ツール") as demo:
        gr.Markdown("# 🎙️ つよつよ文字起こし＆議事録ツール")
        gr.Markdown(
            f"実行環境: {device_label} ／ LLM: {llm.provider_label(settings)} ／ "
            f"話者分離: {'✅ 利用可能' if dia_available else '➖ 未導入（pip install torch torchaudio speechbrain で有効化）'}"
        )

        last_state = gr.State(None)

        with gr.Tabs():
            # ---------------- 文字起こし ----------------
            with gr.Tab("🎙️ 文字起こし"):
                with gr.Row():
                    with gr.Column(scale=1):
                        files_input = gr.File(
                            label="音声・動画ファイル（複数可 — まとめてドラッグ＆ドロップ）",
                            file_count="multiple", file_types=FILE_TYPES, type="filepath",
                        )
                        with gr.Accordion("オプション", open=True):
                            model_dd = gr.Dropdown(
                                transcriber.MODEL_CHOICES,
                                value=settings["whisper"]["model"],
                                label="精度（モデル）",
                                info="auto=GPUならlarge-v3(最高精度)/CPUならsmall",
                            )
                            lang_dd = gr.Dropdown(
                                ["ja", "auto"], value=settings["whisper"]["language"],
                                label="言語", info="ja=日本語固定（推奨） / auto=自動判定",
                            )
                            noise_cb = gr.Checkbox(
                                value=settings["audio"]["noise_reduction"],
                                label="ノイズ除去（汚い録音の精度UP・処理時間増）",
                            )
                            diarize_cb = gr.Checkbox(
                                value=dia_available, interactive=dia_available,
                                label="話者分離・声紋識別" + ("" if dia_available else "（未導入）"),
                            )
                            proofread_cb = gr.Checkbox(
                                value=settings["postprocess"]["proofread"],
                                label="LLM校正（聞き取れない箇所を文脈から推測→赤字表記）",
                            )
                            minutes_cb = gr.Checkbox(value=True, label="議事録.md を生成")
                            insights_cb = gr.Checkbox(value=True, label="要約と考察.md を生成")
                        start_btn = gr.Button("🚀 処理開始", variant="primary")
                        result_md = gr.Markdown()

                    with gr.Column(scale=1):
                        preview_html = gr.HTML(label="プレビュー")
                        with gr.Accordion("✏️ テキスト修正（保存すると学習して成長します）", open=False):
                            edit_box = gr.Textbox(lines=12, label="transcript.md（最後に処理したファイル）")
                            save_edit_btn = gr.Button("保存して学習", variant="primary")
                            edit_status = gr.Markdown()
                        with gr.Accordion("👥 話者の割り当て（話者A→名前を登録すると次回から自動認識）", open=False):
                            speaker_table = gr.Dataframe(
                                headers=["話者ラベル", "割当する名前"],
                                column_count=(2, "fixed"), interactive=True,
                            )
                            assign_btn = gr.Button("話者を確定して声紋を保存")
                            assign_status = gr.Markdown()

            # ---------------- SRT編集 ----------------
            with gr.Tab("📝 SRT編集"):
                gr.Markdown("最後に処理したファイルの字幕を編集できます。時刻は `HH:MM:SS,mmm` 形式。")
                srt_editor = gr.Dataframe(
                    headers=["番号", "開始", "終了", "テキスト"],
                    datatype=["number", "str", "str", "str"],
                    column_count=(4, "fixed"), interactive=True, wrap=True,
                )
                save_srt_btn = gr.Button("SRTを保存", variant="primary")
                srt_download = gr.File(label="保存されたSRT")

            # ---------------- 検索 ----------------
            with gr.Tab("🔍 検索"):
                gr.Markdown("過去の全議事録・文字起こし・要約を横断検索します（スペース区切りでAND検索）。")
                with gr.Row():
                    search_box = gr.Textbox(label="キーワード", placeholder="例: 見積 坪内", scale=4)
                    search_btn = gr.Button("検索", variant="primary", scale=1)
                search_results = gr.Dataframe(
                    headers=["フォルダ", "ファイル", "行", "内容"],
                    interactive=False, wrap=True,
                )

            # ---------------- Word/PDF出力 ----------------
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

            # ---------------- 人物登録 ----------------
            with gr.Tab("👤 人物登録"):
                gr.Markdown(
                    "呼び名と議事録での正式表記を登録します。例: 表記「坪内」呼び名「ボンギン, ボンギンさん」\n"
                    "登録すると文字起こし・議事録で自動的に正式表記へ統一され、音声認識のヒントにも使われます。"
                )
                people_table = gr.Dataframe(
                    headers=["正式表記", "呼び名（カンマ区切り）", "役職（任意）"],
                    column_count=(3, "fixed"), interactive=True,
                    value=people.to_table(),
                )
                people_save_btn = gr.Button("保存", variant="primary")
                people_status = gr.Markdown()

            # ---------------- 用語辞書 ----------------
            with gr.Tab("📚 用語辞書"):
                gr.Markdown("社名・製品名・専門用語を1行1語で登録すると、音声認識の精度が上がります。")
                glossary_box = gr.Textbox(
                    lines=10, label="用語（1行1語）",
                    value="\n".join(glossary.load_glossary()),
                )
                glossary_save_btn = gr.Button("保存", variant="primary")
                glossary_status = gr.Markdown()
                gr.Markdown("### 学習済み修正（テキスト修正から自動で貯まります）")
                gr.Markdown(f"出現回数 {glossary.PROMOTE_COUNT} 回以上で次回から自動適用されます。")
                corrections_table = gr.Dataframe(
                    headers=["誤", "正", "回数"],
                    column_count=(3, "fixed"), interactive=True,
                    value=load_corrections_table(),
                )
                corrections_save_btn = gr.Button("学習済み修正を保存")
                corrections_status = gr.Markdown()

            # ---------------- 設定 ----------------
            with gr.Tab("⚙️ 設定"):
                gr.Markdown("## LLM（議事録・要約・校正の頭脳）")
                provider_radio = gr.Radio(
                    choices=[("Ollama（ローカル・無料・完全オフライン）", "ollama"),
                             ("Claude API", "anthropic"),
                             ("OpenAI API", "openai"),
                             ("使わない（文字起こしのみ）", "none")],
                    value=settings["llm"]["provider"], label="プロバイダ",
                )
                with gr.Group():
                    ollama_url_box = gr.Textbox(value=settings["llm"]["ollama_url"], label="Ollama URL")
                    ollama_model_box = gr.Textbox(
                        value=settings["llm"]["ollama_model"], label="Ollamaモデル",
                        info="事前に `ollama pull <モデル名>` が必要です",
                    )
                with gr.Group():
                    anthropic_key_box = gr.Textbox(
                        value=settings["llm"]["anthropic_api_key"], label="Claude APIキー",
                        type="password", info="空欄なら環境変数 ANTHROPIC_API_KEY を使用",
                    )
                    anthropic_model_box = gr.Textbox(
                        value=settings["llm"]["anthropic_model"], label="Claudeモデル")
                with gr.Group():
                    openai_key_box = gr.Textbox(
                        value=settings["llm"]["openai_api_key"], label="OpenAI APIキー",
                        type="password", info="空欄なら環境変数 OPENAI_API_KEY を使用",
                    )
                    openai_model_box = gr.Textbox(
                        value=settings["llm"]["openai_model"], label="OpenAIモデル")

                gr.Markdown("## 文字起こしの既定値")
                whisper_model_dd = gr.Dropdown(
                    transcriber.MODEL_CHOICES, value=settings["whisper"]["model"], label="モデル")
                lang_default_dd = gr.Dropdown(
                    ["ja", "auto"], value=settings["whisper"]["language"], label="言語")
                red_threshold_slider = gr.Slider(
                    0.1, 0.9, value=settings["postprocess"]["red_threshold"], step=0.05,
                    label="赤字にする確信度のしきい値（低いほど赤字が減る）",
                )

                with gr.Row():
                    settings_save_btn = gr.Button("設定を保存", variant="primary")
                    test_btn = gr.Button("LLM接続テスト")
                settings_status = gr.Markdown()

        # ---------------- イベント接続 ----------------
        start_btn.click(
            fn=run_batch,
            inputs=[files_input, model_dd, lang_dd, noise_cb, diarize_cb,
                    proofread_cb, minutes_cb, insights_cb],
            outputs=[result_md, preview_html, edit_box, last_state, speaker_table, srt_editor],
        )
        save_edit_btn.click(
            fn=save_edited_transcript,
            inputs=[edit_box, last_state],
            outputs=[edit_status, preview_html, last_state],
        )
        assign_btn.click(
            fn=assign_speakers,
            inputs=[speaker_table, last_state],
            outputs=[assign_status, preview_html, last_state],
        )
        save_srt_btn.click(fn=save_srt_from_editor, inputs=[srt_editor, last_state],
                           outputs=srt_download)
        search_btn.click(fn=do_search, inputs=search_box, outputs=search_results)
        search_box.submit(fn=do_search, inputs=search_box, outputs=search_results)
        refresh_btn.click(fn=refresh_jobs, outputs=[job_dd, md_dd])
        job_dd.change(fn=on_job_change, inputs=job_dd, outputs=md_dd)
        export_btn.click(fn=do_export, inputs=[job_dd, md_dd, docx_cb, pdf_cb],
                         outputs=[export_status, export_files])
        people_save_btn.click(fn=save_people_table, inputs=people_table, outputs=people_status)
        glossary_save_btn.click(fn=save_glossary_text, inputs=glossary_box, outputs=glossary_status)
        corrections_save_btn.click(fn=save_corrections_table, inputs=corrections_table,
                                   outputs=corrections_status)
        settings_save_btn.click(
            fn=save_settings_ui,
            inputs=[provider_radio, ollama_url_box, ollama_model_box,
                    anthropic_key_box, anthropic_model_box,
                    openai_key_box, openai_model_box,
                    whisper_model_dd, lang_default_dd, red_threshold_slider],
            outputs=settings_status,
        )
        test_btn.click(fn=test_llm, outputs=settings_status)

        demo.load(fn=refresh_jobs, outputs=[job_dd, md_dd])

    return demo


demo = build_ui()

if __name__ == "__main__":
    print("==========================================================")
    print("Ready. Browser will open automatically.")
    print("==========================================================")
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)
