"""前処理済み全編WAVをWhisper → Qwen → gemma4で比較するローカル実験。

各stageは対応する仮想環境から別プロセスで実行し、終了してから次へ進む。
入力フォルダには source.json と asr.wav を用意する。音声・結果は公開しない。
"""
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def save_json(path, value):
    """旧チェックポイントを完全に書けたJSONだけで置換する。"""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def transcript_md(path, title, rows, note=""):
    text = f"# {title}\n\n{note}\n\n"
    for row in rows:
        warning = " **要確認・校正対象外**" if row.get("unsafe") else ""
        body = row.get("marked") or row["text"]
        body = re.sub(r'<span style="color:red">(.*?)</span>', r'⟦\1⟧', body)
        text += f"## {stamp(row['start'])}–{stamp(row['end'])}{warning}\n\n{body or '（認識文字なし）'}\n\n"
    temp = path.with_suffix(".md.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def prepare(folder, source):
    from core import audio
    if source is None or not source.is_file():
        raise ValueError("prepareには --audio 元音声ファイル が必要です")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    shutil.copy2(source, folder / ("original" + source.suffix.lower()))
    with source.open("rb") as f:
        fingerprint = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = dict(source=str(source.resolve()), sha256=fingerprint, source_size=source.stat().st_size,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    preprocessing="16kHz mono, dynaudnorm f250 g15 p0.9, EQ80-7000Hz, weak noisereduce0.5")
    save_json(folder / "source.json", metadata)
    normalized = audio.prepare_audio(str(source))
    shutil.copy2(normalized, folder / "normalized.wav")
    enhanced = audio.enhance_for_asr(str(folder / "normalized.wav"), eq=True, noise_reduction=True)
    shutil.copy2(enhanced, folder / "asr.wav")
    metadata.update(duration=audio.get_duration(str(folder / "asr.wav")),
                    preprocess_sec=round(time.monotonic() - started, 2),
                    enhancement_applied=enhanced != str(folder / "normalized.wav"))
    save_json(folder / "source.json", metadata)


def whisper(folder):
    from core import config, glossary, llm, people, transcriber
    final = folder / "01_whisper.json"
    if final.exists():
        print("Whisper already complete", flush=True)
        return
    llm.unload_ollama(config.load_settings())
    started, last_save, last_phase = time.monotonic(), 0.0, ""

    def checkpoint(data):
        nonlocal last_save, last_phase
        now = time.monotonic()
        if data["phase"] == last_phase and now - last_save < 10:
            return
        snapshot = dict(data, complete=False, elapsed_sec=round(now - started, 2))
        save_json(folder / "whisper.partial.json", snapshot)
        transcript_md(folder / "01_Whisper_途中経過.md", "Whisper 途中経過", data["segments"],
                      "処理途中の保存です。最終版は 01_Whisper.md。")
        if data["phase"] == "main_complete":
            save_json(folder / "whisper-main.json", snapshot)
        print(json.dumps({k: v for k, v in snapshot.items() if k != "segments"}), flush=True)
        last_save, last_phase = now, data["phase"]

    terms = people.initial_prompt_terms()
    options_path = folder / "experiment-options.json"
    options = read_json(options_path) if options_path.exists() else {}
    result = transcriber.transcribe(str(folder / "asr.wav"), initial_prompt=glossary.build_initial_prompt(terms),
                                    hotwords=glossary.build_hotwords(terms), speed="accurate", decoding="stable",
                                    checkpoint_cb=checkpoint, silence_guard=options.get("silence_guard", True))
    result.update(complete=True, wall_sec=round(time.monotonic() - started, 2))
    save_json(final, result)
    transcript_md(folder / "01_Whisper.md", "Whisper 全編", result["segments"],
                  f"large-v3 / accurate / stable / silence_guard={result['silence_guard']}。抜けの再認識を含む。話者識別・人による正誤確認は未実施。")
    print(f"Whisper complete: {result['wall_sec']} sec", flush=True)


def qwen(folder):
    import soundfile as sf
    from tools.asr_trial_worker import qwen as generate
    from tools.try_qwen_recovery import whisper_text
    from core.transcriber import _REPEAT_RE
    final = folder / "02_qwen.json"
    if final.exists():
        print("Qwen already complete", flush=True)
        return
    started = time.monotonic()
    info = sf.info(folder / "asr.wav")
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError("asr.wavは16kHzモノラルである必要があります")
    baseline = read_json(folder / "01_whisper.json")
    if not baseline.get("complete"):
        raise ValueError("Whisperの全編処理が完了していません")
    clips = []
    (folder / "clips").mkdir(exist_ok=True)
    # 重複しない30秒窓で全サンプルを一度ずつ渡す。窓の端の語は不利になり得る。
    for i, start in enumerate(range(0, info.frames, 30 * info.samplerate)):
        end = min(start + 30 * info.samplerate, info.frames)
        path = folder / "clips" / f"clip-{i + 1:03d}.wav"
        if not path.exists():
            audio, rate = sf.read(folder / "asr.wav", start=start, stop=end, dtype="float32")
            sf.write(path, audio, rate, subtype="PCM_16")
        a, b = start / info.samplerate, end / info.samplerate
        clips.append(dict(id=f"clip-{i + 1:03d}", start=a, end=b, path=str(path), reason="全編比較",
                          whisper=whisper_text(baseline["segments"], a, b)))
    save_json(folder / "qwen-request.json", {"clips": clips})
    partial = folder / "qwen.partial.json"
    old = read_json(partial) if partial.exists() else {"results": [], "wall_sec": 0}
    done = {r["id"] for r in old["results"]}
    pending = [c for c in clips if c["id"] not in done]
    result = old
    for new in generate({"clips": pending}):
        result = dict(new, results=old["results"] + new["results"],
                      wall_sec=round(old["wall_sec"] + time.monotonic() - started, 2), complete=False)
        for row in result["results"]:
            row["unsafe"] = bool(row["token_limit_reached"] or not row["text"] or _REPEAT_RE.search(row["text"]))
        save_json(partial, result)
        transcript_md(folder / "02_Qwen.md", "Qwen 全編（認識原文）", result["results"],
                      "30秒ごとに独立して認識。Whisperとは異なる認識候補。要確認行も削除せず保存。処理完了は02_qwen.jsonのcompleteで確認。")
    result["complete"] = True
    save_json(final, result)
    print(f"Qwen complete: {result['wall_sec']} sec", flush=True)


def gemma(folder):
    from core import config, llm, minutes
    final = folder / "03_gemma.json"
    if final.exists():
        print("Gemma already complete", flush=True)
        return
    started = time.monotonic()
    settings = config.load_settings()
    source = read_json(folder / "02_qwen.json")
    if not source.get("complete"):
        raise ValueError("Qwenの全編処理が完了していません")
    partial = folder / "gemma.partial.json"
    old = read_json(partial) if partial.exists() else {"results": copy.deepcopy(source["results"]), "wall_sec": 0}
    rows = old["results"]
    for row in rows:
        row.setdefault("speaker", "発言者")
        row.setdefault("marked", row["text"])
    result = dict(model=settings["llm"]["ollama_model"], results=rows, complete=False)
    calls = folder / "gemma_calls"
    calls.mkdir(exist_ok=True)
    original_generate = llm.generate

    def audited_generate(settings, system, user_text, think=None):
        call_id = len(list(calls.glob("*-request.json"))) + 1
        save_json(calls / f"{call_id:03d}-request.json",
                  {"model": settings["llm"]["ollama_model"], "system": system, "user": user_text, "think": think})
        call_start = time.monotonic()
        reply = original_generate(settings, system, user_text, think=think)
        save_json(calls / f"{call_id:03d}-response.json",
                  {"text": reply, "elapsed_sec": round(time.monotonic() - call_start, 2)})
        return reply

    llm.generate = audited_generate

    def checkpoint():
        result.update(wall_sec=round(old["wall_sec"] + time.monotonic() - started, 2),
                      processed=sum(bool(r.get("proofread")) for r in rows),
                      changed=sum(a["text"] != b["text"] for a, b in zip(rows, source["results"])))
        save_json(partial, result)
        transcript_md(folder / "03_gemma4.md", "Qwen → gemma4 校正後", rows,
                      "gemma4は文字だけを校正。音声の聞き直しはしない。⟦ ⟧ はモデル自身が推測として印を付けた箇所（印がない箇所の正しさも未確認）。要確認行（空・生成上限・反復）は原文のまま残す。")
        print(f"Gemma saved {result['processed']}/{len(rows)}, changed {result['changed']}", flush=True)

    try:
        minutes.proofread_utterances(settings, rows, skip=lambda r: r.get("unsafe") or r.get("proofread"),
                                     on_chunk=checkpoint)
        checkpoint()
        pending = [r for r in rows if not r.get("unsafe") and not r.get("proofread")]
        result.update(complete=not pending, fallback=llm.pop_fallback_notice())
        if pending:
            save_json(partial, result)
            raise RuntimeError(f"Gemmaの未処理が{len(pending)}行あります。同じコマンドで再開してください。")
        save_json(final, result)
    finally:
        llm.generate = original_generate
        llm.unload_ollama(settings)
    print(f"Gemma complete: {result['wall_sec']} sec", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "whisper", "qwen", "gemma"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--audio", type=Path, help="prepare時の元音声")
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.stage != "prepare" and not (folder / "asr.wav").exists():
        parser.error("asr.wav がありません")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    if args.stage == "prepare":
        prepare(folder, args.audio)
    else:
        globals()[args.stage](folder)


if __name__ == "__main__":
    main()
