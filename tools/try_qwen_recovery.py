"""Whisper → Qwenの聞き直し → gemma4の校正を、別プロセスで順番に試す。

実録音・候補・レポートは指定のローカル出力先だけに保存する。
元の文字起こしや学習データへは適用しない。Qwenの候補は音声との照合が必要。
"""
import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_clips(missing, segments, duration, limit=16):
    """抜け・低確信度・反復を対象にする。短い窓と前後2秒で計算量を限定する。"""
    targets = [(a, b, "文字のない声") for a, b in missing]
    repeat = re.compile(r"(\S{1,8}?)(?:\s*\1){3,}")
    for s in segments:
        words = s.get("words") or []
        probs = [float(w["prob"]) for w in words if "prob" in w]
        if repeat.search(s["text"]):
            targets.append((s["start"], s["end"], "繰り返し"))
        elif len(probs) >= 3 and sum(p < 0.5 for p in probs) / len(probs) >= 0.5:
            targets.append((s["start"], s["end"], "Whisperの確信度が低い"))
    targets.sort()
    merged = []
    for a, b, reason in targets:
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
            continue
        a, b = max(0, a - 2), min(duration, b + 2)
        if merged and a <= merged[-1][1] and b - merged[-1][0] <= 30:
            merged[-1][1] = max(b, merged[-1][1])
            if reason not in merged[-1][2]:
                merged[-1][2] += "・" + reason
        else:
            merged.append([a, b, reason])
    windows = []
    for a, b, reason in merged:
        while a < b and len(windows) < limit:
            end = min(a + 30, b)
            windows.append({"start": round(a, 3), "end": round(end, 3), "reason": reason})
            a = end
    return windows


def whisper_text(segments, start, end):
    """Qwenへ渡す同じ窓に中心時刻が入るWhisper単語を表示する。"""
    pieces = []
    for s in segments:
        if s.get("words"):
            pieces.extend(w["word"] for w in s["words"] if start <= (w["start"] + w["end"]) / 2 < end)
        elif s["start"] < end and s["end"] > start:
            pieces.append(s["text"])
    return "".join(pieces).strip()


def gpu_memory():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, check=True)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_worker(python, backend, request, output, timeout):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_TELEMETRY="1")
    # 子の終了を待つ。次のモデルをロードする前にCUDAの作業領域も返る。
    command = [str(python), str(Path(__file__).with_name("asr_trial_worker.py")), backend,
               str(request), str(output)]
    proc = subprocess.Popen(command, env=env, cwd=ROOT)
    try:
        code = proc.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if proc.poll() is None:
            if os.name == "nt":  # venvのランチャーの子にも実際のPythonがいる
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.kill()
            proc.wait()
        raise


def render_report(folder, clips, qwen, gemma, stats):
    qindex = {x["id"]: x for x in qwen["results"]}
    gindex = {x["id"]: x for x in gemma.get("results", [])}
    esc = html.escape
    cards = []
    gemma_time = f"{gemma['elapsed_sec']:.1f}秒" if "elapsed_sec" in gemma else "未実行"
    for c in clips:
        q = qindex[c["id"]]
        warnings = "生成上限に達したため、途中で切れている可能性があります。" if q["token_limit_reached"] else ""
        cards.append(f'<section><h2>{c["id"]}: {c["start"]:.1f}〜{c["end"]:.1f}秒 — {esc(c["reason"])}</h2>'
                     f'<audio controls preload="none" src="clips/{esc(Path(c["path"]).name)}"></audio>'
                     f'<div class="columns"><article><h3>Whisper</h3><p>{esc(c["whisper"] or "（文字なし）")}</p></article>'
                     f'<article><h3>Qwenの候補</h3><p>{esc(q["text"] or "（文字なし）")}</p><strong>{warnings}</strong></article>'
                     f'<article><h3>gemma4校正後の候補</h3><p>{esc(gindex.get(c["id"], {}).get("text", "（未実行）"))}</p></article></div></section>')
    report = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Qwen聞き直し比較</title>' \
        '<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;background:#f5f5f5;color:#222}' \
        'section{background:white;padding:20px;margin:20px 0;border-radius:10px}h2{font-size:18px}' \
        '.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}p{white-space:pre-wrap;line-height:1.8}' \
        'strong{color:#a00}audio{width:min(100%,600px)}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>' \
        '<h1>Whisper → Qwen → gemma4 比較</h1><p>候補はすべて未確認です。音声と照合して評価してください。' \
        '元の文字起こしは変更していません。空白が文章で埋まっても、正しく聞き取れた証明にはなりません。</p>' \
        f'<p>{len(clips)}区間 ／ Qwen生成 {stats["qwen_inference_sec"]:.1f}秒（読み込み・前処理を除く） ／ ' \
        f'gemma4校正 {gemma_time} ／ Qwen最大GPU割当 {qwen["peak_allocated_mib"]} MiB</p>' + ''.join(cards) + '</html>'
    (folder / "comparison.html").write_text(report, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True, type=Path, help="16kHzモノラルの前処理済みWAV")
    p.add_argument("--baseline", type=Path, help="同じWAVのWhisper JSON。省略時は別プロセスで認識")
    p.add_argument("--output", required=True, type=Path, help="存在しない出力フォルダ")
    p.add_argument("--max-clips", type=int, default=16)
    p.add_argument("--control", action="append", default=[], help="比較用の追加区間。開始秒:終了秒")
    p.add_argument("--gemma", action="store_true", help="Qwen終了後にgemma4で候補を校正")
    args = p.parse_args()
    import soundfile as sf
    from core import config, llm, transcriber
    info = sf.info(args.audio)
    if info.samplerate != 16000 or info.channels != 1:
        p.error("音声は16kHz・モノラルの前処理済みWAVにしてください。")
    if not 1 <= args.max_clips <= 100:
        p.error("--max-clipsは1〜100にしてください。")
    folder = args.output.resolve()
    if folder.exists():
        p.error("上書きを防ぐため、新しい出力フォルダを指定してください。")
    qpython = ROOT / ".venv-qwen" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not qpython.exists():
        p.error(".venv-qwen がありません。docs/Qwen導入と比較.mdを参照してください。")
    folder.mkdir(parents=True)
    llm.unload_ollama(config.load_settings())
    before = gpu_memory()
    if args.baseline:
        data = json.loads(args.baseline.read_text(encoding="utf-8"))
    else:
        write_json(folder / "whisper-request.json", {"audio": str(args.audio.resolve())})
        run_worker(sys.executable, "whisper", folder / "whisper-request.json", folder / "whisper.json", 1800)
        data = json.loads((folder / "whisper.json").read_text(encoding="utf-8"))
    segments = data.get("segments", data.get("utterances"))
    if segments is None:
        p.error("baselineにsegmentsまたはutterancesが必要です。")
    missing = transcriber._missed_spans(str(args.audio), segments, info.duration)
    clips = select_clips(missing, segments, info.duration, args.max_clips)
    for control in args.control:
        start, end = map(float, control.split(":"))
        if not 0 <= start < end <= info.duration or end - start > 30:
            p.error("比較用の区間は録音内の30秒以内にしてください。")
        clips.append({"start": start, "end": end, "reason": "比較用の区間"})
    clips.sort(key=lambda c: c["start"])
    if not clips:
        print("聞き直し対象がありません。--controlで比較する区間を指定できます。")
        return
    (folder / "clips").mkdir()
    for i, c in enumerate(clips):
        c["id"] = f"clip-{i + 1:02d}"
        c["path"] = str(folder / "clips" / (c["id"] + ".wav"))
        audio, rate = sf.read(args.audio, start=int(c["start"] * 16000), stop=int(c["end"] * 16000), dtype="float32")
        sf.write(c["path"], audio, rate, subtype="PCM_16")
        c["whisper"] = whisper_text(segments, c["start"], c["end"])
    write_json(folder / "request.json", {"clips": clips})
    print(f"Whisperの残る抜け {sum(b-a for a,b in missing):.1f}秒、比較対象 {len(clips)}区間", flush=True)
    before_qwen = gpu_memory()
    qwen_started = time.monotonic()
    run_worker(qpython, "qwen", folder / "request.json", folder / "qwen.json", 1800)
    qwen_total = time.monotonic() - qwen_started
    after_qwen = gpu_memory()
    qwen_result = json.loads((folder / "qwen.json").read_text(encoding="utf-8"))
    gemma_result = {}
    stats = {"baseline_missing_sec": round(sum(b-a for a,b in missing), 1),
             "qwen_inference_sec": round(sum(x["elapsed_sec"] for x in qwen_result["results"]), 2),
             "qwen_total_sec": round(qwen_total, 2),
             "gpu_before_mib": before, "gpu_before_qwen_mib": before_qwen, "gpu_after_qwen_mib": after_qwen}
    # 校正に失敗しても、Qwenまでの比較レポートは残す。
    render_report(folder, clips, qwen_result, gemma_result, stats)
    write_json(folder / "stats.json", stats)
    if args.gemma:
        print("Qwenプロセス終了。gemma4で候補を校正します。", flush=True)
        run_worker(sys.executable, "gemma", folder / "qwen.json", folder / "gemma.json", 1800)
        gemma_result = json.loads((folder / "gemma.json").read_text(encoding="utf-8"))
    stats["gpu_after_gemma_mib"] = gpu_memory()
    write_json(folder / "stats.json", stats)
    render_report(folder, clips, qwen_result, gemma_result, stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    print(str(folder / "comparison.html"), flush=True)


if __name__ == "__main__":
    main()
