"""全編比較のJSONから、ローカル専用のMarkdown・再生付きHTML・集計JSONを作る。"""
import argparse
import difflib
from datetime import datetime
import html
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.full_asr_experiment import read_json, save_json, stamp


def normal(text):
    return re.sub(r"[\s、。？！?!,.・…]", "", text)


def delta(left, right):
    parts = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            parts.append("<del>" + html.escape(left[a:b]) + "</del>")
        if tag in ("replace", "insert"):
            parts.append("<ins>" + html.escape(right[c:d]) + "</ins>")
        if tag == "equal":
            parts.append(html.escape(left[a:b]))
    return "".join(parts)


def report(folder):
    source = read_json(folder / "source.json")
    whisper = read_json(folder / "01_whisper.json")
    qwen = read_json(folder / "02_qwen.json")
    gemma = read_json(folder / "03_gemma.json")
    clips = read_json(folder / "qwen-request.json")["clips"]
    if not all(x.get("complete") for x in (whisper, qwen, gemma)):
        raise ValueError("未完了のstageがあります。完了後に集計してください。")
    qindex = {r["id"]: r for r in qwen["results"]}
    gindex = {r["id"]: r for r in gemma["results"]}
    if set(qindex) != {c["id"] for c in clips} or set(gindex) != set(qindex):
        raise ValueError("区間IDが一致しません")
    stats = {
        "duration_sec": source["duration"], "window_count": len(clips),
        "preprocess_sec": source["preprocess_sec"],
        "whisper_wall_sec": whisper["wall_sec"], "qwen_wall_sec": qwen["wall_sec"],
        "qwen_generation_sec": round(sum(r["elapsed_sec"] for r in qwen["results"]), 2),
        "gemma_wall_sec": gemma["wall_sec"],
        "whisper_characters": sum(len(s["text"]) for s in whisper["segments"]),
        "qwen_characters": sum(len(r["text"]) for r in qwen["results"]),
        "gemma_characters": sum(len(r["text"]) for r in gemma["results"]),
        "whisper_missing_sec": whisper["missing_sec"], "whisper_refilled_sec": whisper["refilled_sec"],
        "qwen_empty_windows": sum(not r["text"] for r in qwen["results"]),
        "qwen_token_limited_windows": sum(r["token_limit_reached"] for r in qwen["results"]),
        "qwen_unsafe_windows": sum(r["unsafe"] for r in qwen["results"]),
        "gemma_changed_windows": sum(qindex[c["id"]]["text"] != gindex[c["id"]]["text"] for c in clips),
        "gemma_nonpunctuation_changed_windows": sum(normal(qindex[c["id"]]["text"]) != normal(gindex[c["id"]]["text"]) for c in clips),
        "whisper_empty_windows": sum(not c["whisper"] for c in clips),
        "qwen_text_in_whisper_empty_windows": sum(not c["whisper"] and bool(qindex[c["id"]]["text"]) for c in clips),
        "fallback": gemma.get("fallback", ""),
    }
    stats["total_compute_sec"] = round(sum(stats[k] for k in
        ("preprocess_sec", "whisper_wall_sec", "qwen_wall_sec", "gemma_wall_sec")), 2)
    stats["whisper_silence_guard"] = whisper.get("silence_guard", True)
    diagnosis = ""
    stopped_path = folder / "whisper_aborted_silence_guard" / "status.json"
    if stopped_path.exists():
        stopped = read_json(stopped_path)
        stop_time = datetime.fromisoformat(stopped["stopped_at"])
        first_line = (stopped_path.parent / "whisper.log").read_text(encoding="utf-8").splitlines()[0]
        start_time = datetime.strptime(first_line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=stop_time.tzinfo)
        stats["aborted_whisper_sec"] = round((stop_time - start_time).total_seconds(), 2)
        trial_path = folder / "stall_no_silence_guard.json"
        trial = read_json(trial_path) if trial_path.exists() else {}
        stats["diagnostic_clip_sec"] = trial.get("wall_sec", 0)
        stats["all_attempts_compute_sec"] = round(stats["total_compute_sec"] + stats["aborted_whisper_sec"] + stats["diagnostic_clip_sec"], 2)
        diagnosis = f'''\n## 今回見つかった処理の遅延と対処

最初はアプリと同じ `hallucination_silence_threshold=2.0` で全編を開始したが、約{stats['aborted_whisper_sec']/60:.1f}分で診断のため中断した。途中保存の到達位置は{stamp(stopped['processed_sec'])}。モデル内部のVAD圧縮後の位置が `481.24 → 482.24 → 483.24秒` と1秒ずつ進む挙動を3回のスタック取得で観測した。faster-whisperの無音に囲まれた誤認を除く経路が、周辺の30秒窓を繰り返し認識していた。

元音声530〜650秒の2分を切り出し、内部スキップだけ `None` にすると **{trial.get('wall_sec', 0):.1f}秒**で終了。文字のない声の診断値は{trial.get('missing_sec', 0):.1f}秒残ったため、完了したことと精度が高いことは区別する。この短い試験は長い試行と区切り・初期文脈が異なるので、厳密な速度倍率は出さない。

全編の最終結果では `silence_guard=False` を使用した。VAD、ビーム5、主処理の既定6段の温度試行、外側の誤認除外、抜けの再認識は維持。内部スキップを外すと怪しい発言が残る可能性もあるため、**アプリの既定設定は変更していない**。

中断・短区間診断を含めた処理時間の合計は **{stats['all_attempts_compute_sec']/60:.1f}分**。上の表は完了した試行だけの時間。実装・調査・待機の時間は別。中断結果は `whisper_aborted_silence_guard/`、スタックは `whisper-stack*.txt`、短区間試験は `stall_no_silence_guard.json` に保存。
'''
    differences, cards, changed_md = [], [], []
    for clip in clips:
        key = clip["id"]
        q, g = qindex[key], gindex[key]
        changed = q["text"] != g["text"]
        ratio = difflib.SequenceMatcher(None, normal(clip["whisper"]), normal(q["text"]), autojunk=False).ratio()
        differences.append(dict(id=key, start=clip["start"], end=clip["end"],
                                whisper_chars=len(clip["whisper"]), qwen_chars=len(q["text"]),
                                text_similarity=round(ratio, 4), unsafe=q["unsafe"], gemma_changed=changed))
        label = f"{stamp(clip['start'])}–{stamp(clip['end'])}"
        warning = "要確認（空・生成上限・反復）。gemma4の校正対象外。" if q["unsafe"] else "正誤は未確認"
        cards.append(f'<section data-unsafe="{int(q["unsafe"])}" data-changed="{int(changed)}">'
                     f'<h2>{label} <small>{html.escape(key)}</small></h2><p>{warning}</p>'
                     f'<audio controls preload="none" src="clips/{html.escape(Path(clip["path"]).name)}"></audio>'
                     f'<div class="columns"><article><h3>Whisper</h3><p>{html.escape(clip["whisper"] or "（文字なし）")}</p></article>'
                     f'<article><h3>Qwen</h3><p>{html.escape(q["text"] or "（文字なし）")}</p></article>'
                     f'<article><h3>gemma4（Qwenからの差分）</h3><p>{delta(q["text"], g["text"]) or "（文字なし）"}</p></article></div></section>')
        if changed:
            changed_md.append(f"## {label}\n\nQwen:\n\n{q['text']}\n\ngemma4:\n\n{g['text']}\n")
    save_json(folder / "stats.json", stats)
    save_json(folder / "differences.json", differences)
    (folder / "校正変更一覧.md").write_text("# gemma4が変更した区間\n\n変更は正しさを保証しません。推測の印は03_gemma4.md参照。\n\n" + "\n".join(changed_md), encoding="utf-8")
    markup = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>全編ASR比較</title>
<style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#20242a}
section{background:white;padding:20px;margin:18px 0;border-radius:10px}h2{font-size:20px}small{color:#64748b;font-size:14px}
.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}p{white-space:pre-wrap;line-height:1.8;overflow-wrap:anywhere}
audio{width:min(100%,650px)}del{color:#9f1239;background:#ffe4e6}ins{color:#166534;background:#dcfce7;text-decoration:none}
label{margin-right:24px}@media(max-width:850px){.columns{grid-template-columns:1fr}}[hidden]{display:none}</style>
<h1>Whisper → Qwen → gemma4 全編比較</h1>
<p>すべて同じ前処理済み音声。Qwenは30秒窓で全編を独立認識し、gemma4はその文字だけを校正しました。
文字が増えたことは認識精度の改善を意味しません。Qwenの生成上限・反復・空行は校正せず原文を保存しています。
右列の赤い取消線は削除、緑は追加です。音声で正誤を確認してください。</p>
<label><input id="unsafe" type="checkbox">要確認の区間だけ</label><label><input id="changed" type="checkbox">gemma4の変更がある区間だけ</label>
'''
    markup += "".join(cards) + '''<script>
function filter(){for(const s of document.querySelectorAll('section'))s.hidden=(document.querySelector('#unsafe').checked&&s.dataset.unsafe!=='1')||(document.querySelector('#changed').checked&&s.dataset.changed!=='1')}
document.querySelectorAll('input').forEach(x=>x.addEventListener('change',filter));
</script></html>'''
    (folder / "comparison.html").write_text(markup, encoding="utf-8")
    summary = f'''# 全編文字起こし 実験レポート

対象: `{Path(source['source']).name}`（{stamp(source['duration'])}、{source['duration']:.3f}秒）  
実施日: {source['created_at']}  
原音SHA-256: `{source['sha256']}`

## 実行結果

全編の3段階が完了。処理時間の合計は **{stats['total_compute_sec']/60:.1f}分**（前処理を含む。準備・検討・段階間の待ち時間は含まない）。

|段階|処理時間|文字数|保存先|
|---|---:|---:|---|
|Whisper large-v3 accurate/stable＋抜けの再認識|{whisper['wall_sec']/60:.2f}分|{stats['whisper_characters']}|[01_Whisper.md](01_Whisper.md)|
|Qwen3-ASR-1.7B 全編独立認識|{qwen['wall_sec']/60:.2f}分|{stats['qwen_characters']}|[02_Qwen.md](02_Qwen.md)|
|gemma4 校正（Qwenの文字が入力）|{gemma['wall_sec']/60:.2f}分|{stats['gemma_characters']}|[03_gemma4.md](03_gemma4.md)|

- 前処理 {source['preprocess_sec']:.2f}秒。{source['preprocessing']}。
- Whisperで再認識後も文字が付かなかった声の区間は **{whisper['missing_sec']:.1f}秒**。VADによる代理指標であり、欠落した発言の正確な秒数ではない。
- Qwenは全{len(clips)}区間。空文字 {stats['qwen_empty_windows']}、生成上限 {stats['qwen_token_limited_windows']}、空・上限・反復のいずれかで校正対象外 **{stats['qwen_unsafe_windows']}区間**（分類は重複あり）。
- Whisperが空だった30秒窓 {stats['whisper_empty_windows']}のうち、Qwenが何らかの文字を出した窓は{stats['qwen_text_in_whisper_empty_windows']}。声の正しい回復と、雑音からの捏造をこの数だけで区別できない。
- gemma4は **{stats['gemma_changed_windows']}区間**を変更。句読点・空白等を除いても違う区間は{stats['gemma_nonpunctuation_changed_windows']}。
- gemma4設定モデル: `{gemma['model']}`。フォールバック: {gemma.get('fallback') or 'なし'}。
- Qwen GPUテンソル最大割当 {qwen['peak_allocated_mib']} MiB、予約 {qwen['peak_reserved_mib']} MiB。Qwen生成部分だけは{stats['qwen_generation_sec']:.2f}秒（表は読み込み・前処理等込み）。

## 比較の条件と限界

WhisperとQwenへ同じ16kHzモノラル波形を渡した。Whisperは全編の通常APIと既存の抜け再認識（silence_guard={whisper.get('silence_guard', True)}）、Qwenは重複しない30秒窓（最後だけ短い）で処理した。モデル以外に区切り方・デコード方式・用語ヒントも異なるため、モデルだけの厳密な優劣試験ではない。30秒の境目でQwenの語頭・語尾が欠ける可能性もある。

QwenはWhisperの文章を修正するモデルではなく、音声から別の候補を作る。今回は比較のため全編を認識し、Whisperへの自動差し込みはしていない。gemma4はQwenの文字を校正した。音声を聞かないため、認識されなかった発言を確実に復元することはできない。極端な短縮・水増し・内容削除は既存のガードで不採用にした。

人が原音と照合した正解文がないため、CER/WER・精度改善率は未測定。文字数・文字列の一致度・自然な文章であることを正解率に読み替えない。話者分離、声紋学習、用語の自動学習は行っていない。
{diagnosis}

## 保存物と確認方法

- [再生付き3列比較](comparison.html): 全区間の音声、Whisper、Qwen、gemma4の差分。要確認・校正変更で絞り込み可能。
- [校正変更一覧](校正変更一覧.md): gemma4が変更した全区間。
- `01_whisper.json`, `02_qwen.json`, `03_gemma.json`: 全編結果・時刻・計測値・状態。
- `whisper-main.json`: 抜けの再認識に入る直前の結果。`*.partial.json`: 各段階の途中保存。
- `source.json`, `stats.json`, `differences.json`: 出どころ、集計、区間ごとの文字数・一致度。
- `original{Path(source['source']).suffix.lower()}`, `normalized.wav`, `asr.wav`, `clips/`: 元音声のコピー・各前処理段階・比較用全区間。
- `gemma_calls/`: 校正への入力とモデルの返答。採用されなかった変更も調べられる。
- `whisper.log`, `qwen.log`, `gemma.log`: 実行ログ。

結果はローカルのこのフォルダに保存。音声認識・校正はローカル実行。会議の実データはGitに追加していない。
'''
    notes = folder / "評価メモ.md"
    if notes.exists():
        summary += "\n\n" + notes.read_text(encoding="utf-8")
    for extra in sorted(folder.glob("noise_profile*/追加実験レポート.md")):
        summary += f"\n\n追加実験: [ノイズ見本を指定した比較]({extra.relative_to(folder).as_posix()})\n"
    (folder / "実験レポート.md").write_text(summary, encoding="utf-8")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    print(report(p.parse_args().folder.resolve()))
