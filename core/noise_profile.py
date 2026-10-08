"""人が選んだノイズ見本による、時間軸を変えない弱いスペクトルゲート。"""
import json
import math
from pathlib import Path
import shutil
import time
import unicodedata


def seconds(value):
    """秒、分:秒、時:分:秒を受け付ける。"""
    parts = unicodedata.normalize("NFKC", str(value)).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("時刻は秒数、分:秒、時:分:秒で指定してください")
    try:
        values = [float(x) for x in parts]
    except ValueError:
        raise ValueError("時刻は秒数、分:秒、時:分:秒で指定してください") from None
    if any(not math.isfinite(x) or x < 0 for x in values) or any(x >= 60 for x in values[1:]):
        raise ValueError("時刻の範囲が不正です")
    return sum(x * 60 ** i for i, x in enumerate(reversed(values)))


def reduce_with_profile(wav_path, output_path, residue_path, start, end, strength=0.35, progress=None):
    """同じ波形の指定区間からノイズ統計を取得。除去音も別途保存して発言の損失を確認できる。"""
    import numpy as np
    import noisereduce as nr
    import soundfile as sf

    src, dst, residue = map(lambda p: Path(p).resolve(), (wav_path, output_path, residue_path))
    if len({src, dst, residue}) != 3 or dst.exists() or residue.exists():
        raise ValueError("元音声や既存の結果を上書きできません。新しい保存先を指定してください")
    info = sf.info(src)
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError("16kHzモノラルWAVを指定してください")
    start, end, strength = seconds(start), seconds(end), float(strength)
    if not 0 <= start < end <= info.duration or not 0.5 <= end - start <= 30:
        raise ValueError("ノイズ見本は録音内の0.5〜30秒の区間を選んでください")
    if not math.isfinite(strength) or not 0 <= strength <= 0.8:
        raise ValueError("除去の強さは0〜0.8で指定してください")
    rate = info.samplerate
    noise, _ = sf.read(src, start=round(start * rate), stop=round(end * rate), dtype="float32")
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms < 1e-7:
        raise ValueError("選択範囲が無音です。実際にノイズが聞こえる区間を選んでください")
    # 30秒ごとに前後2秒を付け、余白を除いた元の長さだけ保存する。
    # 見本は全ブロックで同じものを使い、長時間録音もメモリに全展開しない。
    block, pad = 30 * rate, 2 * rate
    partial = dst.with_name(dst.stem + ".partial.wav")
    residue_partial = residue.with_name(residue.stem + ".partial.wav")
    started, original_power, reduced_power = time.monotonic(), 0.0, 0.0
    with sf.SoundFile(partial, "w", samplerate=rate, channels=1, subtype="PCM_16") as out, \
            sf.SoundFile(residue_partial, "w", samplerate=rate, channels=1, subtype="PCM_16") as removed:
        for offset in range(0, info.frames, block):
            lo, hi = max(0, offset - pad), min(info.frames, offset + block + pad)
            data, _ = sf.read(src, start=lo, stop=hi, dtype="float32")
            filtered = nr.reduce_noise(y=data, sr=rate, y_noise=noise, stationary=True,
                                       prop_decrease=strength, use_torch=False,
                                       clip_noise_stationary=False)
            length = min(block, info.frames - offset)
            original = data[offset - lo:offset - lo + length]
            reduced = np.clip(filtered[offset - lo:offset - lo + length], -1, 1)
            out.write(reduced)
            removed.write(np.clip(original - reduced, -1, 1))
            original_power += float(np.sum(original.astype("float64") ** 2))
            reduced_power += float(np.sum(reduced.astype("float64") ** 2))
            if progress:
                progress((offset + length) / info.frames)
    partial.replace(dst)
    residue_partial.replace(residue)
    return dict(start=start, end=end, strength=strength, duration=info.duration, frames=info.frames,
                noise_rms_db=round(20 * math.log10(noise_rms), 2),
                overall_level_change_db=round(10 * math.log10(max(reduced_power, 1e-20) / max(original_power, 1e-20)), 2),
                elapsed_sec=round(time.monotonic() - started, 2),
                method="noisereduce stationary y_noise; CPU; same profile for all blocks")


def create_trial(source, start, end, folder, strength=0.35, progress=None):
    """元音声から試験フォルダを新規作成。既存の自動ノイズ除去を重ねない。"""
    from . import audio
    import soundfile as sf

    source, folder = Path(source), Path(folder).resolve()
    if not source.is_file():
        raise ValueError("元音声を指定してください")
    # 時刻の書式を、大きい音声を変換する前に確認する。
    start, end = seconds(start), seconds(end)
    folder.mkdir(parents=True, exist_ok=False)
    normalized = audio.prepare_audio(str(source))
    eq = audio.enhance_for_asr(normalized, eq=True, noise_reduction=False)
    shutil.copy2(eq, folder / "before.wav")
    meta = reduce_with_profile(folder / "before.wav", folder / "reduced.wav", folder / "removed.wav",
                               start, end, strength, progress)
    noise, rate = sf.read(folder / "before.wav", start=round(start * 16000), stop=round(end * 16000))
    sf.write(folder / "noise_sample.wav", noise, rate, subtype="PCM_16")
    meta.update(source=str(source.resolve()), preprocessing="dynaudnorm + EQ80-7000; no automatic denoiser",
                sample_label="user-selected noise candidate; no automatic speaker/noise ground truth")
    (folder / "profile.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "確認方法.md").write_text(
        "# ノイズ見本による除去の確認\n\n"
        "before.wav（除去前）とreduced.wav（除去後）を聞き比べてください。\n\n"
        "removed.wavは削った音です。ここに説明者の言葉が聞こえたら、除去を弱めるか見本の範囲を変えてください。"
        "削った音をすべてノイズと判定したわけではありません。\n\n"
        "人のざわめきは説明者の声と重なるため、完全には分離できません。元音声は変更していません。"
        "文字起こしの改善は、同じ区間の比較と原音の確認で評価します。\n",
        encoding="utf-8")
    return dict(folder=str(folder), **meta)
