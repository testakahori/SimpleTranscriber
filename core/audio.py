"""ffmpegによる音声・動画の前処理とノイズ除去"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".ts", ".m4v", ".webm"}
SUPPORTED_EXTS = sorted(AUDIO_EXTS | VIDEO_EXTS)


class AudioError(Exception):
    pass


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise AudioError(
            "ffmpeg が見つかりません。https://ffmpeg.org からインストールするか、"
            "`pip install imageio-ffmpeg` を実行してください。"
        )


def prepare_audio(input_path: str) -> str:
    """入力（音声/動画）を 16kHz モノラル・音量を揃えた wav に変換して一時ファイルパスを返す。
    話者分離・声紋登録/照合はこの音声を使う（文字起こし用の補正は enhance_for_asr）。
    音声トラックが無い場合はエラーメッセージ付きで失敗する。"""
    ffmpeg = find_ffmpeg()
    src = Path(input_path)
    if not src.exists():
        raise AudioError(f"ファイルが見つかりません: {input_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        # 声の大きさを揃える（マイクから遠い人の小さな声を持ち上げる）。
        # 実会議録音で認識できる文字数が約12%増えた
        "-af", "dynaudnorm=f=250:g=15:p=0.9",
        "-acodec", "pcm_s16le",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        Path(out_path).unlink(missing_ok=True)
        if "does not contain any stream" in stderr or "Output file does not contain" in stderr:
            raise AudioError(f"音声トラックが見つかりません: {src.name}")
        raise AudioError(f"音声変換に失敗しました ({src.name}): {stderr[-500:]}")

    return out_path


# Whisper用イコライザー: 空調・机の振動などの低音(80Hz未満)と、声に無い高域(7kHz超)を落とす
ASR_EQ = "highpass=f=80,lowpass=f=7000"


def enhance_for_asr(wav_path: str, eq: bool = True, noise_reduction: bool = False) -> str:
    """文字起こし(Whisper)専用に聞き取りやすくした別ファイルを作って返す。
    話者分離・声紋照合は声質を変えないよう prepare_audio の音声をそのまま使うこと。
    何もしない設定なら元のパスを返す（呼び出し側は戻り値 != wav_path の時だけ削除する）。
    時間軸は変えない（無音を切らない）ので、タイムスタンプは元音声とそのまま一致する。"""
    if not (eq or noise_reduction):
        return wav_path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name
    try:
        if eq:
            proc = subprocess.run(
                [find_ffmpeg(), "-y", "-i", wav_path, "-af", ASR_EQ, "-acodec", "pcm_s16le", out_path],
                capture_output=True, text=True, errors="replace")
            if proc.returncode != 0:
                raise AudioError(proc.stderr[-300:])
        else:
            shutil.copyfile(wav_path, out_path)
        if noise_reduction:
            _denoise(out_path)
        return out_path
    except Exception as e:
        logging.warning(f"文字起こし用の音声補正に失敗したため元の音声を使います: {e}")
        Path(out_path).unlink(missing_ok=True)
        return wav_path


def _denoise(wav_path: str) -> str:
    """DeepFilterNet があれば優先（全体を一度に・既定の強さで処理）、無ければ noisereduce で
    弱めのスペクトラルゲート。
    どちらも失敗したら元ファイルを返す（呼び出し側でwarning済み）。"""
    # 1) DeepFilterNet（高品質・任意インストール）
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        audio, _ = load_audio(wav_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(wav_path, enhanced, df_state.sr())
        logging.info("ノイズ除去: DeepFilterNet を使用しました")
        return wav_path
    except ImportError:
        pass

    # 2) noisereduce（標準同梱）
    import noisereduce as nr
    import soundfile as sf
    import numpy as np
    info = sf.info(wav_path)
    rate, total = info.samplerate, info.frames
    # 長時間録音でもメモリを食わないよう5分ずつ処理（前後2秒の余白を付けて継ぎ目を目立たなくする）
    block, pad = rate * 300, rate * 2
    out = np.empty(total, dtype=np.float32)
    for start in range(0, total, block):
        a, b = max(0, start - pad), min(total, start + block + pad)
        data, _ = sf.read(wav_path, start=a, stop=b, dtype="float32")
        # 強く掛けると声まで削れて認識文字数が約7%減った（実会議録音）。控えめに掛ける
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.5)
        end = min(start + block, total)
        out[start:end] = reduced[start - a:start - a + (end - start)]
    sf.write(wav_path, out, rate, subtype="PCM_16")
    logging.info("ノイズ除去: noisereduce を使用しました")
    return wav_path


def get_duration(wav_path: str) -> float:
    import soundfile as sf
    try:
        info = sf.info(wav_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def make_preview_audio(input_path: str, out_path: str) -> str | None:
    """ブラウザ再生用の軽量音声（AAC 48kbps モノラル）を作る。失敗時は None。"""
    try:
        ffmpeg = find_ffmpeg()
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "24000",
               "-c:a", "aac", "-b:a", "48k", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode == 0 and Path(out_path).exists():
            return str(out_path)
        logging.warning(f"再生用音声の作成に失敗: {proc.stderr[-300:]}")
    except Exception as e:
        logging.warning(f"再生用音声の作成に失敗: {e}")
    return None
