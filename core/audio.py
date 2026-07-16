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


def prepare_audio(input_path: str, noise_reduction: bool = False) -> str:
    """入力（音声/動画）を 16kHz モノラル wav に変換して一時ファイルパスを返す。
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

    if noise_reduction:
        try:
            out_path = _denoise(out_path)
        except Exception as e:
            logging.warning(f"ノイズ除去に失敗したためスキップします: {e}")

    return out_path


def _denoise(wav_path: str) -> str:
    """DeepFilterNet があれば優先、無ければ noisereduce でスペクトラルゲート。
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
    data, rate = sf.read(wav_path)
    reduced = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.9)
    sf.write(wav_path, reduced, rate)
    logging.info("ノイズ除去: noisereduce を使用しました")
    return wav_path


def get_duration(wav_path: str) -> float:
    import soundfile as sf
    try:
        info = sf.info(wav_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0
