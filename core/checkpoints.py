"""音声認識の途中結果。完成データと分け、全編完了と取り違えないようにする。"""
import datetime
import copy
import hashlib
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from . import output
from . import utterances as U

FILENAME = "checkpoint.json"
TRANSCRIPT = "文字起こし_途中.md"
SNAPSHOT_TRANSCRIPT = "文字起こし_途中_表示分.md"
READ_ONLY_NOTICE = (
    "⚠️ 途中保存は閲覧・ダウンロード用です。話者識別や補完は未完了の場合があります。"
    "停止した処理は、履歴から開いて「▶ 途中保存から再開」を押してください。"
    "実行中は重ねて再開できません。編集は全編の処理が完了してから行えます。"
)
PHASES = {"prepared": "音声準備完了", "main": "音声認識", "main_complete": "主認識終了・抜けの確認前",
          "recovery": "抜けの補完", "postprocess": "認識終了・話者識別／整形前"}
RESUME_AUDIO = ("resume_source.wav", "resume_asr.wav")


@contextmanager
def job_lock(job_dir: Path):
    """OSが解放するロック。強制終了後に古いロックファイルが残っても再開できる。"""
    with open(job_dir / ".transcription.lock", "a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise ValueError("この文字起こしは実行中です。終了してから再開してください。") from e
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def keep_audio(job_dir: Path, wav: str, asr_wav: str) -> dict:
    """再開に必要な正規化音声・認識音声を保持する。Gradioの一時アップロードに依存しない。"""
    records = {}
    for key, source, name in (("source", wav, RESUME_AUDIO[0]), ("asr", asr_wav, RESUME_AUDIO[1])):
        if key == "asr" and asr_wav == wav:
            records[key] = dict(records["source"])
            continue
        with tempfile.NamedTemporaryFile(dir=job_dir, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(source, "rb") as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            output.retry_file_operation(lambda: os.replace(tmp, job_dir / name))
        finally:
            tmp.unlink(missing_ok=True)
        records[key] = {"filename": name, "sha256": file_hash(job_dir / name)}
    return records


def audio_paths(job_dir: Path, data: dict) -> tuple[str, str]:
    paths = []
    for key in ("source", "asr"):
        record = data.get("resume_audio", {}).get(key, {})
        name = record.get("filename")
        if name not in RESUME_AUDIO:
            raise ValueError("再開用音声の記録が不正です。")
        path = job_dir / name
        if not path.is_file() or file_hash(path) != record.get("sha256"):
            raise ValueError("再開用音声が見つからないか変更されています。元の保存フォルダを復元してください。")
        paths.append(str(path))
    return tuple(paths)


def cleanup_audio(job_dir: Path):
    # 完成後のみ呼ぶ。自分で作った固定名のファイルだけを削除する。
    for name in RESUME_AUDIO:
        (job_dir / name).unlink(missing_ok=True)


def validate(data: dict) -> None:
    """壊れた進捗を再開位置と解釈して、未認識の範囲を飛ばさない。"""
    try:
        if data["phase"] not in PHASES or data.get("schema_version", 1) not in (1, 2):
            raise ValueError()
        duration, position = float(data["duration"]), float(data["processed_sec"])
        if not math.isfinite(duration) or not math.isfinite(position) or not 0 <= position <= duration or duration <= 0:
            raise ValueError()
        if data["phase"] in ("main_complete", "recovery", "postprocess") and position != duration:
            raise ValueError()
        if data["phase"] == "prepared" and (position != 0 or data["segments"]):
            raise ValueError()
        for seg in data["segments"]:
            if not 0 <= float(seg["start"]) <= float(seg["end"]) <= duration + 1:
                raise ValueError()
        if data["phase"] in ("main", "prepared") and any(s["end"] > position + 1 for s in data["segments"]):
            raise ValueError()
        plan = data.get("recovery")
        if plan is not None:
            if not isinstance(plan["next_index"], int) or not 0 <= plan["next_index"] <= len(plan["spans"]):
                raise ValueError()
            for a, b in plan["spans"]:
                if not 0 <= a < b <= duration:
                    raise ValueError()
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("途中保存の進捗データが不正なため、再開できません。") from e


def notice(job: dict) -> str:
    if not job.get("incomplete"):
        return ""
    cp = job.get("checkpoint", {})
    position = U.format_hms(cp.get("processed_sec", 0))
    total = U.format_hms(job.get("duration", 0))
    phase = PHASES.get(cp.get("phase"), "音声認識")
    return (f"⚠️ **途中保存・全編未完了** — {phase} / 処理位置 {position} / 音声全体 {total}"
            f" / 保存日時 {cp.get('saved_at', '不明')}。処理位置は認識できた発言の量ではありません。\n\n"
            + ("⏸ 保存して一時停止しました。「▶ 途中保存から再開」で続行できます。\n\n"
               if cp.get("paused") else "処理エラーで終了しました。\n\n" if cp.get("interrupted") else "")
            + ("この旧形式の保存には音声・設定がありません。同じ元音声を1件だけ選び、"
               "画面の認識設定を当時の値に合わせてから再開してください（同名・同じ長さを確認しますが、"
               "元音声との完全一致は検証できません）。\n\n" if not job.get("resumable_audio") else "")
            + READ_ONLY_NOTICE)


def as_job(job_dir: Path, data: dict) -> dict:
    preview = job_dir / "audio.m4a"
    return {
        "job_dir": str(job_dir), "source": data["source"], "duration": data["duration"],
        "meeting_date": data.get("meeting_date", ""), "memo": data.get("memo", ""),
        "meta_line": "途中保存・全編未完了（話者未確定）", "incomplete": True,
        "resumable_audio": bool(data.get("resume_audio")),
        "checkpoint": {k: data.get(k) for k in ("phase", "processed_sec", "saved_at", "interrupted", "paused")},
        "utterances": U.build_utterances(data.get("segments", []), None),
        "cluster_embeddings": {}, "audio_path": str(preview) if preview.exists() else None,
    }


def markdown(job: dict) -> str:
    return notice(job) + "\n\n" + U.render_md(
        job["utterances"], job["source"], job.get("duration", 0), job.get("meta_line", ""), plain=True)


class Writer:
    """新しいセグメントを受け取った時に保存。通常10秒間隔、段階終了とエラー時は即保存。"""
    def __init__(self, job_dir: Path, source: str, duration: float, meeting_date: str, memo: str, seed=None):
        self.job_dir = job_dir
        self.base = dict(source=source, duration=duration, meeting_date=meeting_date, memo=memo)
        self.latest = copy.deepcopy(seed)
        self.last_saved = float("-inf")

    def __call__(self, snapshot: dict):
        previous = self.latest or {}
        self.latest = {**previous, **snapshot}
        now = time.monotonic()
        if (now - self.last_saved >= 10 or snapshot["phase"] != previous.get("phase")
                or snapshot["phase"] != "main"):
            self._save()
            self.last_saved = now

    def _save(self):
        data = {**self.latest, **self.base, "schema_version": 2 if self.base.get("resume_audio") else 1,
                "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        # JSONが正本。本文の書き込み失敗でも、履歴を開けばJSONから表示・再出力できる。
        output.save_json(self.job_dir, FILENAME, data)
        output.write_text(self.job_dir, TRANSCRIPT, markdown(as_job(self.job_dir, data)))

    def interrupted(self):
        if self.latest is not None:
            self.latest = {**self.latest, "interrupted": True, "paused": False}
            self._save()

    def pause(self):
        # 10秒間隔の制限を使わず、受け取った最新の本文・進捗を確実に保存してから返す。
        self.latest = {**self.latest, "paused": True, "interrupted": False}
        self._save()
