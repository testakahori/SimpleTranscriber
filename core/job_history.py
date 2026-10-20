"""編集前の本文を保存し、古い画面からの上書きを防ぐ。履歴は会議フォルダ内だけに置く。"""
import datetime
import hashlib
import json
import logging
import uuid
from pathlib import Path

from . import checkpoints, output


class ConflictError(ValueError):
    pass


def revision(folder: Path) -> str | None:
    path = Path(folder) / "utterances.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load(folder: Path):
    path = folder / "utterances.json"
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def ensure_current(job: dict) -> None:
    if "_revision" in job and job["_revision"] != revision(Path(job["job_dir"])):
        raise ConflictError("別の画面でこの結果が更新されています。編集内容を控え、「保存済みの内容に戻す」で最新の本文を読み直してください。")


def save(job: dict, writer, action="発言の修正", backup=True) -> None:
    folder = Path(job["job_dir"])
    with checkpoints.job_lock(folder):
        ensure_current(job)
        path = folder / "utterances.json"
        if path.exists() and backup:
            # 壊れた完成データも黙って上書きしない。
            previous = json.loads(path.read_text(encoding="utf-8"))
            history = folder / ".history"
            history.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().astimezone()
            record = {
                "created_at": stamp.isoformat(timespec="seconds"), "action": action,
                "data": previous,
                "cluster_embeddings": output.load_json(folder, "speaker_clusters.json", {}) or {},
            }
            output.save_json(history, stamp.strftime("%Y%m%d_%H%M%S_%f_") + uuid.uuid4().hex[:8] + ".json", record)
        # 完成データの公開前に派生ファイルの更新が失敗した場合も、前回の組を復元する。
        names = ("utterances.json", "speaker_clusters.json", "transcript.md", "文字起こし.md",
                 "transcript.srt", "transcript.vtt")
        before = {name: (folder / name).read_bytes().decode("utf-8")
                  for name in names if (folder / name).exists()}
        try:
            writer(job)
        except Exception:
            for name in names:
                try:
                    if name in before:
                        output.write_text(folder, name, before[name])
                    else:
                        (folder / name).unlink(missing_ok=True)
                except OSError:
                    logging.exception("保存失敗後の復旧が必要です。編集前の本文は .history に保持: %s", name)
            raise
        job["_revision"] = revision(folder)


def choices(job: dict | None) -> list[tuple[str, str]]:
    if not job or job.get("incomplete"):
        return []
    history = Path(job["job_dir"]) / ".history"
    rows = []
    for path in sorted(history.glob("*.json"), reverse=True):
        record = output.load_json(history, path.name, {}) or {}
        if "data" in record:
            rows.append((f"{record.get('created_at', '')} ／ {record.get('action', '編集')} の前", path.name))
    return rows


def restore(job: dict, name: str, writer) -> dict:
    if name not in {value for _, value in choices(job)}:
        raise ValueError("復元する編集履歴を選んでください。")
    record = output.load_json(Path(job["job_dir"]) / ".history", name)
    restored = {**record["data"], "job_dir": job["job_dir"], "audio_path": job.get("audio_path"),
                "cluster_embeddings": record["cluster_embeddings"], "_revision": job.get("_revision")}
    save(restored, writer, action="編集履歴からの復元")
    return restored
