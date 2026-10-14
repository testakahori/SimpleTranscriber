"""output/ フォルダへの構造化出力"""
import datetime
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from .config import OUTPUT_DIR


def _sanitize(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.strip()[:60] or "untitled"


def create_job_dir(source_filename: str) -> Path:
    """output/YYYY-MM-DD_HHMM_元ファイル名/ を作成して返す（重複時は連番）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _sanitize(Path(source_filename).stem)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = OUTPUT_DIR / f"{stamp}_{stem}"
    job_dir = base
    n = 2
    while job_dir.exists():
        job_dir = Path(f"{base}_{n}")
        n += 1
    job_dir.mkdir(parents=True)
    return job_dir


def retry_file_operation(action):
    """Windowsで一時的にファイルを開かれている場合だけ短く待つ。容量不足等は即時通知。"""
    for attempt in range(6):
        try:
            return action()
        except OSError as e:
            if getattr(e, "winerror", None) not in (32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def write_text(job_dir: Path, filename: str, content: str) -> Path:
    """同じフォルダの一時ファイルから置換し、書き込み失敗時は前回の保存を残す。"""
    path = Path(job_dir) / filename
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        retry_file_operation(lambda: os.replace(temp, path))
    finally:
        if temp is not None:
            try:
                retry_file_operation(lambda: temp.unlink(missing_ok=True))
            except OSError:
                logging.warning("保存用の一時ファイルを削除できませんでした: %s", temp, exc_info=True)
    return path


def write_meta(job_dir: Path, meta: dict) -> Path:
    return write_text(job_dir, "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def list_jobs() -> list[str]:
    """output/ 配下のジョブフォルダ名を新しい順に返す"""
    if not OUTPUT_DIR.exists():
        return []
    dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.name, reverse=True)
    return [d.name for d in dirs]


def list_md_files(job_name: str) -> list[str]:
    job_dir = OUTPUT_DIR / job_name
    if not job_dir.is_dir():
        return []
    return sorted(f.name for f in job_dir.glob("*.md"))


def job_path(job_name: str) -> Path:
    return OUTPUT_DIR / job_name


def save_json(job_dir: Path, filename: str, data) -> Path:
    return write_text(job_dir, filename, json.dumps(data, ensure_ascii=False))


def load_json(job_dir: Path, filename: str, default=None):
    path = Path(job_dir) / filename
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def list_editable_jobs() -> list[str]:
    """完成した発言データ、または途中保存をアプリで開けるジョブ。"""
    return [j for j in list_jobs() if any((OUTPUT_DIR / j / name).exists()
            for name in ("utterances.json", "checkpoint.json"))]
