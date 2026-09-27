"""output/ フォルダへの構造化出力"""
import datetime
import json
import re
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


def write_text(job_dir: Path, filename: str, content: str) -> Path:
    path = job_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def write_meta(job_dir: Path, meta: dict) -> Path:
    path = job_dir / "meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


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
    path = Path(job_dir) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


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
    """発言データ(utterances.json)を持つ＝アプリで開き直せるジョブ"""
    return [j for j in list_jobs() if (OUTPUT_DIR / j / "utterances.json").exists()]
