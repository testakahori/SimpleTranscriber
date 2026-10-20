"""議事録・要約に使った入力を記録し、文字起こしとの食い違いを知らせる。"""
import datetime
import hashlib
import json
import threading
from pathlib import Path

from . import minutes, output

DOCUMENTS = ("議事録.md", "要約と考察.md")
INDEX = "documents.json"
_SAVE_LOCK = threading.Lock()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs(job: dict) -> dict:
    """生成開始時に値を固定する。校正済みフラグなど、AIに渡さない値は含めない。"""
    transcript = minutes.transcript_for_llm(job["utterances"])
    if job.get("memo"):
        transcript = f"（会議メモ: {job['memo']}）\n\n" + transcript
    return {"transcript": transcript, "source": job["source"], "meeting_date": job.get("meeting_date", "")}


def fingerprint(source: dict) -> str:
    return digest(json.dumps(source, sort_keys=True, ensure_ascii=False))


def save(job_dir: Path, filename: str, content: str, source: dict) -> None:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # 議事録と要約を並行生成しても、片方の管理情報を消さない。
    with _SAVE_LOCK:
        records = output.load_json(job_dir, INDEX, {})
        records = records if isinstance(records, dict) else {}
        records[filename] = {"input_hash": fingerprint(source), "content_hash": digest(content),
                             "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        output.write_text(job_dir, filename, content)
        # 本文と管理情報の片方だけが更新された場合も、内容ハッシュの不一致で「不明」にする。
        output.save_json(job_dir, INDEX, records)


def status(job: dict, filename: str) -> str:
    path = Path(job["job_dir"]) / filename
    if filename not in DOCUMENTS or not path.exists():
        return "missing"
    records = output.load_json(path.parent, INDEX, {})
    record = records.get(filename, {}) if isinstance(records, dict) else {}
    if not isinstance(record, dict):
        return "unknown"
    if not record.get("input_hash") or record.get("content_hash") != digest(path.read_text(encoding="utf-8")):
        return "unknown"
    return "current" if record["input_hash"] == fingerprint(inputs(job)) else "stale"


def warning(job: dict, filename: str) -> str:
    state = status(job, filename)
    label = Path(filename).stem
    if state == "stale":
        return (f"⚠️ **{label}は更新前の内容です。** 作成後に文字起こし・話者などが変更されています。"
                f"「{label}を作り直す」を押して作り直してください。")
    if state == "unknown":
        return (f"⚠️ **{label}の更新状況を確認できません。** 作成時の記録がないか、文書が外部で変更されています。"
                "現在の文字起こしと一致するとは限りません。必要に応じて作り直してください。")
    return ""
