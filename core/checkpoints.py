"""音声認識の途中結果。完成データと分け、全編完了と取り違えないようにする。"""
import datetime
import time
from pathlib import Path

from . import output
from . import utterances as U

FILENAME = "checkpoint.json"
TRANSCRIPT = "文字起こし_途中.md"
SNAPSHOT_TRANSCRIPT = "文字起こし_途中_表示分.md"
READ_ONLY_NOTICE = (
    "⚠️ 途中保存は閲覧・ダウンロード用です。話者識別や補完は未完了の場合があります。"
    "処理中なら完了後に履歴から開き直してください。停止した処理の続きからの再開には未対応です。"
    "全編を作る場合は、元音声を選んで最初から実行してください。"
)
PHASES = {"main": "音声認識", "main_complete": "主認識終了・抜けの確認前",
          "recovery": "抜けの補完", "postprocess": "認識終了・話者識別／整形前"}


def notice(job: dict) -> str:
    if not job.get("incomplete"):
        return ""
    cp = job.get("checkpoint", {})
    position = U.format_hms(cp.get("processed_sec", 0))
    total = U.format_hms(job.get("duration", 0))
    phase = PHASES.get(cp.get("phase"), "音声認識")
    return (f"⚠️ **途中保存・全編未完了** — {phase} / 処理位置 {position} / 音声全体 {total}"
            f" / 保存日時 {cp.get('saved_at', '不明')}。処理位置は認識できた発言の量ではありません。\n\n"
            + ("処理エラーで終了しました。\n\n" if cp.get("interrupted") else "")
            + READ_ONLY_NOTICE)


def as_job(job_dir: Path, data: dict) -> dict:
    preview = job_dir / "audio.m4a"
    return {
        "job_dir": str(job_dir), "source": data["source"], "duration": data["duration"],
        "meeting_date": data.get("meeting_date", ""), "memo": data.get("memo", ""),
        "meta_line": "途中保存・全編未完了（話者未確定）", "incomplete": True,
        "checkpoint": {k: data.get(k) for k in ("phase", "processed_sec", "saved_at", "interrupted")},
        "utterances": U.build_utterances(data.get("segments", []), None),
        "cluster_embeddings": {}, "audio_path": str(preview) if preview.exists() else None,
    }


def markdown(job: dict) -> str:
    return notice(job) + "\n\n" + U.render_md(
        job["utterances"], job["source"], job.get("duration", 0), job.get("meta_line", ""), plain=True)


class Writer:
    """新しいセグメントを受け取った時に保存。通常10秒間隔、段階終了とエラー時は即保存。"""
    def __init__(self, job_dir: Path, source: str, duration: float, meeting_date: str, memo: str):
        self.job_dir = job_dir
        self.base = dict(source=source, duration=duration, meeting_date=meeting_date, memo=memo)
        self.latest = None
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
        data = {**self.base, **self.latest, "schema_version": 1,
                "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        # JSONが正本。本文の書き込み失敗でも、履歴を開けばJSONから表示・再出力できる。
        output.save_json(self.job_dir, FILENAME, data)
        output.write_text(self.job_dir, TRANSCRIPT, markdown(as_job(self.job_dir, data)))

    def interrupted(self):
        if self.latest is not None:
            self.latest = {**self.latest, "interrupted": True}
            self._save()
