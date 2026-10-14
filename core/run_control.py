"""画面から作業用プロセスへ、一時停止の要求だけを渡す。"""
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

_active: dict[str, Path] = {}
_lock = threading.Lock()


class PauseRequested(Exception):
    """保存可能な境界で呼び出し元に制御を返す。処理エラーとは区別する。"""


def requested(pause_file: str | None) -> bool:
    return bool(pause_file and Path(pause_file).is_file())


def check(pause_file: str | None) -> None:
    if requested(pause_file):
        raise PauseRequested()


@contextmanager
def session(owner: str | None):
    """実行ごとに新しい要求ファイルを使う。別の画面セッションの処理は止めない。"""
    with tempfile.TemporaryDirectory(prefix="st_pause_") as directory:
        flag = Path(directory) / "pause.request"
        with _lock:
            if owner and owner in _active:
                raise ValueError("この画面では文字起こしが実行中です。終了してから実行してください。")
            if owner:
                _active[owner] = flag
        try:
            yield str(flag)
        finally:
            with _lock:
                if owner:
                    _active.pop(owner, None)


def request_pause(owner: str | None) -> bool:
    with _lock:
        flag = _active.get(owner)
        if flag is None:
            return False
        flag.touch(exist_ok=True)
        return True
