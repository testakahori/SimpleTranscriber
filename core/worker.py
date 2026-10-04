"""GPUを使う処理（Whisper・話者分離・声紋）を別プロセスで動かす。

アプリ本体で一度でもGPUを使うと、モデルを捨てても CUDA の作業領域などで VRAM が約2GB残り、
後の LLM（gemma4 31B）があふれて約7倍遅くなっていた。別プロセスなら処理が終わってプロセスが
抜けた時点で VRAM と RAM が全部OSに返るので、①の後にアプリを再起動しなくてよい。

親: worker.run("core.pipeline:transcribe_files", paths, opts, settings, progress=cb)
子: python -m core.worker <target> <引数のpickle> <結果のpickle>
    進み具合は標準出力に1行ずつ（PROGRESS_TAG 割合 説明）、ログは標準エラー（コンソールに出る）。
"""
import importlib
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

PROGRESS_TAG = "@@progress"
ROOT = Path(__file__).resolve().parent.parent


class WorkerError(RuntimeError):
    pass


def run(target: str, *args, progress=None):
    """target ("モジュール:関数") を別プロセスで実行して戻り値を返す。
    関数は progress=コールバック を受け取る。例外は WorkerError として親に伝える。"""
    with tempfile.TemporaryDirectory(prefix="st_worker_") as tmp:
        arg_path, out_path = Path(tmp) / "args.pkl", Path(tmp) / "result.pkl"
        arg_path.write_bytes(pickle.dumps(args))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
        proc = subprocess.Popen(
            [sys.executable, "-m", "core.worker", target, str(arg_path), str(out_path)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace")
        try:
            for line in proc.stdout:
                if line.startswith(PROGRESS_TAG):
                    _, frac, desc = (line.rstrip("\n").split("\t", 2) + ["", ""])[:3]
                    if progress:
                        try:
                            progress(float(frac), desc)
                        except ValueError:
                            pass
                elif line.strip():  # ライブラリが標準出力に出した文字はログへ
                    logging.info(line.rstrip())
            code = proc.wait()
        except BaseException:  # 画面側で中断された時に子を残さない
            proc.kill()
            raise
        if not out_path.exists():
            raise WorkerError(f"作業用プロセスが異常終了しました（終了コード {code}）。"
                              "メモリ不足の可能性があります。コンソールのログを確認してください。")
        status, value = pickle.loads(out_path.read_bytes())
        if status == "error":
            raise WorkerError(value)
        return value


def _main(target: str, arg_path: str, out_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [worker] %(message)s")
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass

    def progress(frac, desc=""):
        desc = str(desc).replace("\n", " ").replace("\t", " ")
        print(f"{PROGRESS_TAG}\t{float(frac):.4f}\t{desc}", flush=True)

    try:
        mod, fn = target.split(":")
        func = getattr(importlib.import_module(mod), fn)
        args = pickle.loads(Path(arg_path).read_bytes())
        data = pickle.dumps(("ok", func(*args, progress=progress)))
    except Exception as e:
        logging.exception(f"{target} に失敗しました")
        data = pickle.dumps(("error", f"{type(e).__name__}: {e}"))
    Path(out_path).write_bytes(data)


if __name__ == "__main__":
    _main(*sys.argv[1:4])
