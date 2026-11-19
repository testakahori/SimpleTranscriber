"""同じ保存済みWhisper本文を新旧の校正・議事録処理へ渡すローカル比較。

出力は会議内容を含むため公開しない。音声認識の正解率を測るツールではない。
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config, glossary, llm, people, utterances


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(folder, revision):
    """指定コミットの処理本体・全プロンプト・テンプレートを固定する。"""
    oid = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    # gitの日本語パス引用を無効化して取り出す。
    paths = subprocess.check_output(["git", "-c", "core.quotePath=false", "ls-tree", "-r",
                                     "--name-only", oid, "prompts"], cwd=ROOT,
                                    text=True, encoding="utf-8").splitlines()
    for name in ["core/minutes.py", "議事録テンプレート.md"] + paths:
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(["git", "show", oid + ":" + name], cwd=ROOT))
    return oid


def load_version(folder, label):
    spec = importlib.util.spec_from_file_location("core._comparison_" + label, folder / "core/minutes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROMPTS_DIR = folder / "prompts"
    module.MINUTES_TEMPLATE = folder / "議事録テンプレート.md"
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", default="a3e60c4")
    parser.add_argument("--after", default="4ed3789")
    parser.add_argument("--model", help="両群で共通に使う実験用モデル。アプリ設定は変更しない")
    parser.add_argument("--no-fallback", action="store_true", help="別モデルへの自動切り替えを止める")
    parser.add_argument("--batch-size", type=int, help="実験中のOllamaのプロンプト処理バッチ数")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", choices=["proofread", "documents", "all"], default="all")
    args = parser.parse_args()
    folder = args.output.resolve()
    settings = config.load_settings()
    if args.model:
        settings["llm"]["ollama_model"] = args.model
    if args.no_fallback:
        settings["llm"]["ollama_fallback_model"] = ""
    if urlparse(settings["llm"]["ollama_url"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("この比較はローカルOllama専用です。")
    if args.resume:
        manifest = read(folder / "manifest.json")
        settings["llm"] = manifest["llm"]
        if fingerprint(args.audio) != manifest["audio_sha256"]:
            parser.error("再開時の音声が一致しません。")
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "input").mkdir()
        shutil.copy2(args.job / "utterances.json", folder / "input/utterances.json")
        for name in ["meta.json", "議事録.md"]:
            if (args.job / name).exists():
                shutil.copy2(args.job / name, folder / "input" / name)
        for source, name in [(people.PEOPLE_PATH, "people.yaml"), (glossary.GLOSSARY_PATH, "glossary.yaml"),
                             (glossary.LEARNED_PATH, "learned_words.yaml"),
                             (glossary.LEARNED_PATH.with_name("context_examples.yaml"), "context_examples.yaml")]:
            if source.exists():
                shutil.copy2(source, folder / "input" / name)
        manifest = dict(audio=str(args.audio.resolve()), audio_sha256=fingerprint(args.audio),
                        job=str(args.job.resolve()), input_sha256=fingerprint(folder / "input/utterances.json"),
                        llm=settings["llm"], seed=42, batch_size=args.batch_size, versions={},
                        note="既存の同一Whisper本文を使用。辞書とLLM設定を固定。ASR・話者識別は再実行しない。")
        for label, revision in [("before", args.before), ("after", args.after)]:
            manifest["versions"][label] = snapshot(folder / "versions" / label, revision)
        save(folder / "manifest.json", manifest)
    if urlparse(settings["llm"]["ollama_url"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("保存された接続先がローカルではありません。")
    if fingerprint(folder / "input/utterances.json") != manifest["input_sha256"]:
        parser.error("固定した入力本文が変更されています。")
    job = read(folder / "input/utterances.json")
    original_post = llm.httpx.post
    active = {"version": "", "stage": ""}

    def traced_post(url, **kwargs):
        if url.endswith("/api/chat") and "json" in kwargs:
            payload = kwargs["json"]
            payload.setdefault("options", {})["seed"] = manifest["seed"]
            if manifest.get("batch_size"):
                payload["options"]["num_batch"] = manifest["batch_size"]
            logdir = folder / active["version"] / "calls"
            logdir.mkdir(parents=True, exist_ok=True)
            index = len(list(logdir.glob("*.request.json"))) + 1
            stem = f"{index:03d}-{active['stage']}"
            save(logdir / (stem + ".request.json"), payload)
            start = time.monotonic()
            print(f"{active['version']} {active['stage']} request {index}", flush=True)
            try:
                response = original_post(url, **kwargs)
                response_data = response.json()
                save(logdir / (stem + ".response.json"), dict(seconds=time.monotonic() - start,
                     status=response.status_code, response=response_data))
                print(f"{active['version']} {active['stage']} response {index}: {time.monotonic()-start:.1f}s", flush=True)
                if response.status_code == 200 and (not response_data.get("done") or not response_data.get("message", {}).get("content", "").strip()):
                    raise llm.LLMError("Ollamaが完了した本文を返しませんでした。比較を中断して記録を保持します。")
                return response
            except Exception as error:
                save(logdir / (stem + ".error.json"), dict(error=str(error), seconds=time.monotonic()-start))
                raise
        return original_post(url, **kwargs)

    with patch.object(people, "PEOPLE_PATH", folder / "input/people.yaml"), \
            patch.object(glossary, "GLOSSARY_PATH", folder / "input/glossary.yaml"), \
            patch.object(glossary, "LEARNED_PATH", folder / "input/learned_words.yaml"), \
            patch.object(llm.httpx, "post", side_effect=traced_post):
        versions = {label: load_version(folder / "versions" / label, label) for label in manifest["versions"]}
        for stage in (["proofread", "documents"] if args.stages == "all" else [args.stages]):
            for label, module in versions.items():
                active.update(version=label, stage=stage)
                dest = folder / label
                dest.mkdir(exist_ok=True)
                status_file = dest / "status.json"
                status = read(status_file) if status_file.exists() else {}
                if status.get(stage, {}).get("complete"):
                    continue
                start = time.monotonic()
                if stage == "proofread":
                    checkpoint = dest / "utterances.json"
                    rows = read(checkpoint) if checkpoint.exists() else copy.deepcopy(job["utterances"])
                    if not checkpoint.exists():
                        for row in rows:
                            row.pop("proofread", None)
                            row.pop("proofread_pending", None)
                    def progress(fraction):
                        print(f"{label} proofreading {fraction:.1%}", flush=True)
                    module.proofread_utterances(settings, rows, progress_cb=progress,
                        skip=lambda row: row.get("proofread") or row.get("proofread_pending"),
                        on_chunk=lambda: save(checkpoint, rows),
                        memo=job.get("memo", ""))
                    save(checkpoint, rows)
                    (dest / "文字起こし.md").write_text(utterances.render_md(rows, plain=True), encoding="utf-8")
                    status[stage] = dict(complete=True, seconds_this_invocation=time.monotonic()-start,
                        accepted=sum(bool(r.get("proofread")) for r in rows),
                        changed=sum(r["text"] != old["text"] for r, old in zip(rows, job["utterances"])),
                        pending=sum(bool(r.get("proofread_pending")) for r in rows))
                else:
                    rows = read(dest / "utterances.json")
                    transcript = module.transcript_for_llm(rows)
                    draft_file = dest / "議事録_照合前.md"
                    if draft_file.exists():
                        draft = draft_file.read_text(encoding="utf-8")
                    else:
                        active["stage"] = "generate"
                        draft = module.generate_minutes(settings, transcript, job.get("source", ""),
                            job.get("meeting_date", ""), memo=job.get("memo", ""))
                        draft_file.write_text(draft, encoding="utf-8")
                    active["stage"] = "verify"
                    document, report = module.verify_document(settings, transcript, draft,
                        memo=job.get("memo", ""), progress_cb=lambda f, d: print(f"{label} {d}", flush=True))
                    (dest / "議事録.md").write_text(document, encoding="utf-8")
                    save(dest / "verification.json", report)
                    status[stage] = dict(complete=True, seconds=time.monotonic()-start,
                                         verification={k:v for k,v in report.items() if k != "items"})
                status[stage]["fallback"] = llm.pop_fallback_notice()
                save(status_file, status)
                print(label, stage, json.dumps(status[stage], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
