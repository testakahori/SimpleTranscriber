"""合成文でローカルLLMの漢字校正・担当者照合を確認。音声認識の精度試験ではない。"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config, glossary, llm, minutes, people

TEXTS = ["明日は顧客へ家電して、申込内容を確認します。", "店舗で家電製品を購入します。",
         "誰が設定を変更したか追跡できるように、カーサログを保存します。",
         "開始は10時です。", "今回は変更を実施しない方針です。"]
TRANSCRIPT = "[00:00:10] 黒田: 問い合わせには青山と白井が対応します。\n[00:00:20] 黒田: 私は会場の準備をします。"
DRAFT = "【アクションアイテム】\n| タスク | 担当者 | 期限 |\n|---|---|---|\n| 問い合わせに対応（根拠 00:00:10） | 白井と黒田 | 未定 |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="比較用の旧core/minutes.py（信頼できるローカルファイルのみ）")
    parser.add_argument("--baseline-prompts", help="旧コードと同じ版のpromptsフォルダ")
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.baseline_prompts):
        parser.error("比較には --baseline と --baseline-prompts の両方が必要です。")
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    settings = config.load_settings()
    variants = []
    if args.baseline:
        spec = importlib.util.spec_from_file_location("core._accuracy_baseline", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        baseline.PROMPTS_DIR = Path(args.baseline_prompts).resolve()
        variants.append(("before", baseline))
    variants.append(("after", minutes))
    result = {"model": settings["llm"]["ollama_model"], "input": TEXTS, "transcript": TRANSCRIPT, "draft": DRAFT, "runs": []}
    people_data = [{"display_name": n} for n in ("青山", "白井", "黒田")]
    with patch.object(people, "load_people", return_value=people_data), \
            patch.object(glossary, "load_glossary", return_value=["架電", "監査ログ"]), \
            patch.object(glossary, "LEARNED_PATH", destination / "synthetic_corrections.yaml"):
        glossary.save_context_examples([
            dict(before="家電", after="架電", context="電話をかける話。家電製品の家電は直さない。"),
            dict(before="カーサログ", after="監査ログ", context="誰が操作したか追える操作・アクセス記録の話。")])
        for label, module in variants:
            started = time.monotonic()
            print(label + ": proofreading", flush=True)
            rows = [dict(start=i * 10, end=i * 10 + 5, speaker="話者A", text=t, marked=t, words=[])
                    for i, t in enumerate(TEXTS)]
            done, changed = module.proofread_utterances(settings, rows, memo="顧客対応とシステムの操作記録についての会議")
            print(label + ": verifying assignments", flush=True)
            document, report = module.verify_document(settings, TRANSCRIPT, DRAFT)
            fallback = llm.pop_fallback_notice()
            checks = {"phone_kanji": "架電" in rows[0]["text"], "appliance_kept": rows[1]["text"] == TEXTS[1],
                      "audit_log": "監査ログ" in rows[2]["text"], "number_kept": rows[3]["text"] == TEXTS[3],
                      "negation_kept": rows[4]["text"] == TEXTS[4],
                      "correct_pair": "| 青山と白井 |" in document,
                      "wrong_pair_flagged": "| 白井と黒田 |" in document and ("要確認" in document or "確認できず" in document)}
            result["runs"].append(dict(version=label, seconds=round(time.monotonic() - started, 1),
                                        done=done, changed=changed, utterances=rows, document=document,
                                        verification=report, fallback=fallback, checks=checks))
            (destination / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (destination / f"{label}_議事録.md").write_text(document, encoding="utf-8")
            print(json.dumps(dict(version=label, checks=checks), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
