"""試験用の独立プロセス。バックエンドごとに終了してGPUメモリをOSへ返す。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def qwen(request):
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Qwenの比較試験にはCUDA対応GPUが必要です。")
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    cached_config = try_to_load_from_cache(model_id, "config.json")
    revision = Path(cached_config).parent.name if isinstance(cached_config, str) else None
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        local_files_only=True).eval()
    load_sec = time.monotonic() - started
    torch.cuda.reset_peak_memory_stats()
    results = []
    for i, clip in enumerate(request["clips"]):
        print(f"Qwen {i + 1}/{len(request['clips'])}", flush=True)
        inputs = processor.apply_transcription_request(
            audio=clip["path"], language="Japanese").to(model.device, model.dtype)
        limit = min(512, max(96, int((clip["end"] - clip["start"]) * 12) + 32))
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        torch.cuda.synchronize()
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = processor.decode(generated, return_format="transcription_only")[0].strip()
        results.append({"id": clip["id"], "start": clip["start"], "end": clip["end"], "text": text,
                        "elapsed_sec": round(time.monotonic() - started, 2),
                        "generated_tokens": generated.shape[1],
                        "token_limit_reached": generated.shape[1] >= limit})
        # 途中で止まっても、終わった候補は試験フォルダへ残す。
        yield {"model": model_id, "revision": revision,
               "load_sec": round(load_sec, 2), "results": results,
               "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
               "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["whisper", "qwen", "gemma"])
    p.add_argument("request", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    request = json.loads(args.request.read_text(encoding="utf-8"))
    started = time.monotonic()
    if args.backend == "qwen":
        for result in qwen(request):
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.backend == "whisper":
        from core import glossary, people, transcriber
        terms = people.initial_prompt_terms()
        result = transcriber.transcribe(request["audio"], initial_prompt=glossary.build_initial_prompt(terms),
                                        hotwords=glossary.build_hotwords(terms), speed="accurate", decoding="stable")
    else:
        from core import config, llm, minutes
        settings = config.load_settings()
        # 候補のコピーだけを校正する。比較結果は原文の正誤を証明するものではない。
        utterances = [{"speaker": f"{x['start']:.1f}〜{x['end']:.1f}秒の候補",
                       "text": x["text"], "marked": x["text"]}
                      for x in request["results"] if x["text"] and not x["token_limit_reached"]]
        ids = [x["id"] for x in request["results"] if x["text"] and not x["token_limit_reached"]]
        try:
            done, changed = minutes.proofread_utterances(settings, utterances)
            result = {"model": settings["llm"]["ollama_model"], "done": done, "changed": changed,
                      "elapsed_sec": round(time.monotonic() - started, 2),
                      "fallback": llm.pop_fallback_notice(),
                      "results": [{"id": i, "text": u["text"]} for i, u in zip(ids, utterances)]}
        finally:
            llm.unload_ollama(settings)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
