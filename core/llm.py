"""LLM連携: Ollama 上の gemma4 だけで動く（完全ローカル・外部APIには送らない）"""
import logging

import httpx

TIMEOUT = 1800.0  # ローカルLLMは遅いことがあるため長めに


class LLMError(Exception):
    """ユーザー向けメッセージを持つLLMエラー"""
    pass


def provider_label(settings: dict) -> str:
    return f"Ollama ({settings.get('llm', {}).get('ollama_model')})"


DEFAULT_TEMPERATURE = 0.3
# 日本語1文字あたりのトークン数の見積もり。gemma4 の実測（2026-11-13、議事録プロンプト＋技術文書
# 2000字/6000字を Ollama の prompt_eval_count で確認）は 0.58〜0.61 トークン/文字。
# 話し言葉・時刻・話者名が混ざる文字起こしは効率が落ちるので、安全側に 0.8 を使う。
# （以前は 1.1 倍＋出力分 8192 で見積もっていたため、上限 16384 では約4400字しか入らず、
#   10分を超える会議はすべて区間メモに圧縮され、しかもメモ用の区切り（18000字）が
#   上限を超えて先頭が黙って切り捨てられていた）
TOKENS_PER_CHAR = 0.8
OUTPUT_RESERVE = 6144  # 生成分のために空けておくトークン数（議事録1本は実測で3000〜5000トークン）


def generate(settings: dict, system: str, user_text: str, think: bool | None = None,
             temperature: float | None = None, output_reserve: int = OUTPUT_RESERVE) -> str:
    """Ollama でテキスト生成する。失敗時は LLMError を送出。

    think: Ollamaの思考モード。None なら設定値（ollama_think）に従う。
    temperature: 事実の書き写し（校正・議事録・照合）は低め（0.1）、考察は既定（0.3）。
    output_reserve: この呼び出しの出力に見込むトークン数（コンテキスト長の見積もりに使う）。
    """
    return _generate_ollama(settings, system, user_text, think, temperature, output_reserve)


def max_ctx(settings: dict) -> int:
    return int(settings["llm"].get("ollama_max_ctx") or 65536)


def estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def input_budget_chars(settings: dict, system: str, output_reserve: int = OUTPUT_RESERVE) -> int:
    """system を除いて、ユーザー入力に使える文字数（上限に収まる最大）"""
    tokens = max_ctx(settings) - output_reserve - estimate_tokens(system) - 64
    return max(0, int(tokens / TOKENS_PER_CHAR))


def fits_context(settings: dict, system: str, user_text: str, output_reserve: int = OUTPUT_RESERVE) -> bool:
    """Ollamaのコンテキスト上限に収まるか（収まらないと先頭が黙って切り捨てられる）"""
    return len(user_text) <= input_budget_chars(settings, system, output_reserve)


def _is_resource_error(resp) -> bool:
    text = resp.text
    return resp.status_code == 500 and (
        "out of memory" in text or "unexpectedly stopped" in text
        or "failed to allocate" in text or "resource limitations" in text
        or "bad_alloc" in text)  # llama-server がRAM不足で落ちた時の表記


def _retry_on_resource_error(url: str, payload: dict, resp, llm_settings: dict):
    """メモリ不足でモデルが起動/実行できなかった時の粘り:
    1) 少し待って同条件で再試行 2) GPU不足ならGPUに載せる層を減らす
    3) それでも駄目なら軽いモデル（ollama_fallback_model）に切り替える"""
    import time

    if not _is_resource_error(resp):
        return resp
    logging.warning(f"Ollamaがリソース不足で失敗しました。再試行します: {resp.text[:200]}")
    time.sleep(3)
    resp = httpx.post(f"{url}/api/chat", json=payload, timeout=TIMEOUT)
    for num_gpu in (48, 40, 24):
        if not (_is_resource_error(resp) and "cuda" in resp.text.lower()):
            break
        logging.warning(f"VRAM不足のため num_gpu={num_gpu} で再試行します")
        payload["options"]["num_gpu"] = num_gpu
        resp = httpx.post(f"{url}/api/chat", json=payload, timeout=TIMEOUT)
    fallback = (llm_settings.get("ollama_fallback_model") or "").strip()
    if _is_resource_error(resp) and fallback and fallback != payload["model"]:
        try:
            names = [m["name"] for m in httpx.get(f"{url}/api/tags", timeout=5).json().get("models", [])]
        except Exception:
            names = []
        if fallback in names:
            logging.warning(f"{payload['model']} がメモリ不足で動かないため {fallback} に切り替えます")
            _fallback_used.add(f"{payload['model']}→{fallback}")
            payload = dict(payload, model=fallback)
            payload["options"].pop("num_gpu", None)
            resp = httpx.post(f"{url}/api/chat", json=payload, timeout=TIMEOUT)
    return resp


_fallback_used: set = set()


def pop_fallback_notice() -> str:
    """直近の処理で軽いモデルに切り替えたかどうか（UI表示用）"""
    if not _fallback_used:
        return ""
    msg = "、".join(sorted(_fallback_used))
    _fallback_used.clear()
    return f"メモリ不足のため一部を軽いモデルで処理しました（{msg}）"


_ctx_in_use: dict = {}  # モデル名 → このプロセスで使った最大のコンテキスト長


def _num_ctx(settings: dict, system: str, user_text: str, output_reserve: int = OUTPUT_RESERVE) -> int:
    """入力長から必要なコンテキスト長を見積もる。上限（ollama_max_ctx）を超える入力は
    呼び出し側で分割する前提（超えると Ollama が先頭を黙って切り捨てる）。
    一度大きいコンテキストで読み込んだら、以後の呼び出しも同じ長さにそろえる
    （議事録 32768 → 照合 16384 のように変わるたびに Ollama がモデルを読み直して遅くなる）"""
    limit = max_ctx(settings)
    need = estimate_tokens(system + user_text) + output_reserve
    ctx = min(16384, limit)  # 下限を揃えて処理ごとのモデル再読み込み（遅い＋メモリ負荷）を避ける
    while ctx < need and ctx < limit:
        ctx *= 2
    if need > limit:
        logging.warning(f"LLMへの入力（約{need}トークン）が上限 {limit} を超えています。先頭が切り捨てられる可能性があります")
    model = settings["llm"].get("ollama_model", "")
    ctx = max(min(ctx, limit), min(_ctx_in_use.get(model, 0), limit))
    _ctx_in_use[model] = ctx
    return ctx


def _generate_ollama(settings: dict, system: str, user_text: str, think: bool | None = None,
                     temperature: float | None = None, output_reserve: int = OUTPUT_RESERVE) -> str:
    llm = settings["llm"]
    url = (llm.get("ollama_url") or "http://localhost:11434").rstrip("/")
    model = llm.get("ollama_model") or "gemma4:31b"
    if think is None:
        think = bool(llm.get("ollama_think", False))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "think": think,
        "keep_alive": "10m",
        "options": {"num_ctx": _num_ctx(settings, system, user_text, output_reserve),
                    "temperature": DEFAULT_TEMPERATURE if temperature is None else float(temperature)},
    }
    if llm.get("ollama_num_gpu") not in (None, ""):
        payload["options"]["num_gpu"] = int(llm["ollama_num_gpu"])
    try:
        resp = httpx.post(f"{url}/api/chat", json=payload, timeout=TIMEOUT)
        if resp.status_code == 400 and "think" in resp.text:
            # 思考モード非対応のモデル → think指定なしで再試行
            payload.pop("think")
            resp = httpx.post(f"{url}/api/chat", json=payload, timeout=TIMEOUT)
        resp = _retry_on_resource_error(url, payload, resp, llm)
    except httpx.ConnectError:
        raise LLMError(
            f"Ollama ({url}) に接続できません。Ollamaを起動してください。"
            "未インストールの場合は https://ollama.com からインストールできます。"
        )
    except httpx.TimeoutException:
        raise LLMError("Ollamaの応答がタイムアウトしました。モデルが大きすぎる可能性があります。")
    except httpx.HTTPError as e:
        raise LLMError(f"Ollamaとの通信に失敗しました（メモリ不足でOllamaが落ちた可能性があります）: {e}")

    if resp.status_code == 404:
        raise LLMError(
            f"モデル '{model}' がOllamaに見つかりません。"
            f"ターミナルで `ollama pull {model}` を実行してください。"
        )
    if resp.status_code != 200:
        raise LLMError(f"Ollamaエラー (HTTP {resp.status_code}): {resp.text[:300]}")

    try:
        data = resp.json()
        content = data["message"]["content"]
    except Exception:
        raise LLMError(f"Ollamaの応答を解析できませんでした: {resp.text[:300]}")
    return _strip_thinking(content)


def unload_ollama(settings: dict) -> None:
    """Ollamaが保持しているモデルをメモリから降ろす（Whisper用にRAM/VRAMを空ける）"""
    llm = settings.get("llm", {})
    url = (llm.get("ollama_url") or "http://localhost:11434").rstrip("/")
    try:
        loaded = httpx.get(f"{url}/api/ps", timeout=3).json().get("models", [])
        for m in loaded:
            httpx.post(f"{url}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
            logging.info(f"Ollamaのモデル {m['name']} をメモリから解放しました")
    except Exception:
        pass


def _strip_thinking(text: str) -> str:
    """ローカルLLMが出力する <think>...</think> ブロックを除去する"""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def test_connection(settings: dict) -> str:
    """設定タブの「接続テスト」用。成功メッセージ or エラーメッセージを返す。"""
    try:
        reply = generate(settings, "1単語で答えてください。", "「OK」とだけ返してください。")
        label = provider_label(settings)
        return f"✅ 接続成功: {label} — 応答: {reply[:50]}"
    except LLMError as e:
        return f"❌ {e}"
    except Exception as e:
        logging.exception("接続テスト失敗")
        return f"❌ 予期しないエラー: {e}"
