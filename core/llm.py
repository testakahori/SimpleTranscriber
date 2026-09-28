"""LLM連携: Ollama(デフォルト・完全ローカル) / Claude API / OpenAI API を切替可能"""
import json
import logging

import httpx

from .config import get_api_key

TIMEOUT = 1800.0  # ローカルLLMは遅いことがあるため長めに


class LLMError(Exception):
    """ユーザー向けメッセージを持つLLMエラー"""
    pass


def provider_label(settings: dict) -> str:
    llm = settings.get("llm", {})
    provider = llm.get("provider", "ollama")
    if provider == "ollama":
        return f"Ollama ({llm.get('ollama_model')})"
    if provider == "anthropic":
        return f"Claude ({llm.get('anthropic_model')})"
    if provider == "openai":
        return f"OpenAI ({llm.get('openai_model')})"
    return "なし"


def generate(settings: dict, system: str, user_text: str, think: bool | None = None) -> str:
    """設定されたプロバイダでテキスト生成する。失敗時は LLMError を送出。

    think: Ollamaの思考モード。None なら設定値（ollama_think）に従う。
    """
    provider = settings.get("llm", {}).get("provider", "ollama")
    if provider == "none":
        raise LLMError("LLMが「なし」に設定されています。設定タブでプロバイダを選択してください。")
    if provider == "ollama":
        return _generate_ollama(settings, system, user_text, think)
    if provider == "anthropic":
        return _generate_anthropic(settings, system, user_text)
    if provider == "openai":
        return _generate_openai(settings, system, user_text)
    raise LLMError(f"不明なプロバイダです: {provider}")


def fits_context(settings: dict, system: str, user_text: str) -> bool:
    """Ollamaのコンテキスト上限に収まるか（収まらないと先頭が黙って切り捨てられる）"""
    if settings.get("llm", {}).get("provider") != "ollama":
        return True
    max_ctx = int(settings["llm"].get("ollama_max_ctx") or 65536)
    return int((len(system) + len(user_text)) * 1.1) + 8192 <= max_ctx


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


def _num_ctx(settings: dict, system: str, user_text: str) -> int:
    """入力長から必要なコンテキスト長を見積もる（日本語は概ね1文字≒1トークン）"""
    max_ctx = int(settings["llm"].get("ollama_max_ctx") or 65536)
    need = int((len(system) + len(user_text)) * 1.1) + 8192  # 出力分の余裕
    ctx = 16384  # 下限を揃えて処理ごとのモデル再読み込み（遅い＋メモリ負荷）を避ける
    while ctx < need and ctx < max_ctx:
        ctx *= 2
    return min(ctx, max_ctx)


def _generate_ollama(settings: dict, system: str, user_text: str, think: bool | None = None) -> str:
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
        "options": {"num_ctx": _num_ctx(settings, system, user_text), "temperature": 0.3},
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


def _generate_anthropic(settings: dict, system: str, user_text: str) -> str:
    api_key = get_api_key(settings, "anthropic")
    if not api_key:
        raise LLMError(
            "Claude APIキーが未設定です。設定タブで入力するか、"
            "環境変数 ANTHROPIC_API_KEY を設定してください。"
        )
    model = settings["llm"].get("anthropic_model") or "claude-opus-4-8"
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic パッケージが未インストールです。`pip install anthropic` を実行してください。")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        # 長い文字起こしを扱うためストリーミングで受信する
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise LLMError("Claude APIキーが無効です。設定を確認してください。")
    except anthropic.NotFoundError:
        raise LLMError(f"Claudeモデル '{model}' が見つかりません。設定を確認してください。")
    except anthropic.RateLimitError:
        raise LLMError("Claude APIのレート制限に達しました。しばらく待って再実行してください。")
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude APIエラー (HTTP {e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise LLMError("Claude APIに接続できません。ネットワークを確認してください。")

    if message.stop_reason == "refusal":
        raise LLMError("Claudeがこの内容の処理を拒否しました。")
    text = "".join(b.text for b in message.content if b.type == "text")
    if message.stop_reason == "max_tokens":
        text += "\n\n（※出力が長すぎるため途中で打ち切られました）"
    return text


def _generate_openai(settings: dict, system: str, user_text: str) -> str:
    api_key = get_api_key(settings, "openai")
    if not api_key:
        raise LLMError(
            "OpenAI APIキーが未設定です。設定タブで入力するか、"
            "環境変数 OPENAI_API_KEY を設定してください。"
        )
    model = settings["llm"].get("openai_model") or "gpt-4o"
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
            },
            timeout=TIMEOUT,
        )
    except httpx.TimeoutException:
        raise LLMError("OpenAI APIの応答がタイムアウトしました。")
    except httpx.HTTPError as e:
        raise LLMError(f"OpenAI APIに接続できません: {e}")

    if resp.status_code == 401:
        raise LLMError("OpenAI APIキーが無効です。設定を確認してください。")
    if resp.status_code == 404:
        raise LLMError(f"OpenAIモデル '{model}' が見つかりません。設定を確認してください。")
    if resp.status_code == 429:
        raise LLMError("OpenAI APIのレート制限に達しました。しばらく待って再実行してください。")
    if resp.status_code != 200:
        raise LLMError(f"OpenAI APIエラー (HTTP {resp.status_code}): {resp.text[:300]}")

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        raise LLMError(f"OpenAIの応答を解析できませんでした: {resp.text[:300]}")


def unload_ollama(settings: dict) -> None:
    """Ollamaが保持しているモデルをメモリから降ろす（Whisper用にRAM/VRAMを空ける）"""
    llm = settings.get("llm", {})
    if llm.get("provider") != "ollama":
        return
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
