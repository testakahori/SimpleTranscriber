"""LLM連携: Ollama(デフォルト・完全ローカル) / Claude API / OpenAI API を切替可能"""
import json
import logging

import httpx

from .config import get_api_key

TIMEOUT = 600.0  # ローカルLLMは遅いことがあるため長めに


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


def generate(settings: dict, system: str, user_text: str) -> str:
    """設定されたプロバイダでテキスト生成する。失敗時は LLMError を送出。"""
    provider = settings.get("llm", {}).get("provider", "ollama")
    if provider == "none":
        raise LLMError("LLMが「なし」に設定されています。設定タブでプロバイダを選択してください。")
    if provider == "ollama":
        return _generate_ollama(settings, system, user_text)
    if provider == "anthropic":
        return _generate_anthropic(settings, system, user_text)
    if provider == "openai":
        return _generate_openai(settings, system, user_text)
    raise LLMError(f"不明なプロバイダです: {provider}")


def _generate_ollama(settings: dict, system: str, user_text: str) -> str:
    llm = settings["llm"]
    url = (llm.get("ollama_url") or "http://localhost:11434").rstrip("/")
    model = llm.get("ollama_model") or "gemma4"
    try:
        resp = httpx.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "stream": False,
                "options": {"num_ctx": 32768},
            },
            timeout=TIMEOUT,
        )
    except httpx.ConnectError:
        raise LLMError(
            f"Ollama ({url}) に接続できません。Ollamaを起動してください。"
            "未インストールの場合は https://ollama.com からインストールできます。"
        )
    except httpx.TimeoutException:
        raise LLMError("Ollamaの応答がタイムアウトしました。モデルが大きすぎる可能性があります。")

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
    return "".join(b.text for b in message.content if b.type == "text")


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
