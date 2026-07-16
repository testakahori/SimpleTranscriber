"""設定の読み書き（settings.yaml）とパス定義"""
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "settings.yaml"
OUTPUT_DIR = ROOT / "output"
PROFILES_DIR = ROOT / "profiles"
VOICEPRINTS_DIR = PROFILES_DIR / "voiceprints"
CORRECTIONS_DIR = ROOT / "corrections"
PROMPTS_DIR = ROOT / "prompts"

DEFAULTS = {
    "llm": {
        "provider": "ollama",  # ollama | anthropic | openai | none
        "ollama_url": "http://localhost:11434",
        "ollama_model": "gemma4",
        "anthropic_model": "claude-opus-4-8",
        "anthropic_api_key": "",
        "openai_model": "gpt-4o",
        "openai_api_key": "",
    },
    "whisper": {
        "model": "auto",  # auto | large-v3 | medium | small
        "language": "ja",  # ja | auto
    },
    "audio": {
        "noise_reduction": False,
    },
    "postprocess": {
        "red_threshold": 0.5,   # この確信度未満の単語を赤字対象にする
        "proofread": True,      # LLMによる校正・文脈補完
    },
    "diarization": {
        "enabled": True,        # ライブラリ未導入なら自動でスキップ
        "match_threshold": 0.75,
        "cluster_threshold": 0.68,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _deep_merge(DEFAULTS, data)
        except Exception:
            pass
    return copy.deepcopy(DEFAULTS)


def save_settings(settings: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)


def get_api_key(settings: dict, provider: str) -> str:
    """設定 → 環境変数 の順でAPIキーを解決する"""
    llm = settings.get("llm", {})
    if provider == "anthropic":
        return llm.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if provider == "openai":
        return llm.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    return ""


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, PROFILES_DIR, VOICEPRINTS_DIR, CORRECTIONS_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
