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
        # gemma4 最上位（31B dense）の3bit量子化版。16GB VRAMに全部載るので4bit版の約10倍速い
        "ollama_model": "hf.co/unsloth/gemma-4-31B-it-GGUF:UD-IQ3_XXS",
        "ollama_num_gpu": 99,          # 全層をGPUへ（Ollamaの自動見積もりは控えめで一部CPUに逃がすため）
        "ollama_think": False,         # 思考モード（議事録の質↑・時間↑）
        "ollama_max_ctx": 16384,       # これを超える長い会議は区間ごとに要約してから議事録化
        "ollama_fallback_model": "gemma4:12b-it-qat",  # メモリ不足で動かない時の代替
        "anthropic_model": "claude-opus-4-8",
        "anthropic_api_key": "",
        "openai_model": "gpt-4o",
        "openai_api_key": "",
    },
    "whisper": {
        "model": "auto",  # auto | large-v3 | large-v3-turbo | kotoba-whisper-v2.0 | medium | small
        "language": "ja",  # ja | auto
        "speed": "accurate",  # accurate | balanced | fast
    },
    "audio": {
        "asr_eq": True,            # 文字起こし前に低音(80Hz未満)・高域(7kHz超)をカット
        "noise_reduction": True,   # 背景ノイズ除去（弱め。4時間で約1.5分増）
    },
    "postprocess": {
        "red_threshold": 0.5,   # この確信度未満の単語を赤字対象にする
        "proofread": True,      # LLMによる校正・文脈補完
        "remove_fillers": True, # 「えー」「えーっと」「うーん」などを消す
    },
    "diarization": {
        "enabled": True,          # ライブラリ未導入なら自動でスキップ
        "num_speakers": 0,        # 0=自動推定
        "max_speakers": 12,       # 自動推定するときの上限人数
        "hf_token": "",           # 設定するとpyannote（高精度）で話者分離
        "match_similarity": 0.55,    # 声紋DBの人物と判定する類似度
    },
    "subtitle": {
        "max_chars_line": 20,
        "max_lines": 2,
        "max_duration": 6.0,
        "min_duration": 1.0,
        "pause_split": 0.6,
        "speaker_prefix": False,
        "drop_period": True,
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
