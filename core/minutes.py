"""LLMによる文書生成: 校正（赤字補完）・議事録・要約と考察"""
import logging

from . import llm, people
from .config import PROMPTS_DIR


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"プロンプトファイルが見つかりません: {path}")


def proofread_transcript(settings: dict, transcript_md: str) -> str:
    """LLMで文字起こしを校正する。誤認識を文脈から修正し、
    修正・推測箇所は赤span表記のまま維持/追加させる。"""
    system = _load_prompt("校正プロンプト.md")
    system += "\n\n# 人物マスタ\n" + people.people_summary_md()
    result = llm.generate(settings, system, transcript_md)
    return _strip_code_fence(result)


def generate_minutes(settings: dict, transcript_md: str, source_name: str = "") -> str:
    """議事録プロンプトv2で議事録.mdを生成する"""
    system = _load_prompt("議事録プロンプト_v2.md")
    system += "\n\n# 人物マスタ（呼び名→正式表記）\n" + people.people_summary_md()
    user = f"次の文字起こしから議事録を作成してください。\n\n{transcript_md}"
    result = llm.generate(settings, system, user)
    return _strip_code_fence(result)


def generate_insights(settings: dict, transcript_md: str, source_name: str = "") -> str:
    """要約・考察・ネクストアクションを生成する"""
    system = _load_prompt("要約と考察プロンプト.md")
    system += "\n\n# 人物マスタ（呼び名→正式表記）\n" + people.people_summary_md()
    user = f"次の文字起こしを分析してください。\n\n{transcript_md}"
    result = llm.generate(settings, system, user)
    return _strip_code_fence(result)


def _strip_code_fence(text: str) -> str:
    """LLMが全体を ```markdown フェンスで囲んだ場合に剥がす"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped
