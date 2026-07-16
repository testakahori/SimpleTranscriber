"""過去の全出力（議事録・文字起こし・要約）の横断全文検索"""
import re
from pathlib import Path

from .config import OUTPUT_DIR

SNIPPET_WIDTH = 60
MAX_RESULTS = 200


def search_outputs(query: str) -> list[list[str]]:
    """output/ 配下の全 .md をキーワードAND検索する。

    Returns: [[フォルダ, ファイル, 行, 前後スニペット], ...]
    """
    keywords = [k for k in re.split(r"[\s　]+", query.strip()) if k]
    if not keywords or not OUTPUT_DIR.exists():
        return []

    results = []
    for md_file in sorted(OUTPUT_DIR.glob("*/*.md"), reverse=True):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        lower = text.lower()
        # ファイル単位でAND判定
        if not all(k.lower() in lower for k in keywords):
            continue
        first_kw = keywords[0].lower()
        for line_no, line in enumerate(text.split("\n"), 1):
            line_lower = line.lower()
            if any(k.lower() in line_lower for k in keywords):
                snippet = _make_snippet(line, keywords)
                results.append([
                    md_file.parent.name,
                    md_file.name,
                    str(line_no),
                    snippet,
                ])
                if len(results) >= MAX_RESULTS:
                    return results
    return results


def _make_snippet(line: str, keywords: list[str]) -> str:
    clean = re.sub(r"<[^>]+>", "", line).strip()
    clean = clean.replace("**", "")
    for k in keywords:
        idx = clean.lower().find(k.lower())
        if idx >= 0:
            start = max(0, idx - SNIPPET_WIDTH // 2)
            end = min(len(clean), idx + len(k) + SNIPPET_WIDTH // 2)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(clean) else ""
            return f"{prefix}{clean[start:end]}{suffix}"
    return clean[:SNIPPET_WIDTH]
