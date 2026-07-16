"""用語辞書と修正学習（使うほど賢くなる仕組み）"""
import difflib

import yaml

from .config import PROFILES_DIR, CORRECTIONS_DIR

GLOSSARY_PATH = PROFILES_DIR / "glossary.yaml"
LEARNED_PATH = CORRECTIONS_DIR / "learned_words.yaml"

# 学習した修正を自動適用するのに必要な出現回数（誤学習防止）
PROMOTE_COUNT = 2


def load_glossary() -> list[str]:
    if GLOSSARY_PATH.exists():
        try:
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return [str(t) for t in (data.get("terms") or []) if str(t).strip()]
        except Exception:
            return []
    return []


def save_glossary(terms: list[str]) -> None:
    GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    terms = [t.strip() for t in terms if t.strip()]
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"terms": terms}, f, allow_unicode=True, sort_keys=False)


def load_corrections() -> list[dict]:
    if LEARNED_PATH.exists():
        try:
            with open(LEARNED_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("corrections", []) or []
        except Exception:
            return []
    return []


def save_corrections(corrections: list[dict]) -> None:
    LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"corrections": corrections}, f, allow_unicode=True, sort_keys=False)


def learn_from_diff(before: str, after: str) -> int:
    """編集前後の差分から「誤→正」ペアを抽出して学習する。学習件数を返す。"""
    corrections = load_corrections()
    index = {(c.get("wrong"), c.get("right")): c for c in corrections}

    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    learned = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "replace":
            continue
        wrong = before[i1:i2].strip()
        right = after[j1:j2].strip()
        # 短い語句の置換のみ学習対象（文まるごとの書き換えは学習しない）
        if not wrong or not right or len(wrong) > 12 or len(right) > 12:
            continue
        if "\n" in wrong or "\n" in right or wrong == right:
            continue
        key = (wrong, right)
        if key in index:
            index[key]["count"] = int(index[key].get("count", 1)) + 1
        else:
            entry = {"wrong": wrong, "right": right, "count": 1}
            corrections.append(entry)
            index[key] = entry
        learned += 1

    if learned:
        save_corrections(corrections)
    return learned


def apply_corrections(text: str) -> str:
    """出現回数が閾値以上の学習済み修正を適用する"""
    for c in load_corrections():
        if int(c.get("count", 0)) >= PROMOTE_COUNT:
            wrong, right = c.get("wrong"), c.get("right")
            if wrong and right:
                text = text.replace(wrong, right)
    return text


def build_initial_prompt(people_terms: list[str]) -> str:
    """Whisperの認識ヒント（initial_prompt）を用語辞書＋人名から組み立てる"""
    terms = load_glossary() + people_terms
    # 学習済み修正の「正しい語」もヒントに追加
    terms += [c["right"] for c in load_corrections()
              if int(c.get("count", 0)) >= PROMOTE_COUNT and c.get("right")]
    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return ""
    return "議事録。関連用語: " + "、".join(uniq)
