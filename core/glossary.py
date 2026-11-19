"""用語辞書と修正学習（使うほど賢くなる仕組み）"""
import difflib

import yaml

from .config import PROFILES_DIR, CORRECTIONS_DIR

GLOSSARY_PATH = PROFILES_DIR / "glossary.yaml"
LEARNED_PATH = CORRECTIONS_DIR / "learned_words.yaml"

# 学習した修正を自動適用するのに必要な出現回数（誤学習防止）
PROMOTE_COUNT = 2

# 句読点・記号。これを含む修正は「その場の文の直し」で、他の文に当てはめると壊れる
# （「。」→「、」を覚えて全文の句点が消えた事例あり）
_PUNCT_CHARS = set("、。，．,.？！?!・…「」『』（）()　 ")


def _learnable(wrong: str, right: str) -> bool:
    """他の文字起こしにも自動で当てはめてよい「誤→正」か。
    句読点を含むもの・1文字だけの置換（「ー」→「ン」など）は文脈次第なので覚えない。"""
    if not wrong or not right or wrong == right:
        return False
    if len(wrong) > 12 or len(right) > 12 or "\n" in wrong or "\n" in right:
        return False
    if _PUNCT_CHARS & (set(wrong) | set(right)):
        return False
    return len(wrong) >= 2


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
    # 1文字の漢字修正も、文脈ごとAI校正の参考に残す。無条件置換には昇格させない。
    contextual = remember_context(before, after)
    corrections = load_corrections()
    index = {(c.get("wrong"), c.get("right")): c for c in corrections}

    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    learned = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "replace":
            continue
        wrong = before[i1:i2].strip()
        right = after[j1:j2].strip()
        # 短い語句の置換のみ学習対象（文まるごとの書き換え・句読点の直しは学習しない）
        if not _learnable(wrong, right):
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
    return learned or int(contextual)


def load_context_examples() -> list[dict]:
    path = LEARNED_PATH.with_name("context_examples.yaml")
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [e for e in data.get("examples", []) if isinstance(e, dict)
            and isinstance(e.get("before"), str) and isinstance(e.get("after"), str)]


def save_context_examples(examples: list[dict]) -> None:
    from . import output
    path = LEARNED_PATH.with_name("context_examples.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(path.parent, path.name, yaml.safe_dump({"examples": examples}, allow_unicode=True, sort_keys=False))


def remember_context(before: str, after: str) -> bool:
    """人の短い修正を文脈付きで保持。家電→架電の「家→架」も単独置換しない。"""
    if not before or not after or before == after or max(len(before), len(after)) > 300:
        return False
    if difflib.SequenceMatcher(None, before, after, autojunk=False).ratio() < 0.5:
        return False
    examples = load_context_examples()
    if any(e["before"] == before and e["after"] == after for e in examples):
        return False
    examples.append({"before": before, "after": after, "context": "人が修正した発言。同じ話題・意味の場合にだけ参考にする。"})
    save_context_examples(examples)
    return True


def context_hints(text: str, limit: int = 12, max_chars: int = 2000) -> str:
    """現在の校正区間に近い例だけを渡す。辞書の正解を文脈なしで押し付けない。"""
    def bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1) if not _PUNCT_CHARS & set(s[i:i + 2])}
    target = bigrams(text)
    scored = []
    for e in load_context_examples():
        before, after = e["before"].strip(), e["after"].strip()
        if not before or not after or max(len(before), len(after)) > 300:
            continue
        score = len(bigrams(before) & target) + (20 if before in text else 0)
        if score:
            scored.append((score, before, after, str(e.get("context", ""))[:200]))
    if not scored:
        return ""
    header = "\n\n# 文脈付きの修正例（機械的に置換しない。同じ意味の文脈でだけ参考にする）\n"
    selected, size = [], len(header)
    for _, before, after, context in sorted(scored, reverse=True)[:limit]:
        entry = f"- 修正前: {before}\n  修正後: {after}\n  適用場面: {context}"
        if size + len(entry) + 1 <= max_chars:
            selected.append(entry)
            size += len(entry) + 1
    return header + "\n".join(selected) if selected else ""


def learn_pair(wrong: str, right: str, promote: bool = False) -> bool:
    """「誤→正」を1件学習する。promote=True なら即座に自動適用の回数（PROMOTE_COUNT）まで上げる
    （一括置換で人が全文に適用した修正は、次回から自動で使ってよい）。学習できたら True。"""
    wrong, right = (wrong or "").strip(), (right or "").strip()
    if not _learnable(wrong, right):
        return False
    corrections = load_corrections()
    for c in corrections:
        if c.get("wrong") == wrong and c.get("right") == right:
            c["count"] = max(int(c.get("count", 1)) + 1, PROMOTE_COUNT if promote else 0)
            break
    else:
        corrections.append({"wrong": wrong, "right": right, "count": PROMOTE_COUNT if promote else 1})
    save_corrections(corrections)
    return True


def apply_corrections(text: str) -> str:
    """出現回数が閾値以上の学習済み修正を適用する"""
    for c in load_corrections():
        if int(c.get("count", 0)) >= PROMOTE_COUNT:
            wrong, right = c.get("wrong"), c.get("right")
            if _learnable(wrong, right):  # 以前の版で覚えた句読点の直し等は使わない
                text = text.replace(wrong, right)
    return text


def build_initial_prompt(people_terms: list[str]) -> str:
    """Whisperの認識ヒント（initial_prompt）を用語辞書＋人名から組み立てる"""
    terms = load_glossary() + people_terms
    # 学習済み修正の「正しい語」もヒントに追加
    terms += [c["right"] for c in load_corrections()
              if int(c.get("count", 0)) >= PROMOTE_COUNT and _learnable(c.get("wrong"), c.get("right"))]
    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return ""
    return "議事録。関連用語: " + "、".join(uniq)


def build_hotwords(people_terms: list[str]) -> str:
    """faster-whisper の hotwords（固有名詞を優先認識させるヒント）"""
    terms = people_terms + load_glossary()
    seen, uniq = set(), []
    for t in terms:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return " ".join(uniq)
