"""校正で事実を変えないための補助検査。意味の正しさを保証する検査ではない。"""
import difflib
import re
import unicodedata
from collections import Counter

from .utterances import RED_OPEN, RED_CLOSE, RED_RE, strip_marks


def name_map(people, speakers=()):
    candidates = {}
    for person in people:
        display = person.get("display_name", "")
        for alias in [display] + list(person.get("aliases") or []):
            if isinstance(alias, str) and len(alias.strip()) >= 2:
                candidates.setdefault(alias.strip(), set()).add(display)
    for speaker in speakers:
        if speaker and len(speaker) >= 2 and speaker not in candidates:
            candidates.setdefault(speaker, set()).add(speaker)
    return {alias: next(iter(names)) for alias, names in candidates.items() if len(names) == 1}


def names_in(text, names):
    if not names:
        return set()
    pattern = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return {names[m.group()] for m in re.finditer(pattern, text)}


def correction_changes_facts(old, new, names):
    """数字と既知の人物を校正で入れ替えない。登録済み別名の正規化は許可する。"""
    def numbers(t):
        return Counter(re.findall(r"\d+(?:[.,:/]\d+)*", unicodedata.normalize("NFKC", t)))
    if numbers(old) != numbers(new):
        return True
    if names_in(old, names) != names_in(new, names):
        return True
    # 校正は言い換えをしないので、否定・未確定の表現を勝手に削る回答も保留する。
    qualifiers = ("しない", "しません", "できない", "できません", "不要", "未定", "未確定", "保留", "検討中", "かもしれない")
    return any((q in old) != (q in new) for q in qualifiers)


def mark_corrections(old, marked):
    """LLMが確実だと称した修正も可視化する。既存の赤字も残す。"""
    def red_flags(text):
        flags, offset = [False] * len(strip_marks(text)), 0
        for match in RED_RE.finditer(text):
            start = match.start() - offset
            end = start + len(match.group(1))
            flags[start:end] = [True] * (end - start)
            offset += len(RED_OPEN) + len(RED_CLOSE)
        return flags
    new, flags, old_flags = strip_marks(marked), red_flags(marked), red_flags(old)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, strip_marks(old), new, autojunk=False).get_opcodes():
        if op == "equal":
            flags[j1:j2] = [a or b for a, b in zip(flags[j1:j2], old_flags[i1:i2])]
        if op != "equal":
            start, end = (j1, j2) if j2 > j1 else (max(0, j1 - 1), min(len(new), j1 + 1))
            flags[start:end] = [True] * (end - start)
    out, active = [], False
    for char, red in zip(new, flags):
        if red != active:
            out.append(RED_OPEN if red else RED_CLOSE)
            active = red
        out.append(char)
    if active:
        out.append(RED_CLOSE)
    return "".join(out)
