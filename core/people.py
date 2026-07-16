"""人物マスタ（呼び名→正式表記、声紋ファイル、役職）"""
from pathlib import Path

import yaml

from .config import PROFILES_DIR

PEOPLE_PATH = PROFILES_DIR / "people.yaml"


def load_people() -> list[dict]:
    if PEOPLE_PATH.exists():
        try:
            with open(PEOPLE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            people = data.get("people", []) or []
            return [p for p in people if p.get("display_name")]
        except Exception:
            return []
    return []


def save_people(people: list[dict]) -> None:
    PEOPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PEOPLE_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"people": people}, f, allow_unicode=True, sort_keys=False)


def apply_aliases(text: str, people: list[dict] | None = None) -> str:
    """文字起こしテキスト内の呼び名を正式表記に置換する（長い呼び名から順に）"""
    if people is None:
        people = load_people()
    pairs = []
    for p in people:
        display = p.get("display_name", "")
        for alias in (p.get("aliases") or []):
            alias = str(alias).strip()
            if alias and alias != display:
                pairs.append((alias, display))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    for alias, display in pairs:
        text = text.replace(alias, display)
    return text


def initial_prompt_terms(people: list[dict] | None = None) -> list[str]:
    """Whisperの initial_prompt に入れる人名リスト（正式表記＋呼び名）"""
    if people is None:
        people = load_people()
    terms = []
    for p in people:
        if p.get("display_name"):
            terms.append(p["display_name"])
        terms.extend(str(a).strip() for a in (p.get("aliases") or []) if str(a).strip())
    return terms


def people_summary_md(people: list[dict] | None = None) -> str:
    """LLMプロンプトに注入する人物マスタの要約"""
    if people is None:
        people = load_people()
    if not people:
        return "（人物マスタ未登録）"
    lines = []
    for p in people:
        aliases = "、".join(str(a) for a in (p.get("aliases") or []))
        role = p.get("role", "")
        line = f"- 正式表記: {p.get('display_name')}"
        if aliases:
            line += f" ／ 呼び名: {aliases}"
        if role:
            line += f" ／ 役職: {role}"
        lines.append(line)
    return "\n".join(lines)


# --- Gradio Dataframe 変換 ---

def to_table(people: list[dict] | None = None) -> list[list[str]]:
    if people is None:
        people = load_people()
    rows = []
    for p in people:
        rows.append([
            p.get("display_name", ""),
            ", ".join(str(a) for a in (p.get("aliases") or [])),
            p.get("role", "") or "",
        ])
    return rows or [["", "", ""]]


def from_table(rows) -> list[dict]:
    people = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        display = str(row[0]).strip()
        aliases_raw = str(row[1]) if len(row) > 1 and row[1] is not None else ""
        aliases = [a.strip() for a in aliases_raw.replace("、", ",").split(",") if a.strip()]
        role = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        person = {"display_name": display, "aliases": aliases}
        if role:
            person["role"] = role
        people.append(person)
    return people
