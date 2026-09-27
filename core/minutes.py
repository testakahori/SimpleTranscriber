"""LLMによる文書生成: 校正（赤字補完）・議事録・要約と考察"""
import logging
import re

from . import glossary, llm, people
from .config import PROMPTS_DIR
from .utterances import RED_CLOSE, RED_OPEN, RED_RE, format_hms, strip_marks

MINUTES_PROMPT = "議事録プロンプト_v3.md"
PROOFREAD_PROMPT = "校正プロンプト_行単位.md"
PROOFREAD_CHUNK_CHARS = 2500  # 1回のLLM呼び出しで校正する分量

U_OPEN, U_CLOSE = "⟦", "⟧"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"プロンプトファイルが見つかりません: {path}")


def _context_block() -> str:
    block = "\n\n# 人物マスタ（呼び名→正式表記）\n" + people.people_summary_md()
    terms = glossary.load_glossary()
    if terms:
        block += "\n\n# 用語辞書（この表記を正とする）\n" + "、".join(terms)
    return block


# =====================================================================
# 校正（発言単位・構造を壊さない）
# =====================================================================

def _to_uncertain(marked: str) -> str:
    return RED_RE.sub(lambda m: f"{U_OPEN}{m.group(1)}{U_CLOSE}", marked)


def _from_uncertain(text: str) -> str:
    text = re.sub(f"{U_OPEN}(.*?){U_CLOSE}", lambda m: f"{RED_OPEN}{m.group(1)}{RED_CLOSE}", text)
    return text.replace(U_OPEN, "").replace(U_CLOSE, "")


_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def proofread_utterances(settings: dict, utterances: list[dict], progress_cb=None) -> tuple[int, int]:
    """発言リストをその場で校正する。(校正できた発言数, 変更された発言数) を返す。

    LLMの出力で番号が欠けた行は元のまま残す（壊さないことを優先）。
    """
    system = _load_prompt(PROOFREAD_PROMPT) + _context_block()

    chunks, cur, size = [], [], 0
    for i, u in enumerate(utterances):
        cur.append(i)
        size += len(u.get("text", ""))
        if size >= PROOFREAD_CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)

    done = changed = 0
    for ci, idxs in enumerate(chunks):
        lines = []
        for i in idxs:
            u = utterances[i]
            sp = u.get("speaker") or "話者"
            lines.append(f"[{i + 1}] {sp}: {_to_uncertain(u.get('marked') or u['text'])}")
        # 前後の文脈（校正対象外）を少し添える
        ctx_before = " ".join(utterances[j]["text"] for j in range(max(0, idxs[0] - 2), idxs[0]))
        user = ""
        if ctx_before:
            user += f"（参考: 直前の発言）{ctx_before}\n\n"
        user += "\n".join(lines)
        try:
            reply = llm.generate(settings, system, user, think=False)
        except llm.LLMError:
            raise
        except Exception as e:
            logging.warning(f"校正チャンク{ci}で失敗: {e}")
            continue
        for line in reply.splitlines():
            m = _LINE_RE.match(line)
            if not m:
                continue
            n = int(m.group(1)) - 1
            body = m.group(2).strip()
            if n not in idxs or not body:
                continue
            u = utterances[n]
            # LLMが話者名を復唱した場合だけ取り除く（「10:30に集合」等の本文は壊さない）
            sp = (u.get("speaker") or "話者")
            for prefix in (f"{sp}:", f"{sp}："):
                if body.startswith(prefix):
                    body = body[len(prefix):].strip()
            new_marked = _from_uncertain(body)
            new_text = strip_marks(new_marked)
            # 極端な短縮・水増し、まとまった文の削除は誤動作とみなして元のまま残す
            if not (0.5 <= len(new_text) / max(len(u["text"]), 1) <= 1.6):
                continue
            if _drops_content(u["text"], new_text):
                logging.info(f"校正で発言内容が削除されたため不採用: {u['text'][:40]}")
                continue
            if new_text != u["text"]:
                changed += 1
            u["marked"], u["text"] = new_marked, new_text
            u["proofread"] = True
            done += 1
        if progress_cb:
            progress_cb((ci + 1) / len(chunks))
    return done, changed


# =====================================================================
# 議事録・要約
# =====================================================================

_FILLER_RE = re.compile(r"^[、。\s]*(えー+|えっと|あのー*|あー+|うーん|まあ|その|なんか)[、。\s]*$")


def _drops_content(old: str, new: str) -> bool:
    """校正で意味のある文言がまとまって消えていないか（フィラー削除は許容）"""
    import difflib
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        removed = old[i1:i2]
        if op == "delete" and len(removed.strip("、。 　")) >= 6 and not _FILLER_RE.match(removed):
            return True
        if op == "replace" and (i2 - i1) >= 10 and (j2 - j1) < (i2 - i1) * 0.4:
            return True
    return False


def transcript_for_llm(utterances: list[dict]) -> str:
    """LLMに渡すコンパクトな文字起こし（1発言1行・赤字は保持）"""
    lines = []
    for u in utterances:
        sp = u.get("speaker") or "発言者"
        lines.append(f"[{format_hms(u['start'])}] {sp}: {u.get('marked') or u['text']}")
    return "\n".join(lines)


def _header(source_name: str, meeting_date: str) -> str:
    head = ""
    if source_name:
        head += f"ファイル名: {source_name}\n"
    if meeting_date:
        head += f"会議日（録音ファイルの日時）: {meeting_date}\n"
    return head


NOTES_SYSTEM = """あなたは会議の書記です。渡されるのは長い会議の文字起こしの一部です。
後で議事録にまとめるための「詳細メモ」を作ってください。
- 話題ごとに、誰が何を報告・提案・質問・回答したかを時刻付きで漏れなく箇条書きにする
- 決定事項、担当者と期限つきの依頼、未確定・保留の事項、数字・日付・固有名詞は必ず残す
- 赤字（<span style="color:red">〜</span>）の箇所は赤字のまま残す
- 推測で事実を足さない。前置きは書かない"""
CHUNK_CHARS = 18000


_notes_cache: dict = {}  # 文字起こし → 区間メモ（議事録と要約で同じメモを使い回す）


def _shrink_transcript(settings: dict, system: str, transcript: str) -> str:
    """長すぎてLLMの入力上限を超える文字起こしを、区間ごとの詳細メモに圧縮する"""
    if llm.fits_context(settings, system, transcript):
        return transcript
    import hashlib
    key = hashlib.sha1((settings["llm"].get("ollama_model", "") + transcript).encode("utf-8")).hexdigest()
    if key in _notes_cache:
        return _notes_cache[key]
    lines, parts, cur = transcript.split("\n"), [], []
    size = 0
    for line in lines:
        cur.append(line)
        size += len(line)
        if size >= CHUNK_CHARS:
            parts.append("\n".join(cur))
            cur, size = [], 0
    if cur:
        parts.append("\n".join(cur))
    notes = []
    for i, part in enumerate(parts, 1):
        memo = llm.generate(settings, NOTES_SYSTEM + _context_block(),
                            f"（全{len(parts)}区間のうち第{i}区間）\n\n{part}", think=False)
        notes.append(f"## 第{i}区間のメモ\n{_strip_code_fence(memo)}")
    logging.info(f"長時間会議のため {len(parts)} 区間のメモに圧縮してから議事録化します")
    shrunk = ("（注: 会議が長いため、以下は文字起こしを区間ごとに要約した詳細メモです）\n\n"
              + "\n\n".join(notes))
    _notes_cache.clear()  # 直近1件だけ保持（メモリを食わないように）
    _notes_cache[key] = shrunk
    return shrunk


def generate_minutes(settings: dict, transcript: str, source_name: str = "",
                     meeting_date: str = "") -> str:
    system = _load_prompt(MINUTES_PROMPT) + _context_block()
    transcript = _shrink_transcript(settings, system, transcript)
    user = (f"{_header(source_name, meeting_date)}\n"
            f"次の文字起こしから議事録を作成してください。\n\n{transcript}")
    return tidy_minutes(_strip_code_fence(llm.generate(settings, system, user)))


def tidy_minutes(md: str) -> str:
    """Markdownとして崩れないよう整える:
    【見出し】の前後に空行、「日時: …」のような1行項目は改行を保持（行末に2スペース）"""
    out = []
    for line in md.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"【[^】]+】", stripped):
            if out and out[-1] != "":
                out.append("")
            out.extend([stripped, ""])
            continue
        if re.match(r"^[^\s#|\-*>][^:：]{0,12}[:：]\s*\S", stripped) and not stripped.endswith("  "):
            line = line.rstrip() + "  "
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def generate_insights(settings: dict, transcript: str, source_name: str = "",
                      meeting_date: str = "") -> str:
    system = _load_prompt("要約と考察プロンプト.md") + _context_block()
    transcript = _shrink_transcript(settings, system, transcript)
    user = f"{_header(source_name, meeting_date)}\n次の文字起こしを分析してください。\n\n{transcript}"
    return _strip_code_fence(llm.generate(settings, system, user))


def _strip_code_fence(text: str) -> str:
    """LLMが全体を ```markdown フェンスで囲んだ場合に剥がす"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped
