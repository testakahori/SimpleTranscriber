"""LLMによる文書生成: 校正（赤字補完）・議事録・要約と考察・文字起こしとの照合"""
import logging
import re

from . import glossary, llm, people
from .config import PROMPTS_DIR
from .utterances import RED_CLOSE, RED_OPEN, RED_RE, format_hms, strip_marks

MINUTES_PROMPT = "議事録プロンプト_v4.md"
PROOFREAD_PROMPT = "校正プロンプト_行単位.md"
INSIGHTS_PROMPT = "要約と考察プロンプト.md"
VERIFY_PROMPT = "照合プロンプト.md"
PROOFREAD_CHUNK_CHARS = 2500  # 1回のLLM呼び出しで校正する分量
MINUTES_TEMPLATE = PROMPTS_DIR.parent / "議事録テンプレート.md"

# 事実を書き写す処理（校正・議事録・区間メモ・照合）は低温で安定させる。考察だけ既定の 0.3
FACTUAL_TEMPERATURE = 0.1

U_OPEN, U_CLOSE = "⟦", "⟧"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"プロンプトファイルが見つかりません: {path}")


def _context_block(memo: str = "") -> str:
    block = "\n\n# 人物マスタ（呼び名→正式表記）\n" + people.people_summary_md()
    terms = glossary.load_glossary()
    if terms:
        block += "\n\n# 用語辞書（この表記を正とする）\n" + "、".join(terms)
    if memo and memo.strip():
        block += "\n\n# 会議の背景メモ（人が書いた議題・参加者・背景。固有名詞や文脈の手掛かりにする）\n" + memo.strip()
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
MISTAKE_HINTS = 40  # 校正の参考に渡す「過去に直した誤変換」の最大件数


def _mistake_hints() -> str:
    """修正表・一括置換で人が直した「誤→正」を、校正の参考として渡す（同じ誤変換の再発を防ぐ）"""
    pairs = []
    for c in glossary.load_corrections():
        wrong, right = str(c.get("wrong", "")), str(c.get("right", ""))
        if glossary._learnable(wrong, right):
            pairs.append((int(c.get("count", 1)), wrong, right))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: -x[0])
    return ("\n\n# 過去に人が直した誤変換の例（同じ語が出たら正の表記にする）\n"
            + "、".join(f"{w}→{r}" for _, w, r in pairs[:MISTAKE_HINTS]))


def proofread_utterances(settings: dict, utterances: list[dict], progress_cb=None,
                         skip=None, on_chunk=None, memo: str = "") -> tuple[int, int]:
    """発言リストをその場で校正する。(校正できた発言数, 変更された発言数) を返す。

    LLMの出力で番号が欠けた行は元のまま残す（壊さないことを優先）。
    skip: True を返した発言は校正しない（校正済み・手修正済み → 途中から再開できる）
    on_chunk: 1回分（約2500字）終わるごとに呼ぶ（途中保存用）
    memo: 会議メモ（議題・参加者）。同音異義語を文脈で選ぶ手掛かりになる
    """
    system = _load_prompt(PROOFREAD_PROMPT) + _context_block(memo) + _mistake_hints()

    chunks, cur, size = [], [], 0
    for i, u in enumerate(utterances):
        if skip and skip(u):
            continue
        cur.append(i)
        size += len(u.get("text", ""))
        if size >= PROOFREAD_CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)

    done = changed = 0
    for ci, idxs in enumerate(chunks):
        for i in idxs:
            utterances[i].pop("proofread", None)
            utterances[i]["proofread_pending"] = True
        lines = []
        for i in idxs:
            u = utterances[i]
            sp = u.get("speaker") or "話者"
            lines.append(f"[{i + 1}] {sp}: {_to_uncertain(u.get('marked') or u['text'])}")
        # 前後の文脈（校正対象外）を少し添える。直後の発言は、質問→回答の流れで語を選ぶのに効く
        ctx_before = " ".join(utterances[j]["text"] for j in range(max(0, idxs[0] - 2), idxs[0]))
        ctx_after = " ".join(utterances[j]["text"] for j in range(idxs[-1] + 1, min(len(utterances), idxs[-1] + 2)))
        user = ""
        if ctx_before:
            user += f"（参考: 直前の発言）{ctx_before}\n\n"
        user += "\n".join(lines)
        if ctx_after:
            user += f"\n\n（参考: 直後の発言）{ctx_after}"
        try:
            reply = llm.generate(settings, system, user, think=False, temperature=FACTUAL_TEMPERATURE)
        except llm.LLMError:
            raise
        except Exception as e:
            logging.warning(f"校正チャンク{ci}で失敗: {e}")
            continue
        accepted = set()
        for line in reply.splitlines():
            m = _LINE_RE.match(line)
            if not m:
                continue
            n = int(m.group(1)) - 1
            body = m.group(2).strip()
            if n not in idxs or n in accepted or not body:
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
            u.pop("proofread_pending", None)
            accepted.add(n)
            done += 1
        if on_chunk:
            on_chunk()
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


NOTES_SYSTEM = """あなたは会議の書記です。渡されるのは長い会議の文字起こしの一部（各発言に時刻と話者名が付いている）です。
後で議事録にまとめるための「詳細メモ」を作ってください。このメモは文字起こしの代わりに使われるので、意味を変えないことが最優先です。
- 話題ごとに、誰が何を報告・提案・質問・回答・反対したかを、発言の時刻 [hh:mm:ss] 付きで漏れなく箇条書きにする
- 決定事項、担当者と期限つきの依頼、未確定・保留の事項、数字・金額・日付・固有名詞は必ず残す
- 重要な発言（決定・依頼・数字・反対意見・条件）は要約せず、発言のまま「」で引用する
- 否定（〜ない）・条件（〜なら、〜の場合）・推量（〜かもしれない）・質問か断定かを落とさない。誰の主張かを取り違えない
- 赤字（<span style="color:red">〜</span>）の箇所は赤字のまま残す
- 推測で事実を足さない。前置きは書かない"""
NOTES_OUTPUT_RESERVE = 4096  # 区間メモ1本の出力に空けておくトークン数
NOTES_OVERLAP_LINES = 3      # 区間の境目で話が切れないよう、直前の数行を参考として添える
NOTES_OVERLAP_CHARS = 400    # 参考行の合計の上限（長い発言でも予算を超えないよう切り詰める）
NOTES_MAX_DEPTH = 2          # 区間メモ自体が上限を超える超長時間会議は、メモをさらにメモにする回数の上限


_notes_cache: dict = {}  # 文字起こし → 区間メモ（議事録と要約で同じメモを使い回す）


def _split_lines_by_chars(lines: list[str], limit: int) -> list[list[str]]:
    parts, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) + 1 > limit:
            parts.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append(cur)
    return parts


def _overlap_tail(part: list[str]) -> list[str]:
    """直前の区間の末尾数行（合計 NOTES_OVERLAP_CHARS 以内に切り詰める）"""
    tail, size = [], 0
    for line in reversed(part[-NOTES_OVERLAP_LINES:]):
        room = NOTES_OVERLAP_CHARS - size
        if room <= 20:
            break
        line = line if len(line) <= room else "…" + line[-(room - 1):]
        tail.insert(0, line)
        size += len(line) + 1
    return tail


def _notes_for(settings: dict, notes_system: str, text: str, progress_cb=None, depth: int = 1) -> str:
    """文字起こし（またはメモ）を区間に分けて詳細メモにする。区間の大きさは上限から逆算する"""
    limit = (llm.input_budget_chars(settings, notes_system, NOTES_OUTPUT_RESERVE)
             - NOTES_OVERLAP_CHARS - 200)  # 参考行・区間番号の分
    if limit < 1000:
        raise llm.LLMError("LLMのコンテキスト上限（ollama_max_ctx）が小さすぎて、文字起こしを区間に分けられません。")
    parts = _split_lines_by_chars(text.split("\n"), limit)
    notes = []
    for i, part in enumerate(parts, 1):
        if progress_cb:
            progress_cb((i - 1) / len(parts), f"長い会議のため区間メモを作成中... {i}/{len(parts)}区間"
                        + (f"（{depth}段目）" if depth > 1 else ""))
        prev_tail = _overlap_tail(parts[i - 2]) if i > 1 else []
        user = f"（全{len(parts)}区間のうち第{i}区間）\n\n"
        if prev_tail:
            user += "（参考: 直前の区間の末尾。メモの対象外）\n" + "\n".join(prev_tail) + "\n\n（ここから第" + str(i) + "区間）\n"
        user += "\n".join(part)
        memo = llm.generate(settings, notes_system, user, think=False, temperature=FACTUAL_TEMPERATURE,
                            output_reserve=NOTES_OUTPUT_RESERVE)
        notes.append(f"## 第{i}区間のメモ\n{_strip_code_fence(memo)}")
    logging.info(f"長時間会議のため {len(parts)} 区間のメモに圧縮します（{depth}段目）")
    return "\n\n".join(notes)


def _shrink_transcript(settings: dict, system: str, transcript: str, progress_cb=None) -> str:
    """長すぎてLLMの入力上限を超える文字起こしを、区間ごとの詳細メモに圧縮する。
    区間の大きさは上限（ollama_max_ctx）から逆算する（固定の18000字だと上限 16384 では
    先頭が黙って切り捨てられていた）。メモ自体がまだ上限を超える時は、メモをさらに区間メモにする"""
    if llm.fits_context(settings, system, transcript):
        return transcript
    import hashlib
    key = hashlib.sha1((settings["llm"].get("ollama_model", "") + system[:200] + transcript).encode("utf-8")).hexdigest()
    if key in _notes_cache:
        return _notes_cache[key]
    notes_system = NOTES_SYSTEM + _context_block()
    head = ("（注: 会議が長いため、以下は文字起こしを区間ごとに要約した詳細メモです。"
            "各項目の [hh:mm:ss] は元の発言の時刻です）\n\n")
    text = transcript
    for depth in range(1, NOTES_MAX_DEPTH + 1):
        text = _notes_for(settings, notes_system, text, progress_cb, depth)
        if llm.fits_context(settings, system, head + text):
            break
    else:
        logging.warning("区間メモがまだ上限を超えています。先頭が切り捨てられる可能性があります")
    shrunk = head + text
    _notes_cache.clear()  # 直近1件だけ保持（メモリを食わないように）
    _notes_cache[key] = shrunk
    return shrunk


def _template_block() -> str:
    """ルートの 議事録テンプレート.md があれば、その形に合わせるよう指示する（書き換えれば形を変えられる）"""
    if not MINUTES_TEMPLATE.exists():
        return ""
    tpl = MINUTES_TEMPLATE.read_text(encoding="utf-8").strip()
    if not tpl:
        return ""
    return ("\n\n# 議事録テンプレート（見出しの名前・順番・構成はこのテンプレートを最優先で守ること。"
            "議題の数や箇条書きの数は会議の内容に合わせて増減してよい）\n" + tpl)


def generate_minutes(settings: dict, transcript: str, source_name: str = "",
                     meeting_date: str = "", memo: str = "", progress_cb=None) -> str:
    system = _load_prompt(MINUTES_PROMPT) + _template_block() + _context_block(memo)
    transcript = _shrink_transcript(settings, system, transcript, progress_cb)
    if progress_cb:
        progress_cb(0.9, "議事録を作成中...")
    user = (f"{_header(source_name, meeting_date)}\n"
            f"次の文字起こしから議事録を作成してください。\n\n{transcript}")
    return tidy_minutes(_strip_code_fence(llm.generate(settings, system, user, temperature=FACTUAL_TEMPERATURE)))


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
                      meeting_date: str = "", memo: str = "", progress_cb=None) -> str:
    system = _load_prompt(INSIGHTS_PROMPT) + _context_block(memo)
    transcript = _shrink_transcript(settings, system, transcript, progress_cb)
    if progress_cb:
        progress_cb(0.9, "要約と考察を作成中...")
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


# =====================================================================
# 文字起こしとの照合（議事録・要約の各項目を、根拠の発言と突き合わせて直す）
# =====================================================================
#
# 議事録は文字起こし全体を読んで書くため、長い会議では区間メモ経由になり、否定・条件・担当の
# 取り違えが起きる。ここでは文書を項目（箇条書き・表の行・要約の段落）に分け、項目が示す時刻
# [hh:mm:ss] の前後（±75秒）の発言だけを根拠として渡し、「一致／修正／確認できず」を判定させる。
# 会議の長さに関係なく、1項目あたりの根拠は小さく収まる。

_TS_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2}):(\d{2})(?!\d)")
_VERDICT_RE = re.compile(r"^\s*\[(\d+)\]\s*(OK|ＯＫ|修正|確認できず)\s*[:：]?\s*(.*)$")
EVIDENCE_WINDOW_SEC = 75.0
EVIDENCE_MAX_CHARS = 1400
VERIFY_BATCH_ITEMS = 6
VERIFY_OUTPUT_RESERVE = 3072
UNVERIFIED_MARK = "（文字起こしで確認できず）"

# 判定の対象外（AIの所見・解釈・会議概要の形式項目）
_SKIP_SECTIONS = ("AIの意見", "会議概要", "考察")
# 箇条書き以外の段落も対象にする節（要点サマリー・3行要約）
_PARAGRAPH_SECTIONS = ("要点サマリー", "要約")
# 「確認できず」を付けない節（「聞き逃されている論点」のように、発言が無いこと自体を指摘する項目があるため）
_NO_UNVERIFIED_SECTIONS = ("リスク",)
FIX_MIN_SIMILARITY = 0.5   # 修正前後の本文（根拠時刻を除く）の類似度がこれ未満なら、別の項目の内容が混ざったとみなして採用しない
_EVIDENCE_NOTE_RE = re.compile(r"[（(]\s*根拠[^）)]*[）)]")


def _to_sec(m) -> float:
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def _transcript_lines(transcript: str) -> list[tuple[float, str]]:
    """「[hh:mm:ss] 話者: 本文」の発言行 → [(秒, 行)]。時刻で始まらない行（会議メモなど）は根拠に使わない"""
    rows = []
    for line in transcript.split("\n"):
        line = line.strip()
        m = _TS_RE.match(line.lstrip("[").strip()) if line else None
        if m:
            rows.append((_to_sec(m), line))
    return rows


_KEYWORD_RE = re.compile(r"[一-鿿々]{2,}|[ァ-ヺー]{2,}|[A-Za-z0-9０-９]{2,}")


def _keywords(text: str) -> set[str]:
    words = set(_KEYWORD_RE.findall(strip_marks(text)))
    return {w for w in words if w not in ("根拠", "担当", "期限", "未定", "確認", "必要", "事項")}


def _evidence(lines: list[tuple[float, str]], item: str) -> tuple[str, str]:
    """項目の根拠になりそうな発言を集める → (根拠の本文, 取り方)。
    取り方: "time"=項目の時刻の前後 / "keyword"=語の重なり（時刻が無い項目） / ""=見つからない"""
    if not lines:
        return "", ""
    picked, how = set(), ""
    stamps = [_to_sec(m) for m in _TS_RE.finditer(item)]
    for t in stamps:
        for i, (sec, _) in enumerate(lines):
            if abs(sec - t) <= EVIDENCE_WINDOW_SEC:
                picked.add(i)
    if picked:
        how = "time"
    else:
        keys = _keywords(item)
        if keys:
            scored = sorted(((sum(1 for k in keys if k in line), i) for i, (_, line) in enumerate(lines)),
                            key=lambda x: (-x[0], x[1]))
            for score, i in scored[:3]:
                if score <= 0:
                    break
                picked.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(lines))
            if picked:
                how = "keyword"
    if not picked:
        return "", ""
    order = sorted(picked)
    text = "\n".join(lines[i][1] for i in order)
    if len(text) > EVIDENCE_MAX_CHARS:
        # どの根拠時刻にも近い行から順に残す（時刻が無い時は先頭から）
        def distance(i):
            return min(abs(lines[i][0] - t) for t in stamps) if stamps else i
        chosen, size = [], 0
        for i in sorted(order, key=distance):
            if size + len(lines[i][1]) + 1 > EVIDENCE_MAX_CHARS:
                break
            chosen.append(i)
            size += len(lines[i][1]) + 1
        text = "\n".join(lines[i][1] for i in sorted(chosen))
    return text, how


def _section_name(line: str) -> str | None:
    s = line.strip()
    m = re.fullmatch(r"【([^】]+)】", s)
    if m:
        return m.group(1)
    m = re.match(r"^#{1,3}\s+(.+)$", s)
    if m and not s.startswith("### 議題"):
        return re.sub(r"^[^\w一-鿿ぁ-ゖァ-ヺ]+", "", m.group(1)).strip()
    return None


def verifiable_items(md: str) -> list[tuple[int, str]]:
    """照合する項目 [(行番号, 行)]。箇条書き・表の行（見出し行は除く）・要約の段落。"""
    return [(i, line) for i, line, _ in _items_with_sections(md)]


def _items_with_sections(md: str) -> list[tuple[int, str, str]]:
    """[(行番号, 行, 節の名前)]"""
    items, section, table_header_seen = [], "", False
    for i, line in enumerate(md.split("\n")):
        name = _section_name(line)
        if name is not None:
            section, table_header_seen = name, False
            continue
        s = line.strip()
        if not s or any(k in section for k in _SKIP_SECTIONS):
            continue
        if s.startswith("|"):
            if re.fullmatch(r"\|[\s\-:|]+\|", s):
                continue
            if not table_header_seen:
                table_header_seen = True
                continue
            if s.replace("|", "").strip():
                items.append((i, line, section))
            continue
        if re.match(r"^\s*([-*・]|\d+[.．)])\s*\S", s):
            if re.sub(r"^\s*([-*・]|\d+[.．)])\s*", "", s).strip("（）() "):
                items.append((i, line, section))
            continue
        if any(k in section for k in _PARAGRAPH_SECTIONS) and not s.startswith("#"):
            items.append((i, line, section))
    return items


def _split_prefix(line: str) -> tuple[str, str]:
    """箇条書き記号・番号などの接頭辞と本文に分ける（表の行は接頭辞なし）"""
    m = re.match(r"^(\s*(?:[-*・]|\d+[.．)])\s+)(.*)$", line)
    if m:
        return m.group(1), m.group(2)
    return "", line


def _acceptable_fix(old: str, new: str, others=()) -> bool:
    """LLMの「修正」を採用してよいか。書式・長さ・元の項目との近さ・根拠時刻の維持を確かめ、
    同じ回で渡した別の項目の内容（番号の取り違え）が混ざったものは採用しない"""
    import difflib
    new = new.strip()
    if not new or len(new) < 2:
        return False
    if old.strip().startswith("|") != new.startswith("|"):
        return False
    if new.startswith("|") and new.count("|") != old.strip().count("|"):
        return False
    plain_old, plain_new = strip_marks(old).strip(), strip_marks(new)
    ratio = len(plain_new) / max(len(plain_old), 1)
    if not 0.3 <= ratio <= 3.0:
        return False
    if {m.group(0) for m in _TS_RE.finditer(plain_old)} - {m.group(0) for m in _TS_RE.finditer(plain_new)}:
        return False  # 根拠の時刻が消えた・変わった

    def core(text):  # 「（根拠 hh:mm:ss）」は多くの項目で共通なので、似ている度合いからは除く
        return re.sub(r"\s+", "", _EVIDENCE_NOTE_RE.sub("", text))

    own = difflib.SequenceMatcher(None, core(plain_old), core(plain_new), autojunk=False).ratio()
    if own < FIX_MIN_SIMILARITY:
        return False
    for other in others:
        plain_other = strip_marks(other).strip()
        if plain_other != plain_old and \
                difflib.SequenceMatcher(None, core(plain_other), core(plain_new), autojunk=False).ratio() > own:
            return False
    return True


def verify_document(settings: dict, transcript: str, md: str, memo: str = "",
                    progress_cb=None) -> tuple[str, dict]:
    """文書の各項目を文字起こしと照合して直す。(修正後の文書, 集計) を返す。
    集計: {"checked", "fixed", "unverified", "skipped", "failed", "error"}。
    LLMの応答が壊れていた項目は元のまま残す。途中でLLMが落ちた時は、そこまでの修正を残して error に理由を入れる。
    段落（要点サマリー・要約）と、語の重なりでしか根拠を探せなかった項目は、修正せず OK／確認できず だけを使う
    （部分的な根拠で書き直させると内容が縮む）。"""
    lines = md.split("\n")
    items = _items_with_sections(md)
    report = {"checked": 0, "fixed": 0, "unverified": 0, "skipped": 0, "failed": 0, "error": ""}
    if not items:
        return md, report
    evidence_lines = _transcript_lines(transcript)
    system = _load_prompt(VERIFY_PROMPT) + _context_block(memo)
    budget = llm.input_budget_chars(settings, system, VERIFY_OUTPUT_RESERVE)

    entries = []
    for n, (li, line, section) in enumerate(items, 1):
        prefix, body = _split_prefix(line)
        ev, how = _evidence(evidence_lines, body)
        if not ev:
            report["skipped"] += 1  # 根拠候補が探せない項目は判定しない（勝手に消さない）
            continue
        fixable = how == "time" and not any(k in section for k in _PARAGRAPH_SECTIONS)
        markable = not any(k in section for k in _NO_UNVERIFIED_SECTIONS)
        entries.append((n, li, prefix, body, f"[{n}] 下書き: {body}\n根拠候補:\n{ev}", fixable, markable))

    batches, cur, size = [], [], 0
    for e in entries:
        block = e[4]
        if cur and (len(cur) >= VERIFY_BATCH_ITEMS or size + len(block) + 2 > budget):
            batches.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += len(block) + 2
    if cur:
        batches.append(cur)

    for bi, batch in enumerate(batches):
        if progress_cb:
            progress_cb(bi / len(batches), f"文字起こしと照合中... {bi + 1}/{len(batches)}")
        user = "\n\n".join(e[4] for e in batch)
        try:
            reply = llm.generate(settings, system, user, think=False, temperature=FACTUAL_TEMPERATURE,
                                 output_reserve=VERIFY_OUTPUT_RESERVE)
        except llm.LLMError as e:
            report["error"] = str(e)  # ここまでの修正は残す
            report["failed"] += len(batches) - bi
            break
        except Exception as e:
            logging.warning(f"照合バッチ{bi}で失敗: {e}")
            report["failed"] += 1
            continue
        verdicts = {}
        for line in reply.splitlines():
            m = _VERDICT_RE.match(line)
            if m:
                verdicts.setdefault(int(m.group(1)), (m.group(2), m.group(3).strip()))
        bodies = [e[3] for e in batch]
        for n, li, prefix, body, _, fixable, markable in batch:
            if n not in verdicts:
                continue
            kind, text = verdicts[n]
            report["checked"] += 1
            if kind in ("OK", "ＯＫ"):
                continue
            if kind == "確認できず":
                if markable and UNVERIFIED_MARK not in body:
                    lines[li] = _keep_break(lines[li], _append_mark(prefix, body))
                    report["unverified"] += 1
                continue
            if not fixable:
                continue
            # 修正: 接頭辞を復唱していたら剥がす
            fixed = text
            if prefix and fixed.startswith(prefix.strip()):
                fixed = fixed[len(prefix.strip()):].strip()
            if fixed == body or not _acceptable_fix(body, fixed, [b for b in bodies if b != body]):
                continue
            lines[li] = _keep_break(lines[li], prefix + fixed)
            report["fixed"] += 1
    return "\n".join(lines), report


def _keep_break(old_line: str, new_line: str) -> str:
    """tidy_minutes が付けた行末2スペース（改行の維持）を残す"""
    if old_line.endswith("  ") and not new_line.endswith("  "):
        return new_line.rstrip() + "  "
    return new_line


def _append_mark(prefix: str, body: str) -> str:
    if body.strip().startswith("|"):
        cells = body.split("|")
        for k in range(1, len(cells) - 1):
            if cells[k].strip():
                cells[k] = cells[k].rstrip() + UNVERIFIED_MARK + " "
                break
        return "|".join(cells)
    return prefix + body.rstrip() + UNVERIFIED_MARK


def verify_summary(report: dict) -> str:
    """UI向けの一言"""
    if not report:
        return "照合できる項目がありませんでした。"
    if not report.get("checked"):
        if report.get("error") or report.get("failed"):
            reason = f"（{report['error']}）" if report.get("error") else ""
            return f"文字起こしとの照合はAIの応答が得られず完了しませんでした{reason}。照合前の文書を保存しています。"
        return "照合できる項目がありませんでした。"
    msg = f"AIが{report['checked']}項目を文字起こしと照合"
    if report.get("fixed"):
        msg += f"し、{report['fixed']}項目を根拠に合わせて修正"
    if report.get("unverified"):
        msg += f"。{report['unverified']}項目は根拠が見つからず「{UNVERIFIED_MARK}」を付記"
    if report.get("skipped"):
        msg += f"（{report['skipped']}項目は根拠候補を探せず未判定）"
    msg += "しました。"
    if report.get("error"):
        msg += f" 途中でAIの応答が止まったため、残りの{report.get('failed', 0)}回分は照合していません（{report['error']}）。"
    elif report.get("failed"):
        msg += f" {report['failed']}回分は応答が壊れていたため照合していません。"
    return msg
