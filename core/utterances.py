"""文字起こし結果の構造化: 単語 → 発話（時刻・話者・内容）→ Markdown / HTML / 字幕

発話 (utterance) の形:
    {"start": float, "end": float, "speaker": str,
     "text": str,        # プレーンテキスト（編集・検索・LLM入力用）
     "marked": str,      # 低信頼箇所を赤spanで囲んだテキスト（表示用）
     "words": [{"start", "end", "word", "prob"}]}
"""
import html
import re

RED_OPEN = '<span style="color:red">'
RED_CLOSE = "</span>"
RED_RE = re.compile(r'<span style="color:red">(.*?)</span>', re.DOTALL)

SENTENCE_END = "。？！?!"
SPLIT_PUNCT = "。？！?!、，,"
MAX_UTTERANCE_SEC = 45.0  # 同じ話者でもこれを超えたら文末で区切る（読みやすさ）

# 話者ごとの色（見分けやすい順）
SPEAKER_COLORS = [
    "#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#dc2626", "#65a30d", "#c026d3", "#475569",
]


# =====================================================================
# 時刻フォーマット
# =====================================================================

def format_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_time(seconds: float, sep: str = ",") -> str:
    ms_total = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def parse_time(value) -> float:
    """'HH:MM:SS,mmm' / 'HH:MM:SS.mmm' / 'MM:SS' / 秒数 → 秒"""
    if value is None:
        raise ValueError("時刻が空です")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"時刻の形式が不正です: {value}")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# =====================================================================
# 発話の構築
# =====================================================================

def _smooth_word_speakers(words: list[dict], labels: list[str]) -> list[str]:
    """1〜2語だけ・0.6秒未満で話者が入れ替わるチラつきを前後に吸収する"""
    labels = list(labels)
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        dur = words[j]["end"] - words[i]["start"]
        if (j - i + 1) <= 2 and dur < 0.6:
            prev_l = labels[i - 1] if i > 0 else None
            next_l = labels[j + 1] if j + 1 < n else None
            fill = prev_l or next_l
            if prev_l and next_l and prev_l != next_l:
                fill = prev_l
            if fill:
                for k in range(i, j + 1):
                    labels[k] = fill
        i = j + 1
    return labels


def _snap_segment_edges(words: list[dict], labels: list[str]) -> list[str]:
    """Whisperの文の先頭・末尾1〜2語だけ話者が違う場合、文の大半の話者に揃える。
    話者交代は文の切れ目で起きることが多く、境界の声紋窓が前の人の声を拾いやすいため。"""
    labels = list(labels)
    n = len(labels)
    if n < 4:
        return labels
    for edge in ("head", "tail"):
        idx = list(range(n)) if edge == "head" else list(range(n - 1, -1, -1))
        k = 0
        while k < n and labels[idx[k]] == labels[idx[0]]:
            k += 1
        if 0 < k <= 2 and k < n:
            span = [words[i] for i in idx[:k]]
            dur = max(w["end"] for w in span) - min(w["start"] for w in span)
            body = labels[idx[k]]
            if dur < 0.8 and sum(1 for l in labels if l == body) >= n * 0.6:
                for i in idx[:k]:
                    labels[i] = body
    return labels


def _marked_text(words: list[dict], threshold: float) -> str:
    parts, in_red = [], False
    for w in words:
        low = w.get("prob", 1.0) < threshold
        if low and not in_red:
            parts.append(RED_OPEN)
            in_red = True
        elif not low and in_red:
            parts.append(RED_CLOSE)
            in_red = False
        parts.append(w["word"])
    if in_red:
        parts.append(RED_CLOSE)
    text = "".join(parts).strip()
    # 空の赤spanを除去
    return text.replace(RED_OPEN + RED_CLOSE, "")


def strip_marks(text: str) -> str:
    return RED_RE.sub(r"\1", text or "")


def _finish(utt: dict, threshold: float) -> dict:
    words = utt["words"]
    utt["start"] = words[0]["start"]
    utt["end"] = words[-1]["end"]
    utt["text"] = "".join(w["word"] for w in words).strip()
    utt["marked"] = _marked_text(words, threshold)
    return utt


def build_utterances(segments: list[dict], word_speakers: dict | None,
                     threshold: float = 0.5) -> list[dict]:
    """Whisperセグメント（＋単語ごとの話者）→ 発話リスト"""
    # 全単語を一列に並べる（単語情報の無いセグメントは疑似単語1個にする）
    flat_words, flat_labels = [], []
    for si, seg in enumerate(segments):
        words = seg.get("words") or [{
            "start": seg["start"], "end": seg["end"], "word": seg.get("text", ""), "prob": 1.0,
        }]
        labels = (word_speakers or {}).get(si) or [""] * len(words)
        if len(labels) != len(words):
            labels = [labels[0]] * len(words)
        elif word_speakers:
            labels = _snap_segment_edges(words, labels)
        for w, lab in zip(words, labels):
            if not w.get("word"):
                continue
            flat_words.append(dict(w, seg=si))
            flat_labels.append(lab)

    if not flat_words:
        return []
    if word_speakers:
        flat_labels = _smooth_word_speakers(flat_words, flat_labels)

    utterances = []
    cur = None
    for w, lab in zip(flat_words, flat_labels):
        new_turn = cur is None or lab != cur["speaker"]
        if not new_turn and not word_speakers and w["seg"] != cur["last_seg"]:
            # 話者分離なし: Whisperのセグメント境界で区切る
            new_turn = True
        if not new_turn:
            last = cur["words"][-1]
            long_enough = w["start"] - cur["words"][0]["start"] > MAX_UTTERANCE_SEC
            ends_sentence = last["word"].strip()[-1:] in SENTENCE_END
            if (long_enough and ends_sentence) or w["start"] - last["end"] > 8.0:
                new_turn = True
        if new_turn:
            if cur:
                utterances.append(_finish(cur, threshold))
            cur = {"speaker": lab, "words": []}
        cur["words"].append({k: w[k] for k in ("start", "end", "word", "prob")})
        cur["last_seg"] = w["seg"]
    if cur:
        utterances.append(_finish(cur, threshold))
    for u in utterances:
        u.pop("last_seg", None)
    return [u for u in utterances if u["text"]]


def rename_speaker(utterances: list[dict], old: str, new: str) -> int:
    n = 0
    for u in utterances:
        if u.get("speaker") == old:
            u["speaker"] = new
            n += 1
    return n


def speaker_stats(utterances: list[dict]) -> list[tuple[str, float, int]]:
    """[(話者, 発話秒数, 発話回数)] を発話時間の多い順に"""
    stats = {}
    for u in utterances:
        sp = u.get("speaker") or ""
        if not sp:
            continue
        sec, cnt = stats.get(sp, (0.0, 0))
        stats[sp] = (sec + max(0.0, u["end"] - u["start"]), cnt + 1)
    return sorted(((k, v[0], v[1]) for k, v in stats.items()), key=lambda x: -x[1])


def speaker_color_map(utterances: list[dict]) -> dict:
    order = []
    for u in utterances:
        sp = u.get("speaker") or ""
        if sp and sp not in order:
            order.append(sp)
    return {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(order)}


# =====================================================================
# 1文ずつの行（表示・Markdown共通）
# =====================================================================

SENTENCE_SPLIT = "。？！?!"
MAX_ROW_CHARS = 80  # 句点が無く長く続く発話は読点で区切る


def _char_times(u: dict) -> list[tuple[float, float]]:
    """発話テキスト1文字ごとの (開始, 終了) 時刻"""
    text = u.get("text", "")
    words = _aligned_words(u)
    times = []
    for w in words:
        n = len(w["word"])
        if n == 0:
            continue
        step = (w["end"] - w["start"]) / n
        times += [(w["start"] + step * k, w["start"] + step * (k + 1)) for k in range(n)]
    joined = "".join(w["word"] for w in words)
    lead = len(joined) - len(joined.lstrip())
    times = times[lead:lead + len(text)]
    if len(times) != len(text):  # 念のため: 発話全体で均等割り
        n = max(len(text), 1)
        step = (u["end"] - u["start"]) / n
        times = [(u["start"] + step * k, u["start"] + step * (k + 1)) for k in range(len(text))]
    return times


def _split_points(text: str) -> list[int]:
    """文の区切り位置（区切り文字の直後のindex）"""
    cuts, last = [], 0
    for i, ch in enumerate(text):
        if ch in SENTENCE_SPLIT:
            cuts.append(i + 1)
            last = i + 1
        elif i - last >= MAX_ROW_CHARS and ch in "、，,":
            cuts.append(i + 1)
            last = i + 1
    if not cuts or cuts[-1] != len(text):
        cuts.append(len(text))
    return cuts


def _marked_slices(marked: str, plain_len: int):
    """赤spanを含む文字列について、プレーン文字index → marked内の位置と「赤の中か」を返す"""
    pos, in_red, i = [], [], 0
    red = False
    while i < len(marked):
        if marked.startswith(RED_OPEN, i):
            red = True
            i += len(RED_OPEN)
            continue
        if marked.startswith(RED_CLOSE, i):
            red = False
            i += len(RED_CLOSE)
            continue
        pos.append(i)
        in_red.append(red)
        i += 1
    return pos, in_red


def sentence_rows(utterances: list[dict]) -> list[dict]:
    """発話を1文ずつに区切った行 [{start, end, speaker, text, marked}]"""
    rows = []
    for u in utterances:
        text = u.get("text", "")
        if not text.strip():
            continue
        marked = u.get("marked") or text
        times = _char_times(u)
        pos, in_red = _marked_slices(marked, len(text))
        use_marked = len(pos) == len(text)
        start = 0
        for cut in _split_points(text):
            piece = text[start:cut]
            if piece.strip():
                lo = start + (len(piece) - len(piece.lstrip()))
                hi = cut - (len(piece) - len(piece.rstrip()))
                if use_marked and hi > lo:
                    frag = marked[pos[lo]:pos[hi - 1] + 1]
                    if in_red[lo] and not frag.startswith(RED_OPEN):
                        frag = RED_OPEN + frag
                    if frag.count(RED_OPEN) > frag.count(RED_CLOSE):
                        frag += RED_CLOSE
                else:
                    frag = text[lo:hi]
                rows.append({"start": times[lo][0] if times else u["start"],
                             "end": times[hi - 1][1] if times else u["end"],
                             "speaker": u.get("speaker", ""),
                             "text": text[lo:hi], "marked": frag.replace(RED_OPEN + RED_CLOSE, "")})
            start = cut
    return rows


# =====================================================================
# Markdown
# =====================================================================

def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}時間{m}分{s}秒" if h else (f"{m}分{s}秒" if m else f"{s}秒")


def render_md(utterances: list[dict], source_name: str = "", duration: float = 0.0,
              meta_line: str = "") -> str:
    lines = []
    if source_name:
        lines.append(f"# 文字起こし: {source_name}")
        lines.append("")
    info = []
    if duration:
        info.append(f"収録時間: {_fmt_dur(duration)}")
    if meta_line:
        info.append(meta_line)
    if info:
        lines.append(" ／ ".join(info))
        lines.append("")

    stats = speaker_stats(utterances)
    if stats:
        total = sum(s[1] for s in stats) or 1.0
        lines.append("| 話者 | 発話時間 | 割合 | 発話数 |")
        lines.append("|---|---|---|---|")
        for sp, sec, cnt in stats:
            lines.append(f"| {sp} | {_fmt_dur(sec)} | {sec / total * 100:.0f}% | {cnt} |")
        lines.append("")

    lines.append('※ <span style="color:red">赤字</span>は音声が不明瞭で認識の確信度が低い、'
                 "またはAIが文脈から推測した箇所です。")
    lines.append("")

    # 1文ずつ「番号 / 時刻 / 話者: 本文」。Markdownで改行が消えないよう行末に2スペース
    for i, r in enumerate(sentence_rows(utterances), 1):
        lines.append(f"{i}  ")
        lines.append(f"{format_srt_time(r['start'])} --> {format_srt_time(r['end'])}  ")
        speaker = r.get("speaker") or ""
        lines.append(f"**{speaker}:** {r['marked']}" if speaker else r["marked"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# =====================================================================
# HTML（アプリ内の見やすい表示）
# =====================================================================

def _escape_keep_red(text: str) -> str:
    po, pc = "\x00RO\x00", "\x00RC\x00"
    t = (text or "").replace(RED_OPEN, po).replace(RED_CLOSE, pc)
    t = html.escape(t)
    return t.replace(po, '<span class="st-red">').replace(pc, "</span>")


VIEW_CSS = """
<style>
.st-wrap{font-family:system-ui,-apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI",sans-serif;
  line-height:1.75;color:var(--body-text-color,#111)}
.st-stats{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}
.st-chip{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;
  background:var(--block-background-fill,#f4f4f5);border:1px solid var(--border-color-primary,#e4e4e7);font-size:13px}
.st-dot{width:10px;height:10px;border-radius:50%;flex:none}
.st-bar{height:6px;border-radius:3px;background:var(--border-color-primary,#e4e4e7);overflow:hidden;width:60px}
.st-bar>i{display:block;height:100%}
.st-list{max-height:640px;overflow-y:auto;padding-right:4px}
.st-u{padding:9px 6px;border-bottom:1px solid var(--border-color-primary,#eee)}
.st-u.st-turn{border-top:2px solid var(--border-color-primary,#ddd)}
.st-no{font-size:12px;color:var(--body-text-color-subdued,#71717a);font-variant-numeric:tabular-nums}
.st-time{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;font-size:12px;
  color:var(--body-text-color-subdued,#71717a);cursor:pointer;user-select:none;display:inline-block}
.st-time:hover{text-decoration:underline}
.st-sp{font-weight:700}
.st-text{white-space:pre-wrap;word-break:break-word}
.st-red{color:#dc2626;background:rgba(220,38,38,.08);border-radius:3px}
.st-legend{font-size:12px;color:var(--body-text-color-subdued,#71717a);margin-bottom:8px}
.st-player{width:100%;margin:0 0 10px}

</style>
"""


def render_html(utterances: list[dict], audio_url: str = "") -> str:
    if not utterances:
        return '<div class="st-wrap">文字起こし結果がありません。</div>'
    colors = speaker_color_map(utterances)
    stats = speaker_stats(utterances)
    total = sum(s[1] for s in stats) or 1.0

    out = [VIEW_CSS, '<div class="st-wrap">']
    if audio_url:
        out.append(f'<audio class="st-player" id="st-player" controls preload="metadata" '
                   f'src="{html.escape(audio_url)}"></audio>')
    if stats:
        out.append('<div class="st-stats">')
        for sp, sec, cnt in stats:
            c = colors.get(sp, "#888")
            pct = sec / total * 100
            out.append(
                f'<span class="st-chip"><span class="st-dot" style="background:{c}"></span>'
                f'<b>{html.escape(sp)}</b>{_fmt_dur(sec)}'
                f'<span class="st-bar"><i style="width:{pct:.0f}%;background:{c}"></i></span>'
                f'{pct:.0f}%</span>'
            )
        out.append("</div>")
    legend = "赤字＝聞き取りに自信がない／AIが推測した箇所"
    if audio_url:
        legend += "　・　時刻をクリックするとその位置から再生"
    out.append(f'<div class="st-legend">{legend}</div>')

    out.append('<div class="st-list">')
    seek_js = ("var a=document.getElementById('st-player');"
               "if(a){a.currentTime=%.2f;a.play();}")
    prev = None
    for i, r in enumerate(sentence_rows(utterances), 1):
        sp = r.get("speaker") or ""
        c = colors.get(sp, "#888")
        onclick = f' onclick="{seek_js % r["start"]}"' if audio_url else ""
        turn = " st-turn" if prev is not None and sp != prev else ""
        prev = sp
        out.append(f'<div class="st-u{turn}"><div class="st-no">{i}</div>')
        out.append(f'<div class="st-time"{onclick}>{format_srt_time(r["start"])} --&gt; '
                   f'{format_srt_time(r["end"])}</div>')
        label = f'<span class="st-sp" style="color:{c}">{html.escape(sp)}:</span> ' if sp else ""
        out.append(f'<div class="st-text">{label}{_escape_keep_red(r["marked"])}</div></div>')
    out.append("</div></div>")
    return "".join(out)


# =====================================================================
# 字幕（SRT / VTT）
# =====================================================================

DEFAULT_SUB_OPTS = {
    "max_chars_line": 20,     # 1行の最大文字数（日本語字幕の目安: 16〜20）
    "max_lines": 2,
    "max_duration": 6.0,      # 1枚の最大表示秒数
    "min_duration": 1.0,      # 1枚の最小表示秒数（読めない速さを防ぐ）
    "pause_split": 0.6,       # この秒数以上の無音で字幕を切り替える
    "speaker_prefix": False,  # 先頭に（話者名）を付ける
    "drop_period": True,      # 字幕慣習に合わせて句点「。」を削除
}


def _visible_len(text: str) -> int:
    return len(text.strip())


def _char_class(ch: str) -> str:
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hira"
    if 0x30A0 <= o <= 0x30FF or ch == "ー":
        return "kata"
    if 0x4E00 <= o <= 0x9FFF or ch in "々〆":
        return "kanji"
    if ch.isdigit():
        return "digit"
    if ch.isascii() and ch.isalpha():
        return "alpha"
    return "other"


def _best_split(text: str, max_chars: int, lines: int) -> int:
    """先頭1行の切れ目を選ぶ（残りは lines-1 行に収める前提）"""
    target = len(text) / lines
    rest_cap = max_chars * (lines - 1)
    best, best_score = None, None
    for i in range(1, len(text)):
        head, rest = text[:i].strip(), text[i:].strip()
        if len(head) > max_chars + 4 or len(rest) > rest_cap + 4 * (lines - 1):
            continue
        score = abs(i - target)
        over = max(0, len(head) - max_chars) + max(0, len(rest) - rest_cap)
        score += over * 3
        a, b = text[i - 1], text[i]
        if _char_class(a) == _char_class(b) and _char_class(a) in ("kata", "digit", "alpha"):
            score += 8  # カタカナ語・数字・英単語の途中では切らない
        elif _char_class(a) == _char_class(b) == "kanji":
            score += 3
        elif a in "をはがでにとへもや" and _char_class(b) != "hira":
            score -= 3  # 助詞の後は自然な切れ目
        if b in "　 ":
            score -= 8  # 文の切れ目（句点を消した跡）で改行するのが最も自然
        elif a in SENTENCE_END:
            score -= 10
        elif a in SPLIT_PUNCT:
            score -= 6
        elif a in "　 ":
            score -= 4
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best if best is not None else int(target)


def _wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    """長い字幕を読点・区切りの良い位置で最大 max_lines 行に改行する"""
    text = text.strip()
    if len(text) <= max_chars or max_lines < 2:
        return text
    lines = min(max_lines, max(2, -(-len(text) // max_chars)))
    i = _best_split(text, max_chars, lines)
    return text[:i].strip() + "\n" + _wrap_lines(text[i:], max_chars, lines - 1)


def _aligned_words(u: dict) -> list[dict]:
    """発話の現在のテキスト（校正・手修正・人名統一後）を、元の単語タイミングに対応付ける。
    テキストが変わっていなければ元の単語をそのまま返す。"""
    import difflib

    text = (u.get("text") or "").strip()
    words = [w for w in (u.get("words") or []) if w.get("word")]
    if not words:
        return [{"start": u["start"], "end": u["end"], "word": text}]
    joined = "".join(w["word"] for w in words)
    if joined.strip() == text:
        return words

    times = []  # 元テキスト1文字ごとの (start, end)
    for w in words:
        n = len(w["word"])
        step = (w["end"] - w["start"]) / n
        times += [(w["start"] + step * k, w["start"] + step * (k + 1)) for k in range(n)]

    new_times = [None] * len(text)
    sm = difflib.SequenceMatcher(None, joined, text, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(j2 - j1):
                new_times[j1 + k] = times[i1 + k]
        elif op in ("replace", "insert"):
            if i2 > i1:
                t0, t1 = times[i1][0], times[i2 - 1][1]
            else:
                t0 = t1 = times[i1 - 1][1] if i1 > 0 else times[0][0]
            span = j2 - j1
            for k in range(span):
                new_times[j1 + k] = (t0 + (t1 - t0) * k / span, t0 + (t1 - t0) * (k + 1) / span)
    return [{"start": a, "end": b, "word": ch} for ch, (a, b) in zip(text, new_times)]


def build_cues(utterances: list[dict], opts: dict | None = None) -> list[dict]:
    """発話 → 字幕キュー [{start, end, speaker, text}]"""
    o = dict(DEFAULT_SUB_OPTS, **(opts or {}))
    max_chars = int(o["max_chars_line"]) * int(o["max_lines"])
    cues = []

    for u in utterances:
        words = _aligned_words(u)
        buf = []

        def flush():
            if not buf:
                return
            text = "".join(w["word"] for w in buf).strip()
            if text:
                cues.append({"start": buf[0]["start"], "end": buf[-1]["end"],
                             "speaker": u.get("speaker", ""), "text": text})
            buf.clear()

        for w in words:
            if buf:
                cur_text = "".join(x["word"] for x in buf)
                too_long = _visible_len(cur_text + w["word"]) > max_chars
                too_slow = w["end"] - buf[0]["start"] > o["max_duration"]
                gap = w["start"] - buf[-1]["end"]
                # 数文字しか溜まっていない所では切らない（「平」「岡です」のような分断を防ぐ）
                paused = gap >= max(o["pause_split"] * 2.5, 1.5) or \
                    (gap >= o["pause_split"] and _visible_len(cur_text) >= 4)
                if too_long or too_slow or paused:
                    # 句読点で切れる位置まで戻せるなら戻す（文の途中で切れにくくする）
                    cut = None
                    if too_long or too_slow:
                        for k in range(len(buf) - 1, 0, -1):
                            if buf[k - 1]["word"].strip()[-1:] in SPLIT_PUNCT:
                                if _visible_len("".join(x["word"] for x in buf[:k])) >= max_chars * 0.4:
                                    cut = k
                                break
                    if cut:
                        rest = buf[cut:]
                        del buf[cut:]
                        flush()
                        buf.extend(rest)
                    else:
                        flush()
            buf.append(w)
            if w["word"].strip()[-1:] in SENTENCE_END and \
                    _visible_len("".join(x["word"] for x in buf)) >= 6:
                flush()
        flush()

    return finalize_cues(_merge_short_cues(cues, o), o)


def _merge_short_cues(cues: list[dict], o: dict) -> list[dict]:
    """表示時間が短すぎる字幕を、同じ話者の隣の字幕とまとめる（読めない速さを防ぐ）"""
    max_chars = int(o["max_chars_line"]) * int(o["max_lines"])
    merged = []
    for c in cues:
        c = dict(c)
        if merged:
            p = merged[-1]
            short = (p["end"] - p["start"] < o["min_duration"]) or \
                (c["end"] - c["start"] < o["min_duration"])
            fits = _visible_len(p["text"] + c["text"]) <= max_chars and \
                c["end"] - p["start"] <= o["max_duration"] * 1.3
            if short and fits and p["speaker"] == c["speaker"] and c["start"] - p["end"] < 1.0:
                sep = "" if p["text"][-1:] in SPLIT_PUNCT else "　"
                p["text"] = p["text"] + sep + c["text"]
                p["end"] = c["end"]
                continue
        merged.append(c)
    return merged


def finalize_cues(cues: list[dict], opts: dict | None = None, add_tail: bool = True) -> list[dict]:
    """表示時間の補正・重なり解消・改行・句点処理（add_tail: 末尾に0.2秒の余韻を足す）"""
    o = dict(DEFAULT_SUB_OPTS, **(opts or {}))
    cues = sorted((dict(c) for c in cues if str(c.get("text", "")).strip()),
                  key=lambda c: c["start"])
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        # 最低表示時間を確保（次の字幕にかぶらない範囲で延長）
        if c["end"] - c["start"] < o["min_duration"]:
            want = c["start"] + o["min_duration"]
            c["end"] = min(want, nxt - 0.05) if nxt is not None else want
        # 少しだけ余韻を持たせる
        tail = c["end"] + (0.2 if add_tail else 0.0)
        c["end"] = min(tail, nxt - 0.05) if nxt is not None else tail
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.3
        text = str(c["text"]).replace("\n", "").strip()
        if o["drop_period"]:
            text = re.sub(r"。(?=.)", "　", text).rstrip("。").strip()
        c["text"] = _wrap_lines(text, int(o["max_chars_line"]), int(o["max_lines"]))
    return cues


def cue_display_text(cue: dict, speaker_prefix: bool) -> str:
    text = cue["text"]
    if speaker_prefix and cue.get("speaker"):
        text = f"（{cue['speaker']}）{text}"
    return text


def cues_to_srt(cues: list[dict], speaker_prefix: bool = False) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out += [str(i), f"{format_srt_time(c['start'])} --> {format_srt_time(c['end'])}",
                cue_display_text(c, speaker_prefix), ""]
    return "\n".join(out)


def cues_to_vtt(cues: list[dict], speaker_prefix: bool = False) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        text = cue_display_text(c, False)
        if speaker_prefix and c.get("speaker"):
            text = f"<v {c['speaker']}>{text}"
        out += [f"{format_srt_time(c['start'], '.')} --> {format_srt_time(c['end'], '.')}", text, ""]
    return "\n".join(out)


_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3}"
_SRT_TIME_RE = re.compile(rf"({_TS})\s*-->\s*({_TS})")


def parse_srt(text: str, known_speakers=None) -> list[dict]:
    """SRT/VTT文字列 → キュー（外部字幕の読み込み用）

    先頭の（名前）は known_speakers に含まれる場合だけ話者として扱う（（拍手）等は本文のまま）。"""
    known = set(known_speakers or [])
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("﻿", ""))
    for block in blocks:
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for idx, line in enumerate(lines):
            m = _SRT_TIME_RE.search(line)
            if m:
                body = "\n".join(lines[idx + 1:]).strip()
                speaker = ""
                sm = re.match(r"^[（(]([^）)]{1,20})[）)]", body)
                if sm and sm.group(1) in known and body[sm.end():].strip():
                    speaker, body = sm.group(1), body[sm.end():].strip()
                vm = re.match(r"^<v ([^>]+)>", body)
                if vm:
                    speaker, body = vm.group(1), body[vm.end():].strip()
                if body:
                    cues.append({"start": parse_time(m.group(1)), "end": parse_time(m.group(2)),
                                 "speaker": speaker, "text": body})
                break
    return cues


def validate_cues(cues: list[dict], opts: dict | None = None) -> list[str]:
    """字幕の品質チェック結果（人が読む警告）"""
    o = dict(DEFAULT_SUB_OPTS, **(opts or {}))
    warns = []
    for i, c in enumerate(cues, 1):
        dur = c["end"] - c["start"]
        text = str(c["text"])
        chars = len(text.replace("\n", ""))
        if dur <= 0:
            warns.append(f"#{i}: 終了時刻が開始時刻以前です")
        elif chars / dur > 12:
            warns.append(f"#{i}: 読む速度が速すぎます（{chars / dur:.1f}文字/秒）")
        if any(len(l) > o["max_chars_line"] + 4 for l in text.split("\n")):
            warns.append(f"#{i}: 1行が長すぎます")
        if text.count("\n") + 1 > o["max_lines"]:
            warns.append(f"#{i}: 行数が多すぎます")
        if i < len(cues) and c["end"] > cues[i]["start"] + 1e-3:
            warns.append(f"#{i}: 次の字幕と時間が重なっています")
    return warns
