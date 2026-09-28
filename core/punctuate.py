"""句読点の推定（Whisperが句読点をほとんど付けない日本語向け）

単語タイムスタンプの「間」と文末表現から「、」「。」「？」を補う。高速・LLM不要。
推定なので完璧ではないが、読みやすさが大きく上がる（AI校正を使うとさらに整う）。
"""
import re

PUNCT = "、。？！?!，,."
# 文末になりやすい表現（直後に間があれば「。」）
SENTENCE_END_RE = re.compile(
    r"(です|ます|ました|でした|ません|ませんでした|でしょう|ましょう|ください|ですね|ますね|"
    r"ですよ|ますよ|ですよね|ますよね|だね|だよ|よね|かな|"
    r"思います|と思う|なります|あります|います|おります|ございます|ある|いる|した|だ)$")
# 文をつなぐ表現。間があっても文はまだ続くことが多いので「、」（長い沈黙の時だけ「。」）
CONJ_END_RE = re.compile(r"(けど|けれど|けれども|ので|から|って|けども)$")
QUESTION_RE = re.compile(r"(ですか|ますか|でしょうか|ましたか|でしたか|のか|んですか|かね|かな|の\?)$")

COMMA_GAP = 0.5      # この秒数以上の間で「、」（0.35秒だと言いよどみの度に入って多すぎた）
PERIOD_GAP = 0.9     # この秒数以上の間で「。」（文末表現があれば SOFT_PERIOD_GAP で可）
SOFT_PERIOD_GAP = 0.25
CONJ_PERIOD_GAP = 1.5  # 「〜けど」「〜ので」の後はここまで黙った時だけ「。」

# 語の頭に来ない文字（この前で区切ると「ワ、ークスペース」「デ。ータ」のように語が割れる）
_NO_WORD_HEAD = "ーぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッヵヶんン"
# 句読点の直前に来ない助詞（「会議、を」のような位置を避ける）
_PARTICLES = {"を", "が", "は", "に", "の", "も", "へ", "や", "ね", "よ"}


def _ends_with_punct(text: str) -> bool:
    return bool(text) and text.rstrip()[-1:] in PUNCT


def _is_japanese(ch: str) -> bool:
    return bool(ch) and ("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" or ch in "々〆ー")


def _is_kanji(ch: str) -> bool:
    return bool(ch) and ("一" <= ch <= "鿿" or ch in "々〆")


def _is_katakana(ch: str) -> bool:
    return bool(ch) and ("ァ" <= ch <= "ヺ" or ch == "ー")


def inside_word(token: str, nxt: str) -> bool:
    """token と次の単語 nxt の間が語の途中か（Whisperの単語は語の切れ端のことが多い）。
    ここに句読点を入れると語が割れて、行もそこで切れてしまう。"""
    t, n = token.strip(), nxt.strip()
    if not t or not n:
        return False
    if n[0] in _NO_WORD_HEAD or t[-1] in "っッ":
        return True
    if _is_katakana(t[-1]) and _is_katakana(n[0]):   # 「グ|ーグル」「エ|ージェント」
        return True
    if t[-1].isdigit() and (n[0].isdigit() or n[0] in "万億千百円%％"):  # 「1|万円」
        return True
    # 漢字1文字の直後が漢字（「平、岡」のような分断を防ぐ）
    return len(t) == 1 and _is_kanji(t) and _is_kanji(n[0])


_tokenizer = None


def _janome():
    global _tokenizer
    if _tokenizer is None:
        try:
            from janome.tokenizer import Tokenizer
            _tokenizer = Tokenizer()
        except Exception:  # 未導入なら辞書判定なしで動く
            _tokenizer = False
    return _tokenizer or None


def _same_morpheme(before: str, char: str, after: str) -> bool:
    """before+char+after を形態素解析して、char と after の先頭が同じ語に入るか"""
    tok = _janome()
    if tok is None or not after:
        return False
    text = before + char + after
    pos = len(before)
    i = 0
    for t in tok.tokenize(text, wakati=True):
        if i <= pos < i + len(t):
            return pos + 1 < i + len(t)
        i += len(t)
    return False


def glue_word_heads(words: list[dict], labels: list | None = None) -> int:
    """語の頭の1文字だけ時刻が前にずれて、続きと離れている単語を続きの直前に寄せる（その場で変更）。
    例:「エ」…9秒…「ビデンス」、「こ」…4秒…「っちが」。Whisperの単語時刻のずれで、
    放置すると1文字だけ句読点が付き、別の行・別の話者になる。寄せた数を返す。"""
    glued = 0
    for k in range(len(words) - 2, -1, -1):
        w, nxt = words[k], words[k + 1]
        if nxt["start"] - w["end"] <= 0.5:
            continue
        head = w["word"].strip().rstrip(PUNCT)
        if not head or len(head) > 2:  # ずれるのは語の頭の1〜2文字だけ
            continue
        after = "".join(x["word"] for x in words[k + 1:k + 4]).strip()
        if re.match(r"ー[一-鿿]", after):  # 「ー番」は「一番」の誤認識で、語の続きではない
            continue
        if not inside_word(head, after):
            if len(head) != 1:
                continue
            before = "".join(x["word"] for x in words[max(0, k - 3):k]).strip()[-8:]
            if not _same_morpheme(before, head, after[:10]):
                continue
        dur = min(w["end"] - w["start"], 0.3)
        w["word"] = w["word"].rstrip().rstrip(PUNCT)
        w["end"] = nxt["start"]
        w["start"] = nxt["start"] - dur
        if labels is not None:
            labels[k] = labels[k + 1]
        glued += 1
    return glued


def _mark_for(tail: str, gap: float, space_break: bool, nxt: str) -> str:
    if QUESTION_RE.search(tail) and (gap >= SOFT_PERIOD_GAP or space_break):
        return "？"
    if CONJ_END_RE.search(tail):
        if gap >= CONJ_PERIOD_GAP:
            return "。"
        return "、" if (gap >= SOFT_PERIOD_GAP or space_break) else ""
    if space_break:
        return "。" if (SENTENCE_END_RE.search(tail) or gap >= SOFT_PERIOD_GAP) else "、"
    if gap >= PERIOD_GAP or (gap >= SOFT_PERIOD_GAP and SENTENCE_END_RE.search(tail)):
        return "。"
    if gap >= COMMA_GAP and nxt.strip() not in _PARTICLES:
        return "、"
    return ""


def punctuate_segments(segments: list[dict]) -> int:
    """Whisperのセグメント列の単語に句読点を追記する（その場で変更）。追加した数を返す。"""
    words = [(si, wi) for si, seg in enumerate(segments) for wi, _ in enumerate(seg.get("words") or [])]
    glue_word_heads([segments[si]["words"][wi] for si, wi in words])
    added = 0
    for k, (si, wi) in enumerate(words):
        w = segments[si]["words"][wi]
        token = w["word"]
        if not token.strip() or _ends_with_punct(token):
            continue
        nxt = segments[words[k + 1][0]]["words"][words[k + 1][1]] if k + 1 < len(words) else None
        gap = nxt["start"] - w["end"] if nxt else 10.0
        if nxt and inside_word(token, nxt["word"]):
            continue
        # 直前の数語をつなげて文末表現を判定（単語が細切れのため）
        tail = "".join(segments[a]["words"][b]["word"] for a, b in words[max(0, k - 3):k + 1]).strip()
        # Whisperは日本語の文の切れ目を空白で表すことがある（特に一括処理モード）
        space_break = False
        if nxt and nxt["word"].startswith((" ", "　")) and _is_japanese(token.strip()[-1:]) \
                and _is_japanese(nxt["word"].strip()[:1]):
            space_break = True
            nxt["word"] = nxt["word"].lstrip(" 　")
        mark = _mark_for(tail, gap, space_break, nxt["word"] if nxt else "")
        if mark:
            w["word"] = token.rstrip() + mark
            added += 1
    for seg in segments:
        if seg.get("words"):
            seg["text"] = "".join(w["word"] for w in seg["words"]).strip()
    return added


def repair_words(words: list[dict]) -> int:
    """以前の版で付けた、語の途中の句読点と「〜けど。」「〜ので。」の「。」を直す（その場で変更）。
    保存済みの結果を作り直す時に使う。直した数を返す。"""
    fixed = 0
    for i, w in enumerate(words):
        token = w.get("word", "")
        if token[-1:] not in "、。":
            continue
        nxt = words[i + 1] if i + 1 < len(words) else None
        body = token[:-1]
        if nxt and body.strip() and inside_word(body, nxt.get("word", "")):
            w["word"] = body
            fixed += 1
            continue
        if token.endswith("。") and nxt and nxt["start"] - w["end"] < CONJ_PERIOD_GAP:
            tail = "".join(x["word"] for x in words[max(0, i - 3):i]) + body
            if CONJ_END_RE.search(tail.strip()):
                w["word"] = body + "、"
                fixed += 1
    return fixed
