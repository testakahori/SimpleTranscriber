"""句読点の推定（Whisperが句読点をほとんど付けない日本語向け）

単語タイムスタンプの「間」と文末表現から「、」「。」「？」を補う。高速・LLM不要。
推定なので完璧ではないが、読みやすさが大きく上がる（AI校正を使うとさらに整う）。
"""
import re

PUNCT = "、。？！?!，,."
# 文末になりやすい表現（直後に間があれば「。」）
SENTENCE_END_RE = re.compile(
    r"(です|ます|ました|でした|ません|ませんでした|でしょう|ましょう|ください|ですね|ますね|"
    r"ですよ|ますよ|ですよね|ますよね|だね|だよ|よね|かな|けど|けれど|ので|から|って|"
    r"思います|と思う|なります|あります|います|おります|ございます|ある|いる|した|だ)$")
QUESTION_RE = re.compile(r"(ですか|ますか|でしょうか|ましたか|でしたか|のか|んですか|かね|かな|の\?)$")

COMMA_GAP = 0.35     # この秒数以上の間で「、」
PERIOD_GAP = 0.9     # この秒数以上の間で「。」（文末表現があれば SOFT_PERIOD_GAP で可）
SOFT_PERIOD_GAP = 0.25


def _ends_with_punct(text: str) -> bool:
    return bool(text) and text.rstrip()[-1:] in PUNCT


def punctuate_segments(segments: list[dict]) -> int:
    """Whisperのセグメント列の単語に句読点を追記する（その場で変更）。追加した数を返す。"""
    words = [(si, wi) for si, seg in enumerate(segments) for wi, _ in enumerate(seg.get("words") or [])]
    added = 0
    for k, (si, wi) in enumerate(words):
        w = segments[si]["words"][wi]
        token = w["word"]
        if not token.strip() or _ends_with_punct(token):
            continue
        if k + 1 < len(words):
            nsi, nwi = words[k + 1]
            gap = segments[nsi]["words"][nwi]["start"] - w["end"]
        else:
            gap = 10.0
        # 直前の数語をつなげて文末表現を判定（単語が細切れのため）
        tail = "".join(segments[a]["words"][b]["word"] for a, b in words[max(0, k - 3):k + 1]).strip()
        mark = ""
        if QUESTION_RE.search(tail) and gap >= SOFT_PERIOD_GAP:
            mark = "？"
        elif gap >= PERIOD_GAP or (gap >= SOFT_PERIOD_GAP and SENTENCE_END_RE.search(tail)):
            mark = "。"
        elif gap >= COMMA_GAP:
            mark = "、"
        if mark:
            w["word"] = token.rstrip() + mark
            added += 1
    for seg in segments:
        if seg.get("words"):
            seg["text"] = "".join(w["word"] for w in seg["words"]).strip()
    return added
