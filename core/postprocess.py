"""文字起こし結果の後処理: 低信頼度の赤字マーキング、テキスト整形"""
import html

RED_OPEN = '<span style="color:red">'
RED_CLOSE = "</span>"


def format_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segment_marked_text(segment: dict, threshold: float) -> str:
    """1セグメントを、低信頼単語を赤spanで囲んだテキストにする。
    単語情報が無い場合はそのままのテキストを返す。"""
    words = segment.get("words") or []
    if not words:
        return segment.get("text", "")

    parts = []
    in_red = False
    for w in words:
        low = w.get("prob", 1.0) < threshold
        token = w.get("word", "")
        if low and not in_red:
            parts.append(RED_OPEN)
            in_red = True
        elif not low and in_red:
            parts.append(RED_CLOSE)
            in_red = False
        parts.append(token)
    if in_red:
        parts.append(RED_CLOSE)
    return "".join(parts).strip()


def build_transcript_md(result: dict, threshold: float, speakers: dict | None = None,
                        source_name: str = "") -> str:
    """タイムスタンプ＋（あれば話者名）付きのMarkdown文字起こしを生成する。
    低信頼箇所は赤spanでマークされる。

    speakers: {segment_index: "話者名"} （話者分離を実行した場合のみ）
    """
    header = ""
    if source_name:
        header = (
            f"# 文字起こし: {source_name}\n\n"
            '※ <span style="color:red">赤字</span>は音声が不明瞭なため'
            "AIが文脈から推測した可能性のある箇所です。\n\n"
        )
    body_lines = []
    for i, seg in enumerate(result.get("segments", [])):
        ts = format_hms(seg["start"])
        text = segment_marked_text(seg, threshold)
        if not text:
            continue
        if speakers and i in speakers:
            body_lines.append(f"**[{ts}] {speakers[i]}:** {text}")
        else:
            body_lines.append(f"**[{ts}]** {text}")
    return header + "\n\n".join(body_lines)


def build_plain_text(result: dict) -> str:
    """マークなしの平文（LLM入力・SRT用）"""
    return "\n".join(seg["text"] for seg in result.get("segments", []) if seg.get("text"))


def build_srt(result: dict) -> str:
    lines = []
    n = 0
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        n += 1
        lines.append(str(n))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def low_confidence_ratio(result: dict, threshold: float) -> float:
    """低信頼単語の割合（meta.json用の品質指標）"""
    total = 0
    low = 0
    for seg in result.get("segments", []):
        for w in (seg.get("words") or []):
            total += 1
            if w.get("prob", 1.0) < threshold:
                low += 1
    return round(low / total, 3) if total else 0.0


def md_to_display_html(md_text: str) -> str:
    """簡易プレビュー用: 赤spanを保持したままHTML化する（太字と改行のみ対応）"""
    # 赤spanを一旦退避してエスケープ
    placeholder_open = "\x00RED_OPEN\x00"
    placeholder_close = "\x00RED_CLOSE\x00"
    text = md_text.replace(RED_OPEN, placeholder_open).replace(RED_CLOSE, placeholder_close)
    text = html.escape(text)
    text = text.replace(placeholder_open, RED_OPEN).replace(placeholder_close, RED_CLOSE)
    # **bold**
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"^# (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = text.replace("\n", "<br>")
    return (
        '<div style="max-height:420px;overflow-y:auto;padding:12px;'
        'border:1px solid #ccc;border-radius:8px;line-height:1.8">'
        f"{text}</div>"
    )
