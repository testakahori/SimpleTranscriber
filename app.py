import gradio as gr
from faster_whisper import WhisperModel
import os
import logging
import time

# 機密情報のロギングを防ぐための設定（標準出力のみ、最小限のフォーマット）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# グローバルにモデルを保持（ロードのオーバーヘッドを防ぐため）
model = None

def load_model():
    global model
    if model is None:
        try:
            # CPU環境を想定したデフォルト設定でロードします
            # GPUがある場合は自動的に速くなりますが、Windows全般の環境を想定しエラーが出にくい設定にします
            logging.info("文字起こしモデル 'small' をロードしています...")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            logging.info("モデルのロードが完了しました。実行準備OKです。")
        except Exception as e:
            logging.error(f"モデルのロードに失敗しました: {e}")
            raise e
    return model

def format_srt_time(seconds):
    """秒数を SRT タイムコード形式 (HH:MM:SS,mmm) に変換する"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_srt(lines, segments_data):
    """テキスト行とセグメント情報からSRT文字列を生成する。
    行数が一致すれば1:1マッピング、不一致なら全体を1ブロックにまとめる。"""
    srt_lines = []
    if segments_data and len(lines) == len(segments_data):
        # 行数一致: タイムスタンプを1:1でマッピング
        for i, (line, (start, end)) in enumerate(zip(lines, segments_data), start=1):
            srt_lines.append(str(i))
            srt_lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
            srt_lines.append(line.strip())
            srt_lines.append("")
    elif segments_data:
        # 行数不一致: 全体の開始〜終了タイムで1ブロック
        total_start = segments_data[0][0]
        total_end = segments_data[-1][1]
        srt_lines.append("1")
        srt_lines.append(f"{format_srt_time(total_start)} --> {format_srt_time(total_end)}")
        srt_lines.extend([l.strip() for l in lines])
        srt_lines.append("")
    else:
        # タイミング情報なし: 00:00:00,000 始まりの1ブロック
        srt_lines.append("1")
        srt_lines.append("00:00:00,000 --> 00:00:00,000")
        srt_lines.extend([l.strip() for l in lines])
        srt_lines.append("")
    return "\n".join(srt_lines)

def transcribe(audio_path):
    if not audio_path:
        return (
            "エラー: 音声ファイルが提供されていません。",
            None, None, [],
            gr.update(interactive=False), gr.update(interactive=False)
        )

    try:
        logging.info("文字起こしを開始します...")
        start_time = time.time()

        m = load_model()

        # vad_filter=Trueを入れることで無音部分のハルシネーション（幻覚）を防ぐ（長尺音声に非常に有効）
        segments, info = m.transcribe(audio_path, beam_size=5, vad_filter=True)

        logging.info(f"検出言語: {info.language} (確率: {info.language_probability:.2f})")

        full_text = ""
        srt_lines_raw = []
        segments_data = []  # [(start, end), ...] タイムスタンプのみ保持

        for i, segment in enumerate(segments, start=1):
            # ログには詳細なテキストは出さない（機密情報保護のため）
            logging.info(f"処理進捗: [{segment.start:.2f}s -> {segment.end:.2f}s]")
            full_text += segment.text + "\n"
            segments_data.append((segment.start, segment.end))
            # SRT形式: 連番 / タイムコード / テキスト / 空行
            srt_lines_raw.append(str(i))
            srt_lines_raw.append(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}")
            srt_lines_raw.append(segment.text.strip())
            srt_lines_raw.append("")

        elapsed_time = time.time() - start_time
        logging.info(f"文字起こし完了。処理時間: {elapsed_time:.2f}秒")

        if not full_text.strip():
            return (
                "音声からテキストが検出されませんでした。",
                None, None, [],
                gr.update(interactive=False), gr.update(interactive=False)
            )

        timestamp = int(time.time())
        base_path = os.path.join(os.getcwd(), f"transcription_result_{timestamp}")

        # TXTファイルとして保存
        txt_path = base_path + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text.strip())

        # SRTファイルとして保存
        srt_path = base_path + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines_raw))

        return (
            full_text.strip(),
            txt_path, srt_path,
            segments_data,
            gr.update(interactive=True),   # edit_btn を有効化
            gr.update(interactive=True),   # save_btn を有効化
        )

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        # 機密情報のため、必要以上にスタックトレースは出力しない
        logging.error("エラーが発生しました。詳細は画面に出力されます。")
        return (
            error_msg, None, None, [],
            gr.update(interactive=False), gr.update(interactive=False)
        )

def enable_edit():
    """編集ボタン: テキストボックスを編集可能にする"""
    return gr.update(interactive=True)

def save_edited(edited_text, segments_data):
    """保存ボタン: 編集済みテキストからTXT/SRTを再生成する"""
    if not edited_text or not edited_text.strip():
        return None, None

    lines = [l for l in edited_text.strip().split("\n") if l.strip()]

    timestamp = int(time.time())
    base_path = os.path.join(os.getcwd(), f"transcription_result_{timestamp}")

    # TXTファイルとして保存
    txt_path = base_path + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(edited_text.strip())

    # SRTファイルとして保存
    srt_path = base_path + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(build_srt(lines, segments_data))

    return txt_path, srt_path

# Gradio UIの構築
with gr.Blocks(title="シンプル文字起こしツール") as demo:
    gr.Markdown("# 🎙️ シンプル文字起こしツール (完全オフライン実行)")
    gr.Markdown("アップロードされた音声ファイル（mp3, wav, mp4など）をテキスト化します。音声データが外部送信されることはありません。")

    segments_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="音声ファイル")
            submit_btn = gr.Button("文字起こし開始", variant="primary")
            gr.Markdown("※長時間のファイル（数十分〜1時間）は、処理にPCのスペック依存でまとまった時間がかかります。\n画面を閉じずにそのままお待ちください。")

        with gr.Column(scale=1):
            text_output = gr.Textbox(label="文字起こし結果", lines=15, interactive=False)
            with gr.Row():
                edit_btn = gr.Button("編集", interactive=False)
                save_btn = gr.Button("保存", interactive=False, variant="primary")
            file_output_txt = gr.File(label="テキストダウンロード (txt)")
            file_output_srt = gr.File(label="字幕ダウンロード (srt)")

    submit_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[text_output, file_output_txt, file_output_srt, segments_state, edit_btn, save_btn]
    )

    edit_btn.click(
        fn=enable_edit,
        outputs=text_output
    )

    save_btn.click(
        fn=save_edited,
        inputs=[text_output, segments_state],
        outputs=[file_output_txt, file_output_srt]
    )

if __name__ == "__main__":
    print("==========================================================")
    print("ブラウザで http://127.0.0.1:7860 を開いて操作してください。")
    print("==========================================================")
    # localhostで起動。外部公開(share=False)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
