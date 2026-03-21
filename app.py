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

def transcribe(audio_path):
    if not audio_path:
        return "エラー: 音声ファイルが提供されていません。", None

    try:
        logging.info("文字起こしを開始します...")
        start_time = time.time()
        
        m = load_model()
        
        # vad_filter=Trueを入れることで無音部分のハルシネーション（幻覚）を防ぐ（長尺音声に非常に有効）
        segments, info = m.transcribe(audio_path, beam_size=5, vad_filter=True)
        
        logging.info(f"検出言語: {info.language} (確率: {info.language_probability:.2f})")
        
        full_text = ""
        for segment in segments:
            # ログには詳細なテキストは出さない（機密情報保護のため）
            logging.info(f"処理進捗: [{segment.start:.2f}s -> {segment.end:.2f}s]")
            full_text += segment.text + "\n"
        
        elapsed_time = time.time() - start_time
        logging.info(f"文字起こし完了。処理時間: {elapsed_time:.2f}秒")
        
        if not full_text.strip():
            return "音声からテキストが検出されませんでした。", None
            
        # txtファイルとして保存
        output_file_name = f"transcription_result_{int(time.time())}.txt"
        
        # カレントディレクトリに保存
        output_file_path = os.path.join(os.getcwd(), output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(full_text.strip())
            
        return full_text.strip(), output_file_path
        
    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        # 機密情報のため、必要以上にスタックトレースは出力しない
        logging.error("エラーが発生しました。詳細は画面に出力されます。")
        return error_msg, None

# Gradio UIの構築
with gr.Blocks(title="シンプル文字起こしツール") as demo:
    gr.Markdown("# 🎙️ シンプル文字起こしツール (完全オフライン実行)")
    gr.Markdown("アップロードされた音声ファイル（mp3, wav, mp4など）をテキスト化します。音声データが外部送信されることはありません。")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="音声ファイル")
            submit_btn = gr.Button("文字起こし開始", variant="primary")
            gr.Markdown("※長時間のファイル（数十分〜1時間）は、処理にPCのスペック依存でまとまった時間がかかります。\n画面を閉じずにそのままお待ちください。")
            
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="文字起こし結果", lines=15)
            file_output = gr.File(label="結果ファイルのダウンロード (txt)")
            
    submit_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[text_output, file_output]
    )

if __name__ == "__main__":
    print("==========================================================")
    print("ブラウザで http://127.0.0.1:7860 を開いて操作してください。")
    print("==========================================================")
    # localhostで起動。外部公開(share=False)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
