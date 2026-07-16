# Research

## 文字起こしエンジン

| 項目 | 内容 |
|------|------|
| ライブラリ | faster-whisper |
| モデル名 | `small`（コード: `WhisperModel("small", ...)` で確認） |
| 実行デバイス | CPU（`device="cpu"`） |
| 量子化 | int8（`compute_type="int8"`） |
| ビームサイズ | 5（`beam_size=5`） |
| VADフィルタ | 有効（`vad_filter=True`）— 無音区間スキップ、ハルシネーション抑制 |
| 言語検出 | 自動（`info.language` / `info.language_probability` をログ出力） |
| モデルロード | グローバルキャッシュ（2回目以降はリロード不要） |

## 依存ライブラリ（requirements.txt より）

```
faster-whisper
gradio
```

インストール済み venv から確認できる実際の主要依存関係：

| パッケージ | バージョン（venv確認） |
|-----------|----------------------|
| faster-whisper | （未確認：バージョン固定なし） |
| gradio | （未確認：バージョン固定なし） |
| ctranslate2 | 4.7.1 |
| onnxruntime | 1.24.2 |
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| av | 16.1.0（音声デコード） |
| huggingface_hub | インストール済み（モデル自動DL用） |

## 対応入力形式

app.py の `gr.File(file_types=[...])` から確認：

```
.mp3 / .wav / .mp4 / .m4a / .ogg / .flac / .aac / .webm
```

## 出力形式

| 形式 | ファイル名パターン | 内容 |
|------|-------------------|------|
| TXT | `transcription_result_{timestamp}.txt` | セグメントテキストを改行結合した平文 |
| SRT | `transcription_result_{timestamp}.srt` | 番号 / タイムコード / テキスト のSRT標準形式 |

## 処理フロー

```
1. ユーザーが音声ファイルをアップロード
2. transcribe() 呼び出し
3. load_model() → WhisperModel("small", cpu, int8) をグローバルにキャッシュ
4. m.transcribe(audio_path, beam_size=5, vad_filter=True)
5. segments イテレータを逐次処理
   - full_text に結合
   - segments_data [(start, end, text), ...] に蓄積
   - SRT行 (srt_lines_raw) に追記
6. TXTファイル書き出し（カレントディレクトリ）
7. SRTファイル書き出し（カレントディレクトリ）
8. Gradio UIに結果返却 + ダウンロードリンク表示
```

## SRTタイムコード形式

`HH:MM:SS,mmm`（例: `00:01:23,456`）
`format_srt_time()` / `parse_srt_time()` で相互変換。

## SRT編集機能の内部構造

- Gradio `Dataframe`（headers: 番号 / 開始 / 終了 / テキスト）
- セグメント削除（`delete_srt_row()`）で行番号振り直し
- 「SRTを保存してダウンロード」で `save_srt_from_editor()` 実行
