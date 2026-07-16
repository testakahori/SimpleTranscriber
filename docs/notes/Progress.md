# Progress

## 実装済み機能

### コア文字起こし

- [x] faster-whisper `small` モデルによる音声→テキスト変換
- [x] CPU / int8量子化で動作（GPU不要）
- [x] VADフィルタ（無音区間スキップ）によるハルシネーション抑制
- [x] 自動言語検出（`info.language` / `info.language_probability`）
- [x] beam_size=5 による精度優先デコード
- [x] モデルのグローバルキャッシュ（起動後2回目以降は即時処理）

### 入出力

- [x] 対応フォーマット: mp3 / wav / mp4 / m4a / ogg / flac / aac / webm
- [x] TXTファイル出力（`transcription_result_{timestamp}.txt`）
- [x] SRTファイル出力（`transcription_result_{timestamp}.srt`）
- [x] タイムスタンプ付きファイル名で上書き衝突なし

### GradioUI（文字起こしタブ）

- [x] 音声ファイルアップロード欄
- [x] 「文字起こし開始」ボタン
- [x] 結果テキストボックス表示（初期は読み取り専用）
- [x] 「編集」ボタンでテキストボックスを編集可能化
- [x] 「保存」ボタンで編集後テキストをTXT/SRTとして再保存
- [x] TXT / SRT ダウンロードリンク

### GradioUI（SRT編集タブ）

- [x] セグメント一覧 Dataframe（番号 / 開始 / 終了 / テキスト）
- [x] セル直接編集（時刻・テキスト）
- [x] 行番号指定による行削除（番号振り直し付き）
- [x] 編集済みSRT保存＆ダウンロード

### 配布・起動

- [x] `run.bat` によるワンクリック起動（Windows）
- [x] 初回 venv 自動作成 + `pip install -r requirements.txt` 自動実行
- [x] `inbrowser=True` でブラウザ自動起動（127.0.0.1:7860）
- [x] `KMP_DUPLICATE_LIB_OK=TRUE` 環境変数設定（Intel MKL競合回避）
