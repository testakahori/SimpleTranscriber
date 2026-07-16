# Project

## 概要

**SimpleTranscriber** は、音声・動画ファイルをローカルPC上で完全オフライン動作する文字起こしツール。
APIキー不要、音声データの外部送信なし。

## 目的

- 会議録音・YouTube動画音声などを安全にテキスト化する
- 技術知識不要のワンクリック起動（`run.bat` ダブルクリックのみ）
- SRT字幕形式での出力にも対応し、動画制作・議事録作成を効率化

## 技術スタック

| 分類 | 内容 |
|------|------|
| 言語 | Python |
| 文字起こしエンジン | faster-whisper（Whisper smallモデル / CPU / int8量子化） |
| UIフレームワーク | Gradio（Blocks API / タブUI） |
| 実行バックエンド | CTranslate2 + ONNX Runtime |
| 起動方式 | run.bat（venv自動構築 + pip install） |
| サーバーアドレス | 127.0.0.1:7860（ローカルのみ、share=False） |

## 主な機能

- 音声ファイルアップロード（mp3 / wav / mp4 / m4a / ogg / flac / aac / webm）
- faster-whisper による自動言語検出 + beam_size=5 + VADフィルタ付き文字起こし
- 結果テキストのGradio画面内表示・編集・再保存
- TXTファイルダウンロード（`transcription_result_{timestamp}.txt`）
- SRTファイルダウンロード（`transcription_result_{timestamp}.srt`）
- SRT編集タブ：セグメント単位の時刻・テキスト編集、行削除、再ダウンロード

## 現状

- コア機能（文字起こし→TXT/SRT出力）は実装済み・動作確認済み
- Windows向け run.bat による配布形態が整備済み
- Mac対応・GitHub同期は未対応（[[Dashboard]] 参照）
