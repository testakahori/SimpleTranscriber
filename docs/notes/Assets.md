# Assets

## 出力ファイル（生成物）

アプリ実行後、プロジェクトルートに以下のファイルが生成される。

| ファイルパターン | 種別 | 内容 |
|----------------|------|------|
| `transcription_result_{timestamp}.txt` | テキスト | 文字起こし結果の平文テキスト（改行区切り） |
| `transcription_result_{timestamp}.srt` | 字幕 | SRT形式（番号 / `HH:MM:SS,mmm --> HH:MM:SS,mmm` / テキスト） |

- `{timestamp}` は `int(time.time())` — Unix秒（例: `transcription_result_1719600000.txt`）
- 文字起こし実行のたびに新規ファイルが生成される（上書きなし）
- 編集後「保存」ボタンを押した場合も新規タイムスタンプで別ファイルが生成される

## ログファイル

| ファイル | 内容 |
|---------|------|
| `error.log` | エラーログ（（未確認）— ファイルが存在する場合） |
| `output.log` | 出力ログ（（未確認）— ファイルが存在する場合） |

アプリ内のロギングは `logging.basicConfig(level=INFO)` で標準出力に出力される。
ファイルへのロギング設定はコード上では未確認。

## その他ファイル

| ファイル | 説明 |
|---------|------|
| `app.py` | メインアプリケーション（286行） |
| `requirements.txt` | 依存ライブラリ定義（`faster-whisper` / `gradio`） |
| `run.bat` | Windows向け起動スクリプト |
| `venv/` | Python仮想環境（gitignore推奨） |
