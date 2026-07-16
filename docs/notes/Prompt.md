# Prompt

## 開発・改善依頼に使えるプロンプト例

### モデルサイズ選択UIの追加

```
app.py の文字起こし画面に、Whisperモデルサイズ（tiny / base / small / medium）を
ドロップダウンで選択できるUIを追加してください。
選択したモデルは load_model() でロードし直してください。
既存の機能は一切壊さないこと。
```

### 言語強制指定オプション

```
app.py の transcribe() に language パラメータを追加し、
Gradio UI 上でドロップダウン（自動検出 / 日本語 / 英語）で選べるようにしてください。
faster_whisper の transcribe() に language 引数として渡すこと。
```

### 出力先フォルダの指定

```
文字起こし結果ファイル（txt / srt）の保存先を、
UIで指定できるようにしてください（デフォルトはカレントディレクトリ）。
```

### run.sh（Mac/Linux対応）の作成

```
run.bat と同等の動作をする run.sh を作成してください。
- bash で動作すること
- venv が存在しない場合のみ python3 -m venv venv を実行
- pip install -r requirements.txt を実行
- KMP_DUPLICATE_LIB_OK=TRUE を export
- python app.py を実行
```
