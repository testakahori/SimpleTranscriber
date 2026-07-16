# 🎙️ つよつよ文字起こし＆議事録ツール (SimpleTranscriber)

音声・動画を投げるだけで、高精度な文字起こし・議事録・要約と考察が `output/` フォルダに整理されて出てくるローカルツールです。使うほど賢くなります。

## 特徴

- **完全ローカル動作（デフォルト）**: 文字起こしはPC内で完結。LLMもOllama（ローカル）がデフォルト。
- **GPU自動検出**: NVIDIA GPUがあれば `large-v3` モデルで高精度・高速に文字起こし。
- **音声も動画もOK**: mp3/wav/m4a などの音声、mp4/mov/mkv などの動画に対応。
- **複数ファイル一括処理**: まとめてドラッグ＆ドロップすると順番に自動処理。
- **ノイズ除去**: 汚い録音でも前処理で精度UP（オプション）。
- **赤字補完**: 聞き取れなかった箇所をLLMが文脈から推測し、<span style="color:red">赤字</span>で表記。
- **議事録＆要約と考察の自動生成**: すべてMarkdown (.md) で出力。
- **人物マスタ**: 「ボンギンさん」→議事録では「坪内」のような呼び名⇔表記の変換を登録可能。
- **話者分離・声紋識別**（オプション）: 一度名前を割り当てると次回から「この声は坪内さん」と自動認識。
- **成長する辞書**: テキスト修正を保存すると誤変換を学習し、次回から自動適用。
- **Word/PDF出力**: 議事録をそのまま提出できる形式に変換（赤字も維持）。
- **横断検索**: 「あの件いつ話したっけ？」を過去の全出力から全文検索。

## 起動方法

| OS | 方法 |
|----|------|
| Windows | `run.bat` をダブルクリック |
| Mac / Linux | ターミナルで `./run.sh`（初回は `chmod +x run.sh`） |

初回はライブラリの自動ダウンロードがあります（数分〜10分）。準備完了後、ブラウザで `http://127.0.0.1:7860` が開きます。

## LLM設定（議事録・要約・校正の頭脳）

「⚙️ 設定」タブで選択できます。**使う人次第で自由に選べます。**

| モード | 必要なもの | 特徴 |
|--------|-----------|------|
| **Ollama**（デフォルト） | [Ollama](https://ollama.com) をインストールし `ollama pull gemma4` | 無料・完全オフライン・機密会議OK |
| **Claude API** | APIキー（設定タブ or 環境変数 `ANTHROPIC_API_KEY`） | 最高品質 |
| **OpenAI API** | APIキー（設定タブ or 環境変数 `OPENAI_API_KEY`） | GPT系を使いたい人向け |
| 使わない | — | 文字起こしのみ実行 |

※ クラウドモードでは文字起こし**テキスト**が外部APIに送信されます（音声ファイル自体は送信されません）。

## 出力フォルダ構造

```
output/
└── 2026-07-16_1430_定例会議/
    ├── transcript.md      ← 文字起こし（タイムスタンプ・話者・赤字付き）
    ├── transcript.srt     ← 字幕ファイル
    ├── 議事録.md          ← 議事録（決定事項・アクションプラン等）
    ├── 要約と考察.md      ← 3行要約・考察・ネクストアクション
    └── meta.json          ← 処理情報
```

## 話者分離・声紋識別を有効にする（オプション）

やや重いライブラリのため任意インストールです:

```
venv\Scripts\pip install torch torchaudio speechbrain   (Windows)
venv/bin/pip install torch torchaudio speechbrain        (Mac)
```

初回実行時に話者埋め込みモデル（約80MB）を自動ダウンロードします。
文字起こし後に「話者A → 坪内」のように割り当てて確定すると声紋が保存され、次回から自動で名前が付きます。

## ファイル構成

```
SimpleTranscriber/
├── app.py               # Gradio UI（メイン）
├── core/                # 機能モジュール群
│   ├── transcriber.py   #   faster-whisper（GPU自動検出）
│   ├── audio.py         #   ffmpeg変換・ノイズ除去
│   ├── diarization.py   #   話者分離・声紋（オプション）
│   ├── llm.py           #   Ollama / Claude / OpenAI 切替
│   ├── minutes.py       #   議事録・要約・校正の生成
│   ├── people.py        #   人物マスタ
│   ├── glossary.py      #   用語辞書・修正学習
│   ├── export.py        #   Word / PDF 変換
│   ├── search.py        #   横断全文検索
│   └── ...
├── prompts/             # LLMプロンプト（カスタマイズ可）
├── profiles/            # 人物マスタ・声紋・用語辞書（成長データ）
├── corrections/         # 学習済み修正
├── output/              # 全出力
├── settings.yaml        # 設定（自動生成）
├── run.bat              # Windows起動
└── run.sh               # Mac/Linux起動
```

## 対応入力形式

音声: mp3 / wav / m4a / ogg / flac / aac / wma / opus
動画: mp4 / mov / avi / mkv / wmv / flv / ts / m4v / webm

## 注意事項

- 終了時はブラウザを閉じるだけでなく、コマンドプロンプト（ターミナル）も閉じてください。
- `large-v3` はCPUだと非常に遅いため、GPU非搭載PCでは `small` / `medium` を推奨します（autoは自動選択）。
- ffmpeg が必要です（Windowsは同梱検出、Macは `brew install ffmpeg`）。

## 動作環境

- Windows / macOS / Linux
- Python 3.10+
- （推奨）NVIDIA GPU + CUDA
