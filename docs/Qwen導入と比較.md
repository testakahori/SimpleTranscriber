# Qwenで聞き直す比較試験

Whisperで文字にならなかった声・確信度の低い発言・繰り返しを含む発言を、Qwen3-ASR-1.7Bで聞き直す試験用のツールです。必要に応じて、Qwenが終了してからgemma4で候補を校正します。

生成された文は未確認の候補です。音声との照合をせずに、元の文字起こしや用語辞書・声紋へ反映しません。アプリの既定の認識方式も変更しません。

## 導入環境

- Windows・NVIDIA GPU・Python 3.12を対象にしています。
- `.venv-qwen` に独立して導入します。既存アプリの `venv` は使用するライブラリを変更しません。
- [Qwen公式のTransformers版](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)を使用します。同じ1.7Bモデルを標準のTransformersで動かす配布形式です。
- モデル取得時だけHugging Faceへ接続します。比較処理はキャッシュ済みモデルを `local_files_only=True` で読み込み、gemma4は既存設定のローカルOllamaを使用します。
- Qwenの開始前にOllamaのモデルを解放し、Qwenプロセスが終了してからgemma4を起動します。試験終了後もOllamaのモデルを解放します。

プロジェクトのルートで、順番に実行します（`uv` が必要です）。

```powershell
uv venv .venv-qwen --python venv\Scripts\python.exe
uv pip install --python .venv-qwen\Scripts\python.exe -r requirements-qwen.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --system-certs
```

このPCではDドライブへの大量ファイルのコピーが遅かったため、環境の実体を `%USERPROFILE%\.cache\simpletranscriber-qwen-20261008` に作り、プロジェクトの `.venv-qwen` からジャンクションで参照しています。コマンドは上記と同じ `.venv-qwen\Scripts\python.exe` で実行できます。QwenのモデルキャッシュもCドライブにあります。

続いてモデルを取得します。認識対象の音声を外部へ送る処理はありません。

```powershell
@'
import truststore
truststore.inject_into_ssl()
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-ASR-1.7B-hf', token=False,
      allow_patterns=['*.json', '*.txt', '*.jinja', '*.safetensors']))
'@ | .\.venv-qwen\Scripts\python.exe -
```

## 比較の実行

音量正規化・EQ等を済ませた16kHzモノラルWAVを用意します。長い会議全体の前に、数分の抜粋で比較してください。音声の切り出し位置を変えず、WhisperとQwenへ同じ波形を渡すことが大切です。

```powershell
.\venv\Scripts\python.exe tools\try_qwen_recovery.py --audio "前処理済み.wav" --output "output\qwen_trial_01" --gemma
```

- `--baseline Whisper結果.json` を付けると、同じWAVを以前に認識した結果を再利用します。`segments` または `utterances` 配列を持つJSONに対応します。WAVとタイムスタンプの原点が同じであることが必要です。
- `--control 90:120` で、認識がうまくいった30秒以内の区間も追加できます。複数回指定できます。
- `--max-clips 16` で聞き直しの対象数を制限します。既定は16区間です。長い抜けは30秒以内に分け、最初の区間から上限まで処理します。
- `--gemma` を省略するとQwenまでで終わります。
- 出力先は存在しないフォルダを指定します。以前の試験結果への上書きを防ぐためです。

## 結果の確認

出力先の `comparison.html` をブラウザで開くと、各区間の音声とWhisper・Qwen・gemma4校正後の文が並びます。`clips` フォルダも同じ場所に置いてください。

`request.json` は対象区間、`qwen.json` はQwenの候補と処理時間・GPUメモリ、`gemma.json` は校正結果、`stats.json` は処理間のGPU使用量を記録します。生成上限に達した候補は途中で切れた可能性があるため、レポートに表示し、gemma4校正から外します。

Qwenはこの試験では単語時刻を生成しません。前後2秒の余白を含む窓全体の候補なので、そのまま抜け区間へ差し込むと前後の文字と重複する可能性があります。自動差し込みや「何秒ぶん正しく回復した」という評価には使用しません。採用する場合は、音声との照合と単語時刻の対応付けが別途必要です。

比較フォルダには会議内容が含まれます。`output/` 内に保存し、Gitには追加しないでください。

## 2026-10-08の実測

RTX 5060 Ti 16GB / Python 3.12.10 / torch 2.14.0+cu130 / Transformers 5.19.0 / accelerate 1.15.0。モデルは `Qwen/Qwen3-ASR-1.7B-hf`、取得したリビジョンは `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` です。

| 試験 | Qwenの対象 | Qwen生成時間 | gemma4校正 | 観測 |
|---|---:|---:|---:|---|
| 雑音の多い会議5分から選択 | 9区間（抜け・低確信度・反復8区間＋通常発話1区間） | 50.0秒 | 88.6秒・8候補 | 全区間に文字は出たが、不自然な短い文もある。1区間は生成上限に達し、校正から除外 |
| 聞き取りやすい会議2分をWhisperから再実行 | 2区間（20秒＋25秒） | 26.4秒 | 15.5秒・2候補 | Whisperとおおむね同じ内容。一部の漢字・語の表記は異なる。Whisper自体は29.6秒で完了 |

Qwen生成時間は最初の呼び出しの初期化を含みますが、プロセス起動・モデル読み込み・音声前処理は含みません。聞き取りやすい会議のQwenプロセス全体は56.1秒でした。gemma4は設定中の31Bモデルを使用し、12Bへのフォールバックは発生していません。

Qwenの最大GPUテンソル割当は4146 MiB、最大予約領域は4596 MiBでした（GPU全体の使用量とは異なります）。Qwen終了前後で、GPU全体の使用量は雑音試験が1081→1081 MiB、聞き取りやすい会議が598→598 MiB。Qwenを終了してからgemma4を起動し、モデルが同時にGPUへ残らないことを確認しました。

**判断**: Qwenを比較用・確認用として残します。雑音で認識できない声を常に正しく補えるわけではなく、自動穴埋めの既定にはしません。人が音声と照合した正解文がないため、CERや精度改善率は未測定です。候補の数や文字数を、回復できた発言の数とはみなしません。

ローカルの試験結果:

- `output/qwen_trial_20261008_noisy/comparison.html`
- `output/qwen_trial_20261008_clean/comparison.html`

CPUで動く回帰テストは `venv\Scripts\python.exe -m unittest discover -s tests -v`（14件）。抜け区間の切り出し・時間上限・比較する単語の範囲・HTMLのエスケープ・タイムアウト時の子プロセス停止と、既存の文字起こし処理を確認しています。
