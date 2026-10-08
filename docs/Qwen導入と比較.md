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

CPUで動く回帰テストは `venv\Scripts\python.exe -m unittest discover -s tests -v`。抜け区間の切り出し・時間上限・比較する単語の範囲・HTMLのエスケープ・タイムアウト時の子プロセス停止と、既存の文字起こし処理を確認しています。

## 全編を比較し、途中結果も保存する

`tools/full_asr_experiment.py` は、WhisperとQwenで全編をそれぞれ認識し、Qwenの文をgemma4で校正する別の実験です。対象を抜けの候補だけに絞る前節とは異なります。Qwenは重複しない30秒窓なので、Whisperとは区切り方が異なり、窓の境界の語が欠けることがあります。モデルだけの厳密な精度比較ではありません。

新しいフォルダで前処理し、各コマンドが正常終了してから次を実行してください。WhisperとQwenは同じ `asr.wav` を使います。

```powershell
.\venv\Scripts\python.exe tools\full_asr_experiment.py prepare output\full_asr_trial_01 --audio "D:\録音\meeting.m4a"
.\venv\Scripts\python.exe tools\full_asr_experiment.py whisper output\full_asr_trial_01
.\.venv-qwen\Scripts\python.exe tools\full_asr_experiment.py qwen output\full_asr_trial_01
.\venv\Scripts\python.exe tools\full_asr_experiment.py gemma output\full_asr_trial_01
.\venv\Scripts\python.exe tools\report_full_asr_experiment.py output\full_asr_trial_01
```

- Whisperは約10秒ごと、Qwenは1窓ごと、gemma4は約2500字ごとに途中結果を保存します。JSONは一時ファイルを完全に書いてから置き換えます。ディスクへの保存失敗は処理を止めて通知します。
- 完了済みのstageは同じコマンドで再計算しません。Qwenは未処理の窓、gemma4は未処理の行から再開できます。**Whisperが中断された場合は先頭から再計算**しますが、途中の文字は `01_Whisper_途中経過.md` に残ります。再実行前に必要な途中結果を別名で保管してください。
- フォルダ内の音声・入力JSONを途中で差し替えないでください。設定を変えた比較は別フォルダで行います。
- Qwenの空文字・生成上限・同じ語の4回以上の反復を含む窓は `unsafe` とし、gemma4へ渡さず原文を残します。この判定は正誤判定ではなく、確認の目印です。
- gemma4が推測と印を付けた箇所はMarkdownで `⟦ ⟧` を残します。校正への入力・モデルが返した原文も `gemma_calls/` に保存します。校正ガードで採用されなかった案も追跡できます。
- 最後のコマンドは、全編が完了していることを検証し、`実験レポート.md`・`comparison.html`・`校正変更一覧.md`・集計JSONを作ります。HTMLでは区間の音声、3段階の文、校正差分を確認できます。

### 雑音で無音スキップが繰り返される場合の試験設定

2026-10-08の全編試験で、faster-whisper 1.2.1 の `hallucination_silence_threshold=2.0` が、誤認らしい窓で1秒ずつしか進まず周辺を繰り返し認識する現象を観測しました。スタックで圧縮後の音声位置が481.24→482.24→483.24秒と進んでいました。

試験フォルダの `experiment-options.json` に次を保存してからWhisperを実行すると、その内部スキップだけを無効にできます。

```json
{"silence_guard": false}
```

VAD、主処理の温度試行、外側の誤認除外、抜けの再認識は維持します。内部で消していた怪しい文が出力に残る可能性があるため、**通常アプリの既定設定は変更していません**。試験のJSON・Markdownには使用設定を記録します。

62分14秒の全編試験では、この設定のWhisperが18分42秒、Qwenが7分39秒、gemma4が6分33秒で完了しました。前処理込み33.5分（中断した最初の試行と診断を含めると41.6分）。Whisper 13731字→Qwen 16204字→gemma4 16072字ですが、正解文との比較ではないため精度改善率ではありません。Qwen125窓のうち生成上限等の21窓は校正対象外として原文を保持し、残り104窓をgemma4へ渡しました。

全データとレポートは `output/20261008_会議_7_全編実験/` に保存しています。追加した「冒頭10秒をノイズ見本にする」試験では、固定した6窓で認識の明確な改善を確認できず、常用は見送りました。結果の詳細と判断は [引継ぎの書](引継ぎの書.md) を参照してください。
