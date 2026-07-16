# Release

## 配布形態

- Windowsフォルダ一式の手渡し（zip配布等）
- エンドユーザーは `run.bat` をダブルクリックするだけで起動
- Python / pip がPCにインストールされていることが前提

## 起動フロー（run.bat）

```
1. chcp 65001（UTF-8コードページ設定）
2. venv\Scripts\activate.bat が存在しない場合 → python -m venv venv を実行（初回のみ）
3. venv を activate
4. pip install -r requirements.txt --quiet（毎回実行・差分のみ更新）
5. KMP_DUPLICATE_LIB_OK=TRUE を設定
6. python app.py を実行
7. ブラウザが自動起動 → http://127.0.0.1:7860
```

## 初回セットアップ所要時間

- venv作成 + 依存ライブラリDL: 数分〜10分程度（ネット速度・PC性能による）
- Whisperモデル（smallサイズ）の自動DL: 初回文字起こし実行時に発生（数百MB）

## バージョン情報

- リリース日: （未確認）
- バージョン番号管理: （未確認・現時点でバージョン番号なし）
- 動作確認OS: Windows（run.bat による。Mac / Linux 対応は未実装）

## 終了方法

- ブラウザタブを閉じるだけでは不十分
- コマンドプロンプトウィンドウの「×」ボタンで閉じることでサーバー停止

## 注意事項

- `share=False`・`server_name="127.0.0.1"` のためローカルネットワーク外からはアクセス不可
- 出力ファイル（`transcription_result_*.txt` / `.srt`）はアプリ起動時のカレントディレクトリに生成される
