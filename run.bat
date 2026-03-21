@echo off
chcp 65001 > nul
title シンプル文字起こしツール
echo =========================================
echo  シンプル文字起こしツール 起動準備中...
echo =========================================

REM 仮想環境の作成と有効化 (存在しない場合)
if not exist "venv\Scripts\activate.bat" (
    echo [情報] 初回セットアップ: 独自の処理環境を構築しています[venv]...
    echo これは初回のみ数分かかります。
    python -m venv venv
)

echo [情報] 処理環境を読み込んでいます...
call venv\Scripts\activate

echo [情報] 必要な設定(ライブラリ)を確認しています...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo.
    echo [エラー] 設定の準備に失敗しました。
    pause
    exit /b %errorlevel%
)

echo.
echo [情報] 準備完了。文字起こしツールを起動します！
echo =========================================
echo ※ 自動的にブラウザが開きます。
echo ※ 終了するときは、この黒い画面の右上の閉じるボタンで終了してください。
echo =========================================
python app.py

pause
