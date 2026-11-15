@echo off
REM 매일 실시간 기록을 돌리는 예시. 아래 두 경로를 내 컴퓨터에 맞게 바꿔서 run_live.bat 으로 저장
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d C:\path\to\crypto-storm-model
"C:\path\to\python.exe" ai11_live.py >> data_ai\live_run.log 2>&1
