@echo off
REM ============================================================
REM  财务报销审核受控 Agent —— 启动
REM  双击运行，然后浏览器打开 http://127.0.0.1:8000/audit
REM ============================================================
setlocal
cd /d %~dp0

echo.
echo   ==========================================
echo    财务报销审核受控 Agent
echo   ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   [错误] 还没安装环境。请先双击 setup.bat。
    echo.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs

echo   审核台: http://127.0.0.1:8000/audit
echo.
echo   提示：演示前请先在页面上点「清空演示数据」，
echo         否则排练跑过的单子会触发"重复报销"，开场基线会翻车。
echo.
echo   按 Ctrl+C 停止服务。
echo.

".venv\Scripts\python.exe" server.py
pause
