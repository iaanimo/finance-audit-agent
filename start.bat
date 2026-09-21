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
echo   本脚本以【演示模式】启动（--demo），页面上会出现「清空演示数据」按钮。
echo   提示：演示前请先点它一次，否则排练跑过的单子会触发"重复报销"，
echo         开场基线会翻车。
echo.
echo   注意：那个按钮会物理删除审核单与审计轨迹。会计凭证依法最低保管 30 年，
echo         真实部署请直接运行 server.py（不加 --demo），按钮不会出现。
echo.
echo   按 Ctrl+C 停止服务。
echo.

".venv\Scripts\python.exe" server.py --demo
pause
