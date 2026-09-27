@echo off
REM ============================================================
REM  财务报销审核受控 Agent —— 启动
REM  双击运行，然后浏览器打开 http://127.0.0.1:8100/audit
REM ============================================================
REM 注意：本文件必须保持 GBK(936) 编码 + CRLF 行尾，两者都别改 ——
REM   存成 UTF-8：中文 Windows 控制台是 936 代码页，窗口里会刷一屏乱码；
REM   行尾存成 LF：中文行会把下一行"吞"进去，表现为刷一屏"不是内部或外部命令"，
REM   严重时后面的 set 变量没生效、启动参数为空（--port 没值会直接启动失败）。
setlocal
cd /d %~dp0

REM 端口：8100 是刻意避开 8000 的 —— 8000 太常用，本机其它本地服务很容易先占上，
REM 而本脚本开头会杀掉占端口的人，两边都用 8000 就会互相踢。要再换端口，只改这一行。
set PORT=8100

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

REM 防呆：先停掉可能残留的旧服务。stale 进程带着**旧路由表**跑新页面
REM （静态文件即时生效、后端路由启动时加载），表现为莫名的 405/404。
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
    taskkill /F /PID %%p >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo   审核台: http://127.0.0.1:%PORT%/audit
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

".venv\Scripts\python.exe" server.py --demo --port %PORT%
pause
