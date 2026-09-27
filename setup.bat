@echo off
REM 注意：本文件必须保持 GBK(936) 编码 + CRLF 行尾，两者都别改 ——
REM   存成 UTF-8：中文 Windows 控制台是 936 代码页，窗口里会刷一屏乱码；
REM   行尾存成 LF：中文行会把下一行"吞"进去，表现为刷一屏"不是内部或外部命令"。
REM ============================================================
REM  财务报销审核受控 Agent —— 首次安装
REM  创建虚拟环境并安装依赖。只需要跑一次。
REM ============================================================
setlocal
cd /d %~dp0

echo.
echo   ==========================================
echo    财务报销审核受控 Agent  ^|  环境安装
echo   ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo   [错误] 找不到 python 命令。
    echo          请先安装 Python 3.10+ 并勾选 "Add to PATH"。
    echo.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo   [提示] .venv 已存在，跳过创建。
) else (
    echo   [1/2] 创建虚拟环境 .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo   [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

echo   [2/2] 安装依赖（首次约需 1-3 分钟）...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo   安装完成。接下来：
echo     1. 复制 .env.example 为 .env，填入 API Key（可选，不填也能演示）
echo     2. 双击 start.bat 启动
echo     3. 浏览器打开 http://127.0.0.1:8100/audit
echo.
pause
