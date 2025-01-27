@echo off
chcp 65001
:: 检查Python是否安装
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 未安装，请先安装 Python。
    exit /b
)

:: 创建虚拟环境
python -m venv venv

:: 激活虚拟环境并安装requirements.txt中的包
call venv\Scripts\activate

cd app

:: 升级pip
python.exe -m pip install --upgrade pip

:: 安装requirements.txt中的包
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt 文件不存在，请确保该文件存在于当前目录。
)

echo 虚拟环境已创建并安装了所需的pip包。 

pause