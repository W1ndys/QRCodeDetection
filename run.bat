@echo off
:: 切换到脚本所在目录
cd /d %~dp0

:: 激活虚拟环境（假设虚拟环境文件夹名为 venv）
call venv\Scripts\activate.bat

:: 运行 Python 脚本
python main.py

:: 暂停以查看输出（可选）
pause 


