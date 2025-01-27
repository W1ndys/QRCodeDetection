@echo off

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo 虚拟环境不存在，请先运行 create_venv_windows.bat 创建虚拟环境。
    exit /b
)

:: 打开一个新的命令提示符窗口并激活虚拟环境
start cmd /k "call venv\Scripts\activate"

echo 已打开一个新的命令提示符窗口并进入虚拟环境。 