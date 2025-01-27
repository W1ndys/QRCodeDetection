#!/bin/bash

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "虚拟环境不存在，请先运行 create_venv_linux.sh 创建虚拟环境。"
    exit
fi

# 打开一个新的终端窗口并激活虚拟环境
gnome-terminal -- bash -c "source venv/bin/activate; exec bash"

echo "已打开一个新的终端窗口并进入虚拟环境。" 