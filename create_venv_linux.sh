#!/bin/bash

# 检查Python是否安装
if ! command -v python3 &> /dev/null
then
    echo "Python3 未安装，请先安装 Python3。"
    exit
fi

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装requirements.txt中的包
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt 文件不存在，请确保该文件存在于当前目录。"
fi

echo "虚拟环境已创建并安装了所需的pip包。" 