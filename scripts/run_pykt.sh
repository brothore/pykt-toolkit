#!/bin/bash
#运行pykt的环境配置
# 要添加的行
NEW_PATH='export PYTHONPATH="/root/autodl-tmp/pykt-toolkit/:$PYTHONPATH"'
pip install pandas scikit-learn matplotlib seaborn wandb
# 检查是否已经存在
if grep -qF "$NEW_PATH" ~/.bashrc; then
    echo "PYTHONPATH already set in ~/.bashrc"
else
    # 添加到文件末尾
    echo "$NEW_PATH" >> ~/.bashrc
    echo "PYTHONPATH added to ~/.bashrc"
    
    # 可选：立即生效
    source ~/.bashrc
    echo "Reloaded ~/.bashrc"
fi
