#!/bin/bash
#运行pykt的环境配置
# 要添加的行
NEW_PATH='export PYTHONPATH="/root/:$PYTHONPATH"'
pip install pandas scikit-learn matplotlib seaborn wandb
# 检查是否已经存在
if grep -qF "$NEW_PATH" ~/.zshrc; then
    echo "PYTHONPATH already set in ~/.zshrc"
else
    # 添加到文件末尾
    echo "$NEW_PATH" >> ~/.zshrc
    echo "PYTHONPATH added to ~/.zshrc"
    
    # 可选：立即生效
    source ~/.zshrc
    echo "Reloaded ~/.zshrc"
fi
