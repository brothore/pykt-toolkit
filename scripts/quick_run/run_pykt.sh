#!/bin/bash

# 1. 定义统一的路径变量
PROJECT_ROOT="/root/pykt-toolkit"
CONFIG_FILE="$HOME/.zshrc"

# 2. 运行pykt的环境配置
echo "Installing python dependencies..."
pip install pandas scikit-learn matplotlib seaborn wandb retrying einops

# 3. 定义要写入的配置行
# 注意：这里 \$PYTHONPATH 加上了转义，防止在脚本运行时被展开，保留到 zshrc 中
EXPORT_CMD="export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""

# 注意：这里 \$PWD 和 basename 前面都加了转义，确保写入文件的是命令本身，而不是当前目录的名字
ALIAS_CMD="alias ce='[ \"\$(basename \"\$PWD\")\" != \"examples\" ] && cd ${PROJECT_ROOT}/examples || :'"

# 4. 检查是否已经存在配置
if grep -qF "${PROJECT_ROOT}" "$CONFIG_FILE"; then
    echo "Configuration already exists in $CONFIG_FILE"
else
    # 添加到文件末尾 (先加个空行以防万一)
    echo "" >> "$CONFIG_FILE"
    echo "# pykt-toolkit environment settings" >> "$CONFIG_FILE"
    echo "$EXPORT_CMD" >> "$CONFIG_FILE"
    echo "$ALIAS_CMD" >> "$CONFIG_FILE"
    
    echo "Successfully added PYTHONPATH and alias to $CONFIG_FILE"
    
    # 5. 提示用户手动生效 (脚本内 source 无法影响父 Shell)
    echo "-------------------------------------------------------"
    echo "Please run the following command to apply changes:"
    echo "  source $CONFIG_FILE"
    echo "-------------------------------------------------------"
fi