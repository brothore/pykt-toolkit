#!/bin/bash

# --- 获取脚本自身的目录 ---
# BASH_SOURCE[0] 是脚本的路径。
# dirname 获取该路径的目录部分。
# readlink -f 确保路径是绝对路径（对于某些系统，如 macOS，可能需要 greadlink -f 或其他方法）。
# 推荐的通用方法是：
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定义公钥文件和目标文件/目录
PUBLIC_KEY_FILE="$SCRIPT_DIR/id_rsa.pub" # <--- 修改点：使用脚本目录
SSH_DIR="$HOME/.ssh"
AUTH_KEYS_FILE="$SSH_DIR/authorized_keys"


echo "--- 1. 检查并创建 $SSH_DIR 目录 ---"

# 1. 检查 .ssh 目录是否存在，如果不存在则创建
if [ ! -d "$SSH_DIR" ]; then
    echo "目录 $SSH_DIR 不存在，正在创建..."
    mkdir -p "$SSH_DIR"
    if [ $? -ne 0 ]; then
        echo "错误：创建目录 $SSH_DIR 失败。"
        exit 1
    fi
fi

# 2. 设置 .ssh 目录权限 (700: 只有所有者可读/写/执行)
echo "设置目录 $SSH_DIR 权限为 700..."
chmod 700 "$SSH_DIR"
if [ $? -ne 0 ]; then
    echo "错误：设置目录 $SSH_DIR 权限失败。"
    exit 1
fi

echo "--- 2. 转换和添加公钥到 authorized_keys ---"

# 3. 检查公钥文件是否存在于脚本所在的目录下
if [ ! -f "$PUBLIC_KEY_FILE" ]; then
    # 提示信息也应该更新，以反映正在检查的路径
    echo "错误：公钥文件 $PUBLIC_KEY_FILE (脚本所在目录) 不存在。"
    exit 1
fi

# 4. 将 id_rsa.pub 的内容追加到 authorized_keys
echo "将 $PUBLIC_KEY_FILE 的内容追加到 $AUTH_KEYS_FILE..."
cat "$PUBLIC_KEY_FILE" >> "$AUTH_KEYS_FILE"
if [ $? -ne 0 ]; then
    echo "错误：追加公钥内容失败。"
    exit 1
fi

# 5. 设置 authorized_keys 文件权限 (600: 只有所有者可读/写)
echo "设置文件 $AUTH_KEYS_FILE 权限为 600..."
chmod 600 "$AUTH_KEYS_FILE"
if [ $? -ne 0 ]; then
    echo "错误：设置文件 $AUTH_KEYS_FILE 权限失败。"
    exit 1
fi

echo ""
echo "--- 3. 部署完成 ---"
echo "SSH 公钥已成功添加到 $AUTH_KEYS_FILE。"
echo "目录 $SSH_DIR 权限为 $(stat -c "%a" "$SSH_DIR") (应为 700)。"
echo "文件 $AUTH_KEYS_FILE 权限为 $(stat -c "%a" "$AUTH_KEYS_FILE") (应为 600)。"
echo "现在您可以使用匹配的私钥通过 SSH 连接到此账户。"