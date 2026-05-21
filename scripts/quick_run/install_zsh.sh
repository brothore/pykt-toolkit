#!/bin/bash

echo "--- 1. 安装 Zsh ---"
# 安装 zsh
echo "尝试安装 zsh..."
if apt install -y zsh; then
    echo "zsh 安装成功。"
else
    echo "zsh 安装失败，请检查 apt 源或权限。"
    exit 1
fi

echo "--- 2. 安装 Oh-My-Zsh ---"
# 使用 Gitee 官方镜像安装，并设置 RUNZSH=no 和 CHSH=no 防止脚本中断
echo "尝试安装 Oh-My-Zsh..."
if RUNZSH=no CHSH=no REMOTE=https://gitee.com/mirrors/oh-my-zsh.git sh -c "$(curl -fsSL https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh)"; then
    echo "Oh-My-Zsh 安装成功。"
else
    echo "Oh-My-Zsh 安装失败。"
    exit 1
fi

# 确保 .zshrc 存在
ZSHRC_FILE="$HOME/.zshrc"
if [ ! -f "$ZSHRC_FILE" ]; then
    echo "警告：$ZSHRC_FILE 文件不存在，Oh-My-Zsh 可能未正确安装。"
    exit 1
fi

echo "--- 3. 克隆 Powerlevel10k 主题 ---"
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
P10K_DIR="${ZSH_CUSTOM}/themes/powerlevel10k"

echo "尝试克隆 Powerlevel10k 到 $P10K_DIR..."
if git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git "$P10K_DIR"; then
    echo "Powerlevel10k 克隆成功。"
else
    echo "Powerlevel10k 克隆失败，请检查网络连接。"
fi

echo "--- 4. 安装必备插件 ---"
# 将原本的 GitHub 地址替换为 Gitee 镜像，防止 AutoDL 连 GitHub 超时报错
echo "正在安装 zsh-autosuggestions 和 zsh-syntax-highlighting..."
git clone https://gitee.com/phatboy/zsh-autosuggestions.git ${ZSH_CUSTOM}/plugins/zsh-autosuggestions
git clone https://gitee.com/hjkl01/zsh-syntax-highlighting.git ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting

echo "--- 5. 修改 ~/.zshrc 配置 ---"
# 替换主题
OLD_THEME_LINE=$(grep -E '^\s*ZSH_THEME=' "$ZSHRC_FILE")
NEW_THEME_LINE='ZSH_THEME="powerlevel10k/powerlevel10k"'

if [[ -n "$OLD_THEME_LINE" ]]; then
    sed -i 's|^[[:space:]]*ZSH_THEME=.*|'"$NEW_THEME_LINE"'|' "$ZSHRC_FILE"
    echo "主题设置已更新为 Powerlevel10k。"
else
    echo "$NEW_THEME_LINE" >> "$ZSHRC_FILE"
fi

# 启用插件 (找到 plugins=(git) 并替换为包含新插件的列表)
sed -i 's/^plugins=(/plugins=(zsh-autosuggestions zsh-syntax-highlighting /' "$ZSHRC_FILE"
echo "插件配置已更新。"

echo ""
echo "--- 安装完成 ---"
echo "脚本执行完毕。请执行 'chsh -s $(which zsh)' 来将默认 shell 设置为 zsh。"
echo ""
echo "要立即切换到 Zsh 并进行 Powerlevel10k 初始化配置，请执行:"
echo "exec zsh"