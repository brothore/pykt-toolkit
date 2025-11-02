#!/bin/bash



echo "--- 1. 安装 Zsh 和 Oh-My-Zsh ---"
# 使用  安装 zsh
echo "尝试安装 zsh..."
if  apt install -y zsh; then
    echo "zsh 安装成功。"
else
    echo "zsh 安装失败，请检查 apt 源或权限。"
    exit 1
fi

# 安装 Oh-My-Zsh
echo "尝试安装 Oh-My-Zsh..."
if curl -sSL https://gitee.com/sgfoot/library/raw/master/oh-my-zsh/install.sh | bash; then
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

echo "--- 2. 克隆 Powerlevel10k 主题 ---"
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
P10K_DIR="${ZSH_CUSTOM}/themes/powerlevel10k"

echo "尝试克隆 Powerlevel10k 到 $P10K_DIR..."
if git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git "$P10K_DIR"; then
    echo "Powerlevel10k 克隆成功。"
else
    echo "Powerlevel10k 克隆失败，请检查 git 是否安装或网络连接。"
fi

echo "--- 3. 修改 ~/.zshrc 设置 Powerlevel10k 为主题 ---"
# 定义要查找和替换的模式
# Oh-My-Zsh 的主题设置通常是 ZSH_THEME="xxx"
OLD_THEME_LINE=$(grep -E '^\s*ZSH_THEME=' "$ZSHRC_FILE")
NEW_THEME_LINE='ZSH_THEME="powerlevel10k/powerlevel10k"'

if [[ -n "$OLD_THEME_LINE" ]]; then
    echo "在 $ZSHRC_FILE 中找到原有主题设置：$OLD_THEME_LINE"
    
    # 使用 sed 进行替换
    # 注意：这里使用 gnu-sed (在大多数 linux 发行版中就是 sed)
    # \s* 匹配零个或多个空格
    # -i 选项表示直接修改文件
    if sed -i 's|^[[:space:]]*ZSH_THEME=.*|'"$NEW_THEME_LINE"'|' "$ZSHRC_FILE"; then
        echo "主题设置已更新为：$NEW_THEME_LINE"
    else
        echo "警告：使用 sed 替换主题行失败，请手动检查 $ZSHRC_FILE。"
        echo "您需要将文件中的 ZSH_THEME=\"...\" 行替换为："
        echo "$NEW_THEME_LINE"
        exit 1
    fi
else
    echo "未在 $ZSHRC_FILE 中找到 ZSH_THEME 设置行，尝试在文件末尾添加..."
    echo "$NEW_THEME_LINE" >> "$ZSHRC_FILE"
    echo "主题设置已添加到文件末尾。"
fi


echo ""
echo "--- 4. 安装完成 ---"
echo "脚本执行完毕。请执行 'chsh -s $(which zsh)' 来将默认 shell 设置为 zsh (如果尚未设置)。"
echo "然后，请退出当前终端并重新打开，或手动执行 'zsh'，Powerlevel10k 会引导你进行配置。"

# 提示用户切换 shell
echo ""
echo "要立即切换到 Zsh，请执行:"
echo "exec zsh"
echo ""