#!/bin/bash

# Zsh自动安装配置脚本
# 需要手动操作的部分已用注释标出

# 安装zsh和依赖
echo "正在更新软件包列表并安装zsh及相关组件..."
apt update
apt install -y zsh git curl fonts-powerline

# 安装oh-my-zsh
echo "正在安装oh-my-zsh..."
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 安装常用插件
echo "正在安装插件..."
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/marlonrichert/zsh-autocomplete ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autocomplete

# 安装Powerlevel10k主题
echo "正在安装Powerlevel10k主题..."
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
ZSHRC="$HOME/.zshrc"

echo "正在优化 .zshrc 配置..."

# --- 1. 替换 ZSH_THEME ---
# 无论原来是什么主题，都替换为 p10k
sed -i 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' "$ZSHRC"

# --- 2. 替换 plugins 块 (多行替换) ---
# 使用 perl 匹配从 plugins=( 到 ) 的所有内容并替换
perl -i -0777 -pe 's/plugins=\(.*?\)/plugins=(\n  git\n  zsh-autosuggestions\n  zsh-syntax-highlighting\n  zsh-autocomplete\n  docker\n  sudo\n  copyfile\n  history\n)/gs' "$ZSHRC"

# --- 3. 添加 PYTHONPATH 和 network_turbo (如果不存在则添加) ---
# 检查是否已有 network_turbo，没有则追加到末尾
if ! grep -q "source /etc/network_turbo" "$ZSHRC"; then
    echo -e "\n# 自动添加的学术加速和路径设置" >> "$ZSHRC"
    echo "source /etc/network_turbo" >> "$ZSHRC"
fi

# 检查是否已有 PYTHONPATH，没有则追加
if ! grep -q "export PYTHONPATH=\"/root/autodl-tmp/pykt-toolkit/" "$ZSHRC"; then
    echo "export PYTHONPATH=\"/root/autodl-tmp/pykt-toolkit/:\$PYTHONPATH\"" >> "$ZSHRC"
fi

echo "✅ .zshrc 修改完成！"