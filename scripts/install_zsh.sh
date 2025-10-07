#!/bin/bash
# Zsh 自动化安装与配置脚本

# --- 1. 安装 Zsh 和依赖 ---

echo "--- 正在安装 Zsh 和必要的工具 ---"

# 尝试安装 zsh (可能需要  权限，并根据不同系统尝试不同包管理器)
if ! command -v zsh &> /dev/null; then
    echo "正在安装 zsh..."
     apt install -y zsh ||  yum install -y zsh || brew install zsh
fi

# 检查/安装 Git
if ! command -v git &> /dev/null; then
    echo "正在安装 Git..."
     apt-get install -y git ||  yum install -y git || brew install git
fi

# 检查/安装 curl
if ! command -v curl &> /dev/null; then
    echo "正在安装 curl..."
     apt-get install -y curl ||  yum install -y curl || brew install curl
fi

# --- 2. 安装 Oh My Zsh ---

if [ ! -d "${HOME}/.oh-my-zsh" ]; then
    echo "--- 正在安装 Oh My Zsh (使用官方链接) ---"
    # 使用官方 GitHub 仓库地址安装，并跳过用户交互
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
else
    echo "Oh My Zsh 已安装，跳过安装。"
fi

# 修复/设置 ZSH_CUSTOM 路径变量
ZSH_CUSTOM="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

# --- 3. 安装 Autojump 二进制包 ---

if ! command -v autojump &> /dev/null; then
    echo "--- 正在安装 autojump (目录跳转) ---"
    # Autojump 作为系统包安装，以便其二进制文件可用
     apt-get install -y autojump ||  yum install -y autojump || brew install autojump
else
    echo "autojump 已安装，跳过安装。"
fi

# --- 4. 安装常用插件 (使用官方 GitHub 链接) ---

# 定义要安装的插件列表 (格式: 仓库拥有者/仓库名)
plugin_repos=(
    "zsh-users/zsh-syntax-highlighting" # 语法高亮
    "zsh-users/zsh-autosuggestions"     # 自动建议
    "zsh-users/zsh-completions"         # 自动补全
)

# 用于存储成功安装的插件名称，用于写入 .zshrc
declare -a plugins_list=() 
# 确保 oh-my-zsh 默认插件也包含
plugins_list+=("git")

echo "--- 正在安装 Oh My Zsh 插件 (使用官方链接) ---"
for plugin_repo in "${plugin_repos[@]}"; do
    # 提取插件名 (即仓库名)
    plugin_name="${plugin_repo##*/}"
    plugin_dir="${ZSH_CUSTOM}/plugins/${plugin_name}"

    if [ ! -d "$plugin_dir" ]; then
        echo "正在克隆插件: $plugin_name"
        # 使用官方 GitHub 地址克隆
        git clone --depth 1 "https://github.com/${plugin_repo}.git" "$plugin_dir"
        echo "✅ 已安装 $plugin_name"
    else
        echo "插件 $plugin_name 已存在，跳过克隆。"
    fi
    # 无论是否克隆，都将插件名添加到列表中，用于写入 .zshrc
    plugins_list+=("$plugin_name")
done

# --- 5. 安装 Powerlevel10k 主题 (使用官方 GitHub 链接) ---

p10k_theme_dir="${ZSH_CUSTOM}/themes/powerlevel10k"
if [ ! -d "$p10k_theme_dir" ]; then
    echo "--- 正在安装 Powerlevel10k 主题 (使用官方链接) ---"
    # 使用官方 GitHub 仓库地址克隆
    git clone --depth=1 "https://github.com/romkatv/powerlevel10k.git" "$p10k_theme_dir"
    echo "✅ Powerlevel10k 安装完成。"
else
    echo "Powerlevel10k 主题已存在，跳过安装。"
fi


# --- 6. 配置 .zshrc ---

echo "--- 正在写入/优化 .zshrc 配置 ---"
# 插件列表用空格分隔
plugins_line=$(IFS=' '; echo "${plugins_list[*]}")

cat << EOF > ~/.zshrc # 覆盖写入配置
# --- Oh My Zsh Core Configuration ---
export ZSH="\${HOME}/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"
# 自动加载的插件列表 (zsh-completions, zsh-syntax-highlighting, zsh-autosuggestions)
plugins=(${plugins_line})

# 加载核心框架
source \$ZSH/oh-my-zsh.sh

# --- Autojump Configuration (更健壮的加载) ---
# 尝试加载 autojump (适用于不同系统/安装方式)
if [ -f /usr/share/autojump/autojump.sh ]; then
    source /usr/share/autojump/autojump.sh
elif [ -f /usr/share/autojump/autojump.zsh ]; then
    source /usr/share/autojump/autojump.zsh
elif [ -f /usr/local/etc/profile.d/autojump.sh ]; then
    source /usr/local/etc/profile.d/autojump.sh
elif [ -f /opt/homebrew/etc/profile.d/autojump.sh ]; then
    source /opt/homebrew/etc/profile.d/autojump.sh
fi

# --- Powerlevel10k Configuration ---
# 自动应用 p10k 配置 (首次运行将启动配置向导)
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# --- 其他配置 ---
# 自动补全系统 (zsh-completions) 已通过 plugins 数组加载
# 语法高亮和自动建议也已通过 plugins 数组加载

EOF

# --- 7. 设为默认 Shell ---

# 仅当当前默认 shell 不是 zsh 时才更改
if [ "$SHELL" != "$(which zsh)" ]; then
    echo "--- 正在设置 zsh 为默认 Shell ---"
    # 使用 which zsh 确保路径正确
    chsh -s "$(which zsh)"
    echo "✅ 已设置 zsh 为默认终端，需要重新登录或重启终端生效。"
else
    echo "zsh 已是默认 Shell，跳过设置。"
fi

echo "--------------------------------------"
echo "🎉 安装和配置脚本完成！"
echo "--------------------------------------"
echo "后续步骤："
echo "1. **重启终端**或 **注销/重新登录**，使 Zsh 生效。"
echo "2. 启动 Zsh 后，**Powerlevel10k 配置向导**将自动运行。"
echo "3. 如果向导未自动运行，请手动运行：\033[1m\033[33mp10k configure\033[0m"
echo "4. 完成配置后，享受你的新终端！"