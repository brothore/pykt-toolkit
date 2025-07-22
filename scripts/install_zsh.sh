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
