#!/bin/bash
# 文件名：setup-remotes.sh
# 功能：验证并正确配置远程仓库
git config --global user.email "brothore@outlook.com"
git config --global user.name "brothore"
# 检查是否在Git仓库中
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "错误：当前目录不是Git仓库"
    exit 1
fi

# 删除可能存在的旧配置（安全方式）
git remote remove origin 2>/dev/null
git remote remove upstream 2>/dev/null

# 配置个人远程仓库
git remote add origin https://github.com/brothore/pykt-toolkit.git

# 配置上游原始仓库
git remote add upstream https://github.com/pykt-team/pykt-toolkit.git

# 设置推送默认策略（安全推荐）
git config push.default current

# 验证配置
echo "✅ 远程仓库配置完成"
echo "当前远程仓库配置："
git remote -v

# 测试连接性
echo "测试连接性..."
git fetch origin --dry-run
git fetch upstream --dry-run

echo "🎉 配置成功！您的仓库已准备好："
echo " - origin 指向您的个人仓库 (推送/拉取)"
echo " - upstream 指向原始团队仓库 (仅拉取)"