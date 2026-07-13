#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== 启用网络加速 ==="
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
else
  echo "错误：未找到 /etc/network_turbo"
  exit 1
fi

REPO_URL="https://github.com/brothore/pykt-toolkit.git"
TARGET_DIR="${1:-pykt-toolkit}"

if [[ -e "$TARGET_DIR" ]]; then
  echo "错误：目标路径已存在：$TARGET_DIR"
  echo "请更换目录名，例如：bash clone_pykt.sh pykt-toolkit-new"
  exit 1
fi

echo "=== 克隆仓库 ==="
cd /root/autodl-tmp/
DATA_DIR="${TARGET_DIR}/data"
mkdir -p "$DATA_DIR"

for archive in assist2009.zip nips_task34.zip peiyou.zip; do
  [[ -f "./$archive" ]] || { echo "缺少文件：$archive"; exit 1; }
  cp -f "./$archive" "$DATA_DIR/"
done

echo "完成：$(cd "$TARGET_DIR" && pwd)"