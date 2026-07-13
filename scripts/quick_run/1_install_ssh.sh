#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PUBLIC_KEY_FILE="${SCRIPT_DIR}/id_rsa.pub"
SSH_DIR="${HOME}/.ssh"
SSH_PUBLIC_KEY_FILE="${SSH_DIR}/id_rsa.pub"
AUTH_KEYS_FILE="${SSH_DIR}/authorized_keys"

if [[ ! -s "$PUBLIC_KEY_FILE" ]]; then
  echo "错误：公钥文件不存在或为空：$PUBLIC_KEY_FILE"
  exit 1
fi

echo "创建并设置 SSH 目录权限：$SSH_DIR"
install -d -m 700 "$SSH_DIR"

echo "覆盖公钥文件：$SSH_PUBLIC_KEY_FILE"
install -m 644 "$PUBLIC_KEY_FILE" "$SSH_PUBLIC_KEY_FILE"

echo "覆盖 authorized_keys：$AUTH_KEYS_FILE"
install -m 600 "$PUBLIC_KEY_FILE" "$AUTH_KEYS_FILE"

echo
echo "部署完成："
echo "  公钥文件：$SSH_PUBLIC_KEY_FILE"
echo "  授权文件：$AUTH_KEYS_FILE"
echo "  .ssh 权限：$(stat -c '%a' "$SSH_DIR")"
echo "  authorized_keys 权限：$(stat -c '%a' "$AUTH_KEYS_FILE")"