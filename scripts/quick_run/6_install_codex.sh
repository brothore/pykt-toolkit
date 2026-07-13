#!/usr/bin/env bash
set -Ee -o pipefail

NVM_VERSION="v0.40.3"
NVM_DIR="${HOME}/.nvm"
ZSHRC_FILE="${HOME}/.zshrc"

echo "=== 0. 启用网络 / VPN ==="
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi
if [[ -r /etc/ssl/certs/autodl-signed.pem ]]; then
  export NODE_EXTRA_CA_CERTS="/etc/ssl/certs/autodl-signed.pem"
fi

echo "=== 1. 安装基础依赖 ==="
if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  echo "错误：需要 root 权限或 sudo。"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y curl ca-certificates git

echo "=== 2. 安装 / 加载 nvm ==="
if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
  curl -fsSL \
    "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" \
    | bash
fi

# shellcheck disable=SC1091
source "${NVM_DIR}/nvm.sh"

echo "=== 3. 安装 Node.js LTS（含 npm） ==="
nvm install --lts
nvm use --lts
nvm alias default 'lts/*'

echo "Node.js: $(node --version)"
echo "npm:     $(npm --version)"

echo "=== 4. 安装 / 更新 Codex CLI ==="
npm install --global @openai/codex

echo "=== 5. 配置 Zsh 自动加载 nvm ==="
touch "${ZSHRC_FILE}"

if ! grep -qF '# >>> nvm >>>' "${ZSHRC_FILE}"; then
  cat >> "${ZSHRC_FILE}" <<'EOF'

# >>> nvm >>>
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
# <<< nvm <<<
EOF
fi

echo "=== 6. 验证 ==="
codex --version

echo
echo "安装完成。重新打开 Zsh，或执行："
echo "source ~/.zshrc"
echo "codex"