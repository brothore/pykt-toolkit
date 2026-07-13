#!/usr/bin/env bash
set -e

ZSHRC="$HOME/.zshrc"

touch "$ZSHRC"

if ! grep -qF '# >>> local proxy >>>' "$ZSHRC"; then
  cat >> "$ZSHRC" <<'EOF'

# >>> local proxy >>>
export http_proxy="http://127.0.0.1:7892"
export https_proxy="http://127.0.0.1:7892"
export HTTP_PROXY="http://127.0.0.1:7892"
export HTTPS_PROXY="http://127.0.0.1:7892"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
# <<< local proxy <<<
EOF

  echo "代理配置已写入 $ZSHRC"
else
  echo "代理配置已存在，未重复写入。"
fi