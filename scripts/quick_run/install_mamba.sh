#!/usr/bin/env bash
# 在 torch 2.8.0+cu128 / cp312 / cxx11abiTRUE 环境下安装 mamba-ssm，不升级 torch
#
# 关键: 直接拿 GitHub Release 上「精确匹配本环境 ABI」的预编译 wheel,
#       绕开 setup.py 的「猜 URL 再 fallback 源码编」逻辑 (本地没 nvcc 编不了)。
#
# 已验证的 wheel 组合:
#   causal-conv1d 1.6.1  @ tag v1.6.2  + cu12 torch2.8 cp312 abiTRUE
#   mamba-ssm     2.3.1  @ tag v2.3.1  + cu12 torch2.8 cp312 abiTRUE
#
# 用法: bash scripts/quick_run/install_mamba.sh
set -euo pipefail

# 1) 网络加速 (AutoDL)
source /etc/network_turbo

# 2) 前置校验: 必须是 torch 2.8.x + cp312, 否则直接退出
python - <<'PY'
import sys, torch
v = torch.__version__
assert v.startswith("2.8."), f"需要 torch 2.8.x, 当前 {v}。先跑 restore_torch28.sh 恢复环境。"
assert sys.version_info[:2] == (3, 12), f"需要 Python 3.12, 当前 {sys.version_info[:2]}"
abi = torch._C._GLIBCXX_USE_CXX11_ABI
assert abi is True, f"需要 cxx11 ABI = True, 当前 {abi}"
triton_v = "N/A"
import importlib.util
if importlib.util.find_spec("triton"):
    import triton; triton_v = triton.__version__
print(f"[ok] torch={v}, cuda={torch.version.cuda}, py=3.12, cxx11abi=True, triton={triton_v}")
PY

# 3) 本地 wheel 优先, 缺了才回退到 GitHub URL
#    --no-deps: 切断 mamba-ssm 2.3.1 对 triton>=3.5 / tilelang / quack-kernels 的传染,
#               防止 pip 顺手把 torch 又升到 2.12
LOCAL_DIR="/root/autodl-tmp/pykt-toolkit/packages"
CCONV_WHL_NAME="causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
MAMBA_WHL_NAME="mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
CCONV_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2/${CCONV_WHL_NAME}"
MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/${MAMBA_WHL_NAME}"

resolve_wheel() {
    # $1 = 本地文件名, $2 = 在线 URL; 输出可被 pip install 的路径或 URL
    local local_path="${LOCAL_DIR}/$1"
    if [ -f "$local_path" ]; then
        echo "$local_path"
    else
        echo "$2"
    fi
}

CCONV_SRC="$(resolve_wheel "$CCONV_WHL_NAME" "$CCONV_URL")"
MAMBA_SRC="$(resolve_wheel "$MAMBA_WHL_NAME" "$MAMBA_URL")"

echo
echo "==[1/3]== 安装 causal-conv1d ..."
echo "         源: $CCONV_SRC"
pip install --no-input --no-deps "$CCONV_SRC"

echo
echo "==[2/3]== 安装 mamba-ssm (--no-deps) ..."
echo "         源: $MAMBA_SRC"
pip install --no-input --no-deps "$MAMBA_SRC"

# 4) 补齐 mamba-ssm 运行时真正需要的轻量包 (einops 是 Mamba 层 import 时的硬依赖)
echo
echo "==[3/3]== 补齐运行依赖 (einops) ..."
pip install --no-input einops

# 5) 自检: import + GPU forward 都过才算成功
echo
echo "==[check]== 自检 ..."
python - <<'PY'
import torch
from mamba_ssm import Mamba
assert torch.cuda.is_available(), "CUDA 不可用, 无法验证 mamba CUDA 扩展"
m = Mamba(d_model=64, d_state=16).cuda()
y = m(torch.randn(2, 10, 64, device="cuda"))
assert y.shape == (2, 10, 64), y.shape
# 顺手验证 causal_conv1d_cuda 扩展也活
from causal_conv1d import causal_conv1d_fn
print(f"[ok] Mamba forward = {tuple(y.shape)}, causal_conv1d_fn 可调用")
PY

echo
echo "[done] mamba-ssm 安装完成, 可以重跑训练脚本"
