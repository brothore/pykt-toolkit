#!/usr/bin/env bash
# restore_torch28.sh
# 从 torch 2.12 / mamba-ssm 2.3 / cu13 混乱状态恢复到 torch 2.8.0+cu128
# 用法: bash scripts/quick_run/restore_torch28.sh
#
# 处理顺序: 卸应用层(mamba) -> 卸 torch 2.12/triton 3.7 -> 卸 cu13 套件 -> 装回 torch 2.8 -> 自检
# 不会动: torchvision 0.23.0+cu128, 所有 nvidia-*-cu12 后缀的包

set -euo pipefail

source /etc/network_turbo

PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "[info] site-packages: $PY_SITE"

# ──────────── Step 1/5: 卸 mamba-ssm 套件 ────────────
echo
echo "==[1/5]== 卸载 mamba-ssm 及其独有依赖 ..."
pip uninstall -y \
  mamba-ssm causal-conv1d \
  tilelang quack-kernels apache-tvm-ffi torch-c-dlpack-ext \
  nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base || true

# 清掉按 torch 2.12 ABI 预编译的 .so，避免后续残留污染
for stem in selective_scan_cuda causal_conv1d_cuda; do
  for f in "$PY_SITE/${stem}".*.so; do
    [ -e "$f" ] && { echo "  rm $f"; rm -f "$f"; }
  done
done

# ──────────── Step 2/5: 卸 torch 2.12 + triton 3.7 ────────────
echo
echo "==[2/5]== 卸载 torch 2.12 / triton 3.7（保留 torchvision）..."
pip uninstall -y torch triton || true

# ──────────── Step 3/5: 卸 cu13 系列 nvidia 包 + cuda-toolkit 套件 ────────────
echo
echo "==[3/5]== 卸载 cu13 系列 nvidia 包与 cuda-toolkit 套件 ..."
# 注意: 卸的是「无 -cu12 后缀」+「-cu13 后缀」的；所有 -cu12 后缀的包保留
pip uninstall -y \
  cuda-toolkit cuda-bindings cuda-python cuda-pathfinder \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-nvjitlink nvidia-nvtx \
  nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 || true

# ──────────── Step 4/5: 装回 torch 2.8.0+cu128 ────────────
echo
echo "==[4/5]== 安装 torch 2.8.0+cu128（来自 PyTorch 官方索引）..."
# +cu128 的 wheel 只在 PyTorch 官方索引里有, aliyun 镜像没有
# 用 --index-url 把官方索引置为主索引; triton 3.4 等子依赖在这个索引里也都有
#
# --force-reinstall 关键: Step 3 卸 cu13 包时会顺手清掉与 cu12 共享的 nvidia/<sub>/lib/
# 目录, 误删 cu12 的 .so 文件 (典型受害者: nccl-cu12 / cusparselt-cu12)。
# pip 默认看到元数据"已满足"就跳过, .so 不会被补回, 后续 import torch 就报
# undefined symbol / cannot open shared object。--force-reinstall 强制重写所有
# wheel 内文件, 把 14 个 nvidia-*-cu12 包的 .so 一并落盘修复。
pip install --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128"

# ──────────── Step 5/5: 自检 ────────────
echo
echo "==[5/5]== 自检 ..."
python - <<'PY'
import sys, os, glob
try:
    # 0) nvidia-*-cu12 包文件完整性: pip 元数据在场但 lib/ 为空 = 之前误删
    sp_nvidia = "/root/miniconda3/lib/python3.12/site-packages/nvidia"
    broken = []
    if os.path.isdir(sp_nvidia):
        for pkg in sorted(os.listdir(sp_nvidia)):
            libdir = os.path.join(sp_nvidia, pkg, "lib")
            if os.path.isdir(libdir) and not glob.glob(os.path.join(libdir, "*.so*")):
                broken.append(pkg)
    if broken:
        print(f"[FAIL] 以下 nvidia 包 lib/ 为空: {broken}")
        print("       修复: pip install --no-cache-dir --force-reinstall --no-deps " +
              " ".join(f'nvidia-{p}-cu12' for p in broken))
        sys.exit(2)

    import torch, torchvision, triton
    print(f"[ok ] torch       = {torch.__version__}   (cuda={torch.version.cuda})")
    print(f"[ok ] torchvision = {torchvision.__version__}")
    print(f"[ok ] triton      = {triton.__version__}")

    # 验证 torchvision C++ 算子 ABI（torch 2.12 时这里会 RuntimeError: operator torchvision::nms does not exist）
    boxes  = torch.tensor([[0,0,10,10],[1,1,11,11]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8])
    keep = torchvision.ops.nms(boxes, scores, 0.5).tolist()
    print(f"[ok ] torchvision.ops.nms -> {keep}")

    # 验证 CUDA 真能用
    if torch.cuda.is_available():
        print(f"[ok ] cuda avail  = True | device = {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] cuda avail = False  (CPU-only 环境可忽略)")

    assert torch.__version__.startswith("2.8."), f"torch 版本不对: {torch.__version__}"
    print("\n[done] torch 2.8.0+cu128 恢复完成，可以重跑修复版的 install_mamba.sh")
except Exception as e:
    print(f"[FAIL] 自检失败: {type(e).__name__}: {e}")
    sys.exit(1)
PY
