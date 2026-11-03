#!/usr/bin/env bash
# Linux GPU 综合诊断
# 可指定 Python：PYTHON=/path/to/python bash gpu_diag.sh

set -u
set -o pipefail
export LC_ALL=C

has() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n========== %s ==========\n' "$*"; }

# 限制部分诊断命令的运行时间，避免驱动异常时一直卡住。
run() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    local rc=0
    if has timeout; then
        timeout 30s "$@" 2>&1 || rc=$?
    else
        "$@" 2>&1 || rc=$?
    fi
    if (( rc == 124 )); then
        printf '[超时] 命令超过 30 秒\n'
    elif (( rc != 0 )); then
        printf '[返回码 %s]\n' "$rc"
    fi
    return 0
}

section "1. 系统与运行环境"
date -Is
uname -a
printf '当前用户：'; id
printf '系统：'
if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    printf '%s\n' "${PRETTY_NAME:-unknown}"
else
    printf 'unknown\n'
fi

if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
    echo "容器环境：检测到容器标记"
fi
if has systemd-detect-virt; then
    run systemd-detect-virt
fi
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "WSL 环境：已检测到"
fi

section "2. GPU 相关环境变量"
for name in \
    CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES \
    CUDA_DEVICE_ORDER CUDA_HOME CUDA_PATH \
    HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL \
    ROCM_PATH HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH
do
    if [[ -v "$name" ]]; then
        printf '%s=%s\n' "$name" "${!name}"
    else
        printf '%s=<未设置>\n' "$name"
    fi
done
echo '提示：可见设备变量为空或设为 -1，可能会屏蔽 GPU。'

section "3. PCI GPU 与绑定驱动"
if has lspci; then
    lspci -nnk | grep -A 4 -Ei \
        'VGA compatible controller|3D controller|Display controller' \
        || echo "未发现 PCI 显示/计算设备；容器或 WSL 中可能不完整。"
else
    echo "未安装 lspci，改查 sysfs。"
fi

for dev in /sys/bus/pci/devices/*; do
    [[ -r "$dev/class" ]] || continue
    cls=$(cat "$dev/class")
    case "$cls" in
        0x03*)
            printf '\nPCI=%s vendor=%s device=%s class=%s\n' \
                "${dev##*/}" \
                "$(cat "$dev/vendor")" \
                "$(cat "$dev/device")" "$cls"
            if [[ -L "$dev/driver" ]]; then
                printf 'driver=%s\n' "$(basename "$(readlink "$dev/driver")")"
            else
                echo "driver=<未绑定>"
            fi
            ;;
    esac
done

section "4. 内核驱动"
if has lsmod; then
    lsmod | grep -Ei \
        '^(nvidia|nouveau|amdgpu|radeon|i915|xe)([[:space:]_])' \
        || echo "未看到常见 GPU 模块；WSL、容器或内建驱动场景需结合后续结果。"
fi
if [[ -r /proc/driver/nvidia/version ]]; then
    cat /proc/driver/nvidia/version
fi
if has mokutil; then
    run mokutil --sb-state
fi

section "5. GPU 设备节点与当前用户权限"
shopt -s nullglob
nodes=(/dev/nvidia* /dev/dri/card* /dev/dri/renderD* /dev/kfd /dev/dxg)
if (( ${#nodes[@]} == 0 )); then
    echo "未发现常见 GPU 设备节点。"
else
    for node in "${nodes[@]}"; do
        [[ -e "$node" ]] || continue
        ls -ld "$node"
        if [[ -c "$node" ]]; then
            r=no; w=no
            [[ -r "$node" ]] && r=yes
            [[ -w "$node" ]] && w=yes
            printf '  当前用户权限：read=%s write=%s\n' "$r" "$w"
        fi
    done
fi

section "6. NVIDIA 驱动、显存、温度、进程"
smi=""
if has nvidia-smi; then
    smi=$(command -v nvidia-smi)
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    smi=/usr/lib/wsl/lib/nvidia-smi
fi

if [[ -n "$smi" ]]; then
    run "$smi"
    run "$smi" \
        --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free \
        --format=csv
    run "$smi" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv
    run "$smi" -L
else
    echo "未找到 nvidia-smi。"
fi

section "7. AMD / ROCm"
if has amd-smi; then
    run amd-smi version
    run amd-smi list
    run amd-smi metric
elif has rocm-smi; then
    run rocm-smi
else
    echo "未找到 amd-smi / rocm-smi。"
fi

if has rocminfo; then
    echo "rocminfo 摘要："
    if has timeout; then
        timeout 30s rocminfo 2>&1 |
            grep -Ei 'Name:|Marketing Name:|Vendor Name:|Device Type:|error|failed' |
            head -n 100
    else
        rocminfo 2>&1 |
            grep -Ei 'Name:|Marketing Name:|Vendor Name:|Device Type:|error|failed' |
            head -n 100
    fi
fi

section "8. 工具链与动态库"
if has nvcc; then
    run nvcc --version
else
    echo "nvcc：未找到；运行预编译的 PyTorch 通常不要求安装 nvcc。"
fi
if has hipcc; then
    run hipcc --version
fi

ldconfig_bin=""
if has ldconfig; then
    ldconfig_bin=$(command -v ldconfig)
elif [[ -x /sbin/ldconfig ]]; then
    ldconfig_bin=/sbin/ldconfig
fi
if [[ -n "$ldconfig_bin" ]]; then
    "$ldconfig_bin" -p 2>/dev/null |
        grep -Ei 'libcuda\.so|libcudart\.so|libnvidia-ml\.so|libamdhip64\.so|libhsa-runtime64\.so|libze_loader\.so' \
        || echo "动态库缓存中未发现常见 GPU 库；Python 环境仍可能自带运行库。"
fi

section "9. 当前 Python 环境与实际 GPU 计算"
py="${PYTHON:-}"
if [[ -z "$py" ]]; then
    if has python3; then
        py=python3
    elif has python; then
        py=python
    fi
fi

if [[ -z "$py" ]]; then
    echo "未找到 Python，跳过框架测试。"
else
    printf '选择的 Python：%s\n' "$py"

    pycmd=("$py")
    if has timeout; then
        pycmd=(timeout 120s "$py")
    fi

    "${pycmd[@]}" - <<'PY'
import sys
import traceback
from importlib import metadata

print("解释器：", sys.executable)
print("Python：", sys.version.replace("\n", " "))

for pkg in (
    "torch", "torchvision", "tensorflow",
    "onnxruntime", "onnxruntime-gpu",
    "cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x",
):
    try:
        print(f"包版本：{pkg}=={metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        pass

try:
    import torch
except ModuleNotFoundError as exc:
    print("\n[未完成计算验证] PyTorch 或其依赖未安装：", exc)
    print("请在实际运行项目的 Python/虚拟环境中重新执行脚本。")
    sys.exit(0)
except Exception:
    print("\n[异常] PyTorch 导入失败：")
    traceback.print_exc()
    sys.exit(1)

print("\nPyTorch：", torch.__version__)
print("构建 CUDA 版本：", getattr(torch.version, "cuda", None))
print("构建 HIP 版本：", getattr(torch.version, "hip", None))
print("CUDA/HIP 后端可用：", torch.cuda.is_available())

backends = []
if torch.cuda.is_available():
    backends.append(("cuda", torch.cuda))
    print("CUDA/HIP 设备数：", torch.cuda.device_count())
    print("PyTorch 编译架构：", torch.cuda.get_arch_list())
elif getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
    print("尝试初始化 CUDA/HIP，获取具体错误：")
    try:
        torch.cuda.init()
    except Exception:
        traceback.print_exc()
else:
    print("当前 PyTorch 未包含 CUDA/HIP 后端。")

xpu = getattr(torch, "xpu", None)
if xpu is not None:
    try:
        print("Intel XPU 后端可用：", xpu.is_available())
        if xpu.is_available():
            backends.append(("xpu", xpu))
    except Exception:
        traceback.print_exc()

passed = failed = 0
for kind, api in backends:
    for i in range(api.device_count()):
        device = f"{kind}:{i}"
        print(f"\n--- 实际计算测试：{device} ---")
        try:
            print("设备名称：", api.get_device_name(i))
            if kind == "cuda":
                props = api.get_device_properties(i)
                print("总显存 GiB：", round(props.total_memory / 1024**3, 2))
                print("设备属性：", props)

            # 约 12 MiB 的矩阵数据，验证分配、传输、矩阵乘法和同步。
            # 使用恒等矩阵，使结果可以直接与输入比较。
            with torch.inference_mode():
                cpu = torch.randn(1024, 1024, dtype=torch.float32)
                a = cpu.to(device)
                identity = torch.eye(1024, device=device)
                result = a @ identity
                api.synchronize(i)
                actual = result.cpu()
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError("计算结果包含 NaN 或 Inf")
                if not torch.allclose(actual, cpu, rtol=1e-4, atol=1e-4):
                    raise RuntimeError("GPU 计算结果校验失败")
                print("[通过] 显存分配、CPU/GPU 传输、矩阵乘法、同步和数值校验")
                passed += 1
                del a, identity, result, actual, cpu
                api.empty_cache()
        except Exception:
            failed += 1
            print("[失败] 实际 GPU 计算异常：")
            traceback.print_exc()

print("\n========== 计算诊断结论 ==========")
if passed and not failed:
    print(f"GPU 在当前 PyTorch 环境中可用，{passed} 个设备通过实际计算。")
elif passed and failed:
    print(f"GPU 部分可用：{passed} 个设备通过，{failed} 个设备失败。")
elif failed:
    print("框架发现了 GPU，但实际计算失败，请查看上方异常。")
else:
    print("当前 PyTorch 环境没有可用于计算的 GPU 后端。")
    print("结合硬件、驱动、设备权限、容器透传和 PyTorch 构建版本继续定位。")
PY
    rc=$?
    if (( rc == 124 )); then
        echo "[异常] Python GPU 测试超过 120 秒，可能卡在驱动初始化或计算阶段。"
    elif (( rc != 0 )); then
        printf '[异常] Python 测试退出码：%s\n' "$rc"
    fi
fi

section "10. 内核 GPU 错误日志"
if has dmesg; then
    kernel_log=$(dmesg --color=never 2>&1)
    rc=$?
    if (( rc == 0 )); then
        printf '%s\n' "$kernel_log" |
            grep -Ei \
            'NVRM|Xid|GPU.*(fault|hang|reset|error)|amdgpu.*(fail|error|timeout)|i915.*(fail|error|hang)|xe.*(fail|error|hang)|fallen off' |
            tail -n 60 \
            || echo "未发现匹配的 GPU 错误日志。"
    else
        printf '%s\n' "$kernel_log"
        echo "当前用户无法读取内核日志，本项跳过。"
    fi
fi

section "诊断完成"
echo "判断计算是否可用，优先看第 9 项的实际计算结果。"
echo "nvidia-smi 成功仅说明驱动管理接口可访问。"
echo "nvidia-smi 显示的 CUDA Version 是驱动支持能力，不等同于已安装的 CUDA Toolkit 版本。"
echo "此脚本用于收集诊断信息；脚本本身的退出码不代表 GPU 健康状态。"