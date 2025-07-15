import sys
from pathlib import Path

# 确保路径正确
mamba_path = Path("/root/miniconda3/lib/python3.10/site-packages/mamba_ssm/ops")
sys.path.append(str(mamba_path))

try:
    from mamba_ssm.ops.triton import k_activations
    print("✅ k_activations 导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")