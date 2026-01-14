try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("✅ 基础组件加载成功！")
except ImportError as e:
    print(f"❌ 加载失败：{e}")