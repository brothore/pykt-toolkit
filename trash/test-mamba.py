import torch
from mamba_ssm import Mamba2

# 设置参数
batch, length, dim = 2, 64, 384  # 批量大小、序列长度、模型维度

# 创建输入张量
x = torch.randn(batch, length, dim).to("cuda")

# 初始化 Mamba2 模型
model = Mamba2(
    d_model=dim,     # 模型维度
    d_state=64,      # SSM 状态扩展因子
    d_conv=4,        # 局部卷积宽度
    expand=2,        # 块扩展因子
    headdim=48       # 头维度
).to("cuda")

# 前向传播
y = model(x)

# 验证输出形状
assert y.shape == x.shape
print("测试成功！输出形状:", y.shape)