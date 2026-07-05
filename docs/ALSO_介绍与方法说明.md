# ALSO 介绍与方法说明

## 1. 项目背景

ALSO 是一个面向深度学习训练场景的分布鲁棒优化器（Distributionally Robust Optimizer, DRO）实现，代码位于 [ALSO](../ALSO)。它的目标是解决现实世界中训练数据存在异质性时，模型训练容易偏向“平均样本”而忽略“困难样本/小群体”的问题。

在许多实际任务中，数据并不均匀，例如：

- 类别不平衡；
- 不同来源的数据分布不同；
- 某些群体样本数量较少，但更重要或更难学。

传统深度学习训练通常对所有样本一视同仁，而 DRO 的思路是让模型在“最坏情况分布”下也表现较好。ALSO 的核心价值在于：它试图把 DRO 的思想和深度学习中的标准训练流程更好地衔接起来。

---

## 2. 项目构成

当前子项目主要包含以下内容：

- [ALSO/README.md](../ALSO/README.md)：说明项目动机、方法背景与使用方式。
- [ALSO/also.py](../ALSO/also.py)：ALSO 的核心实现，定义了优化器类。
- [ALSO/example.ipynb](../ALSO/example.ipynb)：一个端到端的使用示例。
- [ALSO/paper](../ALSO/paper)：论文相关实验与材料。

从结构上看，它是一个“优化器层”的研究实现，而不是一个独立的知识追踪模型工程。

---

## 3. ALSO 的核心思想

ALSO 的核心目标是同时做两件事：

1. 更新模型参数，让模型在当前损失上更好地拟合数据；
2. 动态调整不同样本或不同组的权重，让训练过程更加关注更困难、也更容易被忽略的部分。

它可以把训练看成一个“模型参数更新 + 数据权重更新”的联合过程。

### 3.1 为什么需要它

传统训练方法往往假设所有样本都同样重要，这会带来几个问题：

- 训练会偏向多数类；
- 少数群体的错误被掩盖；
- 模型在实际分布变化时鲁棒性不足。

DRO 的思想正是为了缓解这些问题，但传统 DRO 在深度学习中容易遇到以下难点：

- 和 Adam 这类常用优化器兼容性不足；
- 训练时是 mini-batch 方式，组权重更新不容易直接落地；
- 神经网络通常是非凸问题，理论分析和实现都更复杂。

ALSO 正是为了解决这些落地问题而设计的。

### 3.2 它的主要特点

ALSO 的主要特点包括：

- 支持自适应更新，类似 Adam 的参数更新方式；
- 支持对样本或组进行动态加权；
- 兼容标准深度学习训练流程；
- 可以在非凸目标函数下进行收敛分析；
- 适合处理数据异质性和分布偏移问题。

---

## 4. 方法说明

从实现上看，ALSO 主要引入了两个关键对象：

- 模型参数 $\theta$：负责学习任务目标；
- 组权重分布 $\pi$：负责控制不同样本组在训练中的重要性。

在每一步训练中，ALSO 会：

1. 根据当前的权重分布 $\pi$ 计算一批样本的加权损失；
2. 使用这些损失更新模型参数；
3. 再根据当前损失结果更新 $\pi$，让更“困难”的组获得更大的影响。

这种机制可以理解为一种“对抗性地调整训练权重”的方法：它不是被动地接受数据分布，而是主动地让训练过程更关注最容易造成模型失效的部分。

### 4.1 关键参数

在 [ALSO/also.py](../ALSO/also.py) 中，主要有以下几类参数：

- 模型更新相关：
  - `lr`
  - `weight_decay`
  - `betas`
- 权重分布更新相关：
  - `pi_lr`
  - `pi_decay`
  - `pi_reg`
  - `pi_init`
- 训练尺度相关：
  - `n_groups`
  - `batch_size`
  - `loss_scale`

其中 `n_groups` 表示分组数量；如果你希望对“每个样本单独建组”，就可以让它等于样本数；如果你希望按类别做分组，就可以按类别数设置。

---

## 5. 使用方式

ALSO 的使用方式相对直接，通常分为以下几个步骤：

1. 初始化优化器；
2. 定义损失函数，并确保损失是逐样本返回的；
3. 在训练循环中构造一个闭包函数；
4. 调用 `optimizer.step(...)` 执行一次更新。

### 5.1 典型用法

在示例中，使用方式基本如下：

```python
from also import ALSO

optimizer = ALSO(
    params=model.parameters(),
    n_groups=2,
    batch_size=512,
)

loss_fn = nn.BCEWithLogitsLoss(reduction='none')

for X, y in train_loader:
    def closure(w, scale):
        optimizer.zero_grad()
        preds = model(X).flatten()
        losses = loss_fn(preds, y.flatten().float())
        losses = losses * scale
        loss = (w * losses).sum()
        loss.backward()
        return losses, losses.mean().item()

    optimizer.step(closure=closure, groups_indexes=y)
```

### 5.2 需要注意的点

- 损失函数必须使用 `reduction='none'`，因为 ALSO 需要每个样本的独立损失值；
- 需要为每个 batch 提供 `groups_indexes`，用于指明当前样本属于哪个组；
- 如果是类别不平衡任务，通常会把类别作为分组维度。

---

## 6. 与论文的关系

论文文件为 [papers/Aligning Distributionally Robust Optimization with Practical.pdf](../papers/Aligning%20Distributionally%20Robust%20Optimization%20with%20Practical.pdf)。从论文标题和仓库说明可知，这个项目是对论文方法的官方实现。

论文的核心主张可以概括为：

- 传统 DRO 和深度学习实践之间存在明显差距；
- ALSO 通过“自适应的模型更新 + 可随机更新的样本/组权重”来更好地适配实际训练管线；
- 它在数据异质性较强的任务上，通常比传统训练方法和已有 DRO 方法更有优势。

---

## 7. 适用场景

ALSO 更适合以下场景：

- 类别不平衡较严重的分类任务；
- 多源数据训练或存在分布偏移的问题；
- 希望模型在少数群体或困难样本上表现更稳健时。

如果你的任务数据较均衡，且训练已经稳定，那么普通 Adam 或 SGD 可能已经足够。

---

## 8. 一句话总结

ALSO 是一个把“分布鲁棒优化”思想真正落到深度学习训练实践中的优化器：它不只是让模型学得更好，而是让训练过程在数据异质性下更具鲁棒性。
