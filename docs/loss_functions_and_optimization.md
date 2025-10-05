# 损失函数与优化系统说明文档

本文档详细说明了 Mosaic 项目中损失函数和优化系统的实现架构和位置。

## 1. 核心架构概览

Mosaic 的损失函数和优化系统基于以下几个核心组件：

- **损失函数抽象基类** (`LossTerm`)：定义统一的损失接口
- **结构预测损失函数**：实现具体的蛋白质设计损失
- **优化器**：在概率单纯形上进行梯度优化
- **模型集成层**：将损失函数与特定模型结合

## 2. 损失函数实现位置

### 2.1 基础抽象类

**位置**: [`../src/mosaic/common.py`](../src/mosaic/common.py)

```python
class LossTerm(eqx.Module):
    """损失函数的抽象基类"""
    def __call__(self, sequence, output, key):
        # 返回 (loss_value, auxiliary_info)
        pass

class LinearCombination(eqx.Module):
    """线性组合多个损失项"""
    pass
```

**核心功能**：
- 定义了所有损失函数的统一接口
- 支持损失函数的线性组合（如 `2.0 * loss1 + loss2`）
- 提供算术运算符重载

### 2.2 结构预测损失函数

**位置**: [`../src/mosaic/losses/structure_prediction.py`](../src/mosaic/losses/structure_prediction.py)

**主要损失类**：

#### `BinderTargetContact` (第211-247行)
- **作用**：优化结合物与目标蛋白的接触
- **参数**：
  - `contact_distance`: 接触距离阈值 (默认20.0Å)
  - `paratope_idx`: 结合位点残基索引
  - `epitope_idx`: 表位残基索引
- **机制**：通过距离图 (distogram) 计算接触概率

#### `WithinBinderContact` (第180-208行)
- **作用**：控制结合物内部的接触
- **参数**：
  - `max_contact_distance`: 最大接触距离 (默认14.0Å)
  - `min_sequence_separation`: 最小序列分离 (默认8)
  - `num_contacts_per_residue`: 每个残基的接触数 (默认25)
- **机制**：促进序列上相距较远但空间上接近的残基形成接触

#### 其他损失函数
- `HelixLoss`: 促进螺旋结构形成
- `DistogramRadiusOfGyration`: 控制蛋白质紧密度
- `IPTMLoss`, `PAELoss`, `PLDDTLoss`: 结构质量评估损失

### 2.3 模型特异性损失包装器

#### Boltz2 损失包装器

**位置**: [`../src/mosaic/losses/boltz2.py`](../src/mosaic/losses/boltz2.py)

```python
class Boltz2Loss(LossTerm):
    """Boltz2 模型的损失包装器"""
    joltz2: joltz.Joltz2  # JAX 转换的 Boltz2 模型
    features: PyTree      # 输入特征
    loss: LossTerm | LinearCombination  # 组合损失函数
    recycling_steps: int = 0   # 循环步数
    sampling_steps: int = 25   # 采样步数
```

**核心流程** (第360-381行)：
1. 更新序列特征：`set_binder_sequence(sequence, features)`
2. 创建模型输出：`Boltz2Output(...)`
3. 计算损失：`self.loss(sequence, output, key)`

#### Boltz1 损失包装器

**位置**: [`../src/mosaic/losses/boltz.py`](../src/mosaic/losses/boltz.py)

类似 Boltz2Loss 的实现，使用 Boltz1 模型。

## 3. 优化器实现位置

### 3.1 主优化算法

**位置**: [`../src/mosaic/optimizers.py`](../src/mosaic/optimizers.py)

#### `simplex_APGM` 函数 (第234-346行)

**算法**: 概率单纯形上的加速近端梯度法 (Accelerated Proximal Gradient Method)

**关键参数**：
- `loss_function`: 要优化的损失函数
- `x`: 初始序列 (概率分布形式)
- `n_steps`: 优化步数
- `stepsize`: 步长
- `momentum`: 动量参数
- `scale`: 正则化参数 (>1.0 鼓励稀疏解)
- `logspace`: 是否在对数空间优化 (对应 Bregman 方法)

**核心算法流程**：
```python
for _iter in range(n_steps):
    # 1. 计算当前点的损失和梯度
    (value, aux), g = _eval_loss_and_grad(x, loss_function, key)
    
    # 2. 梯度裁剪
    if gradient_norm > max_gradient_norm:
        g = g * (max_gradient_norm / gradient_norm)
    
    # 3. 梯度下降 + 投影到单纯形
    x_new = projection_simplex(scale * (x - stepsize * g))
    
    # 4. 动量更新
    x = x_new + momentum * (x_new - x_prev)
```

### 3.2 损失和梯度计算

#### `_eval_loss_and_grad` 函数 (第34-75行)

**功能**：
- 计算损失值和梯度
- 处理 NaN 值（替换为大数值）
- 梯度去中心化：`g - g.mean(axis=-1, keepdims=True)`

#### `projection_simplex` 函数 (第214-232行)

**功能**：将向量投影到概率单纯形上
- 确保所有元素非负
- 确保每个位置的概率和为1

## 4. 模型集成层

### 4.1 结构预测模型抽象基类

**位置**: [`../src/mosaic/structure_prediction.py`](../src/mosaic/structure_prediction.py)

```python
class StructurePredictionModel(ABC):
    @abstractmethod
    def build_loss(self, *, loss, features, recycling_steps=1, sampling_steps=None) -> LossTerm:
        """构建模型特异性的损失函数"""
        pass
```

### 4.2 Boltz2 模型实现

**位置**: [`../src/mosaic/models/boltz2.py`](../src/mosaic/models/boltz2.py)

```python
class Boltz2(eqx.Module, StructurePredictionModel):
    def build_loss(self, *, loss, features, recycling_steps=1, sampling_steps=None):
        return Boltz2Loss(
            joltz2=self.model,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps - 1,  # Boltz2 的索引问题
            sampling_steps=sampling_steps or 25,
            deterministic=True
        )
```

## 5. 使用流程分析

基于 [`examples/boltz_binder_design.py`](examples/boltz_binder_design.py) 的实际使用：

### 5.1 损失函数构建 (第147-164行)

```python
# 1. 定义基本损失项
loss_terms = [
    2.0 * sp.BinderTargetContact(),    # 结合物-目标接触
    sp.WithinBinderContact()           # 结合物内部接触
]

# 2. 可选添加 ProteinMPNN 损失
if use_mpnn:
    loss_terms.append(5.0 * InverseFoldingSequenceRecovery(mpnn, temp=0.01))

# 3. 组合所有损失项
combined_loss = sum(loss_terms[1:], loss_terms[0])

# 4. 构建模型相关的损失函数
loss = model.build_loss(loss=combined_loss, features=features)
```

### 5.2 优化过程 (第166-207行)

```python
# 第一阶段：初始优化
_, PSSM = simplex_APGM(
    loss_function=loss,
    n_steps=n_optimization_steps,
    x=initial_x,                    # 随机初始化的软序列
    stepsize=stepsize,
    momentum=momentum
)

# 第二阶段：序列锐化
PSSM_sharper, _ = simplex_APGM(
    loss_function=loss,
    n_steps=n_sharpening_steps,
    x=PSSM,                         # 使用第一阶段的结果
    stepsize=sharpening_stepsize,
    scale=sharpening_scale,         # >1.0 促进离散化
    momentum=momentum
)
```

## 6. 关键设计特点

### 6.1 模块化设计
- **损失函数独立**：每个损失函数都是独立的模块
- **模型解耦**：损失函数不依赖特定的结构预测模型
- **组合灵活**：支持任意线性组合损失函数

### 6.2 JAX 兼容性
- **JIT 编译**：所有函数支持 JAX JIT 加速
- **自动微分**：使用 JAX 的 `value_and_grad` 计算梯度
- **设备兼容**：支持 CPU/GPU/TPU

### 6.3 概率序列优化
- **软序列表示**：序列用概率分布 `(L, 20)` 表示
- **单纯形约束**：每个位置的20个氨基酸概率和为1
- **渐进离散化**：通过调整 `scale` 参数从软序列到硬序列

## 7. 扩展指南

### 7.1 添加新的损失函数

1. 继承 `LossTerm` 基类
2. 实现 `__call__` 方法
3. 返回 `(loss_value, aux_info)` 元组

### 7.2 集成新的结构预测模型

1. 继承 `StructurePredictionModel`
2. 实现抽象方法（`build_loss`, `predict` 等）
3. 创建对应的 `Output` 和 `Loss` 类

### 7.3 自定义优化算法

参考 `simplex_APGM` 的实现模式，关键是：
- 处理概率单纯形约束
- 集成损失函数接口
- 支持梯度裁剪和正则化

## 8. 性能优化建议

### 8.1 JIT 编译
- 损失函数计算会被 JIT 编译，首次运行较慢
- 避免在损失函数中使用 Python 控制流

### 8.2 内存管理
- 大型模型需要注意 GPU 内存使用
- 考虑使用梯度累积处理大批量

### 8.3 超参数调优
- `stepsize`: 建议从 `0.1 * sqrt(binder_length)` 开始
- `scale`: 从 1.0 开始，逐步增加到 1.5-2.0
- `momentum`: 通常使用 0.0-0.9
