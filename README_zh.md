# Mosaic：功能化多目标蛋白质设计框架

基于连续松弛的函数化、多目标蛋白质设计框架。

## 概述

蛋白质设计任务几乎总是涉及多个必须满足或优化的约束条件或属性。例如，在结合子（binder）设计中，可能需要同时确保：
- 与预定靶标的结合概率高
- 与类似脱靶蛋白的结合概率低
- 在细菌中表达良好
- 高度可溶

最近机器学习在蛋白质特性预测方面的应用激增，为这些特性产生了相当准确的预测器。目前缺乏的是一个将这些不同预测器高效灵活地结合成一个设计/筛选/排序框架的方法。

---

## 安装

我们推荐使用 `uv`，例如克隆仓库后运行 `uv sync --group jax-cuda` 来安装依赖项。

要运行示例 notebook，可以尝试 `source .venv/bin/activate`，然后运行 `marimo edit examples/example_notebook.py`。

> 您可能需要为特定软件包和机器添加各种 `uv` 覆盖，请查看 [pyproject.toml](pyproject.toml)

> 您需要GPU或TPU兼容版本的JAX进行结构预测。您可能需要手动安装，即 `uv add jax[cuda12]`。

要自动下载AF2权重，您需要安装 `aria2`：`apt-get install aria2`。

## 介绍

该项目结合了两个简单组件来构建强大的蛋白质设计框架：

- 基于梯度的连续松弛序列空间优化（如 [ColabDesign](https://github.com/sokrypton/ColabDesign)、RSO、BindCraft等）
- 函数化的模块接口，易于组合多个学习或手工设计的损失项和优化算法（如 [A high-level programming language for generative protein design](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf) 等）

关键观察是，我们可以同时将此连续松弛用于多个学习目标项[^1]。

这使我们能够轻松构建结合多个学习势能的客观函数，并高效地优化它们：

```python
combined_loss = (
    Boltz1Loss(
        model=model,
        name="ART2B",
        loss=4 * sp.BinderTargetContact()
        + sp.RadiusOfGyration(target_radius=15.0)
        + sp.WithinBinderContact()
        + 0.3 * sp.HelixLoss()
        + ProteinMPNNLoss(mpnn, num_samples = 8),
        features=boltz_features,
        recycling_steps=0,
    )
    + 0.5 * esm_loss
    + trigram_ll
    + 0.1 * StabilityModel.from_pretrained(esm)
    + 0.5
    * Boltz1Loss(
        model=model,
        name="mono",
        loss=0.2 * sp.PLDDTLoss()
        + sp.RadiusOfGyration(target_radius=15.0)
        + 0.3 * sp.HelixLoss(),
        features=monomer_features,
        recycling_steps=0,
    )
)

_, logits_combined_objective = simplex_APGM(
    loss_function=combined_loss,
    n_steps=150,
    x=np.random.randn(binder_length, 20) * 0.1,
    stepsize=0.1,
)

```

这里我们使用约5个不同的模型来构建损失函数：[Boltz-1](https://github.com/jwohlwend/boltz) 结构预测模型（被使用*两次*：一次预测结合子-靶标复合物，一次预测结合子单体）、ESM2、ProteinMPNN、n-gram模型和在 [mega-scale](https://www.nature.com/articles/s41586-023-06328-6) 数据集上训练的稳定度模型。

定义额外的损失项非常简单，它们是JIT兼容的可调用pytrees：

```python
class LogPCysteine(LossTerm):
    def __call__(self, soft_sequence: Float[Array, "N 20"], key = None):
        mean_log_p = jnp.log(soft_sequence[:, IDX_CYS] + 1E-8).mean()
        return mean_log_p, {"log_p_cys": mean_log_p}
```

自定义损失项没有理由不能涉及更昂贵的（可微分的）操作，例如运行ProteinX或 [EVOLVEpro风格的适应度预测器](https://www.science.org/doi/10.1126/science.adr6006)。

[marimo notebook](examples/example_notebook.py) 提供了一些展示如何工作的示例。

> **警告**：ColabDesign、BindCraft等是针对非常特定问题经过充分测试和调优的方法。`mosaic` 可能需要大量的手动调优才能工作（调节学习率等），经常产生无法通过简单计算机内测试的蛋白质，必须与标准筛选方法结合使用等。这不适合胆小的人：目标是提供一个框架来为您的应用实现自定义目标函数和优化算法。

切换不同的优化器非常简单。例如，假设我们真的想在超立方体$[0,1]^N$上尝试投影梯度下降。我们可以在几行代码中实现：

```python
def RSO_box(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-1)),
    key=None,
):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

    opt_state = optim.init(x)
    
    for _iter in range(n_steps):
        (v, aux再调整), g = _eval_loss_and_grad(
            x=x,
            loss_function=loss_function,
            key=key
        )
        updates, opt_state = optim.update(g, opt_state, x)
        x = optax.apply_updates(x, updates).clip(0,1)
        key = jax.random.fold_in(key, 0)
        _print_iter(_iter, aux, v)

    return x
```

查看 [optimizers.py](src/mosaic/optimizers.py) 了解不同优化器的一些示例。

## 模型和损失函数

| 包含的模型 |
| --- |
| Boltz-1 |
| Boltz-2 |
| AlphaFold2 |
| [Protenix (mini+tiny)](#protenix) |
| [ProteinMPNN](#proteinmpnn) |
| [ESM](#esm) |
| [稳定性](#stability) |
| [AbLang](#ablang) |
| [三元组](#trigram) |

---

### 结构预测

我们在 `mosaic.structure_prediction` 和 `mosaic.models.*` 中为五个结构预测模型提供了简单的接口：`Boltz1`、`Boltz2`、`AF2`、`ProtenixMini` 和 `ProtenixTiny`。

要进行预测或设计结合子，你需要创建一个 `mosaic.structure_prediction.TargetChain` 对象列表。这是一个简单的数据类，包含蛋白质（或DNA或RNA）序列、告诉模型是否应该使用MSA的标志（`use_msa`），以及可能还有模板结构。

例如，我们可以为IL7Ra使用Protenix进行预测：

```python
import jax
from mosaic.structure_prediction import TargetChain
from mosaic.models.protenix import ProtenixMini

model = ProtenixMini()

target_sequence = "DYSFSCYSQLEVNGSQHSLTCAFEDPDVNTTNLEFEICGALVEVKCLNFRKLQEIYFIETKKFLLIGKSNICVKVGEKSLTCKKIDLTTIVKPEAPFDLSVVYREGANDFVVTFNTSHLQKKYVKVLMHDVAYRQEKDENKWTHVNLSSTKLTLLQRKLQPAAMYEIKVRSIPDHYFKGFWSEWSPSYYFRT"

# 生成特征和将模型输出转换为预测包装器的"writer"对象
target_only_features, target_only_structure = model.target_only_features(
    [TargetChain(target_sequence)]
)

prediction = model.predict(
    features=target_only_features,
    writer=target_only_structure,
    key=jax.random.key(0),
    recycling_steps=10,
)

# prediction包含有用的属性，如 prediction.st、prediction.pae 等
```

这个接口对所有结构预测模型都是相同的，所以在理论上我们应该能够只通过改变一行代码将上面的 `ProtenixMini` 替换为 `Boltz2`！

我们还需要定义一个（模型无关的！）结构预测相关损失函数的集合 [src/mosaic/losses/structure_prediction.py](src/mosaic/losses/structure_prediction.py)。使用提供的接口定义自己的损失函数非常简单。

> 在内部，我们区分三类损失函数：仅依赖trunk的、结构模块的或置信度模块的。出于计算效率，我们仅在需要时才运行结构模块或置信度模块！

继续上面的示例，我们可以构建损失函数并进行设计：

```python
import mosaic.losses.structure_prediction as sp

binder_length = 80

design_features, design_structure = protenix.binder_features(
    binder_length = binder_length, chains = [TargetChain(target_sequence)]
)

loss = protenix.build_loss(
    loss=sp.BinderTargetContact() + sp.WithinBinderContact(), features=design_features, recycling_steps = 3
)

PSSM = jax.nn.softmax(
    0.5
    * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )
)

PSSM,_ = simplex_APGM(
    loss_function=loss,
    x=PSSM,
    n_steps=50,
    stepsize=0.15,
    momentum=0.3,
)
```

> 每个结构预测模型也支持低级损失/接口，如果你想做一些高级的事情（例如，用Boltz或Protenix设计对抗小分子的蛋白质结合子）。

### Protenix

查看 [protenij.py](examples/protenij.py) 了解如何使用这个模型族族的示例。该损失函数支持一些高级功能来加速幻觉，即"预循环"（在设计*前*单独在靶标上运行多个循环迭代）和"协循环"（并行运行循环和优化步骤），但也可以类似地与Boltz或AF2一起使用。

### ProteinMPNN

使用首选ProteinMPNN（可溶或香草）模型：

```python
from mosaic.proteinmpnn.mpnn import ProteinMPNN

mpnn = ProteinMPNN.from_pretrained()
```

在最简单的情况下，我们有一个单链结构或复合物，其中我们正在设计的蛋白质作为第一条链出现（注意这可以是预测）。然后我们可以将设计的序列在ProteinMPNN下的（负）对数似然构建为损失项：

```python
inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure( gemmi.read_structure("scaffold.pdb"), mpnn)
```

然后可以将其添加到您正在构建的任何总损失函数中。

注意，使用例如 `ClippedLoss(inverse_folding_LL, 2, 100)` 裁剪损失通常很有帮助：过度优化ProteinMPNN似然通常导致同聚物。

### ProteinMPNN + 结构预测

ProteinMPNN也可以与实时结构预测结合。数学上这是：

$-\log P_\theta(s | AF2(s))$

序列 $s$ 在*该序列的预测结构*的逆折叠下的对数似然。这个损失项是 `ProteinMPNNLoss`。

另一个非常有用的损失项是 `InverseFoldingSequenceRecovery`：在使用ProteinMPNN采样后的序列恢复的连续松弛（大约 $\langle s, -E_{z \sim p_\theta(\cdot | AF2(s))} [z] \rangle$）。我们发现这个项通常加速设计并提高筛选通过率。

### ESM

另一个有用的损失项是ESM2蛋白质语言模型的伪似然（通过 [esm2quinox](https://github.com/patrick-kidger/esm2quinox/tree/main)）；它与各种有用的特性相关（可溶性、可表达性等）。

该项可以如下构造：

```python
import esm
import esm2quinox
torch_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
ESM2PLL = ESM2PseudoLikelihood(esm2quinox.from_torch(torch_model))
```

在典型实践中，这个损失应该被裁剪或压扁以避免过度优化（例如 `ClippedLoss(ESM2PLL, 2, 100)`）。

我们还实现了ESMC的相应损失（通过 [esmj](https://github.com/escalante-bio/esmj)）。

```python
from esmj import from_torch
from esm.models.esmc import ESMC as TORCH_ESMC

esm = from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))
ESMCPLL = ESMCPseudoLikelihood(esm)
```

### 稳定性

一个在megascale数据集上训练的简单delta G预测器。这可能是如何训练并在少量数据上添加简单回归头的很好的示例：[train.py](src/mosaic/stability_model/train.py)。

```python
stability_loss = StabilityModel.from_pretrained(esm)
```

### AbLang

[AbLang](https://github.com/oxpig/AbLang)，一个抗体特异性语言模型族。

```python
import ablang
import jablang

heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()

abpll = AbLangPseudoLikelihood(
    model=jablang.from_torch(heavy_ablang.AbLang),
    tokenizer=heavy_ablang.tokenizer,
    stop_grad=True,
)
```

### 三元组

[生成性蛋白质设计的高级编程语言](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf) 中的三元组语言模型。

```python
trigram_ll = TrigramLL.from_pkl()
```

## 优化器和损失变换

我们在 (src/mosaic/optimizers.py) 中包含一些标准优化器。

首先，`simplex_APGM`，这是概率单纯形上的加速近端梯度算法。一个关键超参数是步长，一个合理的初步猜测是 `0.1*np.sqrt(binder_length)`。另一个有用的关键字参数是 `scale`，这对应于 $\ell_2$ 正则化。大于 `1.0` 的值鼓励稀疏解；典型的结合子设计运行可能从 `scale=1.0` 开始以获得初始的软解，然后增加到更高的值以获得离散解。

`simplex_APGM` 也接受关键字参数 `logspace`，如果这设置为真，我们在对数空间中运行算法，这对应于加速近端Bregman方法。在这种情况下 `scale` 对应于熵正则化。

我们还包含一个离散优化算法 `gradient_MCMC`，这是一个MCMC变体，使用目标函数的泰勒近似定义提议分布（见 [Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC](https://arxiv.org/abs/2212.09925)。）该算法对于微调现有设计或连续优化的结果特别有用。

### 损失变换

我们还提供了一些 [损失函数的常见变换](src/mosaic/losses/transformations.py)。值得注意的是 `ClippedLoss`，它包装并裁剪另一个损失项。

`SetPositions` 和 `FixedPositionsPenalty` 对于修复现有设计的某些位置很有用。

`ClippedGradient` 和 `NormedGradient` 分别裁剪和归一化单个损失项的梯度，这在组合具有非常不同梯度范数的预测器时很有用，例如：

```python
loss = ClippedGradient(inverse_folding_LL, 1.0)  
    + ClippedGradient(ablang_pll, 1.0)
    + 0.25 * ClippedGradient(ESMCPLL, 1.0)
```

## 广泛的理论讨论

基于幻觉的蛋白质设计工作流试图解决以下优化问题：

$$\underset{s \in A^n}{\textrm{minimize}}~\ell(s)$$

这里 $A$ 是氨基酸的集合，所以决策变量 $s$ 在长度为 $n$ 的所有蛋白质序列上变化。 $\ell: A^n \rightarrow \mathbf{R}$ 是评估蛋白质 $s$ 质量的损失函数。在典型实践中，$\ell$ 是神经网络输出的某个函数；即，在 [ColabDesign](https://github.com/sokrypton/ColabDesign) 中 $\ell$ 可能是来自AlphaFold的（负）平均pLDDT。

朴素方法的一个挑战是 $A^n$ 极其庞大并且离散优化很困难；虽然已经使用了MCMC和其他离散算法（见例如 [Rives et al](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf)），但它们通常很慢。

ColabDesign、RSO和BindCraft等利用了 $\ell$ 具有特殊结构的事实，允许原始问题的连续松弛：几乎每个神经网络首先将序列 $s$ 编码成独热矩阵 $P \in \mathbf{R}^{(n, c)}$。如果我们考虑 $\ell$ 作为 $\mathbf{R}^{(n, c)}$ 上的函数，我们可以使用自动微分在 $\mathbf{R}^{(n, c)}$ 或 $\Delta_c^n$（$n$ 个概率单纯形的乘积）上进行连续优化。

> 这涉及到优化分布而不是单个点的经典优化技巧。首先，$\underset{x}{\textrm{minimize }}f(x)$ 松弛为 $\underset{p \in \Delta}{\textrm{minimize }}E_p f(x)$。接下来，如果取 $x$ 的期望是有意义的（如在独热序列情况下），我们可以将 $f$ 和 $E$ 互换得到最终松弛：$\underset{p \in \Delta}{\textrm{minimize }} f( E_p x) = \underset{p \in \Delta}{\textrm{minimize }} f(p)$

这个松弛优化问题的解然后必须转换为序列；这里许多不同的方法都有效：RSO使用预测结构的逆折叠，BindCraft/ColabDesign使用带有递增温度的softmax来鼓励独热解，等等。

默认情况下，我们使用一种广义近端梯度方法（带熵正则化的mirror descent）来在单纯形上进行优化并鼓励稀疏解，尽管很容易切换其他优化算法（例如投影梯度下降或与ColabDesign中的softmax组合）。

通常 $\ell$ 由单个神经网络（或同一架构的集成）形成，但在实践中我们有兴趣同时优化不同神经网络预测的不同特性。这具有减少找到所谓的对抗序列的机会的额外好处。

这种损失项的模块化实现也适用于现代基于RL的生成模型对齐方法：这些对齐形式通常可以被视为*摊销优化*。通常，它们训练生成模型以最小化KL散度减去损失函数的组合，该损失函数可以是计算机内预测器的组合。另一个用例是为离散扩散或流模型提供指导。

[^1]: 这要求我们将神经网络视为*简单的参数函数*，可以编程组合；**不是**作为需要大型库（例如PyTorch lightning）、bash脚本或容器的复杂软件包，这在BioML中是常见做法。
