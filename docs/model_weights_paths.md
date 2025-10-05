# 模型权重下载路径说明

本文档详细说明了 MOSAIC 项目中各个模型权重的默认下载路径、配置方式和相关代码位置。

---

## 📦 模型权重路径总览

| 模型 | 默认路径 | 代码位置 | 自动下载 |
|------|---------|---------|---------|
| **Boltz-2** | `~/.boltz/boltz2_conf.ckpt` | `src/mosaic/losses/boltz2.py:24` | ✅ 是 |
| **Boltz-1** | `~/.boltz/boltz1_conf.ckpt` | `src/mosaic/losses/boltz.py:33` | ✅ 是 |
| **AlphaFold2** | `<data_dir>/params/` (默认 `./params/`) | `src/mosaic/af2/alphafold2.py:61` | ✅ 是 |
| **Protenix** | `~/.protenix/` | `src/mosaic/losses/protenix.py:42` | ✅ 是 |
| **ProteinMPNN** | 内置权重 | `src/mosaic/proteinmpnn/weights/` | ⚠️ 已包含 |

---

## 1. Boltz-2 模型

### 默认路径
```
~/.boltz/boltz2_conf.ckpt
```

### 代码实现
**文件**: `src/mosaic/losses/boltz2.py`

```python
def load_boltz2(checkpoint_path=Path("~/.boltz/boltz2_conf.ckpt").expanduser()):
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)
    
    torch_model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        ...
    )
    return joltz.from_torch(torch_model)
```

### 使用方式

#### 默认路径（自动下载）
```python
from mosaic.models.boltz2 import Boltz2

# 使用默认路径 ~/.boltz/boltz2_conf.ckpt
model = Boltz2()
```

#### 自定义路径
```python
from pathlib import Path
from mosaic.models.boltz2 import Boltz2

# 指定自定义路径
custom_path = Path("/your/custom/path/boltz2_conf.ckpt")
model = Boltz2(cache_path=custom_path)
```

### 下载说明
- 首次使用时，如果权重文件不存在，会自动调用 `boltz_main.download_boltz2()` 下载
- 下载到 `~/.boltz/` 目录
- 文件大小约 **2-3 GB**

---

## 2. Boltz-1 模型

### 默认路径
```
~/.boltz/boltz1_conf.ckpt
```

### 代码实现
**文件**: `src/mosaic/losses/boltz.py`

```python
def load_boltz(
    checkpoint_path: Path = Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
):
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        download(cache)  # 调用 boltz.main.download_boltz1
    
    _torch_model = Boltz1.load_from_checkpoint(
        checkpoint_path,
        ...
    )
    return joltz.from_torch(_torch_model)
```

### 使用方式
```python
from mosaic.models.boltz1 import Boltz1

# 使用默认路径
model = Boltz1()

# 自定义路径
model = Boltz1(checkpoint_path=Path("/your/custom/path"))
```

---

## 3. AlphaFold2 模型

### 默认路径
```
<data_dir>/params/
```
其中 `data_dir` 默认为当前工作目录 `"."`，即：
```
./params/
```

### 目录结构
```
./params/
├── params_model_1_multimer_v3.npz
├── params_model_2_multimer_v3.npz
├── params_model_3_multimer_v3.npz
├── params_model_4_multimer_v3.npz
└── params_model_5_multimer_v3.npz
```

### 代码实现
**文件**: `src/mosaic/af2/alphafold2.py`

```python
class AF2:
    def __init__(self, data_dir="."):
        model_name = "model_1_multimer_v3"
        assert "multimer" in model_name, f"{model_name} is not a multimer model"

        # 检查参数文件是否存在
        if not (Path(data_dir)/"params").exists():
            print(f"Could not find AF2 parameters in {data_dir}/params.")
            print(f"Running `download_params.sh .`")
            from subprocess import run
            run(["bash", "download_params.sh", data_dir], check=True)

        # 加载 5 个 AF2 multimer v3 模型
        model_params = [
            data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
            for model_name in tqdm(
                [f"model_{i}_multimer_v3" for i in range(1, 6)],
                desc="Loading AF2 params",
            )
        ]
```

### 使用方式

#### 默认路径（当前目录）
```python
from mosaic.models.af2 import AlphaFold2

# 使用默认路径 ./params/
model = AlphaFold2()
```

#### 自定义路径
```python
from mosaic.models.af2 import AlphaFold2

# 指定自定义数据目录
model = AlphaFold2(data_dir="/path/to/alphafold/data")
# 将会查找 /path/to/alphafold/data/params/
```

### 下载说明

#### 自动下载（推荐）
首次使用时，如果 `./params/` 目录不存在，会自动运行 `download_params.sh` 脚本。

#### 手动下载
```bash
# 下载到当前目录
bash download_params.sh .

# 下载到指定目录
bash download_params.sh /path/to/download
```

**下载脚本说明** (`download_params.sh`):
- 使用 `aria2c` 下载工具（需要先安装：`sudo apt install aria2`）
- 下载源：`https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`
- 文件大小约 **3.5 GB**（压缩包），解压后约 **4 GB**

---

## 4. Protenix 模型

### 默认路径
```
~/.protenix/
```

### 环境变量
```python
os.environ["PROTENIX_DATA_ROOT_DIR"] = str(Path("~/.protenix").expanduser())
```

### 代码实现
**文件**: `src/mosaic/losses/protenix.py`

```python
def _load_model(name="protenix_mini_default_v0.5.0", cache_path=Path("~/.protenix")):
    cache_path = cache_path.expanduser()
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=f"--model_name {name}",
        fill_required_with_null=True,
    )
    configs.update({"load_checkpoint_dir": str(cache_path)})
    configs["data"]["pdb_cluster_file"] = str(cache_path / "clusters-by-entity-40.txt")
    
    # 自动下载
    protenix.inference.download_infercence_cache(configs)
    
    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint = torch.load(checkpoint_path)
    ...
```

### 使用方式
```python
from mosaic.losses.protenix import load_protenix_mini

# 使用默认路径 ~/.protenix/
model = load_protenix_mini()

# 自定义路径
model = load_protenix_mini(cache_path=Path("/your/custom/path"))
```

### 下载文件
- 模型权重：`protenix_mini_default_v0.5.0.pt`
- 聚类文件：`clusters-by-entity-40.txt`

---

## 5. ProteinMPNN 模型

### 默认路径（内置）
```
src/mosaic/proteinmpnn/weights/
├── v_48_020.pt
└── soluble_v_48_020.pt
```

### 代码实现
**文件**: `src/mosaic/proteinmpnn/mpnn.py`

```python
class ProteinMPNN:
    @classmethod
    def from_pretrained(cls, model_name="v_48_020", soluble=False):
        """从预训练权重加载 ProteinMPNN 模型"""
        # 权重已经包含在项目中
        weights_dir = Path(__file__).parent / "weights"
        if soluble:
            checkpoint_path = weights_dir / "soluble_v_48_020.pt"
        else:
            checkpoint_path = weights_dir / f"{model_name}.pt"
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ...
```

### 使用方式
```python
from mosaic.proteinmpnn.mpnn import ProteinMPNN

# 使用默认权重（v_48_020）
mpnn = ProteinMPNN.from_pretrained()

# 使用可溶性版本
mpnn = ProteinMPNN.from_pretrained(soluble=True)
```

### 说明
- ProteinMPNN 的权重文件已经包含在项目仓库中
- 无需额外下载
- 文件大小约 **100-200 MB**

---

## 📋 快速参考表

### 路径配置优先级

| 模型 | 初始化参数 | 默认值 | 说明 |
|------|-----------|--------|------|
| Boltz-2 | `cache_path` | `None` (使用 `~/.boltz/boltz2_conf.ckpt`) | 可选 |
| Boltz-1 | `checkpoint_path` | `~/.boltz/boltz1_conf.ckpt` | 可选 |
| AlphaFold2 | `data_dir` | `"."` (当前目录) | 必须包含 `params/` 子目录 |
| Protenix | `cache_path` | `~/.protenix` | 可选 |
| ProteinMPNN | - | 内置权重 | 不可配置 |

### 存储空间需求

| 模型 | 文件大小 | 备注 |
|------|---------|------|
| Boltz-2 | ~2-3 GB | 单个检查点文件 |
| Boltz-1 | ~2-3 GB | 单个检查点文件 |
| AlphaFold2 | ~4 GB | 5 个模型权重文件 |
| Protenix | ~1-2 GB | 包含模型和聚类文件 |
| ProteinMPNN | ~200 MB | 已包含在仓库中 |
| **总计** | **~12-14 GB** | 如果使用所有模型 |

---

## 🔧 常见问题

### Q1: 如何更改所有模型的默认下载位置？

**A**: 不同模型有不同的配置方式：

```python
from pathlib import Path

# Boltz-2
model_boltz2 = Boltz2(cache_path=Path("/data/models/boltz2_conf.ckpt"))

# AlphaFold2
model_af2 = AlphaFold2(data_dir="/data/models/alphafold")

# Protenix
model_protenix = load_protenix_mini(cache_path=Path("/data/models/protenix"))
```

### Q2: 如何检查模型权重是否已下载？

**A**: 使用以下脚本检查：

```python
from pathlib import Path

def check_model_weights():
    """检查所有模型权重是否存在"""
    
    checks = {
        "Boltz-2": Path("~/.boltz/boltz2_conf.ckpt").expanduser(),
        "Boltz-1": Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
        "AlphaFold2": Path("./params/params_model_1_multimer_v3.npz"),
        "Protenix": Path("~/.protenix/protenix_mini_default_v0.5.0.pt").expanduser(),
        "ProteinMPNN": Path("src/mosaic/proteinmpnn/weights/v_48_020.pt"),
    }
    
    for model_name, path in checks.items():
        status = "✅ 存在" if path.exists() else "❌ 不存在"
        print(f"{model_name:15} {status:10} {path}")

check_model_weights()
```

### Q3: 下载失败怎么办？

**A**: 常见解决方案：

1. **检查网络连接**：确保可以访问 Google Storage 和 GitHub
2. **手动下载**：
   - Boltz: 访问 [Boltz GitHub](https://github.com/jwohlwend/boltz)
   - AlphaFold2: 使用 `download_params.sh` 脚本
3. **使用代理**：配置 HTTP/HTTPS 代理
4. **从其他机器复制**：如果有已下载的权重，可以直接复制

### Q4: 可以共享权重文件吗？

**A**: 可以！权重文件可以在多个项目或用户之间共享：

```bash
# 创建符号链接共享权重
ln -s /shared/models/boltz ~/.boltz
ln -s /shared/models/alphafold/params ./params
```

### Q5: 如何清理不需要的模型权重？

**A**: 直接删除对应目录：

```bash
# 删除 Boltz 权重
rm -rf ~/.boltz

# 删除 AlphaFold2 权重
rm -rf ./params

# 删除 Protenix 权重
rm -rf ~/.protenix
```

---

## 📚 相关文档

- [Boltz 官方文档](https://github.com/jwohlwend/boltz)
- [AlphaFold2 参数下载](https://github.com/deepmind/alphafold#genetic-databases)
- [ProteinMPNN 论文](https://www.science.org/doi/10.1126/science.add2187)

---

## 📝 更新日志

- **2025-10-05**: 创建文档，记录所有模型权重路径
- 如有更新，请维护此文档

---

**维护者**: AI Assistant  
**最后更新**: 2025-10-05
