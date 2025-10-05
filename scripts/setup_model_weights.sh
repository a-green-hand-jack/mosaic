#!/bin/bash
#
# 模型权重下载和软链接设置脚本
#
# 功能：
# 1. 将所有模型权重下载到统一的缓存目录 /ibex/user/wuj0c/cache/weights/
# 2. 在默认路径创建软链接指向缓存目录
#
# 使用方法：
#   bash scripts/setup_model_weights.sh
#
# 作者: AI Assistant
# 日期: 2025-10-05

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置路径
CACHE_ROOT="/ibex/user/wuj0c/cache/weights"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 模型路径配置
BOLTZ2_CACHE="${CACHE_ROOT}/boltz2"
BOLTZ1_CACHE="${CACHE_ROOT}/boltz1"
AF2_CACHE="${CACHE_ROOT}/alphafold2"
PROTENIX_CACHE="${CACHE_ROOT}/protenix"

# 默认路径配置
HOME_DIR="${HOME}"
BOLTZ2_DEFAULT="${HOME_DIR}/.boltz"
AF2_DEFAULT="${PROJECT_ROOT}/params"
PROTENIX_DEFAULT="${HOME_DIR}/.protenix"

# 使用当前的python 环境
python=/ibex/user/wuj0c/Projects/AwesomeProjects/Protein/mosaic/.venv/bin/python
echo "使用python: ${python}"

echo "========================================================================"
echo "              模型权重下载和软链接设置脚本"
echo "========================================================================"
echo ""
log_info "项目根目录: ${PROJECT_ROOT}"
log_info "缓存根目录: ${CACHE_ROOT}"
echo ""

# 创建缓存目录结构
log_info "创建缓存目录结构..."
mkdir -p "${BOLTZ2_CACHE}"
mkdir -p "${BOLTZ1_CACHE}"
mkdir -p "${AF2_CACHE}"
mkdir -p "${PROTENIX_CACHE}"
log_success "缓存目录创建完成"
echo ""

#==============================================================================
# 1. Boltz-2 模型下载
#==============================================================================
echo "========================================================================"
echo "1. 设置 Boltz-2 模型"
echo "========================================================================"

if [ -f "${BOLTZ2_CACHE}/boltz2_conf.ckpt" ]; then
    log_warning "Boltz-2 权重已存在: ${BOLTZ2_CACHE}/boltz2_conf.ckpt"
else
    log_info "开始下载 Boltz-2 模型权重..."
    log_info "这可能需要几分钟时间（约 2-3 GB）..."
    
    # 使用 Python 调用 boltz 的下载函数
    ${python} << EOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}/src")

from pathlib import Path
try:
    from boltz.main import download_boltz2
    cache_path = Path("${BOLTZ2_CACHE}")
    print(f"下载 Boltz-2 到: {cache_path}")
    download_boltz2(cache_path)
    print("Boltz-2 下载完成！")
except Exception as e:
    print(f"下载失败: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Boltz-2 下载完成"
    else
        log_error "Boltz-2 下载失败"
        exit 1
    fi
fi

# 创建软链接
log_info "创建 Boltz-2 软链接..."
if [ -L "${BOLTZ2_DEFAULT}" ] || [ -e "${BOLTZ2_DEFAULT}" ]; then
    log_warning "目标路径已存在: ${BOLTZ2_DEFAULT}"
    read -p "是否删除并重新创建软链接? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${BOLTZ2_DEFAULT}"
        ln -s "${BOLTZ2_CACHE}" "${BOLTZ2_DEFAULT}"
        log_success "Boltz-2 软链接创建成功: ${BOLTZ2_DEFAULT} -> ${BOLTZ2_CACHE}"
    else
        log_warning "跳过 Boltz-2 软链接创建"
    fi
else
    ln -s "${BOLTZ2_CACHE}" "${BOLTZ2_DEFAULT}"
    log_success "Boltz-2 软链接创建成功: ${BOLTZ2_DEFAULT} -> ${BOLTZ2_CACHE}"
fi
echo ""

#==============================================================================
# 2. Boltz-1 模型下载
#==============================================================================
echo "========================================================================"
echo "2. 设置 Boltz-1 模型"
echo "========================================================================"

if [ -f "${BOLTZ1_CACHE}/boltz1_conf.ckpt" ]; then
    log_warning "Boltz-1 权重已存在: ${BOLTZ1_CACHE}/boltz1_conf.ckpt"
else
    log_info "开始下载 Boltz-1 模型权重..."
    log_info "这可能需要几分钟时间（约 2-3 GB）..."
    
    ${python} << EOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}/src")

from pathlib import Path
try:
    from boltz.main import download_boltz1
    cache_path = Path("${BOLTZ1_CACHE}")
    print(f"下载 Boltz-1 到: {cache_path}")
    download_boltz1(cache_path)
    print("Boltz-1 下载完成！")
except Exception as e:
    print(f"下载失败: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Boltz-1 下载完成"
    else
        log_error "Boltz-1 下载失败"
        exit 1
    fi
fi

# Boltz-1 使用同一个 .boltz 目录，只需确保文件存在
if [ -f "${BOLTZ2_DEFAULT}/boltz1_conf.ckpt" ]; then
    log_success "Boltz-1 权重已可访问: ${BOLTZ2_DEFAULT}/boltz1_conf.ckpt"
else
    log_info "复制 Boltz-1 权重到 Boltz 缓存目录..."
    cp "${BOLTZ1_CACHE}/boltz1_conf.ckpt" "${BOLTZ2_CACHE}/"
    log_success "Boltz-1 权重复制完成"
fi
echo ""

#==============================================================================
# 3. AlphaFold2 模型下载
#==============================================================================
echo "========================================================================"
echo "3. 设置 AlphaFold2 模型"
echo "========================================================================"

if [ -d "${AF2_CACHE}/params" ] && [ -f "${AF2_CACHE}/params/params_model_1_multimer_v3.npz" ]; then
    log_warning "AlphaFold2 权重已存在: ${AF2_CACHE}/params/"
else
    log_info "开始下载 AlphaFold2 模型权重..."
    log_info "这可能需要较长时间（约 3.5 GB 压缩包）..."
    
    # 检查 aria2c 是否安装
    if ! command -v aria2c &> /dev/null; then
        log_error "aria2c 未安装，请先安装: sudo apt install aria2"
        log_info "或者使用 conda/mamba: conda install -c conda-forge aria2"
        exit 1
    fi
    
    # 运行下载脚本
    cd "${PROJECT_ROOT}"
    bash download_params.sh "${AF2_CACHE}"
    
    if [ $? -eq 0 ]; then
        log_success "AlphaFold2 下载完成"
    else
        log_error "AlphaFold2 下载失败"
        exit 1
    fi
fi

# 创建软链接
log_info "创建 AlphaFold2 软链接..."
if [ -L "${AF2_DEFAULT}" ] || [ -e "${AF2_DEFAULT}" ]; then
    log_warning "目标路径已存在: ${AF2_DEFAULT}"
    read -p "是否删除并重新创建软链接? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${AF2_DEFAULT}"
        ln -s "${AF2_CACHE}/params" "${AF2_DEFAULT}"
        log_success "AlphaFold2 软链接创建成功: ${AF2_DEFAULT} -> ${AF2_CACHE}/params"
    else
        log_warning "跳过 AlphaFold2 软链接创建"
    fi
else
    ln -s "${AF2_CACHE}/params" "${AF2_DEFAULT}"
    log_success "AlphaFold2 软链接创建成功: ${AF2_DEFAULT} -> ${AF2_CACHE}/params"
fi
echo ""

#==============================================================================
# 4. Protenix 模型下载
#==============================================================================
echo "========================================================================"
echo "4. 设置 Protenix 模型"
echo "========================================================================"

if [ -f "${PROTENIX_CACHE}/protenix_mini_default_v0.5.0.pt" ]; then
    log_warning "Protenix 权重已存在: ${PROTENIX_CACHE}/protenix_mini_default_v0.5.0.pt"
else
    log_info "开始下载 Protenix 模型权重..."
    log_info "这可能需要几分钟时间（约 1-2 GB）..."
    
    ${python} << EOF
import sys
import os
sys.path.insert(0, "${PROJECT_ROOT}/src")

from pathlib import Path
os.environ["PROTENIX_DATA_ROOT_DIR"] = "${PROTENIX_CACHE}"

try:
    from mosaic.losses.protenix import load_protenix_mini
    cache_path = Path("${PROTENIX_CACHE}")
    print(f"下载 Protenix 到: {cache_path}")
    model = load_protenix_mini(cache_path=cache_path)
    print("Protenix 下载完成！")
except Exception as e:
    print(f"下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Protenix 下载完成"
    else
        log_error "Protenix 下载失败"
        log_warning "Protenix 下载可能需要特定的依赖，可以稍后手动下载"
    fi
fi

# 创建软链接
log_info "创建 Protenix 软链接..."
if [ -L "${PROTENIX_DEFAULT}" ] || [ -e "${PROTENIX_DEFAULT}" ]; then
    log_warning "目标路径已存在: ${PROTENIX_DEFAULT}"
    read -p "是否删除并重新创建软链接? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${PROTENIX_DEFAULT}"
        ln -s "${PROTENIX_CACHE}" "${PROTENIX_DEFAULT}"
        log_success "Protenix 软链接创建成功: ${PROTENIX_DEFAULT} -> ${PROTENIX_CACHE}"
    else
        log_warning "跳过 Protenix 软链接创建"
    fi
else
    ln -s "${PROTENIX_CACHE}" "${PROTENIX_DEFAULT}"
    log_success "Protenix 软链接创建成功: ${PROTENIX_DEFAULT} -> ${PROTENIX_CACHE}"
fi
echo ""

#==============================================================================
# 5. ProteinMPNN（已包含在项目中，无需下载）
#==============================================================================
echo "========================================================================"
echo "5. 检查 ProteinMPNN 模型"
echo "========================================================================"

MPNN_WEIGHTS="${PROJECT_ROOT}/src/mosaic/proteinmpnn/weights"
if [ -f "${MPNN_WEIGHTS}/v_48_020.pt" ]; then
    log_success "ProteinMPNN 权重已存在（项目内置）: ${MPNN_WEIGHTS}"
else
    log_warning "ProteinMPNN 权重未找到，请检查项目完整性"
fi
echo ""

#==============================================================================
# 总结
#==============================================================================
echo "========================================================================"
echo "                        设置完成总结"
echo "========================================================================"
echo ""

log_info "检查所有模型权重..."
echo ""

check_file() {
    local name=$1
    local path=$2
    if [ -e "${path}" ]; then
        echo -e "${GREEN}✅${NC} ${name}: ${path}"
        return 0
    else
        echo -e "${RED}❌${NC} ${name}: ${path}"
        return 1
    fi
}

# 检查缓存目录
echo "缓存目录："
check_file "Boltz-2    " "${BOLTZ2_CACHE}/boltz2_conf.ckpt"
check_file "Boltz-1    " "${BOLTZ2_CACHE}/boltz1_conf.ckpt"
check_file "AlphaFold2 " "${AF2_CACHE}/params/params_model_1_multimer_v3.npz"
check_file "Protenix   " "${PROTENIX_CACHE}/protenix_mini_default_v0.5.0.pt"
check_file "ProteinMPNN" "${MPNN_WEIGHTS}/v_48_020.pt"
echo ""

# 检查软链接
echo "软链接："
check_file "Boltz      " "${BOLTZ2_DEFAULT}"
check_file "AlphaFold2 " "${AF2_DEFAULT}"
check_file "Protenix   " "${PROTENIX_DEFAULT}"
echo ""

# 显示磁盘使用情况
echo "磁盘使用情况："
du -sh "${CACHE_ROOT}" 2>/dev/null || log_warning "无法获取磁盘使用情况"
echo ""

log_success "所有设置完成！"
echo ""
echo "========================================================================"
echo "下一步："
echo "  1. 验证模型加载："
echo "     python3 -c 'from mosaic.models.boltz2 import Boltz2; Boltz2()'"
echo ""
echo "  2. 运行示例脚本："
echo "     python examples/boltz_binder_design.py --help"
echo ""
echo "  3. 查看文档："
echo "     cat docs/model_weights_paths.md"
echo "========================================================================"
