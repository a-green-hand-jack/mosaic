#!/usr/bin/env python3
"""
Boltz-2 蛋白质结合物设计脚本

使用 Boltz-2 模型进行蛋白质结合物的从头设计。
该脚本实现了基于梯度的序列优化，并可选地使用 ProteinMPNN 进行逆向折叠。

输入: 目标蛋白质序列
输出: ./outputs/boltz_binder_design/
     ├── figures/     - 可视化图表（PAE、pLDDT等）
     ├── structures/  - 预测的结构文件（PDB格式）
     └── logs/        - 执行日志

警告:
    1. 需要 GPU 或 TPU 来运行
    2. JAX 使用 JIT 编译，首次执行可能需要较长时间
    3. 可能需要多次运行优化方法才能获得合理的结合物
    4. 更改目标时可能需要调整超参数
    5. 对于蛋白质靶标的 minibinder 设计，建议使用 BindCraft

作者: AI Assistant
日期: 2025-10-05
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mosaic.optimizers import simplex_APGM
from mosaic.common import TOKENS, LossTerm
import mosaic.losses.structure_prediction as sp
from mosaic.models.boltz2 import Boltz2
from mosaic.models.af2 import AlphaFold2
from mosaic.structure_prediction import TargetChain


def setup_output_environment() -> dict:
    """设置输出目录结构"""
    script_name = Path(__file__).stem
    base_output_dir = Path(__file__).parent / "outputs" / script_name
    
    figures_dir = base_output_dir / "figures"
    structures_dir = base_output_dir / "structures"
    logs_dir = base_output_dir / "logs"
    
    for directory in [figures_dir, structures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / "design.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"📁 输出目录设置: {base_output_dir}")
    
    return {
        'base': base_output_dir,
        'figures': figures_dir,
        'structures': structures_dir,
        'logs': logs_dir
    }


OUTPUT_DIRS = setup_output_environment()


class GumbelPerturbation(LossTerm):
    """Gumbel 扰动损失项，用于逆向折叠时增加随机性"""
    key: jax.Array
    
    def __call__(self, sequence: jax.Array, key: jax.Array) -> Tuple[float, dict]:
        """
        计算 Gumbel 扰动损失
        
        Args:
            sequence: 序列的 one-hot 编码，shape: (L, 20)
            key: JAX 随机数生成器密钥
            
        Returns:
            损失值和辅助信息字典
        """
        v = (jax.random.gumbel(self.key, sequence.shape) * sequence).sum()
        return v, {"gumbel": v}


def design_binder_boltz2(
    target_sequence: str,
    binder_length: int,
    n_optimization_steps: int = 75,
    n_sharpening_steps: int = 50,
    stepsize: float = 0.1,
    sharpening_stepsize: float = 0.5,
    sharpening_scale: float = 1.5,
    momentum: float = 0.0,
    use_mpnn: bool = False,
    random_seed: Optional[int] = None
) -> dict:
    """
    使用 Boltz-2 模型设计蛋白质结合物
    
    Args:
        target_sequence: 目标蛋白质序列
        binder_length: 结合物长度
        n_optimization_steps: 初始优化步数
        n_sharpening_steps: 序列锐化步数
        stepsize: 初始优化步长
        sharpening_stepsize: 锐化步长
        sharpening_scale: 锐化缩放因子
        momentum: 动量参数
        use_mpnn: 是否使用 ProteinMPNN 损失项
        random_seed: 随机种子
        
    Returns:
        包含预测结果和序列的字典
    """
    if random_seed is None:
        random_seed = np.random.randint(100000)
    
    logging.info(f"🧬 开始设计结合物")
    logging.info(f"   目标序列长度: {len(target_sequence)}")
    logging.info(f"   结合物长度: {binder_length}")
    logging.info(f"   随机种子: {random_seed}")
    
    # 初始化模型
    logging.info("📦 加载 Boltz-2 模型...")
    model = Boltz2()
    
    # 准备特征
    logging.info("🔧 准备输入特征...")
    features, structure_writer = model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(target_sequence)]
    )
    
    # 构建损失函数
    logging.info("📊 构建损失函数...")
    loss_terms = [
        2.0 * sp.BinderTargetContact(),
        sp.WithinBinderContact()
    ]
    
    if use_mpnn:
        logging.info("   添加 ProteinMPNN 损失项...")
        from mosaic.proteinmpnn.mpnn import ProteinMPNN
        from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
        
        mpnn = ProteinMPNN.from_pretrained()
        loss_terms.append(5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.01)))
    
    loss = model.build_loss(
        loss=sum(loss_terms[1:], loss_terms[0]),
        features=features
    )
    
    # 第一阶段：初始优化
    logging.info(f"🎯 第一阶段优化 ({n_optimization_steps} 步)...")
    initial_x = jax.nn.softmax(
        0.5 * jax.random.gumbel(
            key=jax.random.key(random_seed),
            shape=(binder_length, 20)
        )
    )
    
    _, PSSM = simplex_APGM(
        loss_function=loss,
        n_steps=n_optimization_steps,
        x=initial_x,
        stepsize=stepsize,
        momentum=momentum
    )
    
    logging.info("   初始优化完成")
    
    # 保存初始 PSSM 可视化
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(PSSM, aspect='auto', cmap='viridis')
    ax.set_xlabel('Amino Acid Position')
    ax.set_ylabel('Residue Index')
    ax.set_title('Initial PSSM after Optimization')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] / 'pssm_initial.png', dpi=300)
    plt.close()
    logging.info(f"   保存初始 PSSM: {OUTPUT_DIRS['figures'] / 'pssm_initial.png'}")
    
    # 第二阶段：序列锐化
    logging.info(f"✨ 第二阶段锐化 ({n_sharpening_steps} 步)...")
    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=n_sharpening_steps,
        x=PSSM,
        stepsize=sharpening_stepsize,
        scale=sharpening_scale,
        momentum=momentum
    )
    
    logging.info("   序列锐化完成")
    
    # 保存锐化后的 PSSM 可视化
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(PSSM_sharper, aspect='auto', cmap='viridis')
    ax.set_xlabel('Amino Acid Position')
    ax.set_ylabel('Residue Index')
    ax.set_title('Sharpened PSSM')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] / 'pssm_sharpened.png', dpi=300)
    plt.close()
    logging.info(f"   保存锐化 PSSM: {OUTPUT_DIRS['figures'] / 'pssm_sharpened.png'}")
    
    # 预测软序列结构
    logging.info("🔮 预测软序列结构...")
    soft_pred = model.predict(
        PSSM=PSSM,
        features=features,
        writer=structure_writer,
        key=jax.random.key(random_seed)
    )
    
    logging.info(f"   软序列 ipTM: {soft_pred.iptm:.4f}")
    
    # 预测锐化序列结构
    logging.info("🔮 预测锐化序列结构...")
    pred = model.predict(
        PSSM=PSSM_sharper,
        features=features,
        writer=structure_writer,
        key=jax.random.key(random_seed + 1)
    )
    
    logging.info(f"   锐化序列 ipTM: {pred.iptm:.4f}")
    
    # 提取最终序列
    binder_seq = "".join(TOKENS[i] for i in PSSM_sharper.argmax(-1))
    logging.info(f"🧬 设计的结合物序列: {binder_seq}")
    
    # 保存结构
    structure_path = OUTPUT_DIRS['structures'] / 'designed_complex.pdb'
    with open(structure_path, 'w') as f:
        f.write(pred.st.make_pdb_string())
    logging.info(f"💾 保存结构文件: {structure_path}")
    
    # 保存 PAE 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(soft_pred.pae, cmap='viridis')
    ax1.set_title('Soft Sequence PAE')
    ax1.set_xlabel('Residue Index')
    ax1.set_ylabel('Residue Index')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(pred.pae, cmap='viridis')
    ax2.set_title('Sharp Sequence PAE')
    ax2.set_xlabel('Residue Index')
    ax2.set_ylabel('Residue Index')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] / 'pae_comparison.png', dpi=300)
    plt.close()
    logging.info(f"   保存 PAE 图: {OUTPUT_DIRS['figures'] / 'pae_comparison.png'}")
    
    # 保存 pLDDT 图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pred.plddt, label='Sharp Sequence', linewidth=2)
    ax.plot(soft_pred.plddt, label='Soft Sequence', linewidth=2, alpha=0.7)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('pLDDT')
    ax.set_title('Predicted LDDT Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] / 'plddt_comparison.png', dpi=300)
    plt.close()
    logging.info(f"   保存 pLDDT 图: {OUTPUT_DIRS['figures'] / 'plddt_comparison.png'}")
    
    return {
        'binder_sequence': binder_seq,
        'soft_prediction': soft_pred,
        'sharp_prediction': pred,
        'pssm': PSSM,
        'pssm_sharper': PSSM_sharper
    }


def design_multiple_binders(
    target_sequence: str,
    binder_length: int,
    n_designs: int = 10,
    **kwargs
) -> List[dict]:
    """
    设计多个结合物
    
    Args:
        target_sequence: 目标蛋白质序列
        binder_length: 结合物长度
        n_designs: 要设计的结合物数量
        **kwargs: 传递给 design_binder_boltz2 的其他参数
        
    Returns:
        设计结果列表
    """
    logging.info(f"🎨 开始批量设计 {n_designs} 个结合物...")
    
    designs = []
    for i in range(n_designs):
        logging.info(f"\n{'='*60}")
        logging.info(f"设计 {i+1}/{n_designs}")
        logging.info(f"{'='*60}")
        
        result = design_binder_boltz2(
            target_sequence=target_sequence,
            binder_length=binder_length,
            random_seed=np.random.randint(100000),
            **kwargs
        )
        
        designs.append(result)
        
        # 保存每个设计的结构
        structure_path = OUTPUT_DIRS['structures'] / f'design_{i+1:02d}.pdb'
        with open(structure_path, 'w') as f:
            f.write(result['sharp_prediction'].st.make_pdb_string())
        
        logging.info(f"✅ 设计 {i+1} 完成，ipTM: {result['sharp_prediction'].iptm:.4f}")
    
    logging.info(f"\n🎉 批量设计完成！共生成 {n_designs} 个结合物")
    
    # 保存所有设计的 ipTM 分布
    iptms = [d['sharp_prediction'].iptm for d in designs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_designs + 1), iptms)
    ax.set_xlabel('Design Number')
    ax.set_ylabel('ipTM Score')
    ax.set_title(f'ipTM Scores for {n_designs} Designs')
    ax.axhline(y=np.mean(iptms), color='r', linestyle='--', label=f'Mean: {np.mean(iptms):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] / 'iptm_distribution.png', dpi=300)
    plt.close()
    logging.info(f"   保存 ipTM 分布图: {OUTPUT_DIRS['figures'] / 'iptm_distribution.png'}")
    
    return designs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用 Boltz-2 模型设计蛋白质结合物",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--target-sequence',
        type=str,
        default="FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
        help="目标蛋白质序列"
    )
    
    parser.add_argument(
        '--binder-length',
        type=int,
        default=75,
        help="结合物长度"
    )
    
    parser.add_argument(
        '--n-designs',
        type=int,
        default=1,
        help="要设计的结合物数量（设置为 >1 进行批量设计）"
    )
    
    parser.add_argument(
        '--optimization-steps',
        type=int,
        default=75,
        help="初始优化步数"
    )
    
    parser.add_argument(
        '--sharpening-steps',
        type=int,
        default=50,
        help="序列锐化步数"
    )
    
    parser.add_argument(
        '--stepsize',
        type=float,
        default=0.1,
        help="初始优化步长"
    )
    
    parser.add_argument(
        '--use-mpnn',
        action='store_true',
        help="是否使用 ProteinMPNN 损失项（会更慢但可能产生更好的序列）"
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help="随机种子（用于可重复性）"
    )
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("Boltz-2 蛋白质结合物设计")
    logging.info("="*80)
    logging.info(f"目标序列: {args.target_sequence[:50]}..." if len(args.target_sequence) > 50 else f"目标序列: {args.target_sequence}")
    logging.info(f"结合物长度: {args.binder_length}")
    logging.info(f"设计数量: {args.n_designs}")
    logging.info(f"使用 MPNN: {args.use_mpnn}")
    logging.info("="*80)
    
    try:
        if args.n_designs > 1:
            # 批量设计
            designs = design_multiple_binders(
                target_sequence=args.target_sequence,
                binder_length=args.binder_length,
                n_designs=args.n_designs,
                n_optimization_steps=args.optimization_steps,
                n_sharpening_steps=args.sharpening_steps,
                stepsize=args.stepsize,
                use_mpnn=args.use_mpnn,
            )
            
            # 输出最佳设计
            best_idx = np.argmax([d['sharp_prediction'].iptm for d in designs])
            logging.info(f"\n🏆 最佳设计: #{best_idx + 1}")
            logging.info(f"   序列: {designs[best_idx]['binder_sequence']}")
            logging.info(f"   ipTM: {designs[best_idx]['sharp_prediction'].iptm:.4f}")
            
        else:
            # 单个设计
            result = design_binder_boltz2(
                target_sequence=args.target_sequence,
                binder_length=args.binder_length,
                n_optimization_steps=args.optimization_steps,
                n_sharpening_steps=args.sharpening_steps,
                stepsize=args.stepsize,
                use_mpnn=args.use_mpnn,
                random_seed=args.random_seed
            )
            
            logging.info(f"\n✅ 设计完成！")
            logging.info(f"   序列: {result['binder_sequence']}")
            logging.info(f"   ipTM: {result['sharp_prediction'].iptm:.4f}")
        
        logging.info(f"\n📁 所有结果保存在: {OUTPUT_DIRS['base']}")
        logging.info(f"   结构文件: {OUTPUT_DIRS['structures']}")
        logging.info(f"   可视化图表: {OUTPUT_DIRS['figures']}")
        logging.info(f"   日志文件: {OUTPUT_DIRS['logs']}")
        
    except Exception as e:
        logging.error(f"❌ 设计过程中出现错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
