import torch
import numpy as np
import pandas as pd
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
from torch.utils.data import Dataset, DataLoader

# 导入必要的模块
from cl_base_model import ContrastiveEncoder

warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedContrastiveDataset(Dataset):
    """
    使用增强数据集的对比学习数据集
    直接读取包含一致性得分和概率分布的增强数据集
    """

    def __init__(self, enhanced_dataset_path: str, weighting_strategy: str = 'linear',
                 weight_threshold: float = 0.3, round_num: int = 2,
                 use_filtered_negatives: bool = False, negative_sample_ratio: float = 0.5):
        """
        初始化增强数据集 - 优化版本

        Args:
            enhanced_dataset_path: 增强数据集路径 (.pkl)
            weighting_strategy: 权重策略 ('linear' 或 'threshold')
            weight_threshold: 权重阈值 (仅threshold策略使用)
            round_num: 当前轮次编号 (用于记录训练历史)
            use_filtered_negatives: 是否使用被筛掉的样本作为负例
            negative_sample_ratio: 负样本采样比例 (相对于batch_size)
        """
        print(f" 加载增强数据集: {enhanced_dataset_path}")
        self.weighting_strategy = weighting_strategy
        self.weight_threshold = weight_threshold
        self.round_num = round_num
        self.enhanced_dataset_path = enhanced_dataset_path

        # 新增：负样本相关参数
        self.use_filtered_negatives = use_filtered_negatives
        self.negative_sample_ratio = negative_sample_ratio

        # 加载增强数据集
        with open(enhanced_dataset_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # 处理不同的数据格式
        if isinstance(loaded_data, dict) and 'samples' in loaded_data:
            # 新格式：包含训练历史的数据集
            print(f" 检测到包含训练历史的数据集格式")
            self.enhanced_samples = loaded_data['samples']
            self.original_metadata = {
                'round_info': loaded_data.get('round_info', {}),
                'metadata': loaded_data.get('metadata', {})
            }
        elif isinstance(loaded_data, list):
            # 旧格式：简单的样本列表
            print(f" 检测到简单数据集格式")
            self.enhanced_samples = loaded_data
            self.original_metadata = None
        else:
            raise ValueError(f"不支持的数据集格式: {type(loaded_data)}")

        print(f" 增强数据集大小: {len(self.enhanced_samples)}")

        # 提取一致性得分
        self.consistency_scores = np.array([sample['consistency_score'] for sample in self.enhanced_samples])

        # 预过滤和权重设置
        self._setup_weights_and_filtering(weighting_strategy, weight_threshold)

        # 新增：设置负样本池（被筛掉的低置信样本）
        self._setup_negative_sample_pool()

        print(f" 最终数据集大小: {len(self.enhanced_samples)}")
        print(f" 权重统计: 最小={np.min(self.sample_weights):.4f}, "
              f"最大={np.max(self.sample_weights):.4f}, "
              f"平均={np.mean(self.sample_weights):.4f}")

        if self.use_filtered_negatives:
            print(f" 负样本池大小: {len(self.negative_sample_pool)}")
            print(f" 负样本采样比例: {self.negative_sample_ratio}")

    def _setup_negative_sample_pool(self):
        """设置负样本池：存储被筛掉的低置信样本"""
        self.negative_sample_pool = []

        if not self.use_filtered_negatives:
            return

        # 收集被筛掉的样本（权重为0的样本）
        for i, (sample, weight) in enumerate(zip(self.enhanced_samples, self.sample_weights)):
            if weight == 0:  # 被筛掉的样本
                negative_sample = {
                    'anchor_content': sample['anchor_content'],
                    'positive_content': sample['positive_content'],
                    'consistency_score': sample['consistency_score'],
                    'sample_index': i,
                    'is_negative': True
                }
                self.negative_sample_pool.append(negative_sample)

        print(f" 负样本池构建完成: {len(self.negative_sample_pool)} 个低置信样本可用作负例")

    def _setup_weights_and_filtering(self, strategy: str, threshold: float):
        """设置权重和预过滤 - 优化版本"""
        scores = self.consistency_scores

        if strategy == 'linear':
            # Linear策略：使用一致性得分作为连续权重，但过滤低于阈值的样本
            try:
                filter_threshold = float(threshold) if threshold is not None else 0.4
            except (TypeError, ValueError):
                print(f" Linear策略：无效阈值 {threshold}，使用默认值 0.4")
                filter_threshold = 0.4

            if filter_threshold <= 0 or filter_threshold >= 1:
                print(f" Linear策略：阈值 {filter_threshold} 超出范围 (0,1)，使用默认值 0.4")
                filter_threshold = 0.4

            print(f" Linear策略：使用原始一致性得分作为权重 [0,1]，过滤阈值={filter_threshold}")

            # 创建mask：低于阈值的样本权重设为0（进入负样本池），高于阈值的保留原始得分
            valid_mask = scores >= filter_threshold
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                print(f" 警告：没有样本高于阈值 {filter_threshold}，保留所有样本")
                self.sample_weights = scores.copy()
            else:
                # 应用阈值过滤：低于阈值→0.0，高于阈值→保留原始得分
                self.sample_weights = np.where(valid_mask, scores, 0.0)

                filtered_count = len(scores) - len(valid_indices)
                print(f" Linear策略：保留 {len(valid_indices)}/{len(scores)} "
                      f"({len(valid_indices)/len(scores):.1%}) 高置信样本，"
                      f"{filtered_count} 个低置信样本进入负样本池")

                # 显示过滤和保留样本的得分范围
                if filtered_count > 0:
                    print(f"  - 过滤样本 [0, {filter_threshold:.2f}) 得分范围: "
                          f"[{np.min(scores[~valid_mask]):.4f}, {np.max(scores[~valid_mask]):.4f}]")
                print(f"  - 保留样本 [{filter_threshold:.2f}, 1.0] 得分范围: "
                      f"[{np.min(scores[valid_indices]):.4f}, {np.max(scores[valid_indices]):.4f}]")
                print(f"  - 保留样本权重范围: "
                      f"[{np.min(self.sample_weights[valid_indices]):.4f}, "
                      f"{np.max(self.sample_weights[valid_indices]):.4f}]")

        elif strategy == 'threshold':
            # Threshold strategy: filter by configurable cutoff and assign binary weights
            try:
                filter_threshold = float(threshold) if threshold is not None else 0.4
            except (TypeError, ValueError):
                print(f" Threshold strategy: invalid threshold {threshold}, fallback to 0.4")
                filter_threshold = 0.4

            if filter_threshold <= 0 or filter_threshold >= 1:
                print(f" Threshold strategy: threshold {filter_threshold} out of (0,1), fallback to 0.4")
                filter_threshold = 0.4

            print(f" Threshold strategy: using cutoff={filter_threshold} to pre-filter low-confidence samples")

            # Keep only samples with confidence >= threshold
            valid_mask = scores >= filter_threshold
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                print(f" Warning: no samples above threshold {filter_threshold}, keep all samples")
                self.sample_weights = np.ones_like(scores)
            else:
                # Binary weights: keep high-confidence samples, drop the rest
                self.sample_weights = np.where(valid_mask, 1.0, 0.0)

                print(f" Threshold strategy: flagged {len(valid_indices)}/{len(scores)} "
                      f"({len(valid_indices)/len(scores):.1%}) high-confidence samples")
                low_range = f"[0, {filter_threshold:.2f})"
                high_range = f"[{filter_threshold:.2f}, 1.0]"
                if np.any(~valid_mask):
                    print(f" Filtered samples {low_range} confidence range: "
                          f"[{np.min(scores[~valid_mask]):.4f}, {np.max(scores[~valid_mask]):.4f}]")
                else:
                    print(" No samples filtered out")
                print(f" Retained samples {high_range} confidence range: "
                      f"[{np.min(scores[valid_indices]):.4f}, {np.max(scores[valid_indices]):.4f}]")

        elif strategy == 'tiered':
            # Tiered策略：分层离散权重
            # [0, 0.4) → 0.0 (作为负例)
            # [0.4, 0.6) → 0.5 (低置信)
            # [0.6, 0.8) → 0.7 (中置信)
            # [0.8, 1.0] → 0.9 (高置信)
            print(f" Tiered策略：分层离散权重")
            print(f"   [0.0, 0.4) → weight=0.0 (作为负例)")
            print(f"   [0.4, 0.6) → weight=0.5 (低置信)")
            print(f"   [0.6, 0.8) → weight=0.7 (中置信)")
            print(f"   [0.8, 1.0] → weight=0.9 (高置信)")

            # 初始化权重数组
            self.sample_weights = np.zeros_like(scores)

            # 分层赋权
            tier1_mask = (scores >= 0.0) & (scores < 0.4)   # 负例
            tier2_mask = (scores >= 0.4) & (scores < 0.6)   # 低置信
            tier3_mask = (scores >= 0.6) & (scores < 0.8)   # 中置信
            tier4_mask = (scores >= 0.8) & (scores <= 1.0)  # 高置信

            self.sample_weights[tier1_mask] = 0.0
            self.sample_weights[tier2_mask] = 0.5
            self.sample_weights[tier3_mask] = 0.7
            self.sample_weights[tier4_mask] = 0.9

            # 统计各层样本数量
            tier1_count = np.sum(tier1_mask)
            tier2_count = np.sum(tier2_mask)
            tier3_count = np.sum(tier3_mask)
            tier4_count = np.sum(tier4_mask)
            total_count = len(scores)

            print(f" Tiered策略样本分布：")
            print(f"   Tier 1 [0.0, 0.4) (负例): {tier1_count}/{total_count} ({tier1_count/total_count:.1%})")
            print(f"   Tier 2 [0.4, 0.6) (低置信): {tier2_count}/{total_count} ({tier2_count/total_count:.1%})")
            print(f"   Tier 3 [0.6, 0.8) (中置信): {tier3_count}/{total_count} ({tier3_count/total_count:.1%})")
            print(f"   Tier 4 [0.8, 1.0] (高置信): {tier4_count}/{total_count} ({tier4_count/total_count:.1%})")
            print(f"   有效训练样本 (weight>0): {tier2_count + tier3_count + tier4_count}/{total_count} "
                  f"({(tier2_count + tier3_count + tier4_count)/total_count:.1%})")

        else:
            raise ValueError(f"不支持的权重策略: {strategy}，请使用 'linear', 'threshold' 或 'tiered'")

    def _get_effective_sample_count(self) -> int:
        """获取有效样本数量（权重>0的样本）"""
        return len([w for w in self.sample_weights if w > 0])

    def save_enhanced_dataset_with_training_history(self, save_path: str):
        """保存包含训练历史信息的完整数据集"""
        print(f" 保存带训练历史的增强数据集: {save_path}")

        # 为每个样本添加本轮训练信息
        for i, sample in enumerate(self.enhanced_samples):
            # 初始化字段（如果不存在）
            if 'training_rounds' not in sample:
                sample['training_rounds'] = {}
            if 'filter_history' not in sample:
                sample['filter_history'] = {}

            # 记录本轮是否参与训练
            is_trained = self.sample_weights[i] > 0
            sample['training_rounds'][f'round_{self.round_num}'] = is_trained

            # 记录过滤信息
            sample['filter_history'][f'round_{self.round_num}'] = {
                'strategy': self.weighting_strategy,
                'threshold': self.weight_threshold,
                'consistency_score': float(self.consistency_scores[i]),
                'weight': float(self.sample_weights[i]),
                'passed': is_trained
            }

            # 更新最后训练轮次
            if is_trained:
                sample['last_trained_round'] = self.round_num

        # 计算训练统计
        trained_count = sum(1 for w in self.sample_weights if w > 0)

        # 准备保存数据结构
        if self.original_metadata:
            # 如果有原始元数据，保持和扩展
            save_data = {
                'samples': self.enhanced_samples,
                'round_info': {
                    **self.original_metadata.get('round_info', {}),
                    'current_round': self.round_num,
                    'strategy': self.weighting_strategy,
                    'threshold': self.weight_threshold,
                    'total_samples': len(self.enhanced_samples),
                    'trained_samples': trained_count,
                    'filter_ratio': trained_count / len(self.enhanced_samples) if len(self.enhanced_samples) > 0 else 0
                },
                'metadata': {
                    **self.original_metadata.get('metadata', {}),
                    'training_history': {
                        f'round_{self.round_num}': {
                            'strategy': self.weighting_strategy,
                            'threshold': self.weight_threshold,
                            'trained_samples': trained_count,
                            'total_samples': len(self.enhanced_samples),
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'consistency_scores': self.consistency_scores.tolist(),
                    'sample_weights': self.sample_weights.tolist(),
                    'dataset_path': self.enhanced_dataset_path
                }
            }
        else:
            # 如果没有原始元数据，创建新的格式
            save_data = {
                'samples': self.enhanced_samples,
                'round_info': {
                    'current_round': self.round_num,
                    'strategy': self.weighting_strategy,
                    'threshold': self.weight_threshold,
                    'total_samples': len(self.enhanced_samples),
                    'trained_samples': trained_count,
                    'filter_ratio': trained_count / len(self.enhanced_samples) if len(self.enhanced_samples) > 0 else 0
                },
                'metadata': {
                    'training_history': {
                        f'round_{self.round_num}': {
                            'strategy': self.weighting_strategy,
                            'threshold': self.weight_threshold,
                            'trained_samples': trained_count,
                            'total_samples': len(self.enhanced_samples),
                            'timestamp': datetime.now().isoformat()
                        }
                    },
                    'consistency_scores': self.consistency_scores.tolist(),
                    'sample_weights': self.sample_weights.tolist(),
                    'dataset_path': self.enhanced_dataset_path
                }
            }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f" 完整增强数据集已保存: {save_path}")
        print(f"   本轮训练样本: {trained_count}/{len(self.enhanced_samples)} ({trained_count/len(self.enhanced_samples)*100:.1f}%)")
        print(f"   权重策略: {self.weighting_strategy}, 阈值: {self.weight_threshold}")

    def __len__(self):
        return len(self.enhanced_samples)

    def __getitem__(self, idx):
        sample = self.enhanced_samples[idx]

        # 返回训练需要的数据
        return {
            'anchor_content': sample['anchor_content'],
            'positive_content': sample['positive_content'],
            'consistency_score': sample['consistency_score'],
            'sample_weight': self.sample_weights[idx],
            'anchor_probs': sample.get('anchor_probs', None),
            'positive_probs': sample.get('positive_probs', None),
            'sample_index': idx
        }

class WeightedContrastiveTrainer:
    """
    简化的加权对比学习训练器
    专门用于第二阶段的加权微调
    """

    def __init__(self, enhanced_dataset_path: str, pretrained_model_path: str,
                 weighting_strategy: str = 'linear', weight_threshold: float = 0.3,
                 batch_size: int = 16, base_lr: float = 5e-6, projection_lr: float = 5e-5,
                 use_peft: bool = True, peft_config: dict = None, round_num: int = 2,
                 use_filtered_negatives: bool = False, negative_sample_ratio: float = 0.5):
        """
        初始化加权对比学习训练器

        Args:
            enhanced_dataset_path: 增强数据集路径
            pretrained_model_path: 预训练对比学习模型路径 (必须指定)
            weighting_strategy: 权重策略
            weight_threshold: 权重阈值
            batch_size: 批次大小
            base_lr: 基础学习率
            projection_lr: 投影头学习率
            use_peft: 是否使用PEFT/LoRA
            peft_config: PEFT配置字典
            round_num: 当前轮次编号 (用于记录训练历史)
            use_filtered_negatives: 是否使用被筛掉的样本作为负例
            negative_sample_ratio: 负样本采样比例
        """
        self.enhanced_dataset_path = enhanced_dataset_path
        self.pretrained_model_path = pretrained_model_path
        self.weighting_strategy = weighting_strategy
        self.weight_threshold = weight_threshold
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.projection_lr = projection_lr
        self.use_peft = use_peft
        self.round_num = round_num  # 新增：轮次信息

        # 新增：负样本相关参数
        self.use_filtered_negatives = use_filtered_negatives
        self.negative_sample_ratio = negative_sample_ratio

        #  新增：PEFT配置
        if peft_config is None:
            self.peft_config = {
                'r': 8,              # LoRA的秩，越小参数越少，常用8, 16, 32
                'lora_alpha': 16,    # LoRA的缩放因子，通常是r的两倍
                'target_modules': ["query", "key", "value"], # 对注意力的Q,K,V应用LoRA
                'lora_dropout': 0.1, # LoRA层的dropout率
                'bias': "none",      # "none", "all", "lora_only"
            }
        else:
            self.peft_config = peft_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置增强数据集
        self._setup_enhanced_dataset()

        # 加载预训练模型
        self._load_pretrained_model()

        # 设置优化器和损失函数
        self._setup_optimizer_and_loss()

        print(f" 加权对比学习训练器初始化完成")
        print(f"   模型路径: {self.pretrained_model_path}")
        print(f"   数据集路径: {self.enhanced_dataset_path}")
        print(f"   权重策略: {self.weighting_strategy}")
        print(f"   使用PEFT: {self.use_peft}")
        if self.use_peft:
            print(f"   PEFT配置: r={self.peft_config['r']}, alpha={self.peft_config['lora_alpha']}")
            print(f"   目标模块: {self.peft_config['target_modules']}")
    def _setup_enhanced_dataset(self):
        """设置增强数据集"""
        print(" 加载增强数据集...")

        # 创建增强数据集
        self.enhanced_dataset = EnhancedContrastiveDataset(
            enhanced_dataset_path=self.enhanced_dataset_path,
            weighting_strategy=self.weighting_strategy,
            weight_threshold=self.weight_threshold,
            round_num=self.round_num,  # 传递轮次信息
            use_filtered_negatives=self.use_filtered_negatives,
            negative_sample_ratio=self.negative_sample_ratio
        )

        # 创建数据整理器，传递负样本池
        collator = EnhancedDataCollator(
            negative_sample_pool=self.enhanced_dataset.negative_sample_pool,
            negative_sample_ratio=self.negative_sample_ratio,
            use_filtered_negatives=self.use_filtered_negatives
        )

        # 创建数据加载器
        self.dataloader = DataLoader(
            self.enhanced_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )

        print(f" 增强数据集设置完成")
        print(f"   数据集大小: {len(self.enhanced_dataset)}")
        print(f"   批次数量: {len(self.dataloader)}")
        print(f"   权重策略: {self.weighting_strategy}")
        if self.weighting_strategy == 'threshold':
            print(f"   过滤阈值: {self.weight_threshold}")
            print(f"   策略说明: 预过滤[0,{self.weight_threshold})低置信样本(→负样本池)，剩余样本等权重=1.0训练")
        elif self.weighting_strategy == 'linear':
            print(f"   过滤阈值: {self.weight_threshold}")
            print(f"   策略说明: 过滤[0,{self.weight_threshold})低置信样本(→负样本池)，剩余样本使用原始一致性得分作为连续权重")
        elif self.weighting_strategy == 'tiered':
            print(f"   策略说明: 分层离散权重 - [0,0.4)→0.0(负例), [0.4,0.6)→0.5, [0.6,0.8)→0.7, [0.8,1.0]→0.9")

    def _load_pretrained_model(self):
        """加载预训练的对比学习模型"""
        print(f" 加载预训练模型: {self.pretrained_model_path}")

        if not os.path.exists(self.pretrained_model_path):
            raise FileNotFoundError(f"预训练模型不存在: {self.pretrained_model_path}")

        # 加载检查点
        checkpoint = torch.load(self.pretrained_model_path, map_location='cpu', weights_only=False)

        # 获取模型配置
        model_type = checkpoint.get('training_model_type', 'modelscope')
        model_identifier = checkpoint.get('training_model_identifier_or_path')
        proj_config = checkpoint.get('projection_head_config', {
            'hidden_dim': 768, 'output_dim': 384, 'dropout_rate': 0.15
        })

        # 重建对比学习编码器
        self.contrastive_encoder = ContrastiveEncoder(
            model_type=model_type,
            model_name_or_path=model_identifier,
            projection_hidden_dim=proj_config['hidden_dim'],
            projection_output_dim=proj_config['output_dim'],
            projection_dropout_rate=proj_config['dropout_rate']
        )

        #  重新初始化投影头（用于微调）
        print(" 重新初始化投影头用于微调...")
        for layer in self.contrastive_encoder.projection_head:
            if hasattr(layer, 'weight') and layer.weight is not None:
                # 只对2维或更高维的权重矩阵应用xavier初始化
                if layer.weight.dim() >= 2:
                    torch.nn.init.xavier_uniform_(layer.weight)
                else:
                    # 对于1维权重（如某些特殊层），使用正态分布初始化
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if hasattr(layer, 'bias') and layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        #  关键修正：参考sup_training.py的成功逻辑，先应用LoRA再加载权重
        # 1. 首先检查checkpoint是否包含PEFT配置
        use_peft_from_checkpoint = checkpoint.get('use_peft', False)
        peft_config_from_checkpoint = checkpoint.get('peft_config', None)

        if use_peft_from_checkpoint and peft_config_from_checkpoint is not None:
            print(" 检测到checkpoint包含PEFT/LoRA配置，先应用LoRA结构...")
            try:
                from peft import LoraConfig, get_peft_model

                # 使用checkpoint中的PEFT配置
                lora_config = LoraConfig(**peft_config_from_checkpoint)
                self.contrastive_encoder.base_model = get_peft_model(
                    self.contrastive_encoder.base_model, lora_config
                )
                print(" LoRA结构已应用")

            except Exception as e:
                print(f" LoRA结构应用失败: {e}")

        # 2. 然后加载权重（现在结构应该匹配了）
        try:
            missing_keys, unexpected_keys = self.contrastive_encoder.load_state_dict(
                checkpoint['contrastive_encoder_state_dict'], strict=False
            )
            print(f" 模型权重加载完成 (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")

            if use_peft_from_checkpoint:
                print(" 预训练模型LoRA权重已成功加载")

        except Exception as e:
            print(f" 权重加载部分失败: {e}")

        # 3. 检查是否需要额外的PEFT配置（用于可能的新参数）
        if self.use_peft:
            try:
                from peft import PeftModel

                if hasattr(self.contrastive_encoder, 'base_model'):
                    base_model = self.contrastive_encoder.base_model

                    if isinstance(base_model, PeftModel):
                        print(" 确认模型已包含LoRA适配器")

                        # 获取当前LoRA配置信息
                        current_config = base_model.peft_config
                        if hasattr(current_config, 'values'):
                            config_info = list(current_config.values())[0] if current_config else None
                        else:
                            config_info = current_config.get('default', None) if hasattr(current_config, 'get') else current_config

                        if config_info:
                            print(f"    LoRA配置: r={getattr(config_info, 'r', 'unknown')}, "
                                  f"alpha={getattr(config_info, 'lora_alpha', 'unknown')}, "
                                  f"target_modules={getattr(config_info, 'target_modules', 'unknown')}")
                            print("    预训练模型LoRA权重成功继承")

                    else:
                        print("    模型不是PeftModel，可能需要重新应用LoRA")
                        if not use_peft_from_checkpoint:
                            print("    应用新的LoRA配置...")
                            from peft import LoraConfig, get_peft_model

                            lora_config = LoraConfig(
                                r=self.peft_config['r'],
                                lora_alpha=self.peft_config['lora_alpha'],
                                target_modules=self.peft_config['target_modules'],
                                lora_dropout=self.peft_config['lora_dropout'],
                                bias=self.peft_config['bias'],
                                task_type="FEATURE_EXTRACTION"
                            )

                            self.contrastive_encoder.base_model = get_peft_model(
                                self.contrastive_encoder.base_model, lora_config
                            )
                            print("    新LoRA配置已应用")

                    # 计算可训练参数统计
                    total_params = sum(p.numel() for p in self.contrastive_encoder.parameters())
                    trainable_params = sum(p.numel() for p in self.contrastive_encoder.parameters() if p.requires_grad)

                    print(f"    参数统计: 总参数={total_params:,}, 可训练参数={trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

                else:
                    print("    未找到base_model属性")
                    self.use_peft = False

            except ImportError:
                print("    PEFT库未安装")
                self.use_peft = False
            except Exception as e:
                print(f"    PEFT配置检查失败: {e}")
                self.use_peft = False

        self.contrastive_encoder.to(self.device)

    def _setup_optimizer_and_loss(self):
        """设置优化器和损失函数"""
        # 设置不同学习率
        base_params = []
        projection_params = []

        for name, param in self.contrastive_encoder.named_parameters():
            if 'projection_head' in name:
                projection_params.append(param)
            else:
                base_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.base_lr},
            {'params': projection_params, 'lr': self.projection_lr}
        ])

        print(f" 优化器设置完成")
        print(f"   基础编码器学习率: {self.base_lr}")
        print(f"   投影头学习率: {self.projection_lr}")

    def train(self, num_epochs: int = 50, save_frequency: int = 10):
        """
        使用增强数据集进行加权对比学习训练

        Args:
            num_epochs: 训练轮数
            save_frequency: 保存频率
        """
        print(f" 开始加权对比学习训练...")
        print(f"   训练轮数: {num_epochs}")
        print(f"   数据集大小: {len(self.enhanced_dataset)}")
        print(f"   批次大小: {self.batch_size}")
        print(f"   负样本策略: in-batch")

        self.contrastive_encoder.train()
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                self.optimizer.zero_grad()

                # 提取批次数据
                anchor_texts = batch['anchor_texts']
                positive_texts = batch['positive_texts']
                weights = batch['sample_weights']
                batch_size = len(anchor_texts)

                if batch_size < 2:
                    continue

                try:
                    # 检查是否有额外的负样本
                    use_negatives = batch.get('use_negatives', False)
                    negative_texts = batch.get('negative_texts', [])

                    # 根据权重策略选择不同的处理方式
                    if self.weighting_strategy == 'threshold':
                        # Threshold策略：所有样本权重都是1，使用普通对比学习
                        anchor_emb = self.contrastive_encoder(anchor_texts)
                        positive_emb = self.contrastive_encoder(positive_texts)

                        if use_negatives and negative_texts:
                            # 使用增强的InfoNCE损失（包含额外负样本）
                            negative_emb = self.contrastive_encoder(negative_texts)
                            loss = self._compute_enhanced_infonce_loss(anchor_emb, positive_emb, negative_emb)
                        else:
                            # 使用标准的in-batch InfoNCE损失
                            loss = self._compute_inbatch_infonce_loss(anchor_emb, positive_emb)

                        final_loss = torch.mean(loss)  # 简单平均，无需加权

                    elif self.weighting_strategy == 'linear' or self.weighting_strategy == 'tiered':
                        # Linear策略：使用一致性得分作为权重 [0,1]
                        # Tiered策略：使用分层离散权重 {0.0, 0.5, 0.7, 0.9}
                        anchor_emb = self.contrastive_encoder(anchor_texts)
                        positive_emb = self.contrastive_encoder(positive_texts)

                        if use_negatives and negative_texts:
                            # 使用增强的InfoNCE损失（包含额外负样本）
                            negative_emb = self.contrastive_encoder(negative_texts)
                            loss = self._compute_enhanced_infonce_loss(anchor_emb, positive_emb, negative_emb)
                        else:
                            # 使用标准的in-batch InfoNCE损失
                            loss = self._compute_inbatch_infonce_loss(anchor_emb, positive_emb)

                        # 应用权重（不归一化）
                        weights_tensor = torch.tensor(weights, device=loss.device, dtype=loss.dtype)
                        # 确保分母是有效样本数，不包含0权重样本
                        valid_weights = weights_tensor[weights_tensor > 0]
                        valid_loss = loss[weights_tensor > 0]

                        if len(valid_weights) > 0:
                            final_loss = torch.sum(valid_loss * valid_weights) / torch.sum(valid_weights)
                        else:
                            # 如果没有有效样本，跳过这个批次
                            continue
                    else:
                        raise ValueError(f"不支持的权重策略: {self.weighting_strategy}")

                    # 反向传播
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.contrastive_encoder.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_loss += final_loss.item()
                    num_batches += 1

                    progress_bar.set_postfix(loss=f"{final_loss.item():.4f}")

                except Exception as e:
                    print(f" 批次处理失败: {e}")
                    continue

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f" 新的最佳损失: {best_loss:.4f}, 保存模型...")
                self._save_model(epoch + 1, best_loss, is_best=True)

            # 定期保存
            if (epoch + 1) % save_frequency == 0:
                print(f" 定期保存模型 (Epoch {epoch + 1})...")
                self._save_model(epoch + 1, avg_loss, is_best=False)

        print(f" 加权训练完成! 最佳损失: {best_loss:.4f}")

        # 保存带训练历史的完整增强数据集
        history_save_path = self.enhanced_dataset_path.replace('.pkl', f'_round{self.round_num}_history.pkl')
        self.enhanced_dataset.save_enhanced_dataset_with_training_history(history_save_path)

    def _compute_enhanced_infonce_loss(self, anchor_emb, positive_emb, negative_emb, temperature=0.07):
        """
        计算增强的InfoNCE损失，包含额外的负样本

        Args:
            anchor_emb: 锚点嵌入 [batch_size, embed_dim]
            positive_emb: 正样本嵌入 [batch_size, embed_dim]
            negative_emb: 额外负样本嵌入 [num_negatives, embed_dim]
            temperature: 温度参数

        Returns:
            loss: 每个样本的损失 [batch_size]
        """
        import torch.nn.functional as F

        batch_size = anchor_emb.size(0)
        num_negatives = negative_emb.size(0)

        # 计算正样本相似度（对角线）
        positive_similarities = torch.sum(anchor_emb * positive_emb, dim=1, keepdim=True) / temperature  # [batch_size, 1]

        # 计算batch内负样本相似度矩阵（去掉对角线）
        inbatch_sim_matrix = torch.matmul(anchor_emb, positive_emb.t()) / temperature  # [batch_size, batch_size]
        # 创建mask去掉对角线正样本
        mask = torch.eye(batch_size, device=anchor_emb.device, dtype=torch.bool)
        inbatch_negatives = inbatch_sim_matrix.masked_select(~mask).view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]

        # 计算额外负样本相似度
        if num_negatives > 0:
            extra_negatives = torch.matmul(anchor_emb, negative_emb.t()) / temperature  # [batch_size, num_negatives]
            # 合并所有负样本
            all_negatives = torch.cat([inbatch_negatives, extra_negatives], dim=1)  # [batch_size, batch_size-1+num_negatives]
        else:
            all_negatives = inbatch_negatives

        # 构建logits：正样本 + 所有负样本
        logits = torch.cat([positive_similarities, all_negatives], dim=1)  # [batch_size, 1+num_negatives]

        # 标签总是0（第一个位置是正样本）
        labels = torch.zeros(batch_size, device=anchor_emb.device, dtype=torch.long)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels, reduction='none')

        return loss

    def _compute_inbatch_infonce_loss(self, anchor_emb, positive_emb, temperature=0.07):
        """计算 in-batch InfoNCE 损失"""
        import torch.nn.functional as F

        batch_size = anchor_emb.size(0)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(anchor_emb, positive_emb.t()) / temperature

        # 对角线元素是正样本
        labels = torch.arange(batch_size, device=anchor_emb.device)

        # 计算 InfoNCE 损失
        loss = F.cross_entropy(sim_matrix, labels, reduction='none')

        return loss

    def _save_model(self, epoch: int, loss: float, is_best: bool = False):
        """保存模型到 iter_model/ 目录，确保格式与第一阶段兼容"""
        #  新增：从增强数据集路径中提取数据比例信息
        data_fraction = "unknown"
        try:
            # 从增强数据集路径中提取数据比例
            # 例如: consistency_scores/frac0.2_xxx/enhanced_dataset_xxx.pkl
            dataset_dir = os.path.dirname(self.enhanced_dataset_path)
            dir_name = os.path.basename(dataset_dir)

            import re
            frac_match = re.search(r'frac([\d\.]+)', dir_name)
            if frac_match:
                data_fraction = f"frac{frac_match.group(1)}"
        except Exception as e:
            print(f" 提取数据比例失败: {e}")
            data_fraction = "unknown"

        # 生成参数命名 - 包含数据比例
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.weighting_strategy
        threshold = self.weight_threshold
        lr_base = self.base_lr
        lr_proj = self.projection_lr

        param_name = f"{data_fraction}_enhanced_{strategy}_th{threshold}_lr{lr_base}_{lr_proj}_{timestamp}"

        save_dir = f"iter_model/{param_name}"
        os.makedirs(save_dir, exist_ok=True)

        #  关键修改：确保检查点格式与第一阶段完全兼容
        # 从预训练模型中获取原始配置信息
        pretrained_checkpoint = torch.load(self.pretrained_model_path, map_location='cpu', weights_only=False)

        # 保存模型检查点 - 兼容第一阶段格式
        checkpoint = {
            # ===== 核心模型状态 =====
            'contrastive_encoder_state_dict': self.contrastive_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),  #  新增：优化器状态
            'epoch': epoch,
            'best_loss': loss,  #  修改：使用与第一阶段一致的键名
            'training_history': {'second_stage_training': True, 'weighted_contrastive_epoch': epoch},  #  兼容字段

            # ===== 模型架构配置 (继承自第一阶段) =====
            'training_model_type': pretrained_checkpoint.get('training_model_type', 'modelscope'),
            'training_model_identifier_or_path': pretrained_checkpoint.get('training_model_identifier_or_path'),
            'projection_head_config': pretrained_checkpoint.get('projection_head_config', {
                'hidden_dim': 768, 'output_dim': 384, 'dropout_rate': 0.15
            }),

            # ===== 第一阶段训练参数 (继承) =====
            'pruning_model_path': pretrained_checkpoint.get('pruning_model_path'),
            'similarity_threshold': pretrained_checkpoint.get('similarity_threshold'),
            'num_negatives': pretrained_checkpoint.get('num_negatives'),
            'pruning_inference_batch_size': pretrained_checkpoint.get('pruning_inference_batch_size'),
            'use_weighted_loss': pretrained_checkpoint.get('use_weighted_loss'),
            'loss_weights': pretrained_checkpoint.get('loss_weights'),
            'adaptive_weighting': pretrained_checkpoint.get('adaptive_weighting'),
            'infonce_mode': pretrained_checkpoint.get('infonce_mode'),
            'positive_pair_strategy': pretrained_checkpoint.get('positive_pair_strategy'),
            'simcse_temperature': pretrained_checkpoint.get('simcse_temperature'),
            'simcse_dropout_rate': pretrained_checkpoint.get('simcse_dropout_rate'),
            'simcse_remove_duplicates': pretrained_checkpoint.get('simcse_remove_duplicates'),
            'hybrid_ratio': pretrained_checkpoint.get('hybrid_ratio'),
            'min_subtree_size_ds1': pretrained_checkpoint.get('min_subtree_size_ds1'),
            'max_samples_per_post_ds1': pretrained_checkpoint.get('max_samples_per_post_ds1'),
            'min_subtree_size_ds2': pretrained_checkpoint.get('min_subtree_size_ds2'),
            'max_samples_per_subtree_ds2': pretrained_checkpoint.get('max_samples_per_subtree_ds2'),

            # ===== PEFT/LoRA配置 =====
            'use_peft': self.use_peft,
            'peft_config': self.peft_config if self.use_peft else None,

            # ===== 第二阶段专有参数 =====
            'enhanced_dataset_path': self.enhanced_dataset_path,
            'pretrained_model_path': self.pretrained_model_path,
            'weighting_strategy': self.weighting_strategy,
            'weight_threshold': self.weight_threshold,
            'batch_size': self.batch_size,
            'base_lr': self.base_lr,
            'projection_lr': self.projection_lr,
            'base_lr_initial': self.base_lr,  #  新增：与第一阶段格式兼容
            'projection_lr_initial': self.projection_lr,  #  新增：与第一阶段格式兼容
            'data_fraction': data_fraction,
            'timestamp': timestamp,
            'is_best': is_best,

            # ===== 阶段标识 =====
            'training_stage': 'weighted_contrastive',  #  新增：标识这是第二阶段训练结果
            'stage_info': {
                'stage': 2,
                'description': 'weighted contrastive learning with enhanced dataset',
                'previous_stage_model': self.pretrained_model_path
            }
        }

        #  处理TextCNN特殊配置（如果适用）
        if pretrained_checkpoint.get('training_model_type') == 'textcnn':
            checkpoint['textcnn_config'] = pretrained_checkpoint.get('textcnn_config')
            checkpoint['vocab'] = pretrained_checkpoint.get('vocab')

        model_filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        model_path = os.path.join(save_dir, model_filename)
        torch.save(checkpoint, model_path)

        print(f" 模型已保存: {model_path}")
        print(f" 检查点格式与第一阶段完全兼容，可用于后续监督学习训练")

        return save_dir

class EnhancedDataCollator:
    """增强数据集的数据整理器，用于 in-batch 负样本和额外负样本"""

    def __init__(self, negative_sample_pool: List = None, negative_sample_ratio: float = 0.5,
                 use_filtered_negatives: bool = False):
        """
        初始化数据整理器

        Args:
            negative_sample_pool: 负样本池（被筛掉的低置信样本）
            negative_sample_ratio: 负样本采样比例
            use_filtered_negatives: 是否使用过滤的负样本
        """
        self.negative_sample_pool = negative_sample_pool or []
        self.negative_sample_ratio = negative_sample_ratio
        self.use_filtered_negatives = use_filtered_negatives

    def __call__(self, batch):
        """处理批次数据"""
        anchor_texts = []
        positive_texts = []
        sample_weights = []
        consistency_scores = []

        # 处理正常的正样本对
        for item in batch:
            anchor_texts.append(item['anchor_content'])
            positive_texts.append(item['positive_content'])
            sample_weights.append(item['sample_weight'])
            consistency_scores.append(item['consistency_score'])

        batch_result = {
            'anchor_texts': anchor_texts,
            'positive_texts': positive_texts,
            'sample_weights': sample_weights,
            'consistency_scores': consistency_scores,
            'batch_size': len(anchor_texts)
        }

        # 如果启用了过滤负样本，添加额外的负样本
        if self.use_filtered_negatives and self.negative_sample_pool:
            negative_texts = self._sample_negative_texts(len(anchor_texts))
            batch_result['negative_texts'] = negative_texts
            batch_result['use_negatives'] = True
        else:
            batch_result['use_negatives'] = False

        return batch_result

    def _sample_negative_texts(self, batch_size: int) -> List[str]:
        """从负样本池中采样负例文本"""
        import random

        if not self.negative_sample_pool:
            return []

        # 计算需要采样的负样本数量
        num_negatives = int(batch_size * self.negative_sample_ratio)
        num_negatives = min(num_negatives, len(self.negative_sample_pool))

        if num_negatives == 0:
            return []

        # 随机采样负样本
        sampled_negatives = random.sample(self.negative_sample_pool, num_negatives)

        # 随机选择使用anchor或positive作为负例（增加多样性）
        negative_texts = []
        for neg_sample in sampled_negatives:
            if random.random() < 0.5:
                negative_texts.append(neg_sample['anchor_content'])
            else:
                negative_texts.append(neg_sample['positive_content'])

        return negative_texts

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='增强数据集加权对比学习训练器')

    parser.add_argument('--enhanced-dataset', '-e', type=str, required=True,
                       help='增强数据集路径 (.pkl) [必需]')

    parser.add_argument('--pretrained-model', '-pm', type=str, required=True,
                       help='预训练对比学习模型路径 (.pth) [必需]')

    parser.add_argument('--weighting-strategy', '-w', type=str, default='linear',
                       choices=['linear', 'threshold', 'tiered'],
                       help='权重策略 (默认: linear)')

    parser.add_argument('--weight-threshold', '-t', type=float, default=0.4,
                       help='权重阈值 (默认: 0.4，仅threshold和tiered策略使用)')

    parser.add_argument('--epochs', '-ep', type=int, default=100,
                       help='训练轮数 (默认: 100)')

    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='批次大小 (默认: 16)')

    parser.add_argument('--base-lr', type=float, default=5e-6,
                       help='基础编码器学习率 (默认: 5e-6)')

    parser.add_argument('--projection-lr', type=float, default=5e-5,
                       help='投影头学习率 (默认: 5e-5)')

    parser.add_argument('--save-frequency', type=int, default=20,
                       help='模型保存频率 (默认: 10)')

    #  新增：PEFT/LoRA相关参数
    parser.add_argument('--use-peft', action='store_true', default=True,
                       help='使用PEFT/LoRA进行参数高效微调 (默认: True)')

    parser.add_argument('--no-peft', action='store_true',
                       help='禁用PEFT/LoRA，使用完整模型微调')

    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA的秩 (默认: 8)')

    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA的缩放因子 (默认: 16)')

    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA的dropout率 (默认: 0.1)')

    parser.add_argument('--lora-target-modules', nargs='+',
                       default=["query", "key", "value"],
                       help='LoRA目标模块 (默认: query key value)')

    parser.add_argument('--lora-bias', type=str, default="none",
                       choices=["none", "all", "lora_only"],
                       help='LoRA bias策略 (默认: none)')

    return parser.parse_args()

def main():
    """主函数"""
    print(" 增强数据集加权对比学习训练器启动...")

    args = parse_arguments()

    # 验证输入文件
    if not os.path.exists(args.enhanced_dataset):
        print(f" 增强数据集不存在: {args.enhanced_dataset}")
        return

    if not os.path.exists(args.pretrained_model):
        print(f" 预训练模型不存在: {args.pretrained_model}")
        return

    try:
        #  处理PEFT参数
        use_peft = args.use_peft and not args.no_peft  # 如果指定了--no-peft则禁用

        peft_config = {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'target_modules': args.lora_target_modules,
            'lora_dropout': args.lora_dropout,
            'bias': args.lora_bias
        }

        #  直接初始化加权训练器
        print(" 初始化加权对比学习训练器...")

        trainer = WeightedContrastiveTrainer(
            enhanced_dataset_path=args.enhanced_dataset,
            pretrained_model_path=args.pretrained_model,
            weighting_strategy=args.weighting_strategy,
            weight_threshold=args.weight_threshold,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            projection_lr=args.projection_lr,
            use_peft=use_peft,
            peft_config=peft_config
        )

        #  开始训练
        trainer.train(
            num_epochs=args.epochs,
            save_frequency=args.save_frequency
        )

        print(" 加权对比学习训练完成!")

    except Exception as e:
        print(f" 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def run_stage2_weighted_contrastive(config: dict, pretrained_path: str,
                                   enhanced_dataset: str, output_dir: str, round_num: int = 2) -> str:
    """
    标准化接口：运行Stage 2+加权对比学习

    Args:
        config: Stage 2配置字典
        pretrained_path: 预训练模型路径
        enhanced_dataset: 增强数据集路径
        output_dir: 输出目录
        round_num: 当前轮次编号

    Returns:
        最佳模型路径（直接保存在output_dir/best_model.pth）
    """
    import torch
    import os

    print(f" Stage 2加权对比学习接口调用")
    print(f"   预训练模型: {pretrained_path}")
    print(f"   增强数据集: {enhanced_dataset}")
    print(f"   输出目录: {output_dir}")

    try:
        # 创建训练器
        trainer = WeightedContrastiveTrainer(
            enhanced_dataset_path=enhanced_dataset,
            pretrained_model_path=pretrained_path,
            weighting_strategy=config.get('weighting_strategy', 'linear'),
            weight_threshold=config.get('weight_threshold', 0.3),
            batch_size=config.get('batch_size', 16),
            base_lr=float(config.get('base_lr', 5e-6)),
            projection_lr=float(config.get('projection_lr', 5e-5)),
            use_peft=config.get('use_peft', True),
            peft_config={
                'r': config.get('lora_r', 8),
                'lora_alpha': config.get('lora_alpha', 16),
                'target_modules': config.get('lora_target_modules', ["query", "key", "value", "dense"]),
                'lora_dropout': config.get('lora_dropout', 0.1),
                'bias': config.get('lora_bias', "none")
            },
            round_num=round_num,  # 传递轮次信息
            use_filtered_negatives=config.get('use_filtered_negatives', False),  # 新增：负样本策略
            negative_sample_ratio=config.get('negative_sample_ratio', 0.5)  # 新增：负样本比例
        )

        # 修改训练器的保存逻辑，使其直接保存到output_dir
        original_save_model = trainer._save_model

        def custom_save_model(epoch, loss, is_best=False):
            """自定义保存函数，直接保存到指定目录"""
            if is_best:
                # 从预训练模型中获取原始配置信息
                pretrained_checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

                checkpoint = {
                    'contrastive_encoder_state_dict': trainer.contrastive_encoder.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': loss,
                    'training_history': {'second_stage_training': True, 'weighted_contrastive_epoch': epoch},

                    # 继承第一阶段的配置
                    'training_model_type': pretrained_checkpoint.get('training_model_type', 'modelscope'),
                    'training_model_identifier_or_path': pretrained_checkpoint.get('training_model_identifier_or_path'),
                    'projection_head_config': pretrained_checkpoint.get('projection_head_config'),

                    # PEFT配置
                    'use_peft': trainer.use_peft,
                    'peft_config': trainer.peft_config if trainer.use_peft else None,

                    # 第二阶段参数
                    'enhanced_dataset_path': trainer.enhanced_dataset_path,
                    'pretrained_model_path': trainer.pretrained_model_path,
                    'weighting_strategy': trainer.weighting_strategy,
                    'weight_threshold': trainer.weight_threshold,
                    'training_stage': 'weighted_contrastive'
                }

                # 处理TextCNN特殊配置
                if pretrained_checkpoint.get('training_model_type') == 'textcnn':
                    checkpoint['textcnn_config'] = pretrained_checkpoint.get('textcnn_config')
                    checkpoint['vocab'] = pretrained_checkpoint.get('vocab')

                # 直接保存到output_dir
                model_path = os.path.join(output_dir, 'best_model.pth')
                os.makedirs(output_dir, exist_ok=True)
                torch.save(checkpoint, model_path)
                print(f" 最佳模型已保存: {model_path}")

        # 替换保存函数
        trainer._save_model = custom_save_model

        # 运行训练
        trainer.train(
            num_epochs=config.get('epochs', 50),
            save_frequency=config.get('save_frequency', 10)
        )

        # 返回最佳模型路径
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        print(f" Stage 2训练完成，模型保存在: {best_model_path}")
        return best_model_path

    except Exception as e:
        print(f" Stage 2训练失败: {e}")
        raise


if __name__ == "__main__":
    main()