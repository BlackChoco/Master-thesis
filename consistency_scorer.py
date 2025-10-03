import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import logging

# 导入必要的模块
from sup_training import SupervisedModel, SupervisedTextDataset, load_pretrained_encoder
from cl_dataset import ContrastiveDataset1, ContrastiveDataset2
from Tree_data_model import PostStorage

class ConsistencyScorer:
    """使用训练好的分类器计算对比学习样本对的一致性得分"""

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化一致性评分器

        Args:
            model_path: 训练好的模型路径（.pth文件）
            device: 计算设备
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)

        # 加载模型信息
        self.model_info = self._load_model_info()

        # 重建模型和分词器
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        # 标签映射
        self.label_to_id = self.model_info['hyperparameters']['label_to_id'] if 'label_to_id' in self.model_info['hyperparameters'] else None
        self.id_to_label = {v: k for k, v in self.label_to_id.items()} if self.label_to_id else None

        print(f" 一致性评分器已初始化")
        print(f"   模型路径: {model_path}")
        print(f"   设备: {self.device}")
        print(f"   模型类型: {self.model_info.get('model_architecture', 'Unknown')}")

    def _load_model_info(self) -> Dict:
        """加载模型信息"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")

        #  检查是否为JSON分类器选择文件
        actual_model_path = self.model_path
        if self.model_path.endswith('.json'):
            print(f" 检测到分类器选择文件，读取实际模型路径: {self.model_path}")

            # 读取JSON文件获取实际的模型路径
            with open(self.model_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 从JSON中提取实际的模型路径
            actual_model_path = json_data.get('model_path')
            if not actual_model_path:
                raise ValueError(f"JSON文件中未找到model_path字段: {self.model_path}")

            # 智能路径解析：适配各种自定义路径
            original_path = actual_model_path

            # 首先尝试直接使用路径
            if os.path.exists(actual_model_path):
                print(f"   直接找到模型文件: {actual_model_path}")
            else:
                # 提取路径的关键部分（从最后一个round或supervised_training开始）
                key_parts = []
                path_parts = actual_model_path.replace('\\', '/').split('/')

                # 找到关键起始点
                start_idx = -1
                for i, part in enumerate(path_parts):
                    if part.startswith('round') or part == 'supervised_training' or part == 'saved_models':
                        start_idx = i
                        break

                if start_idx >= 0:
                    key_parts = path_parts[start_idx:]
                    key_path = '/'.join(key_parts)
                    print(f"   提取关键路径: {key_path}")

                    # 从JSON文件所在位置向上搜索
                    json_dir = os.path.dirname(os.path.abspath(self.model_path))
                    search_dirs = [
                        json_dir,  # JSON文件所在目录
                        os.path.dirname(json_dir),  # 父目录
                        os.path.dirname(os.path.dirname(json_dir)),  # 祖父目录
                        os.path.dirname(os.path.dirname(os.path.dirname(json_dir))),  # 曾祖父目录
                        '.',  # 当前工作目录
                    ]

                    found = False
                    for search_dir in search_dirs:
                        # 尝试直接组合
                        candidate = os.path.join(search_dir, key_path)
                        if os.path.exists(candidate):
                            actual_model_path = os.path.abspath(candidate)
                            print(f"   在 {search_dir} 找到模型: {actual_model_path}")
                            found = True
                            break

                        # 尝试在search_dir下递归查找key_path的最后几个部分
                        if not found and len(key_parts) >= 2:
                            # 使用最后两个部分作为搜索模式
                            pattern = os.path.join(key_parts[-2], key_parts[-1])
                            for root, dirs, files in os.walk(search_dir):
                                if pattern in os.path.join(root, '').replace('\\', '/'):
                                    candidate = os.path.join(root, key_parts[-1])
                                    if os.path.exists(candidate):
                                        actual_model_path = os.path.abspath(candidate)
                                        print(f"   通过模式匹配找到: {actual_model_path}")
                                        found = True
                                        break
                            if found:
                                break

                # 如果还是没找到，尝试从文件名反向查找
                if not os.path.exists(actual_model_path):
                    filename = os.path.basename(actual_model_path)
                    if filename.endswith('.pth'):
                        print(f"   尝试查找文件: {filename}")
                        # 从JSON位置向上搜索这个文件名
                        for search_dir in search_dirs:
                            for root, dirs, files in os.walk(search_dir):
                                if filename in files:
                                    candidate = os.path.join(root, filename)
                                    # 验证路径中包含一些关键词
                                    if any(k in candidate for k in ['supervised_training', 'saved_models', 'best_model']):
                                        actual_model_path = os.path.abspath(candidate)
                                        print(f"   通过文件名搜索找到: {actual_model_path}")
                                        break

            # 检查实际模型文件是否存在
            if not os.path.exists(actual_model_path):
                # 最后尝试：在JSON文件所在目录的父目录查找
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.model_path)))
                for root, dirs, files in os.walk(parent_dir):
                    if 'best_model.pth' in files:
                        candidate = os.path.join(root, 'best_model.pth')
                        if os.path.basename(os.path.dirname(candidate)) in actual_model_path:
                            actual_model_path = candidate
                            print(f"   通过搜索找到模型: {actual_model_path}")
                            break

                if not os.path.exists(actual_model_path):
                    raise FileNotFoundError(f"JSON中指定的模型文件不存在: {actual_model_path}")

            print(f" 实际模型路径: {actual_model_path}")
            # 更新self.actual_model_path为实际路径，供后续使用
            self.actual_model_path = actual_model_path
        else:
            # 直接使用pth文件路径
            self.actual_model_path = self.model_path

        print(f" 加载模型检查点: {self.actual_model_path}")
        checkpoint = torch.load(self.actual_model_path, map_location='cpu', weights_only=False)

        return checkpoint

    def _load_model_and_tokenizer(self) -> Tuple[nn.Module, any]:
        """重建并加载模型和分词器"""
        hyperparams = self.model_info['hyperparameters']

        # 加载编码器
        base_encoder, tokenizer, _ = load_pretrained_encoder(hyperparams['checkpoint_path'])
        if base_encoder is None:
            raise ValueError(f"无法加载基础编码器: {hyperparams['checkpoint_path']}")

        # 重建监督模型结构
        num_labels = len(set(hyperparams.get('label_to_id', {}).values())) if 'label_to_id' in hyperparams else 6

        model = SupervisedModel(
            base_encoder=base_encoder,
            num_labels=num_labels,
            classifier_type=hyperparams.get('classifier_type', 'linear'),
            mlp_hidden_neurons=hyperparams.get('mlp_hidden_neurons', 384)
        )

        # 直接使用非严格模式加载权重，兼容不同的模型结构
        print(" 加载模型权重（非严格模式）...")
        missing_keys, unexpected_keys = model.load_state_dict(self.model_info['model_state_dict'], strict=False)

        # 检查关键的分类器权重是否正确加载
        classifier_keys = [k for k in self.model_info['model_state_dict'].keys() if 'classifier' in k]
        if classifier_keys:
            print(f" 分类器权重已加载: {len(classifier_keys)} 个参数")
        else:
            raise ValueError(" 未找到分类器权重，模型无法正常工作")

        # 检查编码器权重加载情况
        encoder_keys_needed = [k for k in missing_keys if 'base_encoder' in k]
        if len(encoder_keys_needed) > 0:
            print(f" 编码器使用预训练权重 ({len(encoder_keys_needed)} 个参数)，分类器使用训练权重")
        else:
            print(f" 完整模型权重已加载")

        if unexpected_keys:
            print(f"ℹ  跳过了 {len(unexpected_keys)} 个不匹配的参数")

        model.to(self.device)
        model.eval()

        print(f" 模型已加载到 {self.device}")

        return model, tokenizer

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测文本的标签概率

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            (logits, probabilities): logits和概率分布
        """
        if not texts:
            return np.array([]), np.array([])

        all_logits = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="预测中", leave=False):
                batch_texts = texts[i:i + batch_size]

                # 分词
                if hasattr(self.tokenizer, '__call__'):
                    # HuggingFace tokenizer
                    encodings = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=256
                    )
                    input_ids = encodings['input_ids'].to(self.device)
                    attention_mask = encodings['attention_mask'].to(self.device)

                    # 预测
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    # TextCNN tokenizer - 需要适配
                    # 这里假设使用SupervisedTextDataset的逻辑
                    raise NotImplementedError("TextCNN tokenizer support not implemented yet")

                all_logits.append(logits.cpu())

        if all_logits:
            logits_concat = torch.cat(all_logits, dim=0).numpy()
            probabilities = F.softmax(torch.from_numpy(logits_concat), dim=1).numpy()
            return logits_concat, probabilities
        else:
            return np.array([]), np.array([])

    def calculate_pair_consistency(self, anchor_text: str, positive_text: str,
                                 method: str = 'confidence_weighted_dot_product') -> float:
        """
        计算一对文本的一致性得分

        Args:
            anchor_text: 锚点文本
            positive_text: 正样本文本
            method: 一致性计算方法 ('confidence_weighted_dot_product', 'kl_divergence', 'cosine_similarity', 'prediction_agreement')

        Returns:
            一致性得分（0-1之间，1表示完全一致）
        """
        # 预测两个文本的概率分布
        _, anchor_probs = self.predict_batch([anchor_text])
        _, positive_probs = self.predict_batch([positive_text])

        if anchor_probs.size == 0 or positive_probs.size == 0:
            return 0.0

        anchor_prob = anchor_probs[0]
        positive_prob = positive_probs[0]

        if method == 'confidence_weighted_dot_product':
            # 使用置信度加权点积方法: Q = (p^T × q) × (1 - H(p)/log K) × (1 - H(q)/log K)
            # 计算点积
            dot_product = np.dot(anchor_prob, positive_prob)

            # 计算类别数量
            K = len(anchor_prob)
            log_K = np.log(K)

            # 计算归一化熵
            epsilon = 1e-8  # 避免log(0)
            anchor_prob_safe = anchor_prob + epsilon
            positive_prob_safe = positive_prob + epsilon

            # H(p) = -Σ(p_i × log p_i)
            entropy_anchor = -np.sum(anchor_prob_safe * np.log(anchor_prob_safe))
            entropy_positive = -np.sum(positive_prob_safe * np.log(positive_prob_safe))

            # 归一化熵: H(p) / log K
            normalized_entropy_anchor = entropy_anchor / log_K
            normalized_entropy_positive = entropy_positive / log_K

            # 置信度系数: 1 - 归一化熵
            confidence_anchor = 1 - normalized_entropy_anchor
            confidence_positive = 1 - normalized_entropy_positive

            # 最终一致性得分
            consistency = dot_product * confidence_anchor * confidence_positive

        elif method == 'kl_divergence':
            # 使用KL散度计算一致性（越小越一致）
            # 添加小常数避免log(0)
            epsilon = 1e-8
            anchor_prob = anchor_prob + epsilon
            positive_prob = positive_prob + epsilon

            # 计算双向KL散度的平均值
            kl1 = np.sum(anchor_prob * np.log(anchor_prob / positive_prob))
            kl2 = np.sum(positive_prob * np.log(positive_prob / anchor_prob))
            avg_kl = (kl1 + kl2) / 2

            # 转换为0-1之间的一致性得分（KL越小，一致性越高）
            consistency = np.exp(-avg_kl)

        elif method == 'cosine_similarity':
            # 使用余弦相似度
            dot_product = np.dot(anchor_prob, positive_prob)
            norm_anchor = np.linalg.norm(anchor_prob)
            norm_positive = np.linalg.norm(positive_prob)

            if norm_anchor == 0 or norm_positive == 0:
                consistency = 0.0
            else:
                consistency = dot_product / (norm_anchor * norm_positive)
                # 转换到0-1范围
                consistency = (consistency + 1) / 2

        elif method == 'prediction_agreement':
            # 预测标签是否一致
            anchor_pred = np.argmax(anchor_prob)
            positive_pred = np.argmax(positive_prob)

            if anchor_pred == positive_pred:
                # 如果预测一致，使用概率相似度作为置信度
                anchor_conf = anchor_prob[anchor_pred]
                positive_conf = positive_prob[positive_pred]
                consistency = min(anchor_conf, positive_conf)
            else:
                consistency = 0.0

        else:
            raise ValueError(f"不支持的一致性计算方法: {method}")

        return float(consistency)

    def _find_corresponding_contrastive_model(self) -> Optional[str]:
        """查找对应的对比学习模型路径"""
        try:
            # 检查模型信息中是否包含checkpoint_path
            hyperparams = self.model_info.get('hyperparameters', {})
            checkpoint_path = hyperparams.get('checkpoint_path')

            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f" 找到对应的对比学习模型: {checkpoint_path}")
                return checkpoint_path

            # 如果没有找到，尝试自动发现最新的对比学习模型
            model_dirs = []
            if os.path.exists("model"):
                for root, dirs, files in os.walk("model"):
                    for file in files:
                        if file == 'best_contrastive_model.pth':
                            model_dirs.append(os.path.join(root, file))

            if model_dirs:
                # 按修改时间排序，返回最新的
                latest_model = max(model_dirs, key=os.path.getmtime)
                print(f" 推荐最新的对比学习模型: {latest_model}")
                return latest_model

        except Exception as e:
            print(f" 查找对比学习模型时出错: {e}")

        return None

    def score_dataset_pairs(self, dataset_path: str, method: str = 'confidence_weighted_dot_product',
                           max_pairs: Optional[int] = None, batch_size: int = 32,
                           save_enhanced_dataset: bool = True) -> Dict:
        """
        为数据集中的所有正样本对计算一致性得分，并保存增强数据集

        Args:
            dataset_path: 数据集文件路径（.pkl）
            method: 一致性计算方法
            max_pairs: 最大处理对数（用于测试）
            batch_size: 批处理大小
            save_enhanced_dataset: 是否保存增强的数据集

        Returns:
            包含一致性得分的字典
        """
        print(f" 开始为数据集计算一致性得分: {dataset_path}")
        print(f"   方法: {method}")
        print(f"   最大对数: {max_pairs if max_pairs else '全部'}")

        # 加载数据集
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")

        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        # 检查数据集格式并提取实际样本
        if isinstance(dataset, dict) and 'samples' in dataset:
            # 增强数据集格式，包含元数据
            actual_samples = dataset['samples']
            print(f" 检测到增强数据集格式")
        else:
            # 普通数据集格式
            actual_samples = dataset

        # 提取所有正样本对
        pairs = []
        pair_info = []

        dataset_size = len(actual_samples)
        process_size = min(dataset_size, max_pairs) if max_pairs else dataset_size

        print(f"   数据集大小: {dataset_size}")
        print(f"   处理大小: {process_size}")

        for i in tqdm(range(process_size), desc="提取样本对"):
            # 根据数据格式获取样本
            if isinstance(actual_samples, list):
                sample = actual_samples[i]
            elif isinstance(actual_samples, dict):
                # 如果是字典，尝试使用索引作为键
                if i in actual_samples:
                    sample = actual_samples[i]
                else:
                    # 尝试将字典值转为列表
                    sample_list = list(actual_samples.values())
                    if i < len(sample_list):
                        sample = sample_list[i]
                    else:
                        continue
            else:
                sample = actual_samples[i]

            # 检查数据集结构，适配不同的键名
            if isinstance(dataset, ContrastiveDataset1):
                # Dataset1: 父评论-子评论对
                # 检查可能的键名变体
                anchor_text = None
                positive_text = None

                # 尝试不同的键名
                for anchor_key in ['anchor_content', 'anchor_text', 'anchor_texts', 'anchor', 'text']:
                    if anchor_key in sample:
                        anchor_text = sample[anchor_key]
                        break

                for positive_key in ['positive_content', 'positive_text_ds1', 'positive_text', 'positive_texts', 'positive']:
                    if positive_key in sample:
                        positive_text = sample[positive_key]
                        break

                if anchor_text is None or positive_text is None:
                    # 打印样本结构以便调试
                    if i == 0:  # 只在第一个样本时打印
                        print(f"   调试：样本键名: {list(sample.keys())}")
                        print(f"   调试：样本类型: {type(sample)}")
                    continue

                if positive_text is not None:
                    pairs.append((anchor_text, positive_text))
                    pair_info.append({
                        'dataset_type': 'dataset1',
                        'sample_index': i,
                        'anchor_text': anchor_text[:100] + '...' if len(anchor_text) > 100 else anchor_text,
                        'positive_text': positive_text[:100] + '...' if len(positive_text) > 100 else positive_text,
                    })

            elif isinstance(dataset, ContrastiveDataset2):
                # Dataset2: 节点-子树中心对
                anchor_text = None
                positive_content_list = None

                # 尝试不同的键名
                for anchor_key in ['anchor_content', 'anchor_text', 'anchor_texts', 'anchor', 'text']:
                    if anchor_key in sample:
                        anchor_text = sample[anchor_key]
                        break

                for positive_key in ['positive_content', 'positive_content_lists_ds2', 'positive_content_lists', 'positive_texts', 'positive']:
                    if positive_key in sample:
                        positive_content_list = sample[positive_key][0] if sample[positive_key] else []
                        break

                if anchor_text is None:
                    if i == 0:  # 只在第一个样本时打印
                        print(f"   调试：样本键名: {list(sample.keys())}")
                        print(f"   调试：样本类型: {type(sample)}")
                    continue

                if positive_content_list:
                    # 将子树内容合并为一个文本
                    positive_text = ' '.join([str(content) for content in positive_content_list if content])

                    if positive_text.strip():
                        pairs.append((anchor_text, positive_text))
                        pair_info.append({
                            'dataset_type': 'dataset2',
                            'sample_index': i,
                            'anchor_text': anchor_text[:100] + '...' if len(anchor_text) > 100 else anchor_text,
                            'positive_text': positive_text[:100] + '...' if len(positive_text) > 100 else positive_text,
                            'subtree_size': len(positive_content_list)
                        })
            else:
                # 通用处理：尝试直接从样本中提取
                if i == 0:  # 只在第一个样本时打印调试信息
                    print(f"   调试：未知数据集类型，样本键名: {list(sample.keys())}")
                    print(f"   调试：样本类型: {type(sample)}")

                # 尝试通用键名
                anchor_text = None
                positive_text = None

                for anchor_key in ['anchor_content', 'anchor_text', 'anchor_texts', 'anchor', 'text', 'input_text']:
                    if anchor_key in sample:
                        anchor_text = sample[anchor_key]
                        break

                for positive_key in ['positive_content', 'positive_text', 'positive_texts', 'positive', 'target_text']:
                    if positive_key in sample:
                        positive_text = sample[positive_key]
                        break

                if anchor_text is not None and positive_text is not None:
                    pairs.append((anchor_text, positive_text))
                    pair_info.append({
                        'dataset_type': 'unknown',
                        'sample_index': i,
                        'anchor_text': anchor_text[:100] + '...' if len(anchor_text) > 100 else anchor_text,
                        'positive_text': positive_text[:100] + '...' if len(positive_text) > 100 else positive_text,
                    })

        print(f" 提取到 {len(pairs)} 个有效样本对")

        if not pairs:
            print(" 没有找到有效的样本对")
            return {
                'consistency_scores': [],
                'pair_info': [],
                'statistics': {},
                'method': method,
                'dataset_path': dataset_path
            }

        # 批量计算一致性得分
        consistency_scores = []
        anchor_probs_all = None
        positive_probs_all = None

        print(f" 计算一致性得分...")

        if method in ['confidence_weighted_dot_product', 'kl_divergence', 'cosine_similarity']:
            # 批量处理方法
            all_anchor_texts = [pair[0] for pair in pairs]
            all_positive_texts = [pair[1] for pair in pairs]

            print("   预测锚点文本...")
            _, anchor_probs_all = self.predict_batch(all_anchor_texts, batch_size)

            print("   预测正样本文本...")
            _, positive_probs_all = self.predict_batch(all_positive_texts, batch_size)

            print("   计算一致性得分...")
            for i in tqdm(range(len(pairs)), desc="计算一致性"):
                if i < len(anchor_probs_all) and i < len(positive_probs_all):
                    anchor_prob = anchor_probs_all[i]
                    positive_prob = positive_probs_all[i]

                    if method == 'confidence_weighted_dot_product':
                        # 置信度加权点积方法
                        dot_product = np.dot(anchor_prob, positive_prob)

                        K = len(anchor_prob)
                        log_K = np.log(K)

                        epsilon = 1e-8
                        anchor_prob_safe = anchor_prob + epsilon
                        positive_prob_safe = positive_prob + epsilon

                        entropy_anchor = -np.sum(anchor_prob_safe * np.log(anchor_prob_safe))
                        entropy_positive = -np.sum(positive_prob_safe * np.log(positive_prob_safe))

                        normalized_entropy_anchor = entropy_anchor / log_K
                        normalized_entropy_positive = entropy_positive / log_K

                        confidence_anchor = 1 - normalized_entropy_anchor
                        confidence_positive = 1 - normalized_entropy_positive

                        consistency = dot_product * confidence_anchor * confidence_positive

                    elif method == 'kl_divergence':
                        epsilon = 1e-8
                        anchor_prob = anchor_prob + epsilon
                        positive_prob = positive_prob + epsilon

                        kl1 = np.sum(anchor_prob * np.log(anchor_prob / positive_prob))
                        kl2 = np.sum(positive_prob * np.log(positive_prob / anchor_prob))
                        avg_kl = (kl1 + kl2) / 2
                        consistency = np.exp(-avg_kl)

                    elif method == 'cosine_similarity':
                        dot_product = np.dot(anchor_prob, positive_prob)
                        norm_anchor = np.linalg.norm(anchor_prob)
                        norm_positive = np.linalg.norm(positive_prob)

                        if norm_anchor == 0 or norm_positive == 0:
                            consistency = 0.0
                        else:
                            consistency = dot_product / (norm_anchor * norm_positive)
                            consistency = (consistency + 1) / 2

                    consistency_scores.append(float(consistency))
                    #  新增：将一致性得分添加到pair_info中
                    if i < len(pair_info):
                        pair_info[i]['consistency_score'] = float(consistency)
                else:
                    consistency_scores.append(0.0)
                    #  新增：为无效样本也添加得分
                    if i < len(pair_info):
                        pair_info[i]['consistency_score'] = 0.0

        else:
            # 逐对处理方法
            for i, (anchor_text, positive_text) in enumerate(tqdm(pairs, desc="计算一致性")):
                consistency = self.calculate_pair_consistency(anchor_text, positive_text, method)
                consistency_scores.append(consistency)
                #  新增：将一致性得分添加到pair_info中
                if i < len(pair_info):
                    pair_info[i]['consistency_score'] = float(consistency)

        # 计算统计信息
        scores_array = np.array(consistency_scores)
        statistics = {
            'count': len(consistency_scores),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'percentiles': {
                '25': float(np.percentile(scores_array, 25)),
                '75': float(np.percentile(scores_array, 75)),
                '90': float(np.percentile(scores_array, 90)),
                '95': float(np.percentile(scores_array, 95))
            }
        }

        print(f" 一致性得分统计:")
        print(f"   平均值: {statistics['mean']:.4f}")
        print(f"   标准差: {statistics['std']:.4f}")
        print(f"   范围: [{statistics['min']:.4f}, {statistics['max']:.4f}]")
        print(f"   中位数: {statistics['median']:.4f}")

        #  新增：保存增强的数据集 - 传递正确的输出目录
        enhanced_dataset_path = None
        if save_enhanced_dataset and pairs:
            # 获取当前输出目录（将在main函数中设置）
            output_dir = getattr(self, '_current_output_dir', '.')
            enhanced_dataset_path = self._save_enhanced_dataset(
                original_dataset_path=dataset_path,
                pairs=pairs,
                consistency_scores=consistency_scores,
                anchor_probs=anchor_probs_all,
                positive_probs=positive_probs_all,
                method=method,
                output_dir=output_dir
            )

        return {
            'consistency_scores': consistency_scores,
            'pair_info': pair_info,
            'statistics': statistics,
            'method': method,
            'dataset_path': dataset_path,
            'enhanced_dataset_path': enhanced_dataset_path,  #  新增
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat(),
            # 移除不可靠的推荐模型逻辑
        }

    def export_samples_for_review(self, results: Dict, output_dir: str, samples_per_interval: int = 50) -> str:
        """
        按一致性得分区间采样并导出CSV文件供人工review

        Args:
            results: 一致性评分结果
            output_dir: 输出目录
            samples_per_interval: 每个区间的样本数量

        Returns:
            CSV文件路径
        """
        if not results['pair_info'] or not results['consistency_scores']:
            print(" 没有可用的样本数据")
            return None

        import pandas as pd

        pair_info = results['pair_info']
        scores = np.array(results['consistency_scores'])

        print(f" 准备导出人工review样本...")
        print(f"   总样本数: {len(pair_info)}")
        print(f"   得分范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

        # 定义得分区间
        intervals = [
            (0.0, 0.2, "很低一致性"),
            (0.2, 0.4, "低一致性"),
            (0.4, 0.6, "中等一致性"),
            (0.6, 0.8, "高一致性"),
            (0.8, 1.0, "很高一致性")
        ]

        review_samples = []

        for min_score, max_score, category in intervals:
            # 找到在当前区间的样本
            in_interval = []
            for i, (info, score) in enumerate(zip(pair_info, scores)):
                if min_score <= score < max_score or (max_score == 1.0 and score == 1.0):
                    in_interval.append((i, info, score))

            print(f"   {category} [{min_score:.1f}-{max_score:.1f}]: {len(in_interval)} 个样本")

            if len(in_interval) == 0:
                continue

            # 采样
            if len(in_interval) <= samples_per_interval:
                sampled = in_interval
            else:
                # 随机采样
                import random
                random.seed(42)  # 固定种子保证可重现
                sampled = random.sample(in_interval, samples_per_interval)

            # 添加到review列表
            for idx, info, score in sampled:
                review_sample = {
                    '样本编号': idx,
                    '一致性得分': f"{score:.4f}",
                    '得分区间': category,
                    '数据集类型': info['dataset_type'],
                    '原始索引': info['sample_index'],
                    '锚点文本': info.get('anchor_text', ''),
                    '正样本文本': info.get('positive_text', ''),
                    '人工评分': '',  # 空列供人工填写
                    '备注': ''      # 空列供人工填写
                }

                # 如果是dataset2，添加子树大小信息
                if info.get('subtree_size'):
                    review_sample['子树大小'] = info['subtree_size']

                review_samples.append(review_sample)

        if not review_samples:
            print(" 没有可导出的样本")
            return None

        # 创建DataFrame并导出CSV
        df_review = pd.DataFrame(review_samples)

        # 按得分排序，便于review
        df_review = df_review.sort_values('一致性得分', ascending=False)

        # 生成文件名
        model_name = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_path else "unknown"
        dataset_name = os.path.splitext(os.path.basename(results['dataset_path']))[0] if results['dataset_path'] else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_filename = f"review_samples_{model_name}_{dataset_name}_{results['method']}_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        os.makedirs(output_dir, exist_ok=True)
        df_review.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig确保Excel正确显示中文

        print(f" 人工review样本已导出: {csv_path}")
        print(f"   总导出样本: {len(review_samples)} 个")
        print(f"   各区间分布:")
        for category in df_review['得分区间'].value_counts().index:
            count = df_review['得分区间'].value_counts()[category]
            print(f"     {category}: {count} 个")

        return csv_path

    def _save_enhanced_dataset(self, original_dataset_path: str, pairs: List[Tuple],
                              consistency_scores: List[float], anchor_probs: np.ndarray,
                              positive_probs: np.ndarray, method: str, output_dir: str) -> str:
        """
        保存增强的数据集，包含一致性得分和概率分布，并保留训练历史信息

        Args:
            original_dataset_path: 原始数据集路径
            pairs: 样本对列表
            consistency_scores: 一致性得分
            anchor_probs: 锚点概率分布
            positive_probs: 正样本概率分布
            method: 一致性计算方法
            output_dir: 输出目录

        Returns:
            增强数据集的保存路径
        """
        print(f" 保存增强数据集...")

        # 尝试加载原始数据集以检查是否包含训练历史信息
        original_samples = None
        original_metadata = None

        try:
            with open(original_dataset_path, 'rb') as f:
                original_data = pickle.load(f)

            # 检查是否是包含训练历史的格式
            if isinstance(original_data, dict) and 'samples' in original_data:
                print(" 检测到包含训练历史的数据集格式")
                original_samples = original_data['samples']
                original_metadata = {
                    'round_info': original_data.get('round_info', {}),
                    'metadata': original_data.get('metadata', {})
                }
            elif isinstance(original_data, list):
                print(" 检测到简单数据集格式")
                original_samples = original_data

        except Exception as e:
            print(f" 加载原始数据集时出错: {e}")

        # 创建增强数据集
        enhanced_samples = []

        for i, ((anchor_text, positive_text), score) in enumerate(zip(pairs, consistency_scores)):
            enhanced_sample = {
                'anchor_content': anchor_text,
                'positive_content': positive_text,
                'consistency_score': score,
                'sample_index': i,
                'pair_type': 'enhanced_pair'
            }

            # 添加概率分布（如果可用）
            if anchor_probs is not None and i < len(anchor_probs):
                enhanced_sample['anchor_probs'] = anchor_probs[i].tolist()

            if positive_probs is not None and i < len(positive_probs):
                enhanced_sample['positive_probs'] = positive_probs[i].tolist()

            # 保留原始样本的训练历史信息（如果存在）
            if original_samples and i < len(original_samples):
                original_sample = original_samples[i]
                if isinstance(original_sample, dict):
                    # 保留训练历史字段
                    for key in ['training_rounds', 'filter_history', 'last_trained_round']:
                        if key in original_sample:
                            enhanced_sample[key] = original_sample[key]

                    # 保留其他元数据
                    for key in ['sample_id', 'created_time', 'source_dataset']:
                        if key in original_sample:
                            enhanced_sample[key] = original_sample[key]

            enhanced_samples.append(enhanced_sample)

        #  修改：增强数据集保存在同一输出目录下
        enhanced_filename = f"enhanced_dataset_{method}.pkl"
        enhanced_dataset_path = os.path.join(output_dir, enhanced_filename)

        # 准备保存的数据结构
        if original_metadata:
            # 如果原始数据包含元数据，保持相同格式
            save_data = {
                'samples': enhanced_samples,
                'round_info': original_metadata.get('round_info', {}),
                'metadata': {
                    **original_metadata.get('metadata', {}),
                    'consistency_scoring': {
                        'method': method,
                        'timestamp': datetime.now().isoformat(),
                        'original_dataset': original_dataset_path,
                        'total_pairs': len(enhanced_samples)
                    }
                }
            }
        else:
            # 如果原始数据是简单格式，也升级为带元数据的格式
            save_data = {
                'samples': enhanced_samples,
                'metadata': {
                    'consistency_scoring': {
                        'method': method,
                        'timestamp': datetime.now().isoformat(),
                        'original_dataset': original_dataset_path,
                        'total_pairs': len(enhanced_samples)
                    }
                }
            }

        # 保存增强数据集
        with open(enhanced_dataset_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f" 增强数据集已保存: {enhanced_dataset_path}")
        print(f"   包含样本数: {len(enhanced_samples)}")
        print(f"   每个样本包含: anchor_content, positive_content, consistency_score, anchor_probs, positive_probs")

        return enhanced_dataset_path

def save_consistency_results(results: Dict, output_path: str, scorer_instance=None,
                           export_review: bool = True, review_samples: int = 50):
    """保存一致性评分结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f" 一致性评分结果已保存到: {output_path}")

    #  新增：自动导出review样本
    if export_review and scorer_instance:
        try:
            output_dir = os.path.dirname(output_path)
            csv_path = scorer_instance.export_samples_for_review(results, output_dir, review_samples)
            if csv_path:
                print(f" review样本CSV已生成: {csv_path}")
        except Exception as e:
            print(f" 导出review样本时出错: {e}")

def create_experiment_output_dir(model_path: str, dataset_path: str, method: str,
                               base_output_dir: str = "consistency_scores") -> str:
    """
    根据实验参数创建专门的输出目录

    Args:
        model_path: 模型路径
        dataset_path: 数据集路径
        method: 一致性计算方法
        base_output_dir: 基础输出目录

    Returns:
        实验专用输出目录路径
    """
    import re

    #  修改：从完整路径中提取关键信息

    # 1. 从模型路径中提取关键信息
    # 例如: sup_result_hyperparams/iter_mlp_experiment/saved_models/lora_bert_base_chinese_cl_frac0.1_seed123/best_model.pth
    # 提取: lora_bert_base_chinese_cl_frac0.1_seed123
    model_key_info = "unknown_model"
    if model_path:
        # 获取倒数第二级目录名（模型保存目录名）
        path_parts = model_path.replace('\\', '/').split('/')
        if len(path_parts) >= 2:
            model_key_info = path_parts[-2]  # 获取模型目录名
        else:
            model_key_info = os.path.splitext(os.path.basename(model_path))[0]

    # 2. 从数据集路径中提取关键信息
    # 例如: cl_dataset\google-bert_bert-base-chinese_sim_0.75_dataset1.pkl
    # 提取: sim_0.75
    dataset_key_info = "unknown_dataset"
    if dataset_path:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        # 提取sim信息
        sim_match = re.search(r'sim_([\d\.]+)', dataset_name)
        if sim_match:
            dataset_key_info = f"sim_{sim_match.group(1)}"
        else:
            dataset_key_info = dataset_name

    # 3. 从模型信息中提取数据比例
    data_fraction = "unknown"
    if "_frac" in model_key_info:
        frac_match = re.search(r'_frac([\d\.]+)', model_key_info)
        if frac_match:
            data_fraction = f"frac{frac_match.group(1)}"

    # 创建实验目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{data_fraction}_{model_key_info}_{dataset_key_info}_{method}_{timestamp}"

    # 创建完整路径
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f" 实验输出目录: {experiment_dir}")
    print(f"   数据比例: {data_fraction}")
    print(f"   模型信息: {model_key_info}")
    print(f"   数据集信息: {dataset_key_info}")
    return experiment_dir

def discover_datasets(base_dir: str = "cl_dataset") -> List[str]:
    """发现可用的数据集文件"""
    if not os.path.exists(base_dir):
        return []

    datasets = []
    for file in os.listdir(base_dir):
        if file.endswith('.pkl'):
            datasets.append(os.path.join(base_dir, file))

    return sorted(datasets)

def discover_classifiers(base_dir: str = "classifier_selections") -> List[str]:
    """发现可用的分类器"""
    if not os.path.exists(base_dir):
        return []

    classifiers = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_selected.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        model_path = data.get('model_path')
                        if model_path and os.path.exists(model_path):
                            classifiers.append({
                                'json_path': json_path,
                                'model_path': model_path,
                                'data_fraction': data.get('data_fraction'),
                                'experiment': os.path.basename(os.path.dirname(root))
                            })
                except Exception as e:
                    print(f" 读取分类器信息失败: {json_path}, 错误: {e}")

    return classifiers

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='一致性评分器 - 使用训练好的分类器评估对比学习样本对的一致性')

    parser.add_argument('--model-path', '-m', type=str, required=True,
                       help='训练好的模型路径 (.pth文件)')

    parser.add_argument('--dataset-path', '-d', type=str, required=True,
                       help='要评分的数据集路径 (.pkl文件)')

    parser.add_argument('--output-dir', '-o', type=str, default='consistency_scores',
                       help='输出目录 (默认: consistency_scores)')

    parser.add_argument('--method', type=str, default='confidence_weighted_dot_product',
                       choices=['confidence_weighted_dot_product', 'kl_divergence', 'cosine_similarity', 'prediction_agreement'],
                       help='一致性计算方法 (默认: confidence_weighted_dot_product)')

    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='批处理大小 (默认: 32)')

    parser.add_argument('--max-pairs', type=int, default=None,
                       help='最大处理对数，用于测试 (默认: 全部)')

    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (默认: auto)')

    parser.add_argument('--list-datasets', action='store_true',
                       help='列出可用的数据集')

    parser.add_argument('--list-classifiers', action='store_true',
                       help='列出可用的分类器')

    parser.add_argument('--review-samples', type=int, default=50,
                       help='每个得分区间导出的review样本数量 (默认: 50)')

    parser.add_argument('--no-review', action='store_true',
                       help='跳过导出review样本CSV文件')

    return parser.parse_args()

def main():
    """主函数"""
    print(" 一致性评分器启动...")

    args = parse_arguments()

    if args.list_datasets:
        datasets = discover_datasets()
        print(f" 发现 {len(datasets)} 个数据集:")
        for dataset in datasets:
            print(f"  • {dataset}")
        return

    if args.list_classifiers:
        classifiers = discover_classifiers()
        print(f" 发现 {len(classifiers)} 个分类器:")
        for clf in classifiers:
            print(f"  • 实验: {clf['experiment']}, 数据比例: {clf['data_fraction']}")
            print(f"    模型路径: {clf['model_path']}")
            print()
        return

    # 验证输入文件
    if not os.path.exists(args.model_path):
        print(f" 模型文件不存在: {args.model_path}")
        return

    if not os.path.exists(args.dataset_path):
        print(f" 数据集文件不存在: {args.dataset_path}")
        return

    try:
        # 初始化评分器
        scorer = ConsistencyScorer(args.model_path, args.device)

        #  新增：创建实验专用输出目录
        experiment_output_dir = create_experiment_output_dir(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            method=args.method,
            base_output_dir=args.output_dir
        )

        #  设置输出目录到scorer实例，用于保存增强数据集
        scorer._current_output_dir = experiment_output_dir

        # 计算一致性得分
        results = scorer.score_dataset_pairs(
            dataset_path=args.dataset_path,
            method=args.method,
            max_pairs=args.max_pairs,
            batch_size=args.batch_size,
            save_enhanced_dataset=True  #  启用增强数据集保存
        )

        #  修改：使用实验目录和固定文件名
        output_filename = "consistency_results.json"
        output_path = os.path.join(experiment_output_dir, output_filename)

        # 保存结果（传入scorer实例以自动生成review样本）
        save_consistency_results(
            results=results,
            output_path=output_path,
            scorer_instance=scorer,
            export_review=not args.no_review,
            review_samples=args.review_samples
        )

        print(f"\n 一致性评分完成!")
        print(f" 结果保存在: {experiment_output_dir}")
        print(f"    JSON结果: {output_filename}")
        if results['enhanced_dataset_path']:
            print(f"    增强数据集: {results['enhanced_dataset_path']}")
        if results['statistics']:
            print(f" 共处理 {results['statistics'].get('count', 0)} 个样本对")
            print(f" 平均一致性得分: {results['statistics'].get('mean', 0):.4f}")
        else:
            print(f" 没有找到有效的样本对进行评分")

    except Exception as e:
        print(f" 一致性评分过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def run_consistency_scoring_interface(classifier_path: str, dataset_path: str,
                                     config: dict, output_dir: str) -> str:
    """
    标准化接口：运行一致性评分

    Args:
        classifier_path: 分类器路径（JSON文件或pth文件）
        dataset_path: 数据集路径
        config: 一致性评分配置字典
        output_dir: 输出目录

    Returns:
        输出目录路径（包含增强数据集）
    """
    import os

    print(f" 一致性评分接口调用")
    print(f"   分类器: {classifier_path}")
    print(f"   数据集: {dataset_path}")
    print(f"   输出目录: {output_dir}")

    try:
        # 创建评分器
        scorer = ConsistencyScorer(classifier_path, device='auto')

        # 设置输出目录
        scorer._current_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 计算一致性得分
        results = scorer.score_dataset_pairs(
            dataset_path=dataset_path,
            method=config.get('method', 'confidence_weighted_dot_product'),
            max_pairs=config.get('max_pairs', None),
            batch_size=config.get('batch_size', 32),
            save_enhanced_dataset=config.get('save_enhanced_dataset', True)
        )

        # 保存结果
        output_filename = "consistency_results.json"
        output_path = os.path.join(output_dir, output_filename)

        save_consistency_results(
            results=results,
            output_path=output_path,
            scorer_instance=scorer,
            export_review=config.get('export_review', True),
            review_samples=config.get('review_samples', 50)
        )

        # 检查增强数据集
        enhanced_dataset_path = results.get('enhanced_dataset_path')
        if not enhanced_dataset_path:
            # 如果结果中没有，尝试查找
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.startswith('enhanced_dataset') and file.endswith('.pkl'):
                        enhanced_dataset_path = os.path.join(root, file)
                        break
                if enhanced_dataset_path:
                    break

        if enhanced_dataset_path:
            print(f" 一致性评分完成，增强数据集: {enhanced_dataset_path}")
        else:
            print(f" 一致性评分完成，输出目录: {output_dir}")

        return output_dir

    except Exception as e:
        print(f" 一致性评分失败: {e}")
        raise


if __name__ == "__main__":
    main()