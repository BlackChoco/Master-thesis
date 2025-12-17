import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import umap
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ModelScope imports
try:
    from modelscope import AutoModel, AutoTokenizer
except ImportError:
    print("Warning: ModelScope not installed. Using transformers as fallback.")
    from transformers import AutoModel, AutoTokenizer

# Import local modules
from cl_base_model import ContrastiveEncoder as ContrastiveModel


class ModelScopeEncoder:
    """ModelScope BERT编码器"""

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)

        print(f"Loading ModelScope model: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"ModelScope loading failed, trying transformers: {e}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)

        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch_texts = texts[i:i + batch_size]

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用CLS token作为句子表示
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


class TextCNNRandomEncoder:
    """TextCNN随机初始化编码器（用于对比baseline）"""

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)

        print(f"Loading TextCNN model (Random Init with seed=42): {model_path}")

        # 加载checkpoint（仅读取配置，不加载权重）
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # 检查checkpoint格式
        if 'config' not in checkpoint or 'student_state_dict' not in checkpoint:
            raise ValueError(f"Invalid TextCNN checkpoint format. Expected keys: config, student_state_dict")

        config = checkpoint['config']
        model_config = config['model']
        tokenizer_name = checkpoint.get('tokenizer_name', 'google-bert/bert-base-chinese')

        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Embedding dim: {model_config['embedding_dim']}")
        print(f"  Representation dim: {model_config['representation_dim']}")
        print(f"  🎲 Reinitializing with seed=42 (ignoring trained weights)")

        # 导入TextCNN模型类
        try:
            from distill.models import TextCNNStudent
        except ImportError:
            raise ImportError("Cannot import TextCNNStudent. Make sure 'distill' package is available.")

        # 加载tokenizer
        try:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # ✅ 设置随机种子42（确保可复现）
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)

        # ✅ 重建模型（随机初始化，不加载checkpoint权重）
        self.model = TextCNNStudent(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=model_config['embedding_dim'],
            filter_sizes=model_config['filter_sizes'],
            num_filters=model_config['num_filters'],
            representation_dim=model_config['representation_dim'],
            num_labels=None,  # 不需要分类头
            pad_token_id=self.tokenizer.pad_token_id,
            dropout=model_config.get('dropout', 0.1),
            use_projection_head=model_config.get('use_projection_head', True),
            projection_dim=model_config.get('projection_dim', 768)
        ).to(self.device)

        # ✅ 关键：不调用 load_state_dict，保持随机初始化状态

        self.model.eval()

        print(f"TextCNN model (Random Init) loaded successfully on {self.device}")

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本（使用随机初始化的权重）"""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding (Random)", leave=False):
            batch_texts = texts[i:i + batch_size]

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                # TextCNN forward返回 (features, projected, logits)
                features, projected, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False
                )
                # 使用representation features (不是projection head输出)
                embeddings = features.cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


class TextCNNEncoder:
    """TextCNN蒸馏模型编码器"""

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)

        print(f"Loading TextCNN model: {model_path}")

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # 检查checkpoint格式
        if 'config' not in checkpoint or 'student_state_dict' not in checkpoint:
            raise ValueError(f"Invalid TextCNN checkpoint format. Expected keys: config, student_state_dict")

        config = checkpoint['config']
        model_config = config['model']
        tokenizer_name = checkpoint.get('tokenizer_name', 'google-bert/bert-base-chinese')

        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Embedding dim: {model_config['embedding_dim']}")
        print(f"  Representation dim: {model_config['representation_dim']}")

        # 导入TextCNN模型类
        try:
            from distill.models import TextCNNStudent
        except ImportError:
            raise ImportError("Cannot import TextCNNStudent. Make sure 'distill' package is available.")

        # 加载tokenizer
        try:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 重建模型
        self.model = TextCNNStudent(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=model_config['embedding_dim'],
            filter_sizes=model_config['filter_sizes'],
            num_filters=model_config['num_filters'],
            representation_dim=model_config['representation_dim'],
            num_labels=None,  # 不需要分类头
            pad_token_id=self.tokenizer.pad_token_id,
            dropout=model_config.get('dropout', 0.1),
            use_projection_head=model_config.get('use_projection_head', True),
            projection_dim=model_config.get('projection_dim', 768)
        ).to(self.device)

        # 加载模型权重（忽略分类器相关的键）
        state_dict = checkpoint['student_state_dict']

        # 过滤掉分类器权重（因为我们创建模型时 num_labels=None）
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if not k.startswith('classifier.')}

        # 使用 strict=False 允许部分加载
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)

        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys}")

        self.model.eval()

        print(f"TextCNN model loaded successfully on {self.device}")

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch_texts = texts[i:i + batch_size]

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                # TextCNN forward返回 (features, projected, logits)
                features, projected, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False
                )
                # 使用representation features (不是projection head输出)
                embeddings = features.cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


class ContrastiveEncoder:
    """对比学习编码器"""

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)

        print(f"Loading contrastive model: {model_path}")

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # 获取基础模型路径和配置
        model_type = checkpoint.get('training_model_type', 'modelscope')
        model_identifier = checkpoint.get('training_model_identifier_or_path', 'google-bert/bert-base-chinese')
        proj_config = checkpoint.get('projection_head_config', {})

        # ✅ 新增：获取LoRA配置
        use_peft = checkpoint.get('use_peft', False)
        peft_config = checkpoint.get('peft_config', None)

        print(f"  Model type: {model_type}")
        print(f"  Model identifier: {model_identifier}")
        print(f"  Use LoRA: {use_peft}")
        if use_peft and peft_config:
            print(f"  LoRA config: r={peft_config.get('r')}, alpha={peft_config.get('lora_alpha')}, "
                  f"target_modules={peft_config.get('target_modules')}")

        # 重建模型
        if model_type == 'modelscope':
            from cl_base_model import ContrastiveEncoder as CL_ContrastiveEncoder
            self.model = CL_ContrastiveEncoder(
                model_type='modelscope',
                model_name_or_path=model_identifier,
                projection_hidden_dim=proj_config.get('hidden_dim', 512),
                projection_output_dim=proj_config.get('output_dim', 128),
                projection_dropout_rate=proj_config.get('dropout_rate', 0.1)
            ).to(self.device)

            # ✅ 新增：应用LoRA（如果checkpoint中使用了LoRA）
            if use_peft and peft_config:
                print("\n  [DEBUG] Applying LoRA configuration to base model...")
                from peft import get_peft_model, LoraConfig

                print(f"    LoRA rank (r): {peft_config.get('r', 8)}")
                print(f"    LoRA alpha: {peft_config.get('lora_alpha', 16)}")
                print(f"    LoRA dropout: {peft_config.get('lora_dropout', 0.1)}")
                print(f"    Target modules: {peft_config.get('target_modules', ['query', 'key', 'value', 'dense'])}")
                print(f"    Bias: {peft_config.get('bias', 'none')}")

                # 在应用LoRA前检查base_model结构
                print(f"\n    Before applying LoRA:")
                pre_lora_params = list(self.model.base_model.named_parameters())
                print(f"      Total parameters: {len(pre_lora_params)}")
                print(f"      Sample parameter names:")
                for name, _ in pre_lora_params[:3]:
                    print(f"        - {name}")

                lora_config = LoraConfig(
                    r=peft_config.get('r', 8),
                    lora_alpha=peft_config.get('lora_alpha', 16),
                    target_modules=peft_config.get('target_modules', ["query", "key", "value", "dense"]),
                    lora_dropout=peft_config.get('lora_dropout', 0.1),
                    bias=peft_config.get('bias', 'none'),
                    task_type=None  # 对比学习不需要task_type
                )

                # 对base_model应用LoRA
                self.model.base_model = get_peft_model(self.model.base_model, lora_config)

                print(f"\n    After applying LoRA:")
                post_lora_params = list(self.model.base_model.named_parameters())
                lora_added_params = [name for name, _ in post_lora_params if 'lora' in name.lower()]
                print(f"      Total parameters: {len(post_lora_params)}")
                print(f"      LoRA parameters added: {len(lora_added_params)}")
                print(f"      Sample LoRA parameter names:")
                for name in lora_added_params[:5]:
                    print(f"        - {name}")

                trainable_params = sum(p.numel() for p in self.model.base_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.base_model.parameters())
                print(f"\n    ✓ LoRA applied to base_model")
                print(f"      Trainable parameters: {trainable_params:,}")
                print(f"      Total parameters: {total_params:,}")
                print(f"      Trainable %: {100 * trainable_params / total_params:.2f}%")

            # 获取tokenizer
            try:
                from cl_base_model import load_tokenizer_from_modelscope
                self.tokenizer = load_tokenizer_from_modelscope(model_identifier)
            except:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # ✅ 在加载权重前，检查checkpoint中的键
        print("\n  [DEBUG] Analyzing checkpoint state_dict keys...")
        checkpoint_keys = list(checkpoint['contrastive_encoder_state_dict'].keys())
        lora_keys_in_checkpoint = [k for k in checkpoint_keys if 'lora' in k.lower()]
        print(f"  Total keys in checkpoint: {len(checkpoint_keys)}")
        print(f"  LoRA-related keys in checkpoint: {len(lora_keys_in_checkpoint)}")
        if lora_keys_in_checkpoint:
            print(f"    Sample LoRA keys from checkpoint:")
            for key in lora_keys_in_checkpoint[:5]:
                print(f"      - {key}")
        else:
            print(f"    ❌ WARNING: No LoRA keys found in checkpoint!")
            print(f"    Sample checkpoint keys:")
            for key in checkpoint_keys[:10]:
                print(f"      - {key}")

        # ✅ 检查当前模型结构中的键
        print("\n  [DEBUG] Analyzing model structure...")
        model_keys = [name for name, _ in self.model.named_parameters()]
        lora_keys_in_model = [k for k in model_keys if 'lora' in k.lower()]
        print(f"  Total parameters in model: {len(model_keys)}")
        print(f"  LoRA-related parameters in model: {len(lora_keys_in_model)}")
        if lora_keys_in_model:
            print(f"    Sample LoRA params in model:")
            for key in lora_keys_in_model[:5]:
                print(f"      - {key}")
        else:
            print(f"    ❌ WARNING: No LoRA parameters in model structure!")
            print(f"    Sample model parameters:")
            for key in model_keys[:10]:
                print(f"      - {key}")

        # 加载模型权重
        print("\n  Loading model weights from checkpoint...")
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['contrastive_encoder_state_dict'], strict=False)

        # ✅ 新增：打印加载统计信息
        print(f"  Missing keys: {len(missing_keys)}")
        if missing_keys and len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"    - {key}")
        elif missing_keys:
            print(f"    First 10 missing keys:")
            for key in missing_keys[:10]:
                print(f"    - {key}")

        print(f"  Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys and len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"    - {key}")
        elif unexpected_keys:
            print(f"    First 10 unexpected keys:")
            for key in unexpected_keys[:10]:
                print(f"    - {key}")

        if not missing_keys and not unexpected_keys:
            print(f"  ✓ All keys matched perfectly!")

        # ✅ 验证LoRA权重是否真的加载并验证数值
        if use_peft:
            print("\n  [DEBUG] Verifying LoRA weights after loading...")
            lora_params = [(name, param) for name, param in self.model.base_model.named_parameters() if 'lora_' in name.lower()]
            if lora_params:
                print(f"  ✓ Detected {len(lora_params)} LoRA parameters in loaded model")
                # 打印前3个LoRA参数的统计信息
                for i, (name, param) in enumerate(lora_params[:3]):
                    print(f"    [{i+1}] {name}")
                    print(f"        Shape: {param.shape}, Mean: {param.data.mean().item():.6f}, Std: {param.data.std().item():.6f}")
                    print(f"        Min: {param.data.min().item():.6f}, Max: {param.data.max().item():.6f}")

                # 检查是否所有LoRA参数都是零（表示未正确加载）
                all_zero = all(param.abs().max().item() < 1e-8 for _, param in lora_params)
                if all_zero:
                    print(f"  ❌ CRITICAL WARNING: All LoRA parameters are near zero! Weights not loaded correctly!")
                else:
                    print(f"  ✓ LoRA parameters contain non-zero values (weights loaded successfully)")
            else:
                print(f"  ❌ WARNING: No LoRA parameters found in model after loading!")

        self.model.eval()

        print(f"\nContrastive model loaded successfully on {self.device}")
        print("=" * 70)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch_texts = texts[i:i + batch_size]

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                # 获取base model输出（不是投影层输出）
                if hasattr(self.model, 'base_model'):
                    outputs = self.model.base_model(**inputs)
                    # 使用CLS token
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    # Fallback: 使用完整模型输出
                    embeddings = self.model(batch_texts).cpu().numpy()

                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


def load_encoder(encoder_info: Dict, device: str = 'auto'):
    """统一编码器加载接口"""
    if encoder_info['type'] == 'modelscope':
        return ModelScopeEncoder(encoder_info['path'], device)
    elif encoder_info['type'] == 'contrastive':
        return ContrastiveEncoder(encoder_info['path'], device)
    elif encoder_info['type'] == 'textcnn':
        return TextCNNEncoder(encoder_info['path'], device)
    elif encoder_info['type'] == 'textcnn_random':
        return TextCNNRandomEncoder(encoder_info['path'], device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_info['type']}")


def discover_encoder_paths(experiment_path: str, baseline_models: List[str], baseline_names: List[str] = None) -> Dict:
    """发现实验中的所有编码器路径"""
    encoders = {}

    # 处理多个Baseline模型
    if baseline_names is None:
        # 自动生成baseline名称
        baseline_names = []
        for i, model in enumerate(baseline_models):
            # 检查是否是文件路径
            if os.path.exists(model) and model.endswith('.pth'):
                # 从文件路径提取名称
                model_short = os.path.basename(model).replace('.pth', '').replace('-', '_')
            else:
                # 从模型路径提取简短名称
                model_short = model.split('/')[-1].replace('-', '_')
            baseline_names.append(f'baseline_{model_short}' if len(baseline_models) > 1 else 'baseline')

    # 确保名称数量匹配
    if len(baseline_names) != len(baseline_models):
        print(f"Warning: Number of baseline names ({len(baseline_names)}) doesn't match models ({len(baseline_models)})")
        # 补齐或截断名称列表
        while len(baseline_names) < len(baseline_models):
            baseline_names.append(f'baseline_{len(baseline_names)}')
        baseline_names = baseline_names[:len(baseline_models)]

    # 添加所有baseline模型
    for name, model_path in zip(baseline_names, baseline_models):
        # 判断是本地文件还是ModelScope/HuggingFace模型
        if os.path.exists(model_path) and model_path.endswith(('.pth', '.pt')):
            # 本地模型文件，需要检测类型
            # 尝试读取checkpoint判断是TextCNN还是Contrastive模型
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # 检查是否是TextCNN模型
                if 'config' in checkpoint and 'student_state_dict' in checkpoint:
                    encoder_type = 'textcnn'
                    description = f'Baseline (TextCNN): {os.path.basename(model_path)}'

                    # ✅ 添加训练后的TextCNN模型
                    encoders[name] = {
                        'type': encoder_type,
                        'path': model_path,
                        'description': description,
                        'is_baseline': True
                    }

                    # ✅ 自动创建随机初始化版本
                    random_name = f'{name}_random'
                    encoders[random_name] = {
                        'type': 'textcnn_random',
                        'path': model_path,  # 使用相同的checkpoint路径（仅读取配置）
                        'description': f'Random Init TextCNN (seed=42): {os.path.basename(model_path)}',
                        'is_baseline': True
                    }
                    print(f"  ✅ Auto-created random baseline: {random_name}")

                # 检查是否是Contrastive模型
                elif 'contrastive_encoder_state_dict' in checkpoint:
                    encoder_type = 'contrastive'
                    description = f'Baseline (Contrastive): {os.path.basename(model_path)}'
                    encoders[name] = {
                        'type': encoder_type,
                        'path': model_path,
                        'description': description,
                        'is_baseline': True
                    }
                else:
                    print(f"Warning: Unknown checkpoint format for {model_path}, treating as contrastive")
                    encoder_type = 'contrastive'
                    description = f'Baseline (Unknown): {os.path.basename(model_path)}'
                    encoders[name] = {
                        'type': encoder_type,
                        'path': model_path,
                        'description': description,
                        'is_baseline': True
                    }
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {model_path}: {e}")
                print(f"  Treating as contrastive model")
                encoders[name] = {
                    'type': 'contrastive',
                    'path': model_path,
                    'description': f'Baseline (Local): {os.path.basename(model_path)}',
                    'is_baseline': True
                }
        else:
            # ModelScope/HuggingFace模型
            encoders[name] = {
                'type': 'modelscope',
                'path': model_path,
                'description': f'Baseline: {model_path}',
                'is_baseline': True  # 标记为baseline
            }

    # 扫描round目录
    if not os.path.exists(experiment_path):
        print(f"Warning: Experiment path does not exist: {experiment_path}")
        return encoders

    round_dirs = sorted([d for d in os.listdir(experiment_path)
                        if d.startswith('round') and os.path.isdir(os.path.join(experiment_path, d))])

    print(f"Found {len(round_dirs)} round directories: {round_dirs}")

    for round_dir in round_dirs:
        try:
            round_num = int(round_dir.replace('round', ''))
            round_path = os.path.join(experiment_path, round_dir)

            # Round 1: contrastive_training/best_contrastive_model.pth
            if round_num == 1:
                encoder_path = os.path.join(round_path, 'contrastive_training', 'best_contrastive_model.pth')
            # Round 2+: best_model.pth
            else:
                encoder_path = os.path.join(round_path, 'best_model.pth')

            if os.path.exists(encoder_path):
                encoders[f'round{round_num}_encoder'] = {
                    'type': 'contrastive',
                    'path': encoder_path,
                    'round': round_num,
                    'description': f'Round {round_num} Contrastive Encoder',
                    'is_baseline': False  # 标记为非baseline
                }
                print(f"Found encoder for round {round_num}: {encoder_path}")
            else:
                print(f"Warning: Encoder not found for round {round_num}: {encoder_path}")
        except Exception as e:
            print(f"Error processing {round_dir}: {e}")

    print(f"Total discovered encoders: {len(encoders)}")
    return encoders


def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载训练集和测试集"""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found. Train: {train_path}, Test: {test_path}")

    trainset = pd.read_csv(train_path)
    testset = pd.read_csv(test_path)

    print(f"Loaded trainset: {len(trainset)} samples")
    print(f"Loaded testset: {len(testset)} samples")

    return trainset, testset


def process_encoder_embeddings(encoder_name: str, encoder_info: Dict,
                             trainset: pd.DataFrame, testset: pd.DataFrame,
                             output_path: str, device: str = 'auto',
                             text_column: str = 'content',
                             use_cache: bool = True,
                             batch_size: int = 32,
                             max_seq_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """处理单个编码器的embeddings"""

    embedding_dir = os.path.join(output_path, encoder_name, 'embeddings')
    train_emb_path = os.path.join(embedding_dir, 'trainset_embeddings.npy')
    test_emb_path = os.path.join(embedding_dir, 'testset_embeddings.npy')

    # 检查是否已存在embeddings
    if use_cache and os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        print(f"Loading existing embeddings for {encoder_name}")
        train_emb = np.load(train_emb_path)
        test_emb = np.load(test_emb_path)
        return train_emb, test_emb

    print(f"Computing embeddings for {encoder_name}")

    # 加载编码器
    encoder = load_encoder(encoder_info, device)

    try:
        # 编码训练集
        print("Encoding trainset...")
        train_emb = encoder.encode_batch(trainset[text_column].tolist(), batch_size=batch_size)

        # 编码测试集
        print("Encoding testset...")
        test_emb = encoder.encode_batch(testset[text_column].tolist(), batch_size=batch_size)

        # 保存embeddings
        os.makedirs(embedding_dir, exist_ok=True)
        np.save(train_emb_path, train_emb)
        np.save(test_emb_path, test_emb)
        print(f"Saved embeddings to {embedding_dir}")

    finally:
        # 释放GPU内存
        if hasattr(encoder, 'model'):
            del encoder.model
        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return train_emb, test_emb


def generate_umap_visualizations(encoder_name: str, test_embeddings: np.ndarray,
                               test_labels: pd.Series, output_path: str,
                               umap_config: Dict = None,
                               max_labels_display: int = 10):
    """生成UMAP可视化"""

    viz_dir = os.path.join(output_path, encoder_name, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Generating UMAP visualizations for {encoder_name}...")

    # UMAP配置
    if umap_config is None:
        umap_config = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'cosine',
            'random_state': 42
        }

    # 2D UMAP
    print("Computing 2D UMAP...")
    umap_2d = umap.UMAP(n_components=2, **umap_config)
    embedding_2d = umap_2d.fit_transform(test_embeddings)

    # 3D UMAP
    print("Computing 3D UMAP...")
    umap_3d = umap.UMAP(n_components=3, **umap_config)
    embedding_3d = umap_3d.fit_transform(test_embeddings)

    # 准备标签和颜色
    unique_labels = sorted(test_labels.unique())
    label_to_color = {label: f'C{i}' for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in test_labels]

    # 生成2D图像
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                         c=colors, alpha=0.7, s=10)

    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=label_to_color[label],
                                 markersize=8, label=f'Label {label}')
                      for label in unique_labels[:max_labels_display]]  # 只显示前N个标签
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'UMAP 2D Visualization - {encoder_name}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()

    # 保存2D图像
    plt.savefig(os.path.join(viz_dir, 'umap_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 生成3D交互图
    fig_3d = go.Figure()

    for label in unique_labels:
        mask = test_labels == label
        fig_3d.add_trace(go.Scatter3d(
            x=embedding_3d[mask, 0],
            y=embedding_3d[mask, 1],
            z=embedding_3d[mask, 2],
            mode='markers',
            name=f'Label {label}',
            marker=dict(size=3, opacity=0.7),
            text=f'Label {label}',
            visible=True if label in unique_labels[:max_labels_display] else 'legendonly'  # 默认只显示前N个
        ))

    fig_3d.update_layout(
        title=f'UMAP 3D Visualization - {encoder_name}',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=800,
        height=600
    )

    # 保存3D HTML
    html_path = os.path.join(viz_dir, 'umap_3d.html')
    fig_3d.write_html(html_path)

    print(f"Visualizations saved to {viz_dir}")


def calculate_precision_at_k(top_k_indices: np.ndarray, candidate_labels: pd.Series,
                           query_label: int, k: int) -> float:
    """计算Precision@K"""
    if len(top_k_indices) < k:
        k = len(top_k_indices)

    if k == 0:
        return 0.0

    top_k_labels = candidate_labels.iloc[top_k_indices[:k]].values
    correct_count = np.sum(top_k_labels == query_label)

    return correct_count / k


def calculate_mrr(top_k_indices: np.ndarray, candidate_labels: pd.Series,
                 query_label: int) -> float:
    """计算Mean Reciprocal Rank"""
    for i, idx in enumerate(top_k_indices):
        if candidate_labels.iloc[idx] == query_label:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_precision_k_mrr(train_embeddings: np.ndarray, test_embeddings: np.ndarray,
                           train_labels: pd.Series, test_labels: pd.Series,
                           k_values: List[int] = [1, 3, 5, 10],
                           batch_size: int = 100) -> Dict:
    """评估Precision@K和MRR"""

    print("Computing similarities and metrics...")

    results = {
        'precision_at_k': {k: [] for k in k_values},
        'mrr': []
    }

    for i in tqdm(range(0, len(train_embeddings), batch_size), desc="Evaluating"):
        end_idx = min(i + batch_size, len(train_embeddings))
        query_batch = train_embeddings[i:end_idx]

        # 计算当前批次与所有候选的相似度
        similarities = cosine_similarity(query_batch, test_embeddings)

        for j, query_idx in enumerate(range(i, end_idx)):
            query_label = train_labels.iloc[query_idx]
            query_similarities = similarities[j]

            # 排序获取最相似的候选（降序）
            top_indices = np.argsort(query_similarities)[::-1]

            # 计算Precision@K
            for k in k_values:
                precision_k = calculate_precision_at_k(top_indices, test_labels, query_label, k)
                results['precision_at_k'][k].append(precision_k)

            # 计算MRR
            mrr = calculate_mrr(top_indices, test_labels, query_label)
            results['mrr'].append(mrr)

    # 计算平均值
    metrics = {
        'precision_at_k': {k: np.mean(values) for k, values in results['precision_at_k'].items()},
        'mrr': np.mean(results['mrr']),
        'sample_count': len(train_embeddings)
    }

    return metrics


def compute_clustering_metrics(embeddings: np.ndarray,
                              labels: pd.Series,
                              n_clusters: int = None,
                              n_init: int = 10,
                              random_state: int = 42,
                              normalize_embeddings: bool = True) -> Dict:
    """
    使用K-means聚类评估embedding质量

    Args:
        embeddings: 样本嵌入向量 [n_samples, embedding_dim]
        labels: 真实标签 [n_samples]
        n_clusters: 聚类数量（默认为标签数量）
        n_init: K-means初始化次数（多次运行取最佳）
        random_state: 随机种子
        normalize_embeddings: 是否L2归一化（推荐True，K-means对尺度敏感）

    Returns:
        {
            'nmi': float,           # Normalized Mutual Information [0,1], 越高越好
            'acc': float,           # Clustering Accuracy [0,1], 越高越好
            'ari': float,           # Adjusted Rand Index [-1,1], 越高越好
            'n_clusters': int,      # 聚类数量
            'inertia': float,       # K-means目标函数值（类内距离平方和）
            'clustering_time': float
        }
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.metrics.cluster import contingency_matrix
    from sklearn.preprocessing import normalize
    from scipy.optimize import linear_sum_assignment
    import time

    # 1. 确定聚类数量
    if n_clusters is None:
        n_clusters = len(labels.unique())

    print(f"  Running K-means clustering (k={n_clusters})...")

    # 2. 归一化embedding（K-means对尺度敏感）
    if normalize_embeddings:
        embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
        print(f"    Embeddings normalized to unit sphere")
    else:
        embeddings_normalized = embeddings

    # 3. K-means聚类
    start_time = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
        max_iter=300,
        algorithm='lloyd'  # 明确指定算法
    )
    cluster_labels = kmeans.fit_predict(embeddings_normalized)
    clustering_time = time.time() - start_time

    # 4. 计算NMI (Normalized Mutual Information)
    nmi = normalized_mutual_info_score(labels, cluster_labels, average_method='arithmetic')

    # 5. 计算ACC (Clustering Accuracy) - 需要匈牙利算法对齐
    def cluster_acc(y_true, y_pred):
        """
        使用匈牙利算法找最优簇到类别的映射
        """
        cm = contingency_matrix(y_true, y_pred)
        # linear_sum_assignment 求最小化，我们要最大化，所以取负
        row_ind, col_ind = linear_sum_assignment(-cm)
        return cm[row_ind, col_ind].sum() / cm.sum()

    acc = cluster_acc(labels, cluster_labels)

    # 6. 计算ARI (Adjusted Rand Index)
    ari = adjusted_rand_score(labels, cluster_labels)

    # 7. K-means的目标函数值（类内距离平方和）
    inertia = float(kmeans.inertia_)

    print(f"  Clustering Results:")
    print(f"    NMI (Normalized Mutual Info): {nmi:.4f}")
    print(f"    ACC (Clustering Accuracy):    {acc:.4f}")
    print(f"    ARI (Adjusted Rand Index):    {ari:.4f}")
    print(f"    Inertia (within-cluster SSE): {inertia:.2f}")
    print(f"    Time: {clustering_time:.2f}s")

    return {
        'nmi': float(nmi),
        'acc': float(acc),
        'ari': float(ari),
        'n_clusters': int(n_clusters),
        'inertia': inertia,
        'clustering_time': clustering_time
    }


def compute_intra_inter_similarity(embeddings: np.ndarray,
                                   labels: pd.Series,
                                   use_centroids_for_inter: bool = True) -> Dict:
    """
    计算类内、类间余弦相似度及其差值（Margin）

    统一使用余弦相似度（Cosine Similarity）：
    - 类内：越大越好（期望 > 0.7）
    - 类间：越小越好（期望 < 0.3）
    - Margin（类内-类间）：越大越好（期望 > 0.5）

    Args:
        embeddings: 样本嵌入向量 [n_samples, embedding_dim]
        labels: 样本标签 [n_samples]
        use_centroids_for_inter: 类间相似度是否使用类中心（True=快速，False=精确）

    Returns:
        {
            'intra_class_similarity': float,      # 类内平均相似度（越大越好）
            'inter_class_similarity': float,      # 类间平均相似度（越小越好）
            'margin': float,                      # 类内 - 类间（越大越好）
            'num_classes': int,
            'details': {...}                      # 每个类别的详细信息
        }
    """
    from sklearn.metrics.pairwise import cosine_similarity

    unique_labels = sorted(labels.unique())
    num_classes = len(unique_labels)

    # ========== 1. 计算类内相似度 ==========
    intra_similarities = []
    class_details = {}

    print(f"  Computing intra-class similarities for {num_classes} classes...")

    for label in unique_labels:
        mask = (labels == label).values
        class_embeddings = embeddings[mask]
        class_size = len(class_embeddings)

        if class_size < 2:
            # 类别只有1个样本，无法计算类内相似度
            print(f"    Warning: Class '{label}' has only {class_size} sample, skipping intra-class calculation")
            continue

        # 计算该类别内所有样本对的余弦相似度
        sim_matrix = cosine_similarity(class_embeddings)

        # 取上三角（不含对角线，对角线都是1.0）
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        avg_intra_sim = float(np.mean(upper_triangle))
        std_intra_sim = float(np.std(upper_triangle))

        intra_similarities.append(avg_intra_sim)

        class_details[str(label)] = {
            'size': int(class_size),
            'intra_similarity': avg_intra_sim,
            'intra_std': std_intra_sim,
            'intra_min': float(np.min(upper_triangle)),
            'intra_max': float(np.max(upper_triangle))
        }

    avg_intra = float(np.mean(intra_similarities)) if intra_similarities else 0.0

    # ========== 2. 计算类间相似度 ==========
    print(f"  Computing inter-class similarities...")

    if use_centroids_for_inter:
        # 方法1：基于类中心（推荐，快速）
        centroids = []
        for label in unique_labels:
            mask = (labels == label).values
            centroid = embeddings[mask].mean(axis=0, keepdims=True)
            centroids.append(centroid)

        centroids = np.vstack(centroids)  # [num_classes, embedding_dim]

        # 计算类中心之间的余弦相似度
        centroid_sim_matrix = cosine_similarity(centroids)

        # 取上三角（不含对角线）
        inter_similarities = centroid_sim_matrix[np.triu_indices_from(centroid_sim_matrix, k=1)]

        print(f"    Using class centroids: {len(inter_similarities)} class pairs")

    else:
        # 方法2：基于所有跨类样本对（精确但慢）
        print(f"    Warning: Computing all pairwise inter-class similarities (slow for large datasets)")
        inter_similarities = []

        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i+1:]:
                mask_i = (labels == label_i).values
                mask_j = (labels == label_j).values

                emb_i = embeddings[mask_i]
                emb_j = embeddings[mask_j]

                # 计算两个类别之间所有样本对的相似度
                cross_sim = cosine_similarity(emb_i, emb_j)
                inter_similarities.append(np.mean(cross_sim))

    avg_inter = float(np.mean(inter_similarities)) if len(inter_similarities) > 0 else 0.0
    std_inter = float(np.std(inter_similarities)) if len(inter_similarities) > 0 else 0.0

    # ========== 3. 计算Margin（差值） ==========
    margin = avg_intra - avg_inter

    print(f"  Results:")
    print(f"    Intra-class similarity: {avg_intra:.4f}")
    print(f"    Inter-class similarity: {avg_inter:.4f}")
    print(f"    Margin (Intra - Inter): {margin:.4f}")

    return {
        'intra_class_similarity': avg_intra,
        'inter_class_similarity': avg_inter,
        'margin': margin,
        'num_classes': num_classes,
        'num_samples': len(embeddings),
        'inter_std': std_inter,
        'details': class_details
    }


def save_encoder_metrics(encoder_name: str, metrics: Dict, output_path: str):
    """保存单个编码器的指标"""

    encoder_dir = os.path.join(output_path, encoder_name)
    os.makedirs(encoder_dir, exist_ok=True)

    metrics_with_meta = {
        'encoder_name': encoder_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }

    metrics_path = os.path.join(encoder_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_with_meta, f, ensure_ascii=False, indent=2)

    print(f"Metrics saved for {encoder_name}")


def generate_clustering_evolution_plot(all_metrics: Dict, output_dir: str, encoders: Dict):
    """
    生成类内/类间相似度、Margin及K-means指标随round演进的折线图

    只绘制round编码器（排除baseline）
    """
    import matplotlib.pyplot as plt

    # 提取round数据
    round_data = []
    for encoder_name, metrics in all_metrics.items():
        if 'round' not in encoder_name:
            continue

        encoder_info = encoders.get(encoder_name, {})
        round_num = encoder_info.get('round', None)

        if round_num is None:
            continue

        row_data = {
            'round': round_num,
            'encoder': encoder_name,
        }

        # 添加聚类相似度指标
        if 'clustering' in metrics:
            clust = metrics['clustering']
            row_data['intra_sim'] = clust['intra_class_similarity']
            row_data['inter_sim'] = clust['inter_class_similarity']
            row_data['margin'] = clust['margin']

        # 添加K-means指标
        if 'kmeans' in metrics:
            km = metrics['kmeans']
            row_data['nmi'] = km['nmi']
            row_data['acc'] = km['acc']
            row_data['ari'] = km['ari']

        round_data.append(row_data)

    if not round_data:
        print("  No round data found for clustering evolution plot")
        return

    # 转为DataFrame并排序
    df = pd.DataFrame(round_data).sort_values('round')

    print(f"  Generating clustering evolution plot for {len(df)} rounds...")

    # 检查是否有K-means数据
    has_kmeans = 'nmi' in df.columns and df['nmi'].notna().any()

    # ========== 创建折线图 ==========
    if has_kmeans:
        # 有K-means数据：2行3列布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    else:
        # 没有K-means数据：1行3列布局（原有的）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1：类内相似度
    ax1 = axes[0]
    if 'intra_sim' in df.columns:
        ax1.plot(df['round'], df['intra_sim'], 'o-', color='#2E86AB',
                 linewidth=3, markersize=10, label='Intra-class Similarity')
        ax1.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
        ax1.set_title('Intra-class Similarity\n(Higher is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])
        ax1.axhline(0.7, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (>0.7)')
        ax1.legend(loc='lower right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('intra_sim')):
                ax1.text(row['round'], row['intra_sim'] + 0.03, f"{row['intra_sim']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图2：类间相似度
    ax2 = axes[1]
    if 'inter_sim' in df.columns:
        ax2.plot(df['round'], df['inter_sim'], 's-', color='#A23B72',
                 linewidth=3, markersize=10, label='Inter-class Similarity')
        ax2.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
        ax2.set_title('Inter-class Similarity\n(Lower is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1])
        ax2.axhline(0.3, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (<0.3)')
        ax2.legend(loc='upper right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('inter_sim')):
                ax2.text(row['round'], row['inter_sim'] + 0.03, f"{row['inter_sim']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图3：Margin（类内 - 类间）
    ax3 = axes[2]
    if 'margin' in df.columns:
        ax3.plot(df['round'], df['margin'], '^-', color='#F18F01',
                 linewidth=3, markersize=10, label='Margin (Intra - Inter)')
        ax3.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Margin (Intra - Inter)', fontsize=13, fontweight='bold')
        ax3.set_title('Separation Margin\n(Higher is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 添加参考线
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero margin')
        ax3.axhline(0.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (>0.5)')
        ax3.legend(loc='lower right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('margin')):
                ax3.text(row['round'], row['margin'] + 0.02, f"{row['margin']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 如果有K-means数据，添加第二行
    if has_kmeans:
        # 子图4：NMI
        ax4 = axes[3]
        ax4.plot(df['round'], df['nmi'], 'D-', color='#06A77D',
                 linewidth=3, markersize=10, label='NMI')
        ax4.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax4.set_ylabel('NMI Score', fontsize=13, fontweight='bold')
        ax4.set_title('Normalized Mutual Information\n(Higher is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_ylim([0, 1])
        ax4.axhline(0.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (>0.5)')
        ax4.legend(loc='lower right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('nmi')):
                ax4.text(row['round'], row['nmi'] + 0.03, f"{row['nmi']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 子图5：ACC
        ax5 = axes[4]
        ax5.plot(df['round'], df['acc'], 'v-', color='#D62828',
                 linewidth=3, markersize=10, label='ACC')
        ax5.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax5.set_ylabel('Clustering Accuracy', fontsize=13, fontweight='bold')
        ax5.set_title('Clustering Accuracy\n(Higher is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_ylim([0, 1])
        ax5.axhline(0.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (>0.5)')
        ax5.legend(loc='lower right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('acc')):
                ax5.text(row['round'], row['acc'] + 0.03, f"{row['acc']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 子图6：ARI
        ax6 = axes[5]
        ax6.plot(df['round'], df['ari'], 'p-', color='#F77F00',
                 linewidth=3, markersize=10, label='ARI')
        ax6.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax6.set_ylabel('Adjusted Rand Index', fontsize=13, fontweight='bold')
        ax6.set_title('Adjusted Rand Index\n(Higher is Better)',
                      fontsize=14, fontweight='bold', pad=15)
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.set_ylim([-0.2, 1])
        ax6.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random')
        ax6.axhline(0.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (>0.5)')
        ax6.legend(loc='lower right', fontsize=10)

        # 标注数值
        for _, row in df.iterrows():
            if pd.notna(row.get('ari')):
                ax6.text(row['round'], row['ari'] + 0.03, f"{row['ari']:.3f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, 'clustering_evolution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Clustering evolution plot saved: {save_path}")

    # ========== 生成数值表格 ==========
    table_path = os.path.join(output_dir, 'clustering_evolution_table.csv')
    df[['round', 'intra_sim', 'inter_sim', 'margin']].to_csv(
        table_path, index=False, float_format='%.4f'
    )
    print(f"  Clustering evolution table saved: {table_path}")

    # ========== 生成改进分析 ==========
    if len(df) > 1:
        improvement_lines = []
        improvement_lines.append("# Clustering Metrics Improvement Analysis\n\n")

        baseline_row = df.iloc[0]  # Round 1作为baseline

        improvement_lines.append("| Round | Intra Delta | Inter Delta | Margin Delta | Note |\n")
        improvement_lines.append("|-------|-------------|-------------|--------------|------|\n")

        for idx, row in df.iterrows():
            if row['round'] == baseline_row['round']:
                improvement_lines.append(f"| {row['round']} | - | - | - | Baseline |\n")
            else:
                intra_delta = row['intra_sim'] - baseline_row['intra_sim']
                inter_delta = row['inter_sim'] - baseline_row['inter_sim']
                margin_delta = row['margin'] - baseline_row['margin']

                note = []
                if intra_delta > 0.05:
                    note.append("Intra up")
                if inter_delta < -0.05:
                    note.append("Inter down")
                if margin_delta > 0.1:
                    note.append("Strong improvement")

                note_str = ", ".join(note) if note else "-"

                improvement_lines.append(
                    f"| {row['round']} | {intra_delta:+.4f} | {inter_delta:+.4f} | "
                    f"{margin_delta:+.4f} | {note_str} |\n"
                )

        improvement_lines.append("\n**Note**: Delta = Change from Round 1 baseline\n")
        improvement_lines.append("- Intra Delta > 0: Improvement (classes more compact)\n")
        improvement_lines.append("- Inter Delta < 0: Improvement (classes more separated)\n")
        improvement_lines.append("- Margin Delta > 0: Improvement (better separation)\n")

        improvement_path = os.path.join(output_dir, 'clustering_improvement.md')
        with open(improvement_path, 'w', encoding='utf-8') as f:
            f.writelines(improvement_lines)

        print(f"  Improvement analysis saved: {improvement_path}")


def generate_comparison_results(all_metrics: Dict, output_path: str, encoders: Dict):
    """生成对比分析结果"""

    comparison_dir = os.path.join(output_path, 'comparison_results')
    os.makedirs(comparison_dir, exist_ok=True)

    # 准备对比数据
    comparison_data = []
    for encoder_name, metrics in all_metrics.items():
        row = {'encoder': encoder_name, 'mrr': metrics['mrr']}

        # 添加聚类指标（使用差值）
        if 'clustering' in metrics:
            clust = metrics['clustering']
            row['intra_sim'] = clust['intra_class_similarity']
            row['inter_sim'] = clust['inter_class_similarity']
            row['margin'] = clust['margin']  # 差值：类内 - 类间
            row['num_classes'] = clust['num_classes']

        # 添加K-means聚类指标
        if 'kmeans' in metrics:
            km = metrics['kmeans']
            row['nmi'] = km['nmi']
            row['acc'] = km['acc']
            row['ari'] = km['ari']

        for k, precision in metrics['precision_at_k'].items():
            row[f'precision_at_{k}'] = precision
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # 保存完整对比表格
    df_comparison.to_csv(os.path.join(comparison_dir, 'full_metrics_comparison.csv'),
                        index=False, encoding='utf-8-sig')

    # 保存独立的聚类指标表格（相似度）
    if 'intra_sim' in df_comparison.columns:
        clustering_cols = ['encoder', 'intra_sim', 'inter_sim', 'margin', 'num_classes']
        df_clustering = df_comparison[clustering_cols].copy()

        # 按margin排序（从大到小）
        df_clustering = df_clustering.sort_values('margin', ascending=False)

        df_clustering.to_csv(
            os.path.join(comparison_dir, 'clustering_metrics.csv'),
            index=False, encoding='utf-8-sig', float_format='%.4f'
        )
        print(f"  Clustering metrics table saved to clustering_metrics.csv")

    # 保存独立的K-means聚类指标表格
    if 'nmi' in df_comparison.columns:
        kmeans_cols = ['encoder', 'nmi', 'acc', 'ari']
        df_kmeans = df_comparison[kmeans_cols].copy()

        # 按NMI排序（从大到小）
        df_kmeans = df_kmeans.sort_values('nmi', ascending=False)

        df_kmeans.to_csv(
            os.path.join(comparison_dir, 'kmeans_metrics.csv'),
            index=False, encoding='utf-8-sig', float_format='%.4f'
        )
        print(f"  K-means metrics table saved to kmeans_metrics.csv")

    # 生成性能曲线图
    plt.figure(figsize=(12, 8))

    k_values = [k for k in sorted([int(col.split('_')[-1]) for col in df_comparison.columns
                                  if col.startswith('precision_at_')])]

    for _, row in df_comparison.iterrows():
        encoder_name = row['encoder']
        precisions = [row[f'precision_at_{k}'] for k in k_values]
        plt.plot(k_values, precisions, marker='o', label=encoder_name, linewidth=2, markersize=6)

    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Precision@K', fontsize=12)
    plt.title('Precision@K Comparison Across Encoders', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(comparison_dir, 'performance_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 生成聚类指标演进图
    generate_clustering_evolution_plot(all_metrics, comparison_dir, encoders)

    # 生成汇总报告
    generate_summary_report(df_comparison, all_metrics, comparison_dir, encoders)

    print(f"Comparison results saved to {comparison_dir}")


def generate_summary_report(df_comparison: pd.DataFrame, all_metrics: Dict, output_dir: str, encoders: Dict):
    """生成实验汇总报告"""

    report_lines = []
    report_lines.append("# Precision@K and Clustering Metrics Evaluation Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # ========== 新增：聚类指标表格 ==========
    if 'intra_sim' in df_comparison.columns:
        report_lines.append("## Clustering Metrics (Cosine Similarity)\n\n")

        report_lines.append("| Encoder | Intra-class | Inter-class | Margin (Intra-Inter) |\n")
        report_lines.append("|---------|-------------|-------------|---------------------|\n")

        # 按margin排序显示
        df_sorted = df_comparison.sort_values('margin', ascending=False)

        for _, row in df_sorted.iterrows():
            report_lines.append(
                f"| {row['encoder']} | {row['intra_sim']:.4f} | "
                f"{row['inter_sim']:.4f} | **{row['margin']:.4f}** |\n"
            )

        report_lines.append("\n**Interpretation**:\n")
        report_lines.append("- **Intra-class Similarity**: Higher is better (good: >0.7, excellent: >0.8)\n")
        report_lines.append("- **Inter-class Similarity**: Lower is better (good: <0.3, excellent: <0.2)\n")
        report_lines.append("- **Margin (Intra - Inter)**: Higher is better (good: >0.5, excellent: >0.7)\n\n")

        # 最佳聚类性能
        best_intra_idx = df_comparison['intra_sim'].idxmax()
        best_inter_idx = df_comparison['inter_sim'].idxmin()  # 注意：越小越好
        best_margin_idx = df_comparison['margin'].idxmax()

        report_lines.append("### Best Clustering Performance\n\n")
        report_lines.append(f"- **Best Intra-class Similarity**: "
                          f"{df_comparison.loc[best_intra_idx, 'encoder']} "
                          f"({df_comparison.loc[best_intra_idx, 'intra_sim']:.4f})\n")
        report_lines.append(f"- **Best Inter-class Similarity (lowest)**: "
                          f"{df_comparison.loc[best_inter_idx, 'encoder']} "
                          f"({df_comparison.loc[best_inter_idx, 'inter_sim']:.4f})\n")
        report_lines.append(f"- **Best Margin**: "
                          f"{df_comparison.loc[best_margin_idx, 'encoder']} "
                          f"({df_comparison.loc[best_margin_idx, 'margin']:.4f})\n\n")

        # Round演进分析（如果有多个round）
        round_encoders = [name for name in df_comparison['encoder'].values
                         if 'round' in name]

        if len(round_encoders) > 1:
            report_lines.append("### Round-by-Round Evolution\n\n")

            round_data = df_comparison[df_comparison['encoder'].isin(round_encoders)].copy()
            round_data['round_num'] = round_data['encoder'].str.extract(r'round(\d+)').astype(int)
            round_data = round_data.sort_values('round_num')

            report_lines.append("| Round | Intra | Inter | Margin | Trend |\n")
            report_lines.append("|-------|-------|-------|--------|-------|\n")

            for idx, row in round_data.iterrows():
                trend_markers = []

                # 与上一轮对比
                if idx > round_data.index[0]:
                    prev_idx = round_data.index[round_data.index.get_loc(idx) - 1]
                    prev_row = round_data.loc[prev_idx]
                    if row['intra_sim'] > prev_row['intra_sim']:
                        trend_markers.append("Intra up")
                    if row['inter_sim'] < prev_row['inter_sim']:
                        trend_markers.append("Inter down")
                    if row['margin'] > prev_row['margin']:
                        trend_markers.append("Margin up")

                trend_str = ", ".join(trend_markers) if trend_markers else "-"

                report_lines.append(
                    f"| {row['round_num']} | {row['intra_sim']:.4f} | "
                    f"{row['inter_sim']:.4f} | {row['margin']:.4f} | {trend_str} |\n"
                )

            report_lines.append("\n")

    # ========== 新增：K-means聚类指标表格 ==========
    if 'nmi' in df_comparison.columns:
        report_lines.append("## K-means Clustering Metrics\n\n")

        report_lines.append("| Encoder | NMI | ACC | ARI |\n")
        report_lines.append("|---------|-----|-----|-----|\n")

        # 按NMI排序显示
        df_sorted_kmeans = df_comparison.sort_values('nmi', ascending=False)

        for _, row in df_sorted_kmeans.iterrows():
            report_lines.append(
                f"| {row['encoder']} | {row['nmi']:.4f} | "
                f"{row['acc']:.4f} | {row['ari']:.4f} |\n"
            )

        report_lines.append("\n**Interpretation**:\n")
        report_lines.append("- **NMI (Normalized Mutual Information)**: [0,1], higher is better (good: >0.5, excellent: >0.7)\n")
        report_lines.append("- **ACC (Clustering Accuracy)**: [0,1], higher is better (good: >0.5, excellent: >0.7)\n")
        report_lines.append("- **ARI (Adjusted Rand Index)**: [-1,1], higher is better (0=random, good: >0.5, excellent: >0.7)\n\n")

        # 最佳K-means性能
        best_nmi_idx = df_comparison['nmi'].idxmax()
        best_acc_idx = df_comparison['acc'].idxmax()
        best_ari_idx = df_comparison['ari'].idxmax()

        report_lines.append("### Best K-means Performance\n\n")
        report_lines.append(f"- **Best NMI**: "
                          f"{df_comparison.loc[best_nmi_idx, 'encoder']} "
                          f"({df_comparison.loc[best_nmi_idx, 'nmi']:.4f})\n")
        report_lines.append(f"- **Best ACC**: "
                          f"{df_comparison.loc[best_acc_idx, 'encoder']} "
                          f"({df_comparison.loc[best_acc_idx, 'acc']:.4f})\n")
        report_lines.append(f"- **Best ARI**: "
                          f"{df_comparison.loc[best_ari_idx, 'encoder']} "
                          f"({df_comparison.loc[best_ari_idx, 'ari']:.4f})\n\n")

        # Round演进分析（如果有多个round）
        round_encoders_kmeans = [name for name in df_comparison['encoder'].values
                                 if 'round' in name]

        if len(round_encoders_kmeans) > 1:
            report_lines.append("### K-means Round-by-Round Evolution\n\n")

            round_data_kmeans = df_comparison[df_comparison['encoder'].isin(round_encoders_kmeans)].copy()
            round_data_kmeans['round_num'] = round_data_kmeans['encoder'].str.extract(r'round(\d+)').astype(int)
            round_data_kmeans = round_data_kmeans.sort_values('round_num')

            report_lines.append("| Round | NMI | ACC | ARI | Trend |\n")
            report_lines.append("|-------|-----|-----|-----|-------|\n")

            for idx, row in round_data_kmeans.iterrows():
                trend_markers = []

                # 与上一轮对比
                if idx > round_data_kmeans.index[0]:
                    prev_idx = round_data_kmeans.index[round_data_kmeans.index.get_loc(idx) - 1]
                    prev_row = round_data_kmeans.loc[prev_idx]
                    if row['nmi'] > prev_row['nmi']:
                        trend_markers.append("NMI up")
                    if row['acc'] > prev_row['acc']:
                        trend_markers.append("ACC up")
                    if row['ari'] > prev_row['ari']:
                        trend_markers.append("ARI up")

                trend_str = ", ".join(trend_markers) if trend_markers else "-"

                report_lines.append(
                    f"| {row['round_num']} | {row['nmi']:.4f} | "
                    f"{row['acc']:.4f} | {row['ari']:.4f} | {trend_str} |\n"
                )

            report_lines.append("\n")

    # ========== 原有的Precision@K表格 ==========
    report_lines.append("## Precision@K and MRR Results\n\n")

    k_cols = [col for col in df_comparison.columns if col.startswith('precision_at_')]
    k_values = sorted([int(col.split('_')[-1]) for col in k_cols])

    header = "| Encoder | MRR |"
    for k in k_values:
        header += f" P@{k} |"
    report_lines.append(header + "\n")

    separator = "|---------|-----|"
    for k in k_values:
        separator += "-----|"
    report_lines.append(separator + "\n")

    for _, row in df_comparison.iterrows():
        line = f"| {row['encoder']} | {row['mrr']:.4f} |"
        for k in k_values:
            line += f" {row[f'precision_at_{k}']:.4f} |"
        report_lines.append(line + "\n")

    report_lines.append("\n")

    # 最佳性能
    report_lines.append("## Best Precision@K Performers\n\n")

    best_mrr_idx = df_comparison['mrr'].idxmax()
    best_mrr_encoder = df_comparison.loc[best_mrr_idx, 'encoder']
    best_mrr_score = df_comparison.loc[best_mrr_idx, 'mrr']

    report_lines.append(f"- **Best MRR**: {best_mrr_encoder} ({best_mrr_score:.4f})\n")

    for k in k_values:
        col = f'precision_at_{k}'
        best_idx = df_comparison[col].idxmax()
        best_encoder = df_comparison.loc[best_idx, 'encoder']
        best_score = df_comparison.loc[best_idx, col]
        report_lines.append(f"- **Best P@{k}**: {best_encoder} ({best_score:.4f})\n")

    report_lines.append("\n")

    # 改进分析 - 支持多个baseline
    baseline_encoders = [name for name in df_comparison['encoder'].values
                        if name in [enc for enc, info in all_metrics.items()
                                   if encoders.get(enc, {}).get('is_baseline', False)]]

    if baseline_encoders:
        report_lines.append("\n## Improvement Analysis\n")

        for baseline_name in baseline_encoders:
            baseline_row = df_comparison[df_comparison['encoder'] == baseline_name].iloc[0]
            report_lines.append(f"\n### Compared to {baseline_name}\n")

            for _, row in df_comparison.iterrows():
                if row['encoder'] in baseline_encoders:  # Skip other baselines
                    continue

                encoder_name = row['encoder']
                mrr_improvement = ((row['mrr'] - baseline_row['mrr']) / baseline_row['mrr']) * 100

                report_lines.append(f"#### {encoder_name}\n")
                report_lines.append(f"- MRR improvement: {mrr_improvement:+.1f}%\n")

                for k in k_values:
                    col = f'precision_at_{k}'
                    p_improvement = ((row[col] - baseline_row[col]) / baseline_row[col]) * 100
                    report_lines.append(f"- P@{k} improvement: {p_improvement:+.1f}%\n")

                report_lines.append("\n")

    # 保存报告
    report_path = os.path.join(output_dir, 'experiment_summary.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Precision@K and MRR Evaluation with Visualization')
    parser.add_argument('experiment_path', help='Path to iterative experiment directory')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 10, 20],
                       help='K values for Precision@K evaluation (default: 1 3 5 10 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding (default: 32)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto/cpu/cuda, default: auto)')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip UMAP visualization generation')

    # Baseline model configuration
    parser.add_argument('--baseline-models', nargs='+',
                       default=['google-bert/bert-base-chinese'],
                       help='List of baseline model paths (default: google-bert/bert-base-chinese)')
    parser.add_argument('--baseline-names', nargs='+',
                       default=None,
                       help='Names for baseline models in results (auto-generated if not provided)')

    # Dataset configuration
    parser.add_argument('--train-path', default='data/sup_train_data/balanced_trainset.csv',
                       help='Path to training dataset CSV')
    parser.add_argument('--test-path', default='data/sup_train_data/balanced_testset.csv',
                       help='Path to test dataset CSV')
    parser.add_argument('--text-column', default='content',
                       help='Column name containing text (default: content)')
    parser.add_argument('--label-column', default='label',
                       help='Column name containing labels (default: label)')

    # UMAP configuration
    parser.add_argument('--umap-n-neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--umap-metric', default='cosine',
                       help='UMAP metric (default: cosine)')
    parser.add_argument('--umap-random-state', type=int, default=42,
                       help='UMAP random state for reproducibility (default: 42)')
    parser.add_argument('--max-labels-display', type=int, default=10,
                       help='Maximum number of labels to display in visualizations (default: 10)')

    # Evaluation configuration
    parser.add_argument('--similarity-batch-size', type=int, default=100,
                       help='Batch size for similarity computation (default: 100)')
    parser.add_argument('--max-seq-length', type=int, default=256,
                       help='Maximum sequence length for tokenization (default: 256)')

    # Output configuration
    parser.add_argument('--no-cache', action='store_true',
                       help='Do not use cached embeddings')
    parser.add_argument('--output-dir', default=None,
                       help='Custom output directory (default: experiment_path/visualization_precision@k)')

    args = parser.parse_args()

    print("=" * 80)
    print(" Precision@K and MRR Evaluation with Visualization")
    print("=" * 80)
    print(f"Experiment path: {args.experiment_path}")
    print(f"K values: {args.k_values}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    try:
        # 1. 发现编码器
        print("\n1. Discovering encoders...")
        encoders = discover_encoder_paths(args.experiment_path, args.baseline_models, args.baseline_names)

        if not encoders:
            print("No encoders found. Exiting.")
            return

        # 2. 加载数据集
        print("\n2. Loading datasets...")
        trainset, testset = load_datasets(args.train_path, args.test_path)

        # 3. 创建输出目录
        output_path = args.output_dir if args.output_dir else os.path.join(args.experiment_path, 'visualization_precision@k')
        os.makedirs(output_path, exist_ok=True)

        # 4. 保存实验配置
        config = {
            'experiment_path': args.experiment_path,
            'k_values': args.k_values,
            'batch_size': args.batch_size,
            'device': args.device,
            'timestamp': datetime.now().isoformat(),
            'discovered_encoders': encoders
        }

        with open(os.path.join(output_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 5. 处理每个编码器
        all_metrics = {}

        for encoder_name, encoder_info in encoders.items():
            print(f"\n{'='*60}")
            print(f"Processing: {encoder_name}")
            print(f"Description: {encoder_info['description']}")
            print(f"{'='*60}")

            try:
                # 编码并保存embeddings
                train_emb, test_emb = process_encoder_embeddings(
                    encoder_name, encoder_info, trainset, testset, output_path,
                    device=args.device,
                    text_column=args.text_column,
                    use_cache=not args.no_cache,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length
                )

                # 生成可视化
                if not args.skip_visualization:
                    umap_config = {
                        'n_neighbors': args.umap_n_neighbors,
                        'min_dist': args.umap_min_dist,
                        'metric': args.umap_metric,
                        'random_state': args.umap_random_state
                    }
                    generate_umap_visualizations(
                        encoder_name, test_emb, testset[args.label_column],
                        output_path, umap_config, args.max_labels_display
                    )

                # 计算评估指标
                print("Computing evaluation metrics...")
                metrics = evaluate_precision_k_mrr(
                    train_emb, test_emb,
                    trainset[args.label_column], testset[args.label_column],
                    args.k_values, args.similarity_batch_size
                )

                # 计算聚类指标（类内/类间相似度）
                print(f"\nComputing clustering metrics for {encoder_name}...")
                clustering_metrics = compute_intra_inter_similarity(
                    test_emb,  # 使用测试集嵌入
                    testset[args.label_column],
                    use_centroids_for_inter=True  # 使用类中心方法（快速）
                )

                # 计算K-means聚类指标（NMI/ACC）
                print(f"\nComputing K-means clustering metrics for {encoder_name}...")
                kmeans_metrics = compute_clustering_metrics(
                    test_emb,  # 使用测试集嵌入
                    testset[args.label_column],
                    n_clusters=None,  # 自动使用标签数量
                    n_init=10,
                    random_state=42,
                    normalize_embeddings=True  # 归一化到单位球面
                )

                # 合并到metrics中
                metrics['clustering'] = clustering_metrics
                metrics['kmeans'] = kmeans_metrics

                all_metrics[encoder_name] = metrics

                # 保存单个编码器指标
                save_encoder_metrics(encoder_name, metrics, output_path)

                # 打印指标
                print(f"Results for {encoder_name}:")
                print(f"  MRR: {metrics['mrr']:.4f}")
                for k in sorted(args.k_values):
                    print(f"  P@{k}: {metrics['precision_at_k'][k]:.4f}")

            except Exception as e:
                print(f"Error processing {encoder_name}: {e}")
                import traceback
                traceback.print_exc()

        # 6. 生成对比分析
        if len(all_metrics) > 1:
            print(f"\n{'='*60}")
            print("Generating comparison analysis...")
            print(f"{'='*60}")
            generate_comparison_results(all_metrics, output_path, encoders)

        print(f"\n{'='*80}")
        print(" Evaluation Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"Processed encoders: {len(all_metrics)}")

        # 显示最终结果摘要
        if all_metrics:
            print("\nFinal Results Summary:")
            print("-" * 50)
            for encoder_name, metrics in all_metrics.items():
                print(f"{encoder_name}:")
                print(f"  MRR: {metrics['mrr']:.4f}")
                precision_str = ", ".join([f"P@{k}: {v:.4f}" for k, v in sorted(metrics['precision_at_k'].items())])
                print(f"  {precision_str}")

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# =============================================================================
# 常见使用示例 (Common Usage Examples)
# =============================================================================

# 1. 基本用法：评估迭代实验（使用默认baseline）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1
# python exp_visualize_precision@k.py iter_model/frac0.1_round2

# 2. 自定义baseline模型（多个baseline）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --baseline-models google-bert/bert-base-chinese iic/nlp_roberta_backbone_base_std ^
#     --baseline-names bert_baseline roberta_baseline

# 3. 使用TextCNN蒸馏模型作为baseline
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --baseline-models distill_exp/run_001/student_textcnn.pt ^
#     --baseline-names textcnn_distilled

# 4. 多个TextCNN模型对比
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --baseline-models distill_exp/run_001/student_textcnn.pt distill_exp/run_002/student_textcnn.pt ^
#     --baseline-names textcnn_v1 textcnn_v2

# 5. 混合baseline（BERT + TextCNN）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --baseline-models google-bert/bert-base-chinese distill_exp/run_001/student_textcnn.pt ^
#     --baseline-names bert_base textcnn_distilled

# 6. 自定义K值和批次大小
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --k-values 1 3 5 10 20 50 ^
#     --batch-size 64 ^
#     --similarity-batch-size 200

# 7. 自定义数据集路径
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --train-path data/custom_train.csv ^
#     --test-path data/custom_test.csv ^
#     --text-column text ^
#     --label-column category

# 8. 跳过UMAP可视化（加速评估）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 --skip-visualization

# 9. 禁用缓存（强制重新计算embeddings）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 --no-cache

# 10. 自定义输出目录
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --output-dir custom_results/experiment_001

# 11. CPU模式（不使用GPU）
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 --device cpu

# 12. UMAP参数调优
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --umap-n-neighbors 30 ^
#     --umap-min-dist 0.0 ^
#     --umap-metric euclidean

# 13. 完整示例：评估多轮迭代实验并对比TextCNN baseline
# python exp_visualize_precision@k.py iter_model/frac0.1_round3 ^
#     --baseline-models google-bert/bert-base-chinese distill_exp/best_textcnn/student_textcnn.pt ^
#     --baseline-names bert_baseline textcnn_baseline ^
#     --k-values 1 3 5 10 20 ^
#     --batch-size 32 ^
#     --output-dir results/round3_evaluation

# 14. 随机初始化TextCNN对比实验（自动生成）
# 注意：当baseline-models包含.pth/.pt文件且为TextCNN模型时，
# 脚本会自动创建一个随机初始化版本（_random后缀）用于对比实验
# python exp_visualize_precision@k.py iter_model/frac0.1_round1 ^
#     --baseline-models distill_exp/run_001/student_textcnn.pt
# 这将自动创建两个baseline: baseline_student_textcnn (训练后) 和 baseline_student_textcnn_random (随机初始化)

# =============================================================================
# 输出文件说明 (Output Files Description)
# =============================================================================

# 输出目录结构 (experiment_path/visualization_precision@k/):
# ├── config.json                          # 实验配置
# ├── baseline_*/                          # Baseline模型结果
# │   ├── embeddings/                      # 嵌入向量缓存
# │   │   ├── trainset_embeddings.npy
# │   │   └── testset_embeddings.npy
# │   ├── visualization/                   # UMAP可视化
# │   │   ├── umap_2d.png                  # 2D静态图
# │   │   └── umap_3d.html                 # 3D交互图
# │   └── metrics.json                     # 评估指标
# ├── round*_encoder/                      # 各轮次编码器结果（结构同上）
# └── comparison_results/                  # 对比分析结果
#     ├── full_metrics_comparison.csv      # 完整指标对比表
#     ├── clustering_metrics.csv           # 聚类指标表（按margin排序）
#     ├── performance_curves.png           # Precision@K曲线图
#     ├── clustering_evolution.png         # 聚类指标演进图（3子图：Intra/Inter/Margin）
#     ├── clustering_evolution_table.csv   # 聚类演进数据表
#     ├── clustering_improvement.md        # 聚类改进分析（相对Round 1）
#     └── experiment_summary.md            # 实验总结报告

# 关键指标解释 (Key Metrics Interpretation):
# - Precision@K: Top-K检索精度（越高越好，期望 >0.8）
# - MRR: 平均倒数排名（越高越好，期望 >0.7）
# - Intra-class Similarity: 类内余弦相似度（越高越好，good: >0.7, excellent: >0.8）
# - Inter-class Similarity: 类间余弦相似度（越低越好，good: <0.3, excellent: <0.2）
# - Margin (Intra - Inter): 类内外分离度（越高越好，good: >0.5, excellent: >0.7）

# =============================================================================