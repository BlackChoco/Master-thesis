import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import warnings
import random
from typing import List, Dict, Optional
from Tree_data_model import PostStorage
from peft import get_peft_model, LoraConfig

# 导入解耦后的模块
from cl_base_model import ContrastiveEncoder, load_model_from_modelscope, load_tokenizer_from_modelscope
from cl_dataset import ContrastiveDataset1, ContrastiveDataset2, SimCSEDataset, ContrastiveDataCollator, build_vocab_from_post_storage, preprocess_text
from cl_loss import ContrastiveLoss
from cl_utils import build_pruned_forest

warnings.filterwarnings("ignore", category=UserWarning, message=r"Glyph .* missing from font\(s\) Arial\.")


def set_seed(seed: int):
    """
    设置所有随机种子以确保可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ 随机种子已设置为: {seed}")
    print(f"   - Python random: {seed}")
    print(f"   - NumPy: {seed}")
    print(f"   - PyTorch: {seed}")
    if torch.cuda.is_available():
        print(f"   - CUDA: 确定性模式已启用")


def load_trained_model_and_tokenizer(checkpoint_path: str):
    """
    从checkpoint加载完整的训练器状态，并返回训练好的基础模型和分词器。
    """
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 文件 {checkpoint_path} 未找到。")
        return None, None, None

    print(f"正在从 {checkpoint_path} 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=False) # 加载到CPU以避免GPU问题

    model_type = checkpoint['training_model_type']
    model_identifier = checkpoint['training_model_identifier_or_path']
    proj_config = checkpoint['projection_head_config']

    # --- 新增：获取PEFT配置 ---
    use_peft = checkpoint.get('use_peft', False)
    peft_config = checkpoint.get('peft_config', None)
    # -------------------------
    
    encoder = None
    if model_type == 'textcnn':
        textcnn_conf = checkpoint['textcnn_config']
        vocab_data = checkpoint['vocab']
        encoder = ContrastiveEncoder(
            model_type='textcnn',
            vocab=vocab_data,
            textcnn_config=textcnn_conf,
            projection_hidden_dim=proj_config['hidden_dim'],
            projection_output_dim=proj_config['output_dim'],
            projection_dropout_rate=proj_config['dropout_rate']
        )
    elif model_type == 'modelscope':
        encoder = ContrastiveEncoder(
            model_type='modelscope',
            model_name_or_path=model_identifier,
            projection_hidden_dim=proj_config['hidden_dim'],
            projection_output_dim=proj_config['output_dim'],
            projection_dropout_rate=proj_config['dropout_rate']
        )
        # --- 新增：如果使用了PEFT，重新包装模型 ---
        if use_peft:
            print(" 检测到PEFT训练的checkpoint，正在重新应用LoRA配置...")
            lora_config = LoraConfig(**peft_config)
            encoder.base_model = get_peft_model(encoder.base_model, lora_config)
            print(" LoRA配置已重新应用。")
        # ------------------------------------
    else:
        print(f"错误: Checkpoint中未知的模型类型 '{model_type}'")
        return None, None, None

    encoder.load_state_dict(checkpoint['contrastive_encoder_state_dict'])
    encoder.eval() # 设置为评估模式
    print(f" {model_type.upper()} ContrastiveEncoder 加载完成并设置为评估模式。")
    
    # 返回基础模型和分词器
    return encoder.base_model, encoder.tokenizer, model_type


class DynamicContrastiveTrainer:
    def __init__(self, post_storage: PostStorage,
                 seed: int = 42,  # 新增：随机种子
                 training_model_type: str = 'modelscope',
                 training_model_identifier_or_path: Optional[str] = "google-bert/bert-base-chinese",
                 textcnn_config: Optional[Dict] = None,
                 projection_head_config: Optional[Dict] = None,
                 use_peft: bool = False,  # <--- 新增参数：是否使用PEFT
                 peft_config: Optional[Dict] = None, # <--- 新增参数：PEFT配置
                 pruning_model_path: str = "google-bert/bert-base-chinese",
                 similarity_threshold: float = 0.5,
                 num_negatives: int = 2,
                 batch_size: int = 32,
                 pruning_inference_batch_size: int = 32,
                 base_lr: float = 1e-6,
                 projection_lr: float = 1e-4,
                 use_weighted_loss: bool = False,
                 loss_weights: Optional[Dict[str, float]] = None,
                 adaptive_weighting: bool = False,
                 infonce_mode: str = 'unidirectional',
                 min_subtree_size_ds1: int = 2,
                 max_samples_per_post_ds1: Optional[int] = None,
                 min_subtree_size_ds2: int = 4,
                 max_samples_per_subtree_ds2: Optional[int] = None,
                 # --- 新增正样本对策略参数 ---
                 positive_pair_strategy: str = 'comment_reply',
                 infonce_temperature: float = 0.07,  # InfoNCE损失温度系数，用于Dataset1
                 simcse_temperature: float = 0.07,  # SimCSE温度系数，用于simcse_dropout策略
                 simcse_dropout_rate: float = 0.1,
                 simcse_remove_duplicates: bool = True,  # 新增：是否去重
                 hybrid_ratio: float = 0.5,
                 bidirectional_loss: bool = True,  # 新增：控制in-batch模式的方向性
                 output_dir: Optional[str] = None,
                 lightweight_mode: bool = False):  # 新增：轻量级模式（Grid Search用，不保存模型）

        # ✅ 首先设置随机种子，确保所有后续操作可复现
        set_seed(seed)
        self.seed = seed

        self.post_storage = post_storage
        self.training_model_type = training_model_type.lower()
        self.training_model_identifier_or_path = training_model_identifier_or_path
        self.textcnn_config = textcnn_config
        self.use_peft = use_peft
        self.peft_config = peft_config
        self.pruning_model_path = pruning_model_path
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives if infonce_mode != 'in_batch' else 0
        self.batch_size = batch_size
        self.pruning_inference_batch_size = pruning_inference_batch_size
        self.output_dir = output_dir  # 保存输出目录
        self.lightweight_mode = lightweight_mode  # 轻量级模式
        self.infonce_mode = infonce_mode

        self.min_subtree_size_ds1 = min_subtree_size_ds1
        self.max_samples_per_post_ds1 = max_samples_per_post_ds1
        self.min_subtree_size_ds2 = min_subtree_size_ds2
        self.max_samples_per_subtree_ds2 = max_samples_per_subtree_ds2

        self.use_weighted_loss = use_weighted_loss
        if self.use_weighted_loss:
            if loss_weights is None: self.loss_weights = {'dataset1': 0.5, 'dataset2': 0.5}
            else: self.loss_weights = loss_weights.copy()
            weight_sum = sum(self.loss_weights.values())
            if weight_sum > 0:
                for k_lw in self.loss_weights: self.loss_weights[k_lw] /= weight_sum
            else:
                self.loss_weights = {'dataset1': 0.5, 'dataset2': 0.5}
            self.adaptive_weighting = adaptive_weighting
            self.initial_loss_weights = self.loss_weights.copy()
            print(f" 启用损失加权: {self.loss_weights}, 自适应: {self.adaptive_weighting}")
        else:
            self.loss_weights = {'dataset1': 1.0, 'dataset2': 1.0}
            self.adaptive_weighting = False
            print(" 使用独立损失（每个数据集的损失乘以其权重，默认为1.0）")

        # --- 新增：正样本对策略配置 ---
        self.positive_pair_strategy = positive_pair_strategy.lower()
        self.infonce_temperature = infonce_temperature  # InfoNCE温度系数
        self.simcse_temperature = simcse_temperature    # SimCSE温度系数
        self.simcse_dropout_rate = simcse_dropout_rate
        self.simcse_remove_duplicates = simcse_remove_duplicates  # 新增
        self.hybrid_ratio = hybrid_ratio
        self.bidirectional_loss = bidirectional_loss  # 新增：控制in-batch模式的方向性

        # 验证策略参数
        if self.positive_pair_strategy not in ['comment_reply', 'simcse_dropout', 'hybrid']:
            raise ValueError(f"不支持的正样本对策略: {positive_pair_strategy}. 可选: 'comment_reply', 'simcse_dropout', 'hybrid'")

        # 验证温度参数范围
        if not (0.01 <= infonce_temperature <= 1.0):
            raise ValueError(f"infonce_temperature必须在[0.01, 1.0]范围内，得到{infonce_temperature}")
        if not (0.01 <= simcse_temperature <= 1.0):
            raise ValueError(f"simcse_temperature必须在[0.01, 1.0]范围内，得到{simcse_temperature}")

        print(f" 正样本对构造策略: {self.positive_pair_strategy}")
        print(f"   InfoNCE温度系数: {self.infonce_temperature}")
        print(f"   双向损失: {self.bidirectional_loss}")
        if self.positive_pair_strategy == 'simcse_dropout':
            print(f"   SimCSE温度参数: {self.simcse_temperature}")
            print(f"   SimCSE dropout率: {self.simcse_dropout_rate}")
        elif self.positive_pair_strategy == 'hybrid':
            print(f"   混合比例 (SimCSE:评论回复) = {self.hybrid_ratio}:{1-self.hybrid_ratio}")
        # -----------------------------------------

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _default_proj_config = {'hidden_dim': 512, 'output_dim': 256, 'dropout_rate': 0.1}
        self.projection_config = {**_default_proj_config, **(projection_head_config if projection_head_config is not None else {})}


        self.vocab = None
        if self.training_model_type == 'textcnn':
            print(" 使用 TextCNN 模型进行训练。")
            if self.textcnn_config is None:
                raise ValueError("当 training_model_type 为 'textcnn' 时，textcnn_config 是必需的")
            print(" 从 PostStorage 为 TextCNN 构建词汇表...")
            self.vocab, _ = build_vocab_from_post_storage(self.post_storage, min_freq=self.textcnn_config.get('min_vocab_freq', 1))
            self.contrastive_encoder = ContrastiveEncoder(
                model_type='textcnn',
                vocab=self.vocab,
                textcnn_config=self.textcnn_config,
                projection_hidden_dim=self.projection_config['hidden_dim'],
                projection_output_dim=self.projection_config['output_dim'],
                projection_dropout_rate=self.projection_config['dropout_rate']
            ).to(self.device)
            print(f"   TextCNN 模型名称 (标识符): {self.training_model_identifier_or_path}")
        elif self.training_model_type == 'modelscope':
            print(f"  使用 ModelScope 模型进行训练: {self.training_model_identifier_or_path}")
            if self.training_model_identifier_or_path is None:
                 raise ValueError("对于 'modelscope' 模型类型，training_model_identifier_or_path 是必需的。")
            self.contrastive_encoder = ContrastiveEncoder(
                model_type='modelscope',
                model_name_or_path=self.training_model_identifier_or_path,
                projection_hidden_dim=self.projection_config['hidden_dim'],
                projection_output_dim=self.projection_config['output_dim'],
                projection_dropout_rate=self.projection_config['dropout_rate']
            ).to(self.device)
             # --- 新增：应用PEFT/LoRA的逻辑 ---
            if self.use_peft:
                print(" 应用PEFT (LoRA)到基础模型...")
                
                # 定义默认LoRA配置，并与用户传入的配置合并
                default_lora_config = {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ["query", "key", "value"],
                    'lora_dropout': 0.05,
                    'bias': "none",
                }
                final_lora_config_dict = {**default_lora_config, **(self.peft_config or {})}
                
                lora_config = LoraConfig(**final_lora_config_dict)
                
                # 将基础模型包装成PeftModel
                self.contrastive_encoder.base_model = get_peft_model(self.contrastive_encoder.base_model, lora_config)
                
                print(" LoRA应用完成。可训练参数详情:")
                self.contrastive_encoder.base_model.print_trainable_parameters()
            # --- PEFT逻辑结束 ---
        else:
            raise ValueError(f"不支持的训练模型类型: {self.training_model_type}。请选择 'modelscope' 或 'textcnn'。")

        print(f" 初始化剪枝模型: {self.pruning_model_path}")
        # 使用为ModelScope定义的辅助函数加载剪枝模型
        self.pruning_tokenizer = load_tokenizer_from_modelscope(self.pruning_model_path)
        self.pruning_model = load_model_from_modelscope(
            self.pruning_model_path,
            torch_dtype=torch.float32
        )
        self.pruning_model_on_gpu = False

        self.optimizer = torch.optim.AdamW([
            {'params': self.contrastive_encoder.base_model.parameters(), 'lr': base_lr, 'weight_decay': 1e-6, 'initial_lr': base_lr},
            {'params': self.contrastive_encoder.projection_head.parameters(), 'lr': projection_lr, 'weight_decay': 1e-4, 'initial_lr': projection_lr}
        ])

        self.loss_fn1 = ContrastiveLoss(
            temperature=self.infonce_temperature,  # 使用可配置的温度系数
            loss_type='infonce',
            infonce_mode=infonce_mode,
            bidirectional_loss=self.bidirectional_loss  # 传递双向损失参数
        )
        self.loss_fn2 = ContrastiveLoss(
            temperature=0.05,  # Dataset2保持硬编码，避免影响
            loss_type='infonce',
            infonce_mode=infonce_mode
        )

        self.training_history = defaultdict(list)
        self.training_history['dataset_sizes'] = {'dataset1': [], 'dataset2': []}
        if self.use_weighted_loss:
            self.training_history['weight_ds1'] = []
            self.training_history['weight_ds2'] = []
        print(f" DynamicContrastiveTrainer 初始化完成。设备: {self.device}")

    def _load_pruning_model_to_gpu(self):
        if not self.pruning_model_on_gpu and self.device.type == 'cuda':
            print(" 将剪枝模型加载到GPU...")
            self.pruning_model.to(self.device)
            self.pruning_model_on_gpu = True

    def _unload_pruning_model_from_gpu(self):
        if self.pruning_model_on_gpu and self.device.type == 'cuda':
            print(" 从GPU卸载剪枝模型...")
            self.pruning_model.to('cpu')
            self.pruning_model_on_gpu = False
            torch.cuda.empty_cache()

    def _get_pruning_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        # self._load_pruning_model_to_gpu() # 由调用者管理GPU加载/卸载
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
        texts = [t if t else "<empty_text_placeholder>" for t in texts] # 替换空字符串

        all_embeddings_list = []
        self.pruning_model.eval()
        with torch.no_grad():
            # 使用tqdm包裹批处理循环
            for i in tqdm(range(0, len(texts), self.pruning_inference_batch_size), desc="计算剪枝嵌入", unit="batch"):
                batch_texts = texts[i:i + self.pruning_inference_batch_size]
                if not batch_texts: continue
                try:
                    inputs = self.pruning_tokenizer(
                        batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
                    ).to(self.pruning_model.device)
                    outputs = self.pruning_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    if batch_embeddings.dtype in [torch.bfloat16, torch.float16]:
                        batch_embeddings = batch_embeddings.float()
                    all_embeddings_list.append(batch_embeddings.cpu())
                except Exception as e:
                    print(f"获取剪枝嵌入时批处理错误: {e}. 跳过此批次。批次文本示例: {batch_texts[0][:50] if batch_texts else 'N/A'}")
                    # 根据需要，可以为失败的批次添加占位符嵌入，或者简单地跳过
                    # 例如，添加零嵌入：
                    # output_dim = self.pruning_model.config.hidden_size # 假设HF模型
                    # all_embeddings_list.append(torch.zeros((len(batch_texts), output_dim), dtype=torch.float32).cpu())


        if not all_embeddings_list: return np.array([])
        try:
            all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
        except RuntimeError as e:
            print(f"连接剪枝嵌入时出错: {e}")
            # 尝试找出哪个嵌入导致问题
            # for i, emb_tensor in enumerate(all_embeddings_list):
            # print(f"Tensor {i} shape: {emb_tensor.shape}, dtype: {emb_tensor.dtype}")
            return np.array([]) # 返回空数组以避免进一步错误
        return all_embeddings_np

    def _build_and_log_datasets(self):
        print(" 构建/重建数据集...")

        # ========== 步骤1：剪枝森林构建（所有策略都需要） ==========
        print("   收集所有评论用于剪枝模型嵌入...")
        all_comment_texts = []
        comment_nodes_references = []

        for post_id, post_container in self.post_storage.posts.items():
            # Path A: 尝试从 'comments' 字典获取 (如果存在且被填充)
            if hasattr(post_container, 'comments') and isinstance(post_container.comments, dict) and post_container.comments:
                for comment_id, comment_node in post_container.comments.items():
                    content_str = str(comment_node.content) if comment_node.content is not None else ""
                    all_comment_texts.append(content_str)
                    comment_nodes_references.append(comment_node)
            # Path B: 尝试从 'root' 属性遍历
            elif hasattr(post_container, 'root') and post_container.root:
                queue = [post_container.root]
                visited_ids_in_post = set()
                while queue:
                    comment_node = queue.pop(0)
                    if comment_node.comment_id in visited_ids_in_post:
                        continue
                    visited_ids_in_post.add(comment_node.comment_id)

                    content_str = str(comment_node.content) if comment_node.content is not None else ""
                    all_comment_texts.append(content_str)
                    comment_nodes_references.append(comment_node)

                    if hasattr(comment_node, 'children') and comment_node.children:
                        for child_node in comment_node.children:
                            if child_node:
                                queue.append(child_node)

        if all_comment_texts:
            # 对文本进行预处理
            print(f"   对 {len(all_comment_texts)} 条评论进行预处理以用于剪枝...")
            preprocessed_texts_for_pruning = [preprocess_text(text) for text in all_comment_texts]

            print(f"   计算 {len(preprocessed_texts_for_pruning)} 条预处理后评论的剪枝嵌入...")
            self._load_pruning_model_to_gpu()
            pruning_embeddings_np = self._get_pruning_embeddings(preprocessed_texts_for_pruning)
            self._unload_pruning_model_from_gpu()

            if pruning_embeddings_np.ndim == 2 and pruning_embeddings_np.shape[0] == len(comment_nodes_references):
                for i, comment_node in enumerate(comment_nodes_references):
                    if hasattr(comment_node, 'set_embedding'):
                        comment_node.set_embedding(pruning_embeddings_np[i])
                    else:
                        comment_node.embedding = pruning_embeddings_np[i]
                print("   剪枝嵌入已为PostStorage中的所有评论设置。")
            else:
                print(f"   警告: _get_pruning_embeddings 的形状不匹配或结果为空。")
        else:
            print("   在PostStorage中未找到评论进行剪枝嵌入。")

        # ✅ 构建剪枝森林（所有策略都需要）
        build_pruned_forest(self.post_storage, self.similarity_threshold)

        # ========== 步骤2：根据策略构建数据集 ==========
        if self.positive_pair_strategy == 'simcse_dropout':
            print("="*60)
            print(" SimCSE策略检测：从剪枝后的高质量评论中构建单文本数据集")
            print("="*60)

            # ✅ 现在 forest.subtrees 已经构建完成
            self.dataset1 = SimCSEDataset(
                post_storage=self.post_storage,
                remove_duplicates=self.simcse_remove_duplicates,
                min_text_length=5,  # 与 ContrastiveDataset1 一致
                max_samples=self.max_samples_per_post_ds1,
                min_subtree_size=self.min_subtree_size_ds1
            )

            # SimCSE不使用Dataset2
            self.dataset2 = None

            ds1_size = len(self.dataset1)
            ds2_size = 0
            self.training_history['dataset_sizes']['dataset1'].append(ds1_size)
            self.training_history['dataset_sizes']['dataset2'].append(ds2_size)
            print(f"   SimCSEDataset 大小: {ds1_size}")

            if ds1_size == 0:
                print(" 警告: SimCSE数据集为空，训练无法进行。")

            # 创建数据整理器和加载器
            self.collator1 = ContrastiveDataCollator(self.dataset1, num_negatives=0)
            self.collator2 = None

            # ✅ 创建可复现的随机数生成器（用于 DataLoader shuffle）
            g1 = torch.Generator()
            g1.manual_seed(self.seed)

            self.train_loader1 = DataLoader(
                self.dataset1,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collator1,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False,
                generator=g1  # ✅ 绑定随机数生成器，确保 shuffle 可复现
            ) if ds1_size > 0 else None

            self.train_loader2 = None

            print(" SimCSE数据集和 DataLoader 已准备就绪。")

            # 保存SimCSE数据集
            print(" 正在保存构建好的SimCSE数据集...")
            # 使用实验目录而不是固定的cl_dataset目录
            if self.output_dir:
                save_dir = self.output_dir
            else:
                save_dir = "cl_dataset"
                os.makedirs(save_dir, exist_ok=True)

            sanitized_model_name = self.training_model_identifier_or_path.replace('/', '_')
            base_filename = f"{sanitized_model_name}_simcse"

            if ds1_size > 0:
                # 保存SimCSE数据集到实验目录
                ds1_filepath = os.path.join(save_dir, f"dataset1_sim_{self.similarity_threshold}.pkl")
                try:
                    with open(ds1_filepath, 'wb') as f:
                        pickle.dump(self.dataset1, f)
                    print(f"   -> SimCSEDataset 已保存至: {ds1_filepath}")
                except Exception as e:
                    print(f"   ->  保存 SimCSEDataset 失败: {e}")

            # ✅ 新增：为 Round 2+ 准备 comment-reply 数据集
            print("\n" + "="*60)
            print(" 为后续轮次构建 comment-reply 数据集（用于加权对比学习）")
            print("="*60)

            try:
                # 构建 comment-reply 数据集（与 comment-reply 策略相同）
                print("   创建 ContrastiveDataset1（comment-reply 对）...")
                comment_reply_dataset = ContrastiveDataset1(
                    post_storage=self.post_storage,
                    similarity_threshold=self.similarity_threshold,
                    min_subtree_size=self.min_subtree_size_ds1,
                    max_samples_per_post=self.max_samples_per_post_ds1
                )

                comment_reply_size = len(comment_reply_dataset)
                print(f"   ContrastiveDataset1 大小: {comment_reply_size}")

                # 保存 comment-reply 数据集（用于 Round 2+）
                if comment_reply_size > 0:
                    # 保存到实验目录，使用与dataset1类似的命名格式
                    comment_reply_filepath = os.path.join(save_dir, f"dataset1_comment_reply_sim_{self.similarity_threshold}.pkl")
                    try:
                        with open(comment_reply_filepath, 'wb') as f:
                            pickle.dump(comment_reply_dataset, f)
                        print(f"   -> comment-reply 数据集已保存至: {comment_reply_filepath}")
                        print(f"      此数据集将用于 Round 2+ 的加权对比学习")
                    except Exception as e:
                        print(f"   -> ⚠️  保存 comment-reply 数据集失败: {e}")
                else:
                    print(f"   -> ⚠️  comment-reply 数据集为空，未保存")

            except Exception as e:
                print(f"   -> ⚠️  构建 comment-reply 数据集失败: {e}")
                import traceback
                traceback.print_exc()

            print("="*60 + "\n")

            return  # SimCSE策略完成，直接返回

        # ========== Comment-Reply 策略 ==========
        print("   创建 Dataset1...")
        self.dataset1 = ContrastiveDataset1(
            post_storage=self.post_storage,
            similarity_threshold=self.similarity_threshold,
            min_subtree_size=self.min_subtree_size_ds1,
            max_samples_per_post=self.max_samples_per_post_ds1
        )

        # 只有权重>0时才创建Dataset2
        if self.loss_weights.get('dataset2', 0) > 0:
            print("   创建 Dataset2...")
            self.dataset2 = ContrastiveDataset2(
                post_storage=self.post_storage,
                min_subtree_size=self.min_subtree_size_ds2,
                max_samples_per_subtree=self.max_samples_per_subtree_ds2
            )
        else:
            print("   跳过 Dataset2 (权重为0)...")
            self.dataset2 = None

        ds1_size = len(self.dataset1)
        ds2_size = len(self.dataset2) if self.dataset2 else 0
        self.training_history['dataset_sizes']['dataset1'].append(ds1_size)
        self.training_history['dataset_sizes']['dataset2'].append(ds2_size)
        print(f"   Dataset1 大小: {ds1_size}, Dataset2 大小: {ds2_size}")

        if ds1_size == 0 and ds2_size == 0:
            print(" 警告: 两个数据集都为空。训练可能无法有效进行。")

        collator_num_neg = self.num_negatives
        self.collator1 = ContrastiveDataCollator(self.dataset1, num_negatives=collator_num_neg)

        # 只有Dataset2存在且权重>0时才创建collator2
        if self.dataset2 and self.loss_weights.get('dataset2', 0) > 0:
            self.collator2 = ContrastiveDataCollator(self.dataset2, num_negatives=collator_num_neg)
        else:
            self.collator2 = None

        # ✅ 创建可复现的随机数生成器（用于 DataLoader shuffle）
        g1 = torch.Generator()
        g1.manual_seed(self.seed)

        self.train_loader1 = DataLoader(
            self.dataset1,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator1,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            generator=g1  # ✅ 绑定随机数生成器，确保 shuffle 可复现
        ) if ds1_size > 0 else None

        # 只有Dataset2存在且权重>0时才创建train_loader2
        if self.dataset2 and ds2_size > 0 and self.loss_weights.get('dataset2', 0) > 0:
            # ✅ 创建独立的随机数生成器（Dataset2）
            g2 = torch.Generator()
            g2.manual_seed(self.seed)

            self.train_loader2 = DataLoader(
                self.dataset2,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collator2,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False,
                generator=g2  # ✅ 绑定随机数生成器，确保 shuffle 可复现
            )
        else:
            self.train_loader2 = None
        print(" 数据集和 DataLoader 已准备就绪。")

        # --- 新增代码：保存构建好的数据集 ---
        print(" 正在保存构建好的数据集...")
        save_dir = "cl_dataset"
        os.makedirs(save_dir, exist_ok=True)

        # 创建动态文件名
        sanitized_model_name = self.training_model_identifier_or_path.replace('/', '_')
        similarity_str = str(self.similarity_threshold)
        base_filename = f"{sanitized_model_name}_sim_{similarity_str}"

        # 保存 Dataset1
        if ds1_size > 0:
            ds1_filepath = os.path.join(save_dir, f"{base_filename}_dataset1.pkl")
            try:
                with open(ds1_filepath, 'wb') as f:
                    pickle.dump(self.dataset1, f)
                print(f"   -> Dataset1 已保存至: {ds1_filepath}")
            except Exception as e:
                print(f"   ->  保存 Dataset1 失败: {e}")
        else:
            print("   -> Dataset1 为空，不进行保存。")

        # 保存 Dataset2
        if self.dataset2 and ds2_size > 0:
            ds2_filepath = os.path.join(save_dir, f"{base_filename}_dataset2.pkl")
            try:
                with open(ds2_filepath, 'wb') as f:
                    pickle.dump(self.dataset2, f)
                print(f"   -> Dataset2 已保存至: {ds2_filepath}")
            except Exception as e:
                print(f"   ->  保存 Dataset2 失败: {e}")
        else:
            print("   -> Dataset2 为空或未启用，不进行保存。")

    def _process_batch(self, batch: Dict, loss_fn: ContrastiveLoss, dataset_name: str) -> Optional[torch.Tensor]:
        self.optimizer.zero_grad()

        anchor_texts = batch['anchor_texts']
        positive_texts_ds1 = batch.get('positive_texts_ds1', [])
        positive_content_lists_ds2 = batch.get('positive_content_lists_ds2', [])
        negative_texts_flat = batch.get('negative_texts', []) # 扁平列表
        num_negatives_per_anchor = batch.get('num_negatives', 0)

        if not anchor_texts: return None
        loss = None
        processed_anchors_count = 0 # 当前批次中实际处理的锚点数量

        if dataset_name == 'dataset1':
            # ========== SimCSE策略：简化版（使用专用数据集） ==========
            if batch.get('is_simcse', False):
                # SimCSE数据集已经提供了干净的单文本列表
                texts = anchor_texts  # 已经是单文本列表

                if not texts or len(texts) < 2:
                    return None

                # 确保dropout激活
                self.contrastive_encoder.train()

                # 两次前向传播，使用不同的dropout
                anchor_emb = self.contrastive_encoder(texts)
                positive_emb = self.contrastive_encoder(texts)  # 同样文本，不同dropout

                processed_anchors_count = len(texts)
                # 不打印详细信息，避免日志过多

            # ========== 原有策略：pair-based数据集 ==========
            else:
                valid_indices = [i for i, txt in enumerate(positive_texts_ds1) if txt is not None]
                if not valid_indices:
                    return None

                current_anchor_texts = [anchor_texts[i] for i in valid_indices]
                current_positive_texts = [positive_texts_ds1[i] for i in valid_indices]
                processed_anchors_count = len(current_anchor_texts)
                if processed_anchors_count == 0:
                    return None

                # ---  根据策略选择正样本对构造方式 ---
                if self.positive_pair_strategy == 'comment_reply':
                    # 原有的评论-回复策略
                    anchor_emb = self.contrastive_encoder(current_anchor_texts)
                    positive_emb = self.contrastive_encoder(current_positive_texts)

                elif self.positive_pair_strategy == 'hybrid':
                    # 混合策略：部分使用SimCSE（充分利用数据），部分使用评论回复
                    original_count = len(current_anchor_texts)
                    split_point = int(original_count * self.hybrid_ratio)

                    # SimCSE部分：使用部分父子评论的所有文本
                    simcse_anchor_texts = current_anchor_texts[:split_point]
                    simcse_positive_texts = current_positive_texts[:split_point]

                    simcse_all_texts = []
                    simcse_all_texts.extend([t for t in simcse_anchor_texts if t is not None and t.strip()])
                    simcse_all_texts.extend([t for t in simcse_positive_texts if t is not None and t.strip()])

                    # 去重（如果启用）
                    if self.simcse_remove_duplicates:
                        simcse_all_texts = list(dict.fromkeys(simcse_all_texts))

                    simcse_anchor_emb = None
                    simcse_positive_emb = None
                    if simcse_all_texts:
                        self.contrastive_encoder.train()
                        simcse_anchor_emb = self.contrastive_encoder(simcse_all_texts)
                        simcse_positive_emb = self.contrastive_encoder(simcse_all_texts)
                        print(f"混合策略-SimCSE部分：使用 {len(simcse_all_texts)} 个文本")

                    # 评论回复部分
                    reply_anchor_texts = current_anchor_texts[split_point:]
                    reply_positive_texts = current_positive_texts[split_point:]
                    reply_anchor_emb = None
                    reply_positive_emb = None
                    if reply_anchor_texts and reply_positive_texts:
                        reply_anchor_emb = self.contrastive_encoder(reply_anchor_texts)
                        reply_positive_emb = self.contrastive_encoder(reply_positive_texts)
                        print(f"混合策略-评论回复部分：使用 {len(reply_anchor_texts)} 对")

                    # 合并嵌入
                    anchor_emb_parts = [emb for emb in [simcse_anchor_emb, reply_anchor_emb] if emb is not None]
                    positive_emb_parts = [emb for emb in [simcse_positive_emb, reply_positive_emb] if emb is not None]

                    if anchor_emb_parts and positive_emb_parts:
                        anchor_emb = torch.cat(anchor_emb_parts, dim=0)
                        positive_emb = torch.cat(positive_emb_parts, dim=0)
                        processed_anchors_count = anchor_emb.shape[0]
                        print(f"混合策略：总计使用 {processed_anchors_count} 个样本")
                    else:
                        return None

            # 处理负样本（根据策略调整）
            # 注意：SimCSE专用数据集不使用额外负样本，仅使用in-batch负样本
            neg_emb_reshaped = None
            if self.infonce_mode != 'in_batch' and num_negatives_per_anchor > 0 and negative_texts_flat and not batch.get('is_simcse', False):
                if self.positive_pair_strategy == 'hybrid':
                    # 对于混合策略，需要扩展负样本
                    total_needed_negatives = processed_anchors_count * num_negatives_per_anchor

                    extended_negative_texts = []
                    original_neg_count = len(negative_texts_flat)
                    if original_neg_count > 0:
                        for i in range(total_needed_negatives):
                            extended_negative_texts.append(negative_texts_flat[i % original_neg_count])

                        neg_emb_flat = self.contrastive_encoder(extended_negative_texts)
                        if neg_emb_flat.nelement() > 0:
                            try:
                                neg_emb_reshaped = neg_emb_flat.view(processed_anchors_count, num_negatives_per_anchor, -1)
                            except RuntimeError as e:
                                print(f"混合策略 Reshape error: {e}")
                                return None

                else:
                    # 原有的评论回复策略：保持原有逻辑
                    current_negative_texts_for_batch = []
                    for original_batch_idx in valid_indices:
                        start_idx_flat = original_batch_idx * num_negatives_per_anchor
                        end_idx_flat = start_idx_flat + num_negatives_per_anchor
                        current_negative_texts_for_batch.extend(negative_texts_flat[start_idx_flat:end_idx_flat])

                    if current_negative_texts_for_batch:
                        neg_emb_flat = self.contrastive_encoder(current_negative_texts_for_batch)
                        if neg_emb_flat.nelement() > 0:
                            try:
                                neg_emb_reshaped = neg_emb_flat.view(processed_anchors_count, num_negatives_per_anchor, -1)
                            except RuntimeError as e:
                                print(f"DS1 Reshape error: {e}")
                                return None

            if anchor_emb.nelement() > 0 and positive_emb.nelement() > 0:
                # SimCSE专用数据集使用专用温度参数
                current_loss_fn = loss_fn
                if batch.get('is_simcse', False):
                    # 导入ContrastiveLoss
                    from cl_loss import ContrastiveLoss
                    current_loss_fn = ContrastiveLoss(
                        temperature=self.simcse_temperature,
                        loss_type='infonce',
                        infonce_mode=self.infonce_mode,
                        bidirectional_loss=self.bidirectional_loss
                    )

                loss = current_loss_fn(anchor_emb, positive_emb, neg_emb_reshaped if neg_emb_reshaped is not None and neg_emb_reshaped.nelement() > 0 else None)

        elif dataset_name == 'dataset2':
            valid_indices = [i for i, lst in enumerate(positive_content_lists_ds2) if lst is not None and any(s and s.strip() for s in lst)]
            if not valid_indices: return None

            current_anchor_texts = [anchor_texts[i] for i in valid_indices]
            current_positive_lists = [positive_content_lists_ds2[i] for i in valid_indices]
            processed_anchors_count = len(current_anchor_texts)
            if processed_anchors_count == 0: return None

            anchor_emb = self.contrastive_encoder(current_anchor_texts)
            positive_emb_list_for_ds2 = []
            for text_list_for_one_anchor in current_positive_lists:
                valid_texts_in_list = [s for s in text_list_for_one_anchor if s and s.strip()]
                if valid_texts_in_list:
                    # 获取基础嵌入，然后平均，然后投影
                    node_base_embeddings = self.contrastive_encoder.get_base_embeddings(valid_texts_in_list)
                    if node_base_embeddings.nelement() > 0:
                        avg_node_base_embedding = node_base_embeddings.mean(dim=0, keepdim=True)
                        projected_avg_emb = self.contrastive_encoder.projection_head(avg_node_base_embedding)
                        positive_emb_list_for_ds2.append(projected_avg_emb)
                    else: # 如果列表中的所有文本都无效/空，则添加零嵌入
                        proj_output_dim = self.contrastive_encoder.projection_head[-2].out_features
                        positive_emb_list_for_ds2.append(torch.zeros((1, proj_output_dim), device=self.device, dtype=anchor_emb.dtype))
                else: # 如果整个列表为空或仅包含无效字符串
                    proj_output_dim = self.contrastive_encoder.projection_head[-2].out_features
                    positive_emb_list_for_ds2.append(torch.zeros((1, proj_output_dim), device=self.device, dtype=anchor_emb.dtype))

            if not positive_emb_list_for_ds2 or len(positive_emb_list_for_ds2) != processed_anchors_count:
                 print(f"DS2: positive_emb_list 长度 ({len(positive_emb_list_for_ds2)}) 与锚点数量 ({processed_anchors_count}) 不匹配。")
                 return None
            positive_emb = torch.cat(positive_emb_list_for_ds2, dim=0)

            neg_emb_reshaped = None
            if self.infonce_mode != 'in_batch' and num_negatives_per_anchor > 0 and negative_texts_flat:
                current_negative_texts_for_batch = []
                for original_batch_idx in valid_indices:
                    start_idx_flat = original_batch_idx * num_negatives_per_anchor
                    end_idx_flat = start_idx_flat + num_negatives_per_anchor
                    current_negative_texts_for_batch.extend(negative_texts_flat[start_idx_flat:end_idx_flat])

                if current_negative_texts_for_batch:
                    neg_emb_flat = self.contrastive_encoder(current_negative_texts_for_batch)
                    if neg_emb_flat.nelement() > 0:
                        try:
                            neg_emb_reshaped = neg_emb_flat.view(processed_anchors_count, num_negatives_per_anchor, -1)
                        except RuntimeError as e:
                            print(f"DS2 Reshape error: {e}. Anchors: {processed_anchors_count}, Neg per anchor: {num_negatives_per_anchor}, Flat neg shape: {neg_emb_flat.shape}")
                            return None


            if anchor_emb.nelement() > 0 and positive_emb.nelement() > 0:
                loss = loss_fn(anchor_emb, positive_emb, neg_emb_reshaped if neg_emb_reshaped is not None and neg_emb_reshaped.nelement() > 0 else None)

        if loss is not None and loss.requires_grad:
            effective_weight = self.loss_weights.get(dataset_name, 1.0) if self.use_weighted_loss else 1.0
            (loss * effective_weight).backward()
            torch.nn.utils.clip_grad_norm_(self.contrastive_encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss # 返回未缩放的损失用于记录
        return None

    def train(self, num_epochs: int, rebuild_frequency: int, scheduler_patience: int, min_improvement: float):
        # ✅ 新增：处理 epochs=0 的特殊情况（保存原始模型）
        if num_epochs == 0:
            print("="*60)
            print("⚠️  检测到 epochs=0 - 原始模型保存模式")
            print("="*60)
            print("说明：将保存未经训练的原始BERT模型")
            print(f"模型：{self.training_model_identifier_or_path}")
            print(f"目的：用于Round 1去噪实验（先监督学习，再对比学习）")

            # 构建数据集（为了保存dataset.pkl）
            self._build_and_log_datasets()

            # 初始化训练历史（全0）
            self.training_history['epoch'].append(0)
            self.training_history['loss_ds1'].append(0.0)
            self.training_history['loss_ds2'].append(0.0)
            self.training_history['combined_loss'].append(0.0)
            self.training_history['learning_rate_base'].append(
                self.optimizer.param_groups[0]['lr']
            )
            self.training_history['learning_rate_proj'].append(
                self.optimizer.param_groups[1]['lr']
            )

            if self.use_weighted_loss:
                self.training_history['weight_ds1'].append(
                    self.loss_weights.get('dataset1', 1.0)
                )
                self.training_history['weight_ds2'].append(
                    self.loss_weights.get('dataset2', 0.0)
                )

            # 保存原始模型（标记为epoch=0）
            self.save_checkpoint(
                epoch_num=0,
                loss_val=0.0,
                is_best=True,
                final_save=True
            )

            print(f"✅ 原始模型已保存（未训练）")
            print(f"   保存路径: {self.output_dir}")
            print(f"   Epoch: 0")
            print(f"   Loss: 0.0 (占位值)")
            print("="*60)
            return

        # ========== 以下是原有的训练逻辑 ==========
        self.contrastive_encoder.train()
        best_overall_loss = float('inf')
        patience_counter = 0

        # 确保优化器中的 initial_lr 已设置
        for grp in self.optimizer.param_groups:
            if 'initial_lr' not in grp:
                grp['initial_lr'] = grp['lr']


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=scheduler_patience, threshold=min_improvement
        )
        self._build_and_log_datasets() # 初始数据集构建

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            if epoch > 0 and rebuild_frequency > 0 and epoch % rebuild_frequency == 0:
                print(f"Epoch {epoch+1}: 重建数据集...")
                self._build_and_log_datasets()

            self.contrastive_encoder.to(self.device) # 确保模型在正确的设备上
            self.contrastive_encoder.train() # 设置为训练模式

            epoch_losses_ds1 = []
            epoch_losses_ds2 = []

            # Dataset 1 训练循环
            if self.train_loader1 and len(self.train_loader1) > 0:
                print(f"在 Dataset1 上训练 (大小: {len(self.dataset1)} 样本, {len(self.train_loader1)} 批次)")
                progress_bar_ds1 = tqdm(self.train_loader1, desc=f"Epoch {epoch+1} DS1", leave=False, dynamic_ncols=True)
                for batch1 in progress_bar_ds1:
                    loss1_val = self._process_batch(batch1, self.loss_fn1, 'dataset1')
                    if loss1_val is not None:
                        epoch_losses_ds1.append(loss1_val.item())
                        progress_bar_ds1.set_postfix(loss=f"{loss1_val.item():.4f}")
            else:
                print("Dataset1 为空或未加载，跳过训练。")


            # Dataset 2 训练循环
            if (self.train_loader2 and len(self.train_loader2) > 0 and
                self.loss_weights.get('dataset2', 0) > 0):
                print(f"在 Dataset2 上训练 (大小: {len(self.dataset2)} 样本, {len(self.train_loader2)} 批次)")
                progress_bar_ds2 = tqdm(self.train_loader2, desc=f"Epoch {epoch+1} DS2", leave=False, dynamic_ncols=True)
                for batch2 in progress_bar_ds2:
                    loss2_val = self._process_batch(batch2, self.loss_fn2, 'dataset2')
                    if loss2_val is not None:
                        epoch_losses_ds2.append(loss2_val.item())
                        progress_bar_ds2.set_postfix(loss=f"{loss2_val.item():.4f}")
            else:
                print("Dataset2 训练被跳过 (权重为0或未启用)")


            avg_loss_ds1 = np.mean(epoch_losses_ds1) if epoch_losses_ds1 else 0.0
            avg_loss_ds2 = np.mean(epoch_losses_ds2) if epoch_losses_ds2 else 0.0
            current_epoch_combined_loss = 0.0
            w1, w2 = 0.0, 0.0 # 初始化权重

            if self.use_weighted_loss:
                w1_config = self.loss_weights.get('dataset1', 0.5)
                w2_config = self.loss_weights.get('dataset2', 0.5)

                active_datasets = 0
                if epoch_losses_ds1: active_datasets +=1
                if epoch_losses_ds2: active_datasets +=1

                if active_datasets == 0:
                    current_epoch_combined_loss = 0.0
                    w1, w2 = 0.0, 0.0
                elif active_datasets == 1:
                    if epoch_losses_ds1:
                        current_epoch_combined_loss = avg_loss_ds1
                        w1, w2 = 1.0, 0.0
                    else: # epoch_losses_ds2 must be active
                        current_epoch_combined_loss = avg_loss_ds2
                        w1, w2 = 0.0, 1.0
                else: # Both active
                    # Normalize configured weights
                    sum_config_w = w1_config + w2_config
                    if sum_config_w > 0:
                        w1 = w1_config / sum_config_w
                        w2 = w2_config / sum_config_w
                    else:
                        w1, w2 = 0.5, 0.5
                    current_epoch_combined_loss = (w1 * avg_loss_ds1) + (w2 * avg_loss_ds2)
            else: # 不使用加权损失，简单平均或取单个
                if epoch_losses_ds1 and epoch_losses_ds2:
                    current_epoch_combined_loss = (avg_loss_ds1 + avg_loss_ds2) / 2.0
                elif epoch_losses_ds1:
                    current_epoch_combined_loss = avg_loss_ds1
                elif epoch_losses_ds2:
                    current_epoch_combined_loss = avg_loss_ds2
                else:
                    current_epoch_combined_loss = 0.0 # 两个数据集都没有损失

            print(f"Epoch {epoch+1} 总结: 平均 DS1 损失: {avg_loss_ds1:.4f}, 平均 DS2 损失: {avg_loss_ds2:.4f}, 组合损失 (用于调度器): {current_epoch_combined_loss:.4f}")

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss_ds1'].append(avg_loss_ds1)
            self.training_history['loss_ds2'].append(avg_loss_ds2)
            self.training_history['combined_loss'].append(current_epoch_combined_loss)
            if self.use_weighted_loss:
                self.training_history['weight_ds1'].append(w1)
                self.training_history['weight_ds2'].append(w2)

            if self.adaptive_weighting and epoch_losses_ds1 and epoch_losses_ds2 and self.use_weighted_loss:
                # 简单的基于损失的自适应权重调整 (更高级的方法存在)
                # 损失越大，权重应该越大 (假设任务是最小化损失)
                # 或者，如果任务是最大化某个指标，则指标越小，权重越大
                # 这里我们最小化损失，所以损失大的数据集应该获得更多关注（即更高的权重）
                loss_sum_for_weighting = avg_loss_ds1 + avg_loss_ds2
                if loss_sum_for_weighting > 1e-9: # 避免除以零
                    # 权重与损失成正比
                    raw_w1_adaptive = avg_loss_ds1 / loss_sum_for_weighting if avg_loss_ds1 > 0 else 0
                    raw_w2_adaptive = avg_loss_ds2 / loss_sum_for_weighting if avg_loss_ds2 > 0 else 0

                    # 平滑更新，alpha_smooth 决定新计算的权重占多大比重
                    alpha_smooth = 0.3 # 0 表示完全使用初始权重, 1 表示完全使用当前计算的权重
                    self.loss_weights['dataset1'] = (1 - alpha_smooth) * self.initial_loss_weights['dataset1'] + alpha_smooth * raw_w1_adaptive

                    self.loss_weights['dataset2'] = (1 - alpha_smooth) * self.initial_loss_weights['dataset2'] + alpha_smooth * raw_w2_adaptive

                    # 重新归一化
                    current_sum_adapted_weights = self.loss_weights['dataset1'] + self.loss_weights['dataset2']
                    if current_sum_adapted_weights > 0:
                        self.loss_weights['dataset1'] /= current_sum_adapted_weights
                        self.loss_weights['dataset2'] /= current_sum_adapted_weights
                    else: # 如果两个权重都变为零，则回退
                        self.loss_weights['dataset1'], self.loss_weights['dataset2'] = self.initial_loss_weights['dataset1'], self.initial_loss_weights['dataset2']
                    print(f"自适应权重更新: DS1={self.loss_weights['dataset1']:.3f}, DS2={self.loss_weights['dataset2']:.3f}")


            scheduler.step(current_epoch_combined_loss)
            current_lr_base = self.optimizer.param_groups[0]['lr']
            current_lr_proj = self.optimizer.param_groups[1]['lr']
            self.training_history['learning_rate_base'].append(current_lr_base)
            self.training_history['learning_rate_proj'].append(current_lr_proj)
            print(f"当前学习率: Base={current_lr_base:.2e}, Projection={current_lr_proj:.2e}")


            if current_epoch_combined_loss < best_overall_loss - min_improvement:
                if epoch_losses_ds1 or epoch_losses_ds2: # 仅当实际发生训练时才保存
                    best_overall_loss = current_epoch_combined_loss
                    patience_counter = 0
                    print(f" 发现新的最佳模型! 组合损失: {best_overall_loss:.4f}. 保存模型...")
                    self.save_checkpoint(epoch + 1, best_overall_loss, is_best=True)
                else:
                    print("此轮未执行训练步骤。跳过最佳模型检查。")
            else:
                patience_counter += 1
                print(f"耐心计数: {patience_counter}/{scheduler.patience}")

            if patience_counter > scheduler.patience: # 注意：scheduler.patience 是 ReduceLROnPlateau 的参数
                print(" 早停触发。")
                break
            self.plot_training_progress(save_plot=False, show_plot=False) # 生成绘图数据以备保存

        print(" 训练完成。")
        # 保存最终模型，无论是否最佳
        self.save_checkpoint(epoch + 1, current_epoch_combined_loss, is_best=True, final_save=True)
        self.plot_training_progress(save_plot=True, show_plot=True) # 保存并显示最终绘图


    def save_checkpoint(self, epoch_num, loss_val, is_best=False, final_save=False):
        # 只在找到更优模型时才执行保存操作
        if not is_best:
            return

        # ========== 轻量级模式：只保存最小必需的模型文件 ==========
        if self.lightweight_mode:
            # 使用传入的output_dir或默认目录
            if self.output_dir:
                save_dir = self.output_dir
            else:
                model_name = self.training_model_identifier_or_path.replace('/', '_').replace('-', '_')
                strategy_name = self.positive_pair_strategy
                similarity_str = str(self.similarity_threshold).replace('.', 'p')
                experiment_folder_name = f"{model_name}_{strategy_name}_sim{similarity_str}"
                save_dir = os.path.join("model", experiment_folder_name)

            os.makedirs(save_dir, exist_ok=True)

            # 只保存模型权重（最小化文件大小）
            lightweight_state = {
                'contrastive_encoder_state_dict': self.contrastive_encoder.state_dict(),
                'training_model_type': self.training_model_type,
                'training_model_identifier_or_path': self.training_model_identifier_or_path,
                'projection_head_config': self.projection_config,
            }

            filepath = os.path.join(save_dir, "best_contrastive_model.pth")
            torch.save(lightweight_state, filepath)
            print(f"[轻量级模式] 模型已保存至: {filepath} (仅权重，约~400MB)")
            return

        # ========== 以下是正常的保存逻辑（单值运行时使用） ==========
        state = {
            'epoch': epoch_num,
            'contrastive_encoder_state_dict': self.contrastive_encoder.state_dict(),
            # ❌ 移除：不保存optimizer_state_dict（节省~730MB）
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': loss_val,
            'training_history': dict(self.training_history), # 保存训练历史的副本
            'training_model_type': self.training_model_type,
            'training_model_identifier_or_path': self.training_model_identifier_or_path,
            'projection_head_config': self.projection_config,
            'pruning_model_path': self.pruning_model_path,
            'similarity_threshold': self.similarity_threshold,
            'num_negatives': self.num_negatives,
            'batch_size': self.batch_size,
            'pruning_inference_batch_size': self.pruning_inference_batch_size,
            'base_lr_initial': self.optimizer.param_groups[0].get('initial_lr'),
            'projection_lr_initial': self.optimizer.param_groups[1].get('initial_lr'),
            'use_weighted_loss': self.use_weighted_loss,
            'loss_weights': self.loss_weights,
            'adaptive_weighting': self.adaptive_weighting,
            'infonce_mode': self.infonce_mode,
            # --- 新增PEFT状态 ---
            'use_peft': self.use_peft,
            'peft_config': self.peft_config,
            # --------------------
            # --- 新增正样本对策略状态 ---
            'positive_pair_strategy': self.positive_pair_strategy,
            'infonce_temperature': self.infonce_temperature,  # 新增：保存InfoNCE温度
            'simcse_temperature': self.simcse_temperature,
            'simcse_dropout_rate': self.simcse_dropout_rate,
            'simcse_remove_duplicates': self.simcse_remove_duplicates,  # 新增
            'hybrid_ratio': self.hybrid_ratio,
            'bidirectional_loss': self.bidirectional_loss,  # 新增：保存双向损失设置
            # --------------------------------
            'min_subtree_size_ds1': self.min_subtree_size_ds1,
            'max_samples_per_post_ds1': self.max_samples_per_post_ds1,
            'min_subtree_size_ds2': self.min_subtree_size_ds2,
            'max_samples_per_subtree_ds2': self.max_samples_per_subtree_ds2
        }
        if self.training_model_type == 'textcnn':
            state['textcnn_config'] = self.textcnn_config
            state['vocab'] = self.vocab

        #  改进的保存路径：包含模型名、策略、相似度阈值
        # 1. 处理模型名称
        model_name = self.training_model_identifier_or_path.replace('/', '_').replace('-', '_')
        
        # 2. 策略名称
        strategy_name = self.positive_pair_strategy
        if self.positive_pair_strategy == 'hybrid':
            strategy_name = f"hybrid_ratio{self.hybrid_ratio}"
        
        # 3. 相似度阈值
        similarity_str = str(self.similarity_threshold).replace('.', 'p')  # 0.95 -> 0p95
        
        # 4. 组合文件夹名
        experiment_folder_name = f"{model_name}_{strategy_name}_sim{similarity_str}"

        # 使用传入的output_dir或默认目录
        if self.output_dir:
            save_dir = self.output_dir
        else:
            save_dir = os.path.join("model", experiment_folder_name)

        os.makedirs(save_dir, exist_ok=True)

        # 生成损失图的PNG字节
        fig_bytes = self.plot_training_progress(save_plot=False, show_plot=False, return_bytes=True)
        if fig_bytes:
            # ❌ 删除：不再将PNG保存到checkpoint中（优化文件大小）
            # state['loss_plot_png'] = fig_bytes

            # ✅ 保留：将损失图保存为独立的PNG文件
            plot_filepath = os.path.join(save_dir, "training_loss_plot.png")
            with open(plot_filepath, 'wb') as f:
                f.write(fig_bytes)
            print(f" 训练损失图已保存至: {plot_filepath}")

        # 定义并保存最优模型的checkpoint文件
        filepath = os.path.join(save_dir, "best_contrastive_model.pth")
        torch.save(state, filepath)
        print(f" 最优模型已更新并保存至: {filepath}")
        print(f" 实验文件夹: {experiment_folder_name}")

        # 可选：保存基础模型到独立目录
        base_model_dir = os.path.join(save_dir, f"trained_{self.training_model_type}_embedding_model")
        os.makedirs(base_model_dir, exist_ok=True)
        self.contrastive_encoder.save_base_model(base_model_dir)
        print(f" 基础模型已保存至: {base_model_dir}")


    def plot_training_progress(self, save_plot=False, show_plot=True, return_bytes=False):
        if not self.training_history['epoch']:
            if return_bytes: return None
            return

        try:
            import matplotlib
            matplotlib.use('Agg') # 确保非交互式后端
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            print("Matplotlib 未找到。跳过绘图生成。")
            if return_bytes: return None
            return
        except UserWarning: pass # 忽略seaborn样式警告

        fig, ax1 = plt.subplots(figsize=(14, 8)) # 调整图形大小
        epochs_data = self.training_history['epoch']

        color_ds1_loss = 'orangered'
        color_ds2_loss = 'forestgreen'
        color_combined_loss = 'mediumblue'

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', color='black', fontsize=12)

        if 'loss_ds1' in self.training_history and any(v is not None and v > 0 for v in self.training_history['loss_ds1']):
            ax1.plot(epochs_data, self.training_history['loss_ds1'], color=color_ds1_loss, linestyle='--', marker='.', markersize=6, label='DS1 Loss (父子)')
        if 'loss_ds2' in self.training_history and any(v is not None and v > 0 for v in self.training_history['loss_ds2']):
            ax1.plot(epochs_data, self.training_history['loss_ds2'], color=color_ds2_loss, linestyle=':', marker='x', markersize=6, label='DS2 Loss (节点-中心)')
        if 'combined_loss' in self.training_history and any(v is not None for v in self.training_history['combined_loss']):
            ax1.plot(epochs_data, self.training_history['combined_loss'], color=color_combined_loss, linewidth=2.5, marker='o', markersize=4, label='组合损失 (调度器)')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.set_yscale('log') # 尝试对数刻度，如果损失范围很大

        ax2 = ax1.twinx()
        color_lr_base = 'purple'
        color_lr_proj = 'darkcyan'
        color_w1 = 'coral'
        color_w2 = 'lightgreen'

        ax2.set_ylabel('学习率 / 权重', color='dimgray', fontsize=12)
        if 'learning_rate_base' in self.training_history and any(v is not None for v in self.training_history['learning_rate_base']):
            ax2.plot(epochs_data, self.training_history['learning_rate_base'], color=color_lr_base, linestyle='-.', marker='s', markersize=4, label='LR (Base)')
        if 'learning_rate_proj' in self.training_history and any(v is not None for v in self.training_history['learning_rate_proj']):
            ax2.plot(epochs_data, self.training_history['learning_rate_proj'], color=color_lr_proj, linestyle='-.', marker='D', markersize=3, label='LR (Proj)')

        if self.use_weighted_loss and 'weight_ds1' in self.training_history and 'weight_ds2' in self.training_history:
            if any(v is not None for v in self.training_history['weight_ds1']):
                 ax2.plot(epochs_data, self.training_history['weight_ds1'], color=color_w1, linestyle=(0, (3, 5, 1, 5)), marker='^', markersize=5, label='权重 DS1') # (0, (3, 5, 1, 5)) is dashdotdotted
            if any(v is not None for v in self.training_history['weight_ds2']):
                 ax2.plot(epochs_data, self.training_history['weight_ds2'], color=color_w2, linestyle=(0, (3, 5, 1, 5)), marker='v', markersize=5, label='权重 DS2')

        ax2.tick_params(axis='y', labelcolor='dimgray', labelsize=10)
        ax2.set_ylim(bottom=0) # 学习率和权重非负
        # ax2.set_yscale('log') # 如果学习率变化范围也很大

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # 将图例放在图表下方
        fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, fontsize=9)

        model_name_for_title = self.training_model_identifier_or_path.split('/')[-1]
        plt.title(f'训练进度: {model_name_for_title} ({self.training_model_type.upper()})', fontsize=15, pad=25)
        fig.tight_layout(rect=[0, 0.08, 1, 0.95]) # 调整布局为图例留出空间

        plot_bytes_val = None
        if return_bytes or save_plot:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight') # bbox_inches='tight' 确保所有内容都保存
            buf.seek(0)
            plot_bytes_val = buf.getvalue()
            buf.close()

        # if save_plot and plot_bytes_val:
        #     plot_filename = f"training_progress_{self.training_model_identifier_or_path.replace('/', '_')}_{self.training_model_type}.png"
        #     with open(plot_filename, 'wb') as f:
        #         f.write(plot_bytes_val)
        #     print(f" 训练进度图已保存到 {plot_filename}")

        # if show_plot:
        #     # 由于使用了 'Agg' 后端，plt.show() 不会显示任何内容。
        #     # 用户需要打开保存的PNG文件来查看图像。
        #     print("绘图已生成。如果 save_plot=True，请检查保存的PNG文件。")

        plt.close(fig) # 关闭图形以释放内存

        if return_bytes:
            return plot_bytes_val
        return None

def build_pruned_forest(post_storage: PostStorage, similarity_threshold: float):
    """
    基于相似度阈值构建剪枝后的森林
    """
    print(" 构建剪枝森林...")
    post_storage.forests.clear()
    pruning_results = post_storage.prune_all_posts_by_similarity(
        similarity_threshold=similarity_threshold, show_progress=True
    )
    print(f" 森林构建完成: {len(pruning_results)} 个帖子")
    return pruning_results

def fine_tune_contrastive_model(
    model: ContrastiveEncoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveLoss,
    device: torch.device,
    num_epochs: int = 3,
    scheduler_patience: int = 2,
    min_improvement: float = 1e-5
):
    """
    对比模型微调
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, threshold=min_improvement
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches_processed = 0

        progress_bar = tqdm(train_loader, desc=f"微调对比模型 (Epoch {epoch+1}/{num_epochs})", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            anchor_texts = batch['anchor_texts']
            positive_texts_ds1 = batch['positive_texts_ds1']
            negative_texts = batch['negative_texts']
            num_negatives = batch['num_negatives']

            # 过滤掉 positive_texts_ds1 中的 None (来自Dataset2的占位符)
            valid_indices_ds1 = [i for i, txt in enumerate(positive_texts_ds1) if txt is not None]
            
            if valid_indices_ds1:
                anchor_texts_ds1 = [anchor_texts[i] for i in valid_indices_ds1]
                positive_texts_ds1_f = [positive_texts_ds1[i] for i in valid_indices_ds1]

                if anchor_texts_ds1: # 确保列表不为空
                    anchor_emb = model(anchor_texts_ds1)
                    positive_emb_ds1 = model(positive_texts_ds1_f)
                    
                    # 重构与过滤后样本对应的负样本
                    neg_emb = None
                    if negative_texts and num_negatives > 0:
                        # 从扁平列表中提取与 valid_indices_ds1 对应的负样本
                        current_batch_neg_texts = []
                        original_batch_size_collator = len(anchor_texts) # collator处理前的batch大小
                        for orig_idx in valid_indices_ds1:
                            start = orig_idx * num_negatives
                            end = start + num_negatives
                            current_batch_neg_texts.extend(negative_texts[start:end])
                        
                        if current_batch_neg_texts:
                            neg_emb_flat = model(current_batch_neg_texts)
                            neg_emb = neg_emb_flat.view(len(anchor_texts_ds1), num_negatives, -1)
                    
                    if neg_emb is not None and neg_emb.nelement() > 0:
                        loss = loss_fn(anchor_emb, positive_emb_ds1, neg_emb)
                    else:
                        loss = loss_fn(anchor_emb, positive_emb_ds1)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches_processed += 1

            progress_bar.set_postfix(loss=epoch_loss / batches_processed if batches_processed > 0 else epoch_loss)

        avg_loss = epoch_loss / batches_processed if batches_processed > 0 else 0.0
        scheduler.step(avg_loss)

        if avg_loss < best_loss - min_improvement:
            best_loss = avg_loss
            patience_counter = 0
            print(f" 保存当前最佳模型，损失: {best_loss:.4f}")
            # 可以选择保存模型
            # torch.save(model.state_dict(), "best_contrastive_model.pth")
        else:
            patience_counter += 1
            print(f"⏳ 等待更优模型，当前计数器: {patience_counter}/{scheduler_patience}")

        if patience_counter >= scheduler_patience:
            print(" 早停触发，停止训练")








