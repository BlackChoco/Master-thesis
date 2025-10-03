import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
import random
import itertools
import hashlib
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# --- 关键修改: 导入 AutoModel 和 AutoTokenizer ---
from modelscope import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split # 用于从训练集中划分验证集
from peft import get_peft_model, LoraConfig


# 从解耦后的模块导入必要的类
from cl_base_model import ContrastiveEncoder, TextCNNModel, TextCNNTokenizer

# --- 2. 编码器缓存相关类 ---

class EncodedDataset(Dataset):
    """使用预编码特征的数据集，用于冻结编码器的场景"""
    def __init__(self, encoded_features: torch.Tensor, labels: list, label_to_id: dict):
        self.encoded_features = encoded_features
        self.labels = labels
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'encoded_features': self.encoded_features[idx],
            'labels': torch.tensor(self.label_to_id[self.labels[idx]], dtype=torch.long)
        }

def generate_cache_key(model_name: str, texts: list, data_fraction: float = 1.0, max_len: int = 256, round_num: int = None) -> str:
    """为文本列表和模型生成唯一的缓存键，包含轮次信息"""
    # 使用模型名称、文本内容的哈希值、数据比例生成缓存键
    texts_str = ''.join(texts[:100])  # 只用前100个文本计算哈希，避免过长
    content_hash = hashlib.md5(f"{model_name}_{texts_str}_{len(texts)}_{data_fraction}_{max_len}".encode()).hexdigest()[:8]
    # 包含轮次信息和简短hash
    if round_num:
        return f"{model_name}_round{round_num}_frac{data_fraction}_{content_hash}"
    return f"{model_name}_frac{data_fraction}_{content_hash}"

def encode_texts_with_model(texts: list, encoder, tokenizer, device, batch_size: int = 64, max_len: int = 256):
    """使用编码器批量编码文本"""
    encoder.eval()
    encoder = encoder.to(device)  # 确保编码器在正确的设备上
    all_features = []

    print(f" 正在编码 {len(texts)} 个文本...")

    # 创建临时数据集用于批量编码
    temp_dataset = SupervisedTextDataset(
        texts, [0] * len(texts),  # 临时标签，不会使用
        tokenizer, {0: 0}, max_len
    )
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(temp_loader, desc="编码中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 获取编码器输出
            base_output = encoder(input_ids=input_ids, attention_mask=attention_mask)

            # 提取特征（与SupervisedModel.forward中的逻辑相同）
            if isinstance(base_output, dict):
                if 'last_hidden_state' in base_output:
                    features = base_output['last_hidden_state'][:, 0, :]
                elif 'hidden_states' in base_output:
                    features = base_output['hidden_states'][:, 0, :]
                elif 'pooler_output' in base_output and base_output['pooler_output'] is not None:
                    features = base_output['pooler_output']
                else:
                    raise KeyError(f"在模型输出字典中找不到可用的特征键。可用键: {base_output.keys()}")
            elif hasattr(base_output, 'pooler_output') and base_output.pooler_output is not None:
                features = base_output.pooler_output
            elif hasattr(base_output, 'last_hidden_state'):
                features = base_output.last_hidden_state[:, 0, :]
            else:
                features = base_output

            # 确保特征是float32
            if features.dtype != torch.float32:
                features = features.to(torch.float32)

            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)

def load_or_create_cache(cache_key: str, cache_dir: str, texts: list, encoder, tokenizer, device, config):
    """加载或创建编码缓存"""
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        print(f" 从缓存加载编码特征: {cache_path}")
        return torch.load(cache_path, map_location='cpu')
    else:
        print(f" 缓存不存在，正在创建新缓存...")
        os.makedirs(cache_dir, exist_ok=True)

        encoded_features = encode_texts_with_model(
            texts, encoder, tokenizer, device,
            batch_size=config['optimization']['cache_batch_size']
        )

        # 保存缓存
        torch.save(encoded_features, cache_path)
        print(f" 编码特征已缓存到: {cache_path}")

        return encoded_features

def save_best_model_for_seed(model, model_state, hyperparams, best_val_f1, test_metrics, experiment_output_dir):
    """为每个种子保存最优模型"""
    model_name = hyperparams['model_name'].replace('/', '_').replace('-', '_')

    #  修改：在实验目录下创建saved_models子目录
    models_dir = os.path.join(experiment_output_dir, 'saved_models')
    save_dir = os.path.join(models_dir, f"{model_name}_frac{hyperparams['data_fraction']}_seed{hyperparams['seed']}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型权重和元信息
    model_path = os.path.join(save_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model_state,
        'hyperparameters': hyperparams,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'model_architecture': type(model).__name__
    }, model_path)

    print(f" 种子{hyperparams['seed']}的最优模型已保存到: {model_path}")

    return {
        'model_path': model_path,
        'save_dir': save_dir
    }

class CachedSupervisedModel(nn.Module):
    """使用缓存特征的监督模型，用于冻结编码器的场景"""
    def __init__(self, encoder_output_dim: int, num_labels: int, classifier_type: str = 'linear', mlp_hidden_neurons: int = 384):
        super().__init__()

        print(f"缓存模式分类器接收的特征维度: {encoder_output_dim}")

        if classifier_type == 'linear':
            self.classifier = nn.Linear(encoder_output_dim, num_labels)
        elif classifier_type == 'mlp':
            # 两层MLP分类器：指定中间层神经元数量
            self.classifier = nn.Sequential(
                nn.Linear(encoder_output_dim, mlp_hidden_neurons),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_neurons, num_labels)
            )
            print(f"缓存模式MLP分类器结构: 2层，维度变化: {encoder_output_dim} -> {mlp_hidden_neurons} -> {num_labels}")
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}。请选择 'linear' 或 'mlp'。")

    def forward(self, encoded_features):
        return self.classifier(encoded_features)

# --- 3. 修改后的训练和评估函数 ---

def train_epoch_cached(model, data_loader, loss_fn, optimizer, device, scheduler):
    """缓存模式的训练轮次"""
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="训练中", leave=False):
        encoded_features = batch['encoded_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(encoded_features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model_cached(model, data_loader, loss_fn, device, id_to_label: dict):
    """缓存模式的模型评估"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中", leave=False):
            encoded_features = batch['encoded_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(encoded_features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Macro指标：各类别简单平均
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Micro指标：仅计算F1用于验证
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    class_ids = sorted(id_to_label.keys())
    target_names = [f"class_{id_to_label[cid]}" for cid in class_ids]
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=class_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics = {}
    for name in target_names:
        if name in report_dict:
            metrics = report_dict[name].copy()
            metrics.pop('support', None)
            per_class_metrics[name] = metrics

    return {
        # 核心指标：适合均衡数据集的5个关键指标
        "accuracy": accuracy,                # 整体准确率
        "precision": precision_macro,        # Macro精确率
        "recall": recall_macro,             # Macro召回率
        "f1_score": f1_macro,               # Macro F1分数
        "f1_micro": f1_micro,               # Micro F1（验证用，应等于accuracy）
        # 辅助信息
        "loss": avg_loss,
        "per_class_metrics": per_class_metrics
    }

# --- 3. 数据集和模型定义 ---

class SupervisedTextDataset(Dataset):
    """用于有监督文本分类的PyTorch数据集。"""
    def __init__(self, texts: list, labels: list, tokenizer, label_to_id: dict, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用与预训练时相同的分词器
        # 检查分词器类型，因为 TextCNNTokenizer 和 HF Tokenizer 的参数不同
        if isinstance(self.tokenizer, TextCNNTokenizer):
            encoding = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_len
            )
        else: # 假设是 HuggingFace Tokenizer
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_to_id[label], dtype=torch.long)
        }

class SupervisedModel(nn.Module):
    """
    包装预训练的编码器和一个分类头。
    """
    def __init__(self, base_encoder: nn.Module, num_labels: int, classifier_type: str = 'linear', mlp_hidden_neurons: int = 384):
        super().__init__()
        self.base_encoder = base_encoder

        # 从基础编码器获取隐藏层维度
        if hasattr(base_encoder, 'base_dim'): # 适用于我们自定义的TextCNNModel
            hidden_size = base_encoder.base_dim
        elif hasattr(base_encoder.config, 'hidden_size'): # 适用于HuggingFace/ModelScope模型
            hidden_size = base_encoder.config.hidden_size
        else:
            # 尝试从模型输出获取维度（如果模型已加载参数）
            try:
                # 创建一个虚拟输入来推断维度
                dummy_input = {'input_ids': torch.randint(0, 100, (1, 10)), 'attention_mask': torch.ones(1, 10)}
                dummy_output = base_encoder(**dummy_input)

                # 调试：打印输出的所有键
                print(f"调试: 模型输出键: {dummy_output.keys()}")

                # ModelScope 模型通常返回字典，尝试不同的键
                if 'last_hidden_state' in dummy_output:
                    hidden_size = dummy_output['last_hidden_state'].shape[-1]
                elif 'hidden_states' in dummy_output:
                    hidden_size = dummy_output['hidden_states'].shape[-1]
                else:
                    raise KeyError("在模型输出中找不到 'last_hidden_state' 或 'hidden_states'。")

            except Exception as e:
                raise ValueError(f"无法自动确定基础编码器的输出维度: {e}")

        print(f"分类器接收的隐藏层维度: {hidden_size}")

        if classifier_type == 'linear':
            self.classifier = nn.Linear(hidden_size, num_labels)
        elif classifier_type == 'mlp':
            # 两层MLP分类器：指定中间层神经元数量
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_neurons),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_neurons, num_labels)
            )
            print(f"MLP分类器结构: 2层，维度变化: {hidden_size} -> {mlp_hidden_neurons} -> {num_labels}")
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}。请选择 'linear' 或 'mlp'。")

    def forward(self, input_ids, attention_mask):
        # 基础编码器输出
        # ModelScope 模型需要关键字参数
        base_output = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 根据基础编码器的输出类型提取特征
        # ModelScope 模型通常返回一个字典
        if isinstance(base_output, dict):
            if 'last_hidden_state' in base_output:
                # [CLS] token representation
                features = base_output['last_hidden_state'][:, 0, :]
            elif 'hidden_states' in base_output:
                # 有些模型可能只返回 hidden_states
                features = base_output['hidden_states'][:, 0, :]
            elif 'pooler_output' in base_output and base_output['pooler_output'] is not None:
                features = base_output['pooler_output']
            else:
                # 如果都找不到，抛出错误
                raise KeyError(f"在模型输出字典中找不到可用的特征键。可用键: {base_output.keys()}")
        elif hasattr(base_output, 'pooler_output') and base_output.pooler_output is not None:
            features = base_output.pooler_output
        elif hasattr(base_output, 'last_hidden_state'):
            features = base_output.last_hidden_state[:, 0, :]
        else:
            features = base_output

        # --- 修正：确保 features 是 float32 ---
        if features.dtype != torch.float32:
            features = features.to(torch.float32)
        # --------------------------------------

        logits = self.classifier(features)
        return logits

    def freeze_encoder(self):
        """冻结基础编码器的所有参数。"""
        print(" 正在冻结基础编码器的参数...")
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        print(" 基础编码器已冻结。")

    def unfreeze_encoder(self):
        """解冻基础编码器的所有参数。"""
        print(" 正在解冻基础编码器的参数...")
        for param in self.base_encoder.parameters():
            param.requires_grad = True
        print(" 基础编码器已解冻。")


# --- 2. 辅助函数 ---

def set_seed(seed_value=42):
    """为所有相关库设置随机种子以保证可复现性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_data(train_path, test_path):
    """加载训练和测试数据集。"""
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print(f"加载数据: 训练集 {len(df_train)} 行, 测试集 {len(df_test)} 行。")
        return df_train, df_test
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        return None, None

def load_pretrained_encoder(checkpoint_path: str):
    """
    从对比学习阶段的checkpoint加载基础模型和分词器，或直接从ModelScope Hub加载。
    """
    if os.path.exists(checkpoint_path):
        print(f"正在从本地checkpoint {checkpoint_path} 加载...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

        model_type = checkpoint.get('training_model_type', 'hf')
        model_identifier = checkpoint.get('training_model_identifier_or_path')
        proj_config = checkpoint.get('projection_head_config')
        use_peft = checkpoint.get('use_peft', False)
        peft_config = checkpoint.get('peft_config', None)

        temp_encoder = None
        if model_type == 'textcnn':
            textcnn_conf = checkpoint['textcnn_config']
            vocab_data = checkpoint['vocab']
            temp_encoder = ContrastiveEncoder(
                model_type='textcnn', vocab=vocab_data, textcnn_config=textcnn_conf,
                projection_hidden_dim=proj_config['hidden_dim'],
                projection_output_dim=proj_config['output_dim'],
                projection_dropout_rate=proj_config['dropout_rate']
            )
        elif model_type in ['hf', 'modelscope']:
            temp_encoder = ContrastiveEncoder(
                model_type='modelscope', model_name_or_path=model_identifier,
                projection_hidden_dim=proj_config['hidden_dim'],
                projection_output_dim=proj_config['output_dim'],
                projection_dropout_rate=proj_config['dropout_rate']
            )
            # --- LoRA/PEFT模型特殊处理 ---
            if use_peft and peft_config is not None:
                print("检测到PEFT/LoRA模型，正在应用LoRA结构...")
                lora_config = LoraConfig(**peft_config)
                temp_encoder.base_model = get_peft_model(temp_encoder.base_model, lora_config)
                print("LoRA结构已应用。")
        else:
            print(f"错误: Checkpoint中未知的模型类型 '{model_type}'")
            return None, None, None

        # 加载权重时允许strict=False，兼容LoRA权重
        temp_encoder.load_state_dict(checkpoint['contrastive_encoder_state_dict'], strict=False)
        print(f" Checkpoint 加载成功。模型类型: {model_type.upper()}")

        return temp_encoder.base_model, temp_encoder.tokenizer, model_type
    
    # 如果不是本地文件，则尝试从 ModelScope Hub 加载
    else:
        print(f"未找到本地文件，尝试从 ModelScope Hub 加载模型: {checkpoint_path}")
        try:
            # --- 关键修改: 使用 AutoModel 和 AutoTokenizer ---
            base_model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            print(f" 成功从 ModelScope Hub 加载基础模型和分词器。")
            return base_model, tokenizer, 'ms' # 返回 'ms' 作为模型类型
        except Exception as e:
            print(f"错误: 无法从 ModelScope Hub 加载模型 '{checkpoint_path}': {e}")
            return None, None, None


# --- 3. 训练和评估逻辑 ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """执行一个训练轮次。"""
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="训练中", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, loss_fn, device, id_to_label: dict):
    """在验证集或测试集上评估模型。"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # --- 计算整体指标 ---
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Macro指标：各类别简单平均
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Micro指标：仅计算F1用于验证
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    # --- 计算每个类别的指标（保留用于详细分析） ---
    class_ids = sorted(id_to_label.keys())
    target_names = [f"class_{id_to_label[cid]}" for cid in class_ids]
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=class_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    per_class_metrics = {}
    for name in target_names:
        if name in report_dict:
            metrics = report_dict[name].copy()
            metrics.pop('support', None)
            per_class_metrics[name] = metrics

    return {
        # 核心指标：适合均衡数据集的5个关键指标
        "accuracy": accuracy,                # 整体准确率
        "precision": precision_macro,        # Macro精确率
        "recall": recall_macro,             # Macro召回率
        "f1_score": f1_macro,               # Macro F1分数
        "f1_micro": f1_micro,               # Micro F1（验证用，应等于accuracy）
        # 辅助信息
        "loss": avg_loss,
        "per_class_metrics": per_class_metrics
    }

# --- 4. 超参数搜索配置 ---

# 统一的实验配置字典
CONFIG = {
    # 实验元信息
    'experiment_meta': {
        'description': '0.75_0.1_round1__experiment',  # 实验描述标识符
        'experiment_name': 'mlp在6个标签均衡数据集',   # 实验的中文名称
        'purpose': '对比LoRA微调后的BERT与原始BERT在不同数据量下的性能表现（均衡数据集）,在第一阶段对比学习训练的编码器上进行加权对比损失训练',  # 实验目的
        'notes': '使用mlp，测试5个不同数据比例',  # 实验备注
    },

    # 数据配置
    'data': {
        'train_data_path': 'data/sup_train_data/balanced_trainset.csv',
        'test_data_path': 'data/sup_train_data/balanced_testset.csv',
        'validation_split': 0.2,  # 验证集比例（仅在使用比例分割时有效）
        'excluded_labels': [],   # 要过滤的标签
        # 新增：固定数量分割配置
        'use_fixed_split': True,    # 是否使用固定数量分割而非比例分割
        'train_samples_per_label': 500,  # 每个标签的训练样本数
        'val_samples_per_label': 200,    # 每个标签的验证样本数
    },

    # 模型配置
    'models': {
        # 'lora_bert_base_chinese_cl': 'model/google-bert_bert-base-chinese/best_contrastive_model.pth',
        # 'TextCNN_CL_bert': 'model/my_custom_textcnn_v3_bert_pruning_paircl/best_contrastive_model.pth',
        # 'Bert_base_chinese_nocl': 'google-bert/bert-base-chinese',
        '0.75_round1_0.1_cl_bert' : 'iter_model/frac0.1_round1/best_model.pth'
    },

    # 超参数搜索空间
    'hyperparameters': {
        'epochs': [100],                    # 训练轮数      [50,100]
        'batch_size':[16] ,              # 批次大小         [16,32,64,128]
        'learning_rate': [1e-3], # 学习率       [1e-3,1e-4]
        'data_fractions': [1.0, 0.5, 0.2, 0.1, 0.05, 0.02],  # 数据使用比例       [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
        'seeds': [42, 123, 456, 789, 101, 202, 303, 404, 505, 606],             # 随机种子
        'classifier_types': ['mlp'], # 分类器类型
        'mlp_hidden_neurons': [512,384,256],  # MLP隐藏层神经元数量
        'freeze_encoder': [True],     # 是否冻结编码器
    },

    # 实验控制
    'experiment': {
        'base_output_dir': 'sup_result_hyperparams',  # 基础输出目录
        'save_individual_results': True,
        'aggregate_results': True,
        'save_experiment_info': True,  # 保存实验信息
    },

    # 性能优化
    'optimization': {
        'use_encoder_cache': True,       # 冻结编码器时是否使用缓存
        'cache_dir': 'encoder_cache',    # 缓存目录
        'cache_batch_size': 64,          # 缓存时的批次大小
    }
}

def generate_hyperparameter_combinations(config):
    """生成所有超参数组合"""
    hyperparams = config['hyperparameters']

    # 创建超参数组合
    combinations = []
    for (model_name, checkpoint_path), epochs, batch_size, lr, fraction, seed, classifier_type, mlp_hidden_neurons, freeze in itertools.product(
        config['models'].items(),
        hyperparams['epochs'],
        hyperparams['batch_size'],
        hyperparams['learning_rate'],
        hyperparams['data_fractions'],
        hyperparams['seeds'],
        hyperparams['classifier_types'],
        hyperparams['mlp_hidden_neurons'],
        hyperparams['freeze_encoder']
    ):
        # 只有当classifier_type='mlp'时，mlp_hidden_neurons参数才有意义
        # 当classifier_type='linear'时，跳过不同神经元数量的组合
        if classifier_type == 'linear' and mlp_hidden_neurons != hyperparams['mlp_hidden_neurons'][0]:
            continue

        combinations.append({
            'model_name': model_name,
            'checkpoint_path': checkpoint_path,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'data_fraction': fraction,
            'seed': seed,
            'classifier_type': classifier_type,
            'mlp_hidden_neurons': mlp_hidden_neurons,
            'freeze_encoder': freeze,
        })

    return combinations

def run_single_experiment(config, hyperparams, experiment_output_dir=None, round_num=None):
    """运行单次实验"""

    print(f"\n--- 实验配置 ---")
    print(f"模型: {hyperparams['model_name']}")
    print(f"分类器: {hyperparams['classifier_type']}")
    if hyperparams['classifier_type'] == 'mlp':
        print(f"MLP隐藏层神经元: {hyperparams['mlp_hidden_neurons']}")
    print(f"冻结编码器: {hyperparams['freeze_encoder']}")
    print(f"数据比例: {hyperparams['data_fraction']*100}%")
    print(f"学习率: {hyperparams['learning_rate']}")
    print(f"批次大小: {hyperparams['batch_size']}")
    print(f"训练轮数: {hyperparams['epochs']}")
    print(f"随机种子: {hyperparams['seed']}")
    if round_num:
        print(f"实验轮次: 第{round_num}轮")

    # 缓存优化提示
    use_cache = config['optimization']['use_encoder_cache'] and hyperparams['freeze_encoder']
    if use_cache:
        print(" 缓存优化: 启用")
    else:
        print(" 缓存优化: 禁用 (编码器未冻结或缓存功能关闭)")

    # 加载数据
    df_train_full, df_test = load_data(config['data']['train_data_path'], config['data']['test_data_path'])
    if df_train_full is None:
        return None

    # 过滤标签
    for label in config['data']['excluded_labels']:
        df_train_full = df_train_full[df_train_full['label'] != label].reset_index(drop=True)
        df_test = df_test[df_test['label'] != label].reset_index(drop=True)

    #  关键修改：根据配置选择数据分割方式
    if config['data']['use_fixed_split']:
        print(f" 使用固定数量分割: 每个标签 {config['data']['train_samples_per_label']} 训练 + {config['data']['val_samples_per_label']} 验证...")

        # 按标签分组并固定采样
        df_train_list = []
        df_val_list = []

        for label in df_train_full['label'].unique():
            label_data = df_train_full[df_train_full['label'] == label].reset_index(drop=True)

            # 检查每个标签的数据量是否足够
            required_total = config['data']['train_samples_per_label'] + config['data']['val_samples_per_label']
            if len(label_data) < required_total:
                print(f"  警告: 标签 '{label}' 只有 {len(label_data)} 条数据，需要 {required_total} 条")
                print(f"   将使用所有可用数据，按原比例分割...")
                # 如果数据不够，按原比例分割
                label_train, label_val = train_test_split(
                    label_data,
                    test_size=config['data']['validation_split'],
                    random_state=42
                )
            else:
                # 固定种子随机采样
                label_data_shuffled = label_data.sample(n=len(label_data), random_state=42).reset_index(drop=True)

                # 按固定数量分割
                label_train = label_data_shuffled[:config['data']['train_samples_per_label']]
                label_val = label_data_shuffled[config['data']['train_samples_per_label']:config['data']['train_samples_per_label'] + config['data']['val_samples_per_label']]

            df_train_list.append(label_train)
            df_val_list.append(label_val)
            print(f"   标签 '{label}': {len(label_train)} 训练, {len(label_val)} 验证")

        df_train_prelim = pd.concat(df_train_list, ignore_index=True)
        df_val = pd.concat(df_val_list, ignore_index=True)

        print(f" 固定数量分割完成: 训练集 {len(df_train_prelim)} 条, 验证集 {len(df_val)} 条")
    else:
        print(" 使用比例分割，确保实验间数据一致性...")
        df_train_prelim, df_val = train_test_split(
            df_train_full,
            test_size=config['data']['validation_split'],
            random_state=42,  # 固定种子！所有实验使用相同的数据分割
            stratify=df_train_full['label']
        )

    # 数据采样（如果需要）
    if hyperparams['data_fraction'] < 1.0:
        df_train = df_train_prelim.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=hyperparams['data_fraction'], random_state=hyperparams['seed'])
        ).reset_index(drop=True)
    else:
        df_train = df_train_prelim.reset_index(drop=True)

    #  关键修改：在数据分割后设置实验随机种子
    print(f" 设置实验随机种子 {hyperparams['seed']} (影响模型初始化和训练过程)...")
    set_seed(hyperparams['seed'])

    # 生成标签映射
    unique_labels = sorted(df_train_full['label'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)

    # 加载预训练编码器
    base_encoder, tokenizer, _ = load_pretrained_encoder(hyperparams['checkpoint_path'])
    if base_encoder is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保编码器在正确的设备上
    base_encoder = base_encoder.to(device)

    # 根据是否使用缓存选择不同的处理路径
    if use_cache:
        print("\n 使用缓存优化模式...")

        # 为数据集生成缓存 - 保存在实验目录下
        if experiment_output_dir:
            cache_dir = os.path.join(experiment_output_dir, 'encoder_cache')
        else:
            cache_dir = config['optimization']['cache_dir']

        #  修复：基于完整数据集生成缓存键，包含轮次信息
        train_cache_key = generate_cache_key(hyperparams['model_name'], df_train_prelim['content'].tolist(), 1.0, 256, round_num)  # 基于完整训练集
        val_cache_key = generate_cache_key(hyperparams['model_name'], df_val['content'].tolist(), 1.0, 256, round_num)  # 验证集总是100%
        test_cache_key = generate_cache_key(hyperparams['model_name'], df_test['content'].tolist(), 1.0, 256, round_num)  # 测试集总是100%

        # 加载或创建缓存（基于完整数据集）
        train_features_full = load_or_create_cache(train_cache_key, cache_dir, df_train_prelim['content'].tolist(),
                                                  base_encoder, tokenizer, device, config)
        val_features = load_or_create_cache(val_cache_key, cache_dir, df_val['content'].tolist(),
                                          base_encoder, tokenizer, device, config)
        test_features = load_or_create_cache(test_cache_key, cache_dir, df_test['content'].tolist(),
                                           base_encoder, tokenizer, device, config)

        #  关键修复：根据采样后的训练集选择对应的特征
        if hyperparams['data_fraction'] < 1.0:
            # 获取采样后训练集在原始训练集中的索引
            train_indices = df_train_prelim.index[df_train_prelim['content'].isin(df_train['content'])].tolist()
            train_features = train_features_full[train_indices]
            print(f" 从完整训练特征({train_features_full.shape[0]})中选择采样特征({train_features.shape[0]})")
        else:
            train_features = train_features_full

        # 创建缓存数据集
        train_dataset = EncodedDataset(train_features, df_train['label'].tolist(), label_to_id)
        val_dataset = EncodedDataset(val_features, df_val['label'].tolist(), label_to_id)
        test_dataset = EncodedDataset(test_features, df_test['label'].tolist(), label_to_id)

        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

        # 构建缓存模式的监督模型
        encoder_output_dim = train_features.shape[1]
        model = CachedSupervisedModel(
            encoder_output_dim=encoder_output_dim,
            num_labels=num_labels,
            classifier_type=hyperparams['classifier_type'],
            mlp_hidden_neurons=hyperparams['mlp_hidden_neurons'] if hyperparams['classifier_type'] == 'mlp' else 384
        ).to(device)

        # 设置优化器
        optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'])

        # 准备训练
        loss_fn = nn.CrossEntropyLoss()
        total_steps = len(train_loader) * hyperparams['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_val_f1 = -1
        best_model_state = None

        # 训练循环（缓存模式）
        for epoch in range(hyperparams['epochs']):
            print(f"  Epoch {epoch + 1}/{hyperparams['epochs']}")
            train_loss = train_epoch_cached(model, train_loader, loss_fn, optimizer, device, scheduler)
            val_metrics = evaluate_model_cached(model, val_loader, loss_fn, device, id_to_label)
            print(f"  训练损失: {train_loss:.4f} | 验证F1: {val_metrics['f1_score']:.4f} | 验证准确率: {val_metrics['accuracy']:.4f}")

            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_model_state = model.state_dict()
                print(f"   新的最佳验证F1分数: {best_val_f1:.4f}")

        # 测试集评估（缓存模式）
        if best_model_state:
            model.load_state_dict(best_model_state)
        print(" 使用最佳模型在测试集上进行最终评估...")
        test_metrics = evaluate_model_cached(model, test_loader, loss_fn, device, id_to_label)

        #  新增: 保存最优模型
        model_save_info = save_best_model_for_seed(model, best_model_state, hyperparams, best_val_f1, test_metrics, experiment_output_dir)

    else:
        print("\n 使用标准模式...")

        # 标准模式（原有逻辑）
        train_dataset = SupervisedTextDataset(df_train['content'].tolist(), df_train['label'].tolist(), tokenizer, label_to_id)
        val_dataset = SupervisedTextDataset(df_val['content'].tolist(), df_val['label'].tolist(), tokenizer, label_to_id)
        test_dataset = SupervisedTextDataset(df_test['content'].tolist(), df_test['label'].tolist(), tokenizer, label_to_id)

        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

        # 构建监督模型
        model = SupervisedModel(
            base_encoder=base_encoder,
            num_labels=num_labels,
            classifier_type=hyperparams['classifier_type'],
            mlp_hidden_neurons=hyperparams['mlp_hidden_neurons'] if hyperparams['classifier_type'] == 'mlp' else 384
        ).to(device)

        # 设置优化器
        if hyperparams['freeze_encoder']:
            model.freeze_encoder()
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['learning_rate'])
        else:
            model.unfreeze_encoder()
            optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'])

        # 准备训练
        loss_fn = nn.CrossEntropyLoss()
        total_steps = len(train_loader) * hyperparams['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_val_f1 = -1
        best_model_state = None

        # 训练循环
        for epoch in range(hyperparams['epochs']):
            print(f"  Epoch {epoch + 1}/{hyperparams['epochs']}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
            val_metrics = evaluate_model(model, val_loader, loss_fn, device, id_to_label)
            print(f"  训练损失: {train_loss:.4f} | 验证F1: {val_metrics['f1_score']:.4f} | 验证准确率: {val_metrics['accuracy']:.4f}")

            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_model_state = model.state_dict()
                print(f"   新的最佳验证F1分数: {best_val_f1:.4f}")

        # 测试集评估
        if best_model_state:
            model.load_state_dict(best_model_state)
        print(" 使用最佳模型在测试集上进行最终评估...")
        test_metrics = evaluate_model(model, test_loader, loss_fn, device, id_to_label)

        #  新增: 保存最优模型
        model_save_info = save_best_model_for_seed(model, best_model_state, hyperparams, best_val_f1, test_metrics, experiment_output_dir)

    print(f"  测试集结果:")
    print(f"    准确率: {test_metrics['accuracy']:.4f}")
    print(f"    Macro精确率: {test_metrics['precision']:.4f}")
    print(f"    Macro召回率: {test_metrics['recall']:.4f}")
    print(f"    Macro F1分数: {test_metrics['f1_score']:.4f}")
    print(f"    Micro F1分数: {test_metrics['f1_micro']:.4f} (验证: {'✓' if abs(test_metrics['f1_micro'] - test_metrics['accuracy']) < 0.001 else '✗'})")

    # 显示每个类别的详细指标
    print(f"  各类别详细指标:")
    for class_name, metrics in test_metrics['per_class_metrics'].items():
        print(f"    {class_name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    print(f"  说明: P=精确率, R=召回率, F1=F1分数")

    # 添加超参数信息到结果中
    result = {
        'hyperparameters': hyperparams,
        'metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'used_cache': use_cache,
        'model_save_path': model_save_info['model_path']  #  新增模型路径
    }

    return result

def extract_best_results_by_model_and_fraction(results):
    """提取每个模型在每个数据比例下的最优结果"""
    best_results = {}

    for result in results:
        if result is None:
            continue

        hyperparams = result['hyperparameters']
        metrics = result['metrics']

        model_name = hyperparams['model_name']
        data_fraction = hyperparams['data_fraction']
        f1_score = metrics['f1_score']

        key = (model_name, data_fraction)

        if key not in best_results or f1_score > best_results[key]['metrics']['f1_score']:
            best_results[key] = result

    return best_results

def generate_model_comparison_analysis(best_results):
    """生成模型对比分析"""
    # 按模型和数据比例组织结果
    comparison_data = {}
    model_names = set()
    data_fractions = set()

    for (model_name, data_fraction), result in best_results.items():
        model_names.add(model_name)
        data_fractions.add(data_fraction)

        if model_name not in comparison_data:
            comparison_data[model_name] = {}

        comparison_data[model_name][data_fraction] = {
            'metrics': result['metrics'],
            'per_class_metrics': result['metrics'].get('per_class_metrics', {}),
            'hyperparams': result['hyperparameters'],
            'best_val_f1': result['best_val_f1']
        }

    model_names = sorted(model_names)
    data_fractions = sorted(data_fractions, reverse=True)

    # 生成对比表格数据
    comparison_table = {
        'models': model_names,
        'data_fractions': data_fractions,
        'results': {}
    }

    for fraction in data_fractions:
        comparison_table['results'][fraction] = {}
        for model in model_names:
            if model in comparison_data and fraction in comparison_data[model]:
                model_result = comparison_data[model][fraction]
                comparison_table['results'][fraction][model] = {
                    'accuracy': round(model_result['metrics']['accuracy'], 4),
                    'f1_score': round(model_result['metrics']['f1_score'], 4),
                    'precision': round(model_result['metrics']['precision'], 4),
                    'recall': round(model_result['metrics']['recall'], 4),
                    'f1_micro': round(model_result['metrics'].get('f1_micro', 0), 4),
                    'per_class_metrics': model_result['metrics'].get('per_class_metrics', {}),
                    'best_hyperparams': {
                        'learning_rate': model_result['hyperparams']['learning_rate'],
                        'batch_size': model_result['hyperparams']['batch_size'],
                        'epochs': model_result['hyperparams']['epochs'],
                        'seed': model_result['hyperparams']['seed']
                    }
                }
            else:
                comparison_table['results'][fraction][model] = None

    return comparison_table, comparison_data

def group_results_by_model_fraction(results):
    """按(模型,数据比例)分组保存所有种子的结果"""
    grouped = {}

    for result in results:
        if result is None:
            continue

        hyperparams = result['hyperparameters']
        key = f"{hyperparams['model_name']}_frac{hyperparams['data_fraction']}"

        if key not in grouped:
            grouped[key] = []

        grouped[key].append({
            'seed': hyperparams['seed'],
            'test_f1': result['metrics']['f1_score'],
            'test_accuracy': result['metrics']['accuracy'],
            'test_precision': result['metrics']['precision'],
            'test_recall': result['metrics']['recall'],
            'test_f1_micro': result['metrics'].get('f1_micro', 0),
            'per_class_metrics': result['metrics'].get('per_class_metrics', {}),
            'val_f1': result['best_val_f1'],
            'model_path': result['model_save_path'],
            'hyperparameters': hyperparams
        })

    return grouped

def save_experiment_results(results, config, output_dir=None):
    """保存实验结果 - 重新组织的文件夹结构"""
    # 创建实验特定的输出目录
    if output_dir:
        # 使用传入的output_dir（用于iterative_main）
        output_dir = output_dir
    else:
        # 使用配置中的默认目录（用于独立运行）
        experiment_id = config['experiment_meta']['description']
        output_dir = os.path.join(config['experiment']['base_output_dir'], experiment_id)

    os.makedirs(output_dir, exist_ok=True)

    # 创建两个子文件夹
    best_results_dir = os.path.join(output_dir, 'best_results_comparison')
    detailed_results_dir = os.path.join(output_dir, 'detailed_results')
    os.makedirs(best_results_dir, exist_ok=True)
    os.makedirs(detailed_results_dir, exist_ok=True)

    # 保存实验信息到主目录
    if config['experiment']['save_experiment_info']:
        experiment_info = {
            'experiment_meta': config['experiment_meta'],
            'data_config': config['data'],
            'models': config['models'],
            'hyperparameters': config['hyperparameters'],
            'total_experiments': len(results),
            'successful_experiments': sum(1 for r in results if r is not None),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        info_filepath = os.path.join(output_dir, 'experiment_info.json')
        with open(info_filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=4)
        print(f" 实验信息已保存到: {info_filepath}")

    # 保存详细结果到 detailed_results 文件夹
    if config['experiment']['save_individual_results']:
        print(" 保存详细实验结果...")
        for i, result in enumerate(results):
            if result is None:
                continue
            hyperparams = result['hyperparameters']
            filename = (
                f"{hyperparams['model_name']}_{hyperparams['classifier_type']}"
            )
            if hyperparams['classifier_type'] == 'mlp':
                filename += f"_{hyperparams['mlp_hidden_neurons']}neurons"
            filename += (
                f"_freeze{hyperparams['freeze_encoder']}_"
                f"ep{hyperparams['epochs']}_bs{hyperparams['batch_size']}_"
                f"lr{hyperparams['learning_rate']}_"
                f"frac{int(hyperparams['data_fraction']*100)}_"
                f"seed{hyperparams['seed']}.json"
            )
            filepath = os.path.join(detailed_results_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        print(f" 详细结果已保存到: {detailed_results_dir}")

    #  新增: 保存所有种子的完整结果
    print(" 保存所有种子的完整结果...")
    all_seeds_results = group_results_by_model_fraction(results)

    all_seeds_path = os.path.join(output_dir, 'all_seeds_results.json')
    with open(all_seeds_path, 'w', encoding='utf-8') as f:
        json.dump(all_seeds_results, f, ensure_ascii=False, indent=4)
    print(f" 所有种子结果已保存到: {all_seeds_path}")

    # 提取最优结果并生成对比分析
    print(" 提取最优结果并生成对比分析...")
    best_results = extract_best_results_by_model_and_fraction(results)
    comparison_table, comparison_data = generate_model_comparison_analysis(best_results)

    # 保存模型对比结果到 best_results_comparison 文件夹
    model_comparison_path = os.path.join(best_results_dir, 'model_comparison_by_data_fraction.json')
    with open(model_comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_table, f, ensure_ascii=False, indent=4)
    print(f" 模型对比结果已保存到: {model_comparison_path}")

    # 保存最优超参数配置
    best_hyperparams = {}
    for (model_name, data_fraction), result in best_results.items():
        if model_name not in best_hyperparams:
            best_hyperparams[model_name] = {}
        best_hyperparams[model_name][data_fraction] = {
            'hyperparameters': result['hyperparameters'],
            'performance': {
                'f1_score': result['metrics']['f1_score'],
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1_micro': result['metrics'].get('f1_micro', 0)
            },
            'per_class_metrics': result['metrics'].get('per_class_metrics', {}),
            'validation_f1': result['best_val_f1']
        }

    best_hyperparams_path = os.path.join(best_results_dir, 'best_hyperparams_by_model.json')
    with open(best_hyperparams_path, 'w', encoding='utf-8') as f:
        json.dump(best_hyperparams, f, ensure_ascii=False, indent=4)
    print(f"  最优超参数已保存到: {best_hyperparams_path}")

    # 生成性能分析报告
    performance_analysis = {
        'summary': {
            'total_model_data_fraction_combinations': len(best_results),
            'models_tested': list(set(model for model, _ in best_results.keys())),
            'data_fractions_tested': sorted(list(set(fraction for _, fraction in best_results.keys())), reverse=True)
        },
        'best_overall_performance': {},
        'performance_trends': {}
    }

    # 找出每个指标的全局最优结果
    for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
        best_result = max(best_results.values(), key=lambda x: x['metrics'][metric])
        performance_analysis['best_overall_performance'][metric] = {
            'value': best_result['metrics'][metric],
            'model': best_result['hyperparameters']['model_name'],
            'data_fraction': best_result['hyperparameters']['data_fraction'],
            'hyperparameters': best_result['hyperparameters'],
            'per_class_metrics': best_result['metrics'].get('per_class_metrics', {})
        }

    # 分析性能趋势
    for model_name in performance_analysis['summary']['models_tested']:
        model_results = [(fraction, result) for (model, fraction), result in best_results.items() if model == model_name]
        model_results.sort(key=lambda x: x[0], reverse=True)  # 按数据比例降序排列

        performance_analysis['performance_trends'][model_name] = {
            'f1_scores_by_fraction': [(fraction, result['metrics']['f1_score']) for fraction, result in model_results],
            'accuracy_by_fraction': [(fraction, result['metrics']['accuracy']) for fraction, result in model_results]
        }

    performance_analysis_path = os.path.join(best_results_dir, 'performance_analysis.json')
    with open(performance_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(performance_analysis, f, ensure_ascii=False, indent=4)
    print(f" 性能分析报告已保存到: {performance_analysis_path}")

    # 传统的聚合结果分析 - 保存到详细结果文件夹
    if config['experiment']['aggregate_results']:
        print(" 生成传统聚合分析...")
        aggregate_results = {}
        for result in results:
            if result is None:
                continue

            hyperparams = result['hyperparameters']
            metrics = result['metrics']

            # 创建配置键
            config_key = (
                hyperparams['model_name'],
                hyperparams['classifier_type'],
                hyperparams['mlp_hidden_neurons'] if hyperparams['classifier_type'] == 'mlp' else 0,
                hyperparams['freeze_encoder'],
                hyperparams['epochs'],
                hyperparams['batch_size'],
                hyperparams['learning_rate'],
                hyperparams['data_fraction']
            )

            if config_key not in aggregate_results:
                aggregate_results[config_key] = []

            aggregate_results[config_key].append({
                'seed': hyperparams['seed'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_micro': metrics.get('f1_micro', 0),
                'per_class_metrics': metrics.get('per_class_metrics', {})
            })

        # 计算统计信息
        summary_results = {
            'experiment_meta': config['experiment_meta'],  # 添加实验元信息
            'results': {}
        }

        for config_key, runs in aggregate_results.items():
            if len(runs) > 0:
                df_runs = pd.DataFrame(runs)
                summary_results['results'][str(config_key)] = {
                    'config': {
                        'model_name': config_key[0],
                        'classifier_type': config_key[1],
                        'mlp_hidden_neurons': config_key[2],
                        'freeze_encoder': config_key[3],
                        'epochs': config_key[4],
                        'batch_size': config_key[5],
                        'learning_rate': config_key[6],
                        'data_fraction': config_key[7]
                    },
                    'mean_metrics': df_runs.mean().to_dict(),
                    'std_metrics': df_runs.std().to_dict(),
                    'num_runs': len(runs),
                    'all_runs': runs
                }

        # 保存聚合结果到详细结果文件夹
        summary_filepath = os.path.join(detailed_results_dir, 'hyperparameter_search_summary.json')
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=4)
        print(f" 传统聚合结果已保存到: {summary_filepath}")

    print(f"\n 所有结果已保存到: {output_dir}")
    print(f"    最优结果对比: {best_results_dir}")
    print(f"    详细实验结果: {detailed_results_dir}")

    return output_dir


# --- 6. 主函数 ---

def run_supervised_training_interface(encoder_path: str, config: dict, output_dir: str, round_num: int = None) -> str:
    """
    标准化接口：运行监督学习超参数搜索

    Args:
        encoder_path: 编码器模型路径
        config: 监督学习配置字典
        output_dir: 输出目录

    Returns:
        实验结果目录路径
    """
    import os
    import copy

    print(f" 监督学习接口调用")
    print(f"   编码器: {encoder_path}")
    print(f"   输出目录: {output_dir}")
    if round_num:
        print(f"   实验轮次: 第{round_num}轮")

    try:
        # 复制全局CONFIG并修改
        global CONFIG
        config_copy = copy.deepcopy(CONFIG)

        # 更新配置
        config_copy['experiment_meta']['description'] = 'iterative_supervised'
        config_copy['experiment']['base_output_dir'] = output_dir

        # 使用提供的编码器路径
        config_copy['models'] = {
            'iterative_encoder': encoder_path
        }

        # 更新数据配置
        if 'train_data_path' in config:
            config_copy['data']['train_data_path'] = config['train_data_path']
        if 'test_data_path' in config:
            config_copy['data']['test_data_path'] = config['test_data_path']

        # 更新超参数
        if 'data_fractions' in config:
            config_copy['hyperparameters']['data_fractions'] = config['data_fractions']
        if 'epochs' in config:
            config_copy['hyperparameters']['epochs'] = config['epochs']
        if 'batch_size' in config:
            config_copy['hyperparameters']['batch_size'] = config['batch_size']
        if 'learning_rate' in config:
            # Convert learning_rate to float if it's a string (from YAML)
            lr = config['learning_rate']
            if isinstance(lr, list):
                config_copy['hyperparameters']['learning_rate'] = [float(x) for x in lr]
            else:
                config_copy['hyperparameters']['learning_rate'] = [float(lr)]
        if 'seeds' in config:
            config_copy['hyperparameters']['seeds'] = config['seeds']
        if 'classifier_types' in config:
            config_copy['hyperparameters']['classifier_types'] = config['classifier_types']
        if 'mlp_hidden_neurons' in config:
            config_copy['hyperparameters']['mlp_hidden_neurons'] = config['mlp_hidden_neurons']
        if 'freeze_encoder' in config:
            config_copy['hyperparameters']['freeze_encoder'] = config['freeze_encoder']

        # 使用更新后的配置运行实验
        original_config = CONFIG
        CONFIG = config_copy

        # 生成超参数组合并运行
        combinations = generate_hyperparameter_combinations(CONFIG)
        print(f" 生成了 {len(combinations)} 个超参数组合")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        for i, hyperparams in enumerate(combinations):
            print(f"运行组合 {i+1}/{len(combinations)}")
            result = run_single_experiment(CONFIG, hyperparams, output_dir, round_num)
            all_results.append(result)

        # 保存结果 - 使用传入的output_dir而不是CONFIG中的base_output_dir
        experiment_dir = save_experiment_results(all_results, CONFIG, output_dir)

        # 恢复原始配置
        CONFIG = original_config

        print(f" 监督学习完成，结果保存在: {experiment_dir}")
        return experiment_dir

    except Exception as e:
        print(f" 监督学习失败: {e}")
        raise


if __name__ == '__main__':
    print(" 开始超参数搜索实验...")
    print(f" 实验名称: {CONFIG['experiment_meta']['experiment_name']}")
    print(f" 实验目的: {CONFIG['experiment_meta']['purpose']}")
    print(f" 实验备注: {CONFIG['experiment_meta']['notes']}")

    # 生成所有超参数组合
    combinations = generate_hyperparameter_combinations(CONFIG)
    print(f" 总共需要运行 {len(combinations)} 个实验配置")

    #  新增：提前创建实验目录
    experiment_id = CONFIG['experiment_meta']['description']
    experiment_output_dir = os.path.join(CONFIG['experiment']['base_output_dir'], experiment_id)
    os.makedirs(experiment_output_dir, exist_ok=True)
    print(f" 实验输出目录: {experiment_output_dir}")

    # 运行所有实验
    all_results = []
    for i, hyperparams in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"实验进度: {i+1}/{len(combinations)}")
        print(f"{'='*80}")

        # 主函数运行时不传round_num，只在iterative_main调用时传
        result = run_single_experiment(CONFIG, hyperparams, experiment_output_dir, None)
        all_results.append(result)

        # 可选：每完成几个实验就保存一次中间结果
        if (i + 1) % 10 == 0:
            print(f"\n 保存中间结果... (已完成 {i+1}/{len(combinations)} 个实验)")
            experiment_dir = save_experiment_results(all_results, CONFIG)

    # 保存最终结果
    print(f"\n 保存最终实验结果...")
    experiment_dir = save_experiment_results(all_results, CONFIG)

    print(f"\n 所有超参数搜索实验已完成！")
    print(f" 结果保存在: {experiment_dir}")
    print(f" 实验描述: {CONFIG['experiment_meta']['description']}")

# --- 其他实验配置示例 ---

# 你可以复制以下配置示例，修改experiment_meta部分，进行不同的实验对比

EXPERIMENT_CONFIGS = {
    "learning_rate_comparison": {
        'experiment_meta': {
            'description': 'learning_rate_comparison',
            'experiment_name': '学习率对比实验',
            'purpose': '测试不同学习率对模型性能的影响',
            'notes': '固定其他参数，对比1e-3, 2e-3, 5e-3三种学习率',
        },
        'hyperparameters': {
            'epochs': [20],
            'batch_size': [32],
            'learning_rate': [1e-3, 2e-3, 5e-3],
            'data_fractions': [1.0],
            'seeds': [42, 123, 456],
            'classifier_types': ['linear'],
            'mlp_layers': [1],  # linear分类器时MLP层数无意义
            'freeze_encoder': [True],
        },
    },

    "data_efficiency": {
        'experiment_meta': {
            'description': 'data_efficiency',
            'experiment_name': '数据效率分析',
            'purpose': '分析模型在不同数据量下的学习效率',
            'notes': '固定最优参数，测试数据稀缺场景下的性能衰减',
        },
        'hyperparameters': {
            'epochs': [50],
            'batch_size': [32],
            'learning_rate': [1e-3],
            'data_fractions': [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01],
            'seeds': [42, 123, 456, 789, 101],
            'classifier_types': ['linear', 'mlp'],
            'mlp_layers': [1, 2, 3],  # 测试不同MLP层数
            'freeze_encoder': [True],
        },
    },

    "architecture_comparison": {
        'experiment_meta': {
            'description': 'architecture_comparison',
            'experiment_name': '架构对比实验',
            'purpose': '对比Linear Probe和MLP分类器的性能差异',
            'notes': '在相同条件下测试两种分类器架构',
        },
        'hyperparameters': {
            'epochs': [30],
            'batch_size': [16, 32],
            'learning_rate': [1e-3, 2e-3],
            'data_fractions': [1.0, 0.5, 0.2],
            'seeds': [42, 123, 456],
            'classifier_types': ['linear', 'mlp'],
            'mlp_layers': [1, 2, 3],  # 对比不同MLP层数的效果
            'freeze_encoder': [True, False],
        },
    }
}

# 使用方法：
# 1. 将上述任意配置复制到主CONFIG中
# 2. 修改experiment_meta字段来描述你的实验
# 3. 运行脚本即可