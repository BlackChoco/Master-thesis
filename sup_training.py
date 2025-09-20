import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
import random
import itertools
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

# --- 1. 数据集和模型定义 ---

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
    def __init__(self, base_encoder: nn.Module, num_labels: int, classifier_type: str = 'linear', mlp_layers: int = 2):
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
            # 动态构建MLP分类器
            layers = []
            current_dim = hidden_size

            # 添加隐藏层
            for i in range(mlp_layers - 1):
                next_dim = current_dim // 2
                layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                current_dim = next_dim

            # 添加输出层
            layers.append(nn.Linear(current_dim, num_labels))

            self.classifier = nn.Sequential(*layers)
            print(f"MLP分类器结构: {mlp_layers}层，维度变化: {hidden_size} -> ... -> {num_labels}")
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
        print("🧊 正在冻结基础编码器的参数...")
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        print("✅ 基础编码器已冻结。")

    def unfreeze_encoder(self):
        """解冻基础编码器的所有参数。"""
        print("🔥 正在解冻基础编码器的参数...")
        for param in self.base_encoder.parameters():
            param.requires_grad = True
        print("✅ 基础编码器已解冻。")


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
        print(f"✅ Checkpoint 加载成功。模型类型: {model_type.upper()}")

        return temp_encoder.base_model, temp_encoder.tokenizer, model_type
    
    # 如果不是本地文件，则尝试从 ModelScope Hub 加载
    else:
        print(f"未找到本地文件，尝试从 ModelScope Hub 加载模型: {checkpoint_path}")
        try:
            # --- 关键修改: 使用 AutoModel 和 AutoTokenizer ---
            base_model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            print(f"✅ 成功从 ModelScope Hub 加载基础模型和分词器。")
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
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # --- 计算每个类别的指标 ---
    class_ids = sorted(id_to_label.keys())
    # 使用原始标签作为报告中的名称
    target_names = [f"class_{id_to_label[cid]}" for cid in class_ids]
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        labels=class_ids,
        target_names=target_names,
        output_dict=True, 
        zero_division=0
    )
    # 提取每个类别的指标，并移除 'support'
    per_class_metrics = {}
    for name in target_names:
        if name in report_dict:
            metrics = report_dict[name].copy()
            metrics.pop('support', None)
            per_class_metrics[name] = metrics

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "per_class_metrics": per_class_metrics
    }

# --- 4. 超参数搜索配置 ---

# 统一的实验配置字典
CONFIG = {
    # 实验元信息
    'experiment_meta': {
        'description': 'baseline_comparison',  # 实验描述标识符
        'experiment_name': 'Baseline对比实验',   # 实验的中文名称
        'purpose': '对比LoRA微调后的BERT与原始BERT在不同数据量下的性能表现',  # 实验目的
        'notes': '使用linear probe和MLP分类器，测试5个不同数据比例',  # 实验备注
    },

    # 数据配置
    'data': {
        'train_data_path': 'data/sup_train_data/trainset.csv',
        'test_data_path': 'data/sup_train_data/testset.csv',
        'validation_split': 0.2,  # 验证集比例
        'excluded_labels': [5],   # 要过滤的标签
    },

    # 模型配置
    'models': {
        'lora_bert_base_chinese_cl': 'model/google-bert_bert-base-chinese/best_contrastive_model.pth',
        # 'TextCNN_CL_bert': 'model/my_custom_textcnn_v3_bert_pruning_paircl/best_contrastive_model.pth',
        'Bert_base_chinese_nocl': 'google-bert/bert-base-chinese',
    },

    # 超参数搜索空间
    'hyperparameters': {
        'epochs': [50],                    # 训练轮数
        'batch_size': [32],              # 批次大小
        'learning_rate': [1e-3], # 学习率
        'data_fractions': [1.0, 0.5, 0.2, 0.1, 0.05],  # 数据使用比例
        'seeds': [42, 123, 456, 789, 101],             # 随机种子
        'classifier_types': ['linear'], # 分类器类型
        'mlp_layers': [1, 2, 3],          # MLP层数 (仅在classifier_type='mlp'时生效)
        'freeze_encoder': [True],     # 是否冻结编码器
    },

    # 实验控制
    'experiment': {
        'base_output_dir': 'sup_result_hyperparams',  # 基础输出目录
        'save_individual_results': True,
        'aggregate_results': True,
        'save_experiment_info': True,  # 保存实验信息
    }
}

def generate_hyperparameter_combinations(config):
    """生成所有超参数组合"""
    hyperparams = config['hyperparameters']

    # 创建超参数组合
    combinations = []
    for (model_name, checkpoint_path), epochs, batch_size, lr, fraction, seed, classifier_type, mlp_layers, freeze in itertools.product(
        config['models'].items(),
        hyperparams['epochs'],
        hyperparams['batch_size'],
        hyperparams['learning_rate'],
        hyperparams['data_fractions'],
        hyperparams['seeds'],
        hyperparams['classifier_types'],
        hyperparams['mlp_layers'],
        hyperparams['freeze_encoder']
    ):
        # 只有当classifier_type='mlp'时，mlp_layers参数才有意义
        # 当classifier_type='linear'时，跳过mlp_layers > 1的组合
        if classifier_type == 'linear' and mlp_layers > 1:
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
            'mlp_layers': mlp_layers,
            'freeze_encoder': freeze,
        })

    return combinations

def run_single_experiment(config, hyperparams):
    """运行单次实验"""

    # 设置随机种子
    set_seed(hyperparams['seed'])
    print(f"\n--- 实验配置 ---")
    print(f"模型: {hyperparams['model_name']}")
    print(f"分类器: {hyperparams['classifier_type']}")
    if hyperparams['classifier_type'] == 'mlp':
        print(f"MLP层数: {hyperparams['mlp_layers']}")
    print(f"冻结编码器: {hyperparams['freeze_encoder']}")
    print(f"数据比例: {hyperparams['data_fraction']*100}%")
    print(f"学习率: {hyperparams['learning_rate']}")
    print(f"批次大小: {hyperparams['batch_size']}")
    print(f"训练轮数: {hyperparams['epochs']}")
    print(f"随机种子: {hyperparams['seed']}")

    # 加载数据
    df_train_full, df_test = load_data(config['data']['train_data_path'], config['data']['test_data_path'])
    if df_train_full is None:
        return None

    # 过滤标签
    for label in config['data']['excluded_labels']:
        df_train_full = df_train_full[df_train_full['label'] != label].reset_index(drop=True)
        df_test = df_test[df_test['label'] != label].reset_index(drop=True)

    # 划分验证集
    df_train_prelim, df_val = train_test_split(
        df_train_full,
        test_size=config['data']['validation_split'],
        random_state=hyperparams['seed'],
        stratify=df_train_full['label']
    )

    # 数据采样
    if hyperparams['data_fraction'] < 1.0:
        df_train = df_train_prelim.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=hyperparams['data_fraction'], random_state=hyperparams['seed'])
        ).reset_index(drop=True)
    else:
        df_train = df_train_prelim.reset_index(drop=True)

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

    # 创建数据集
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
        mlp_layers=hyperparams['mlp_layers'] if hyperparams['classifier_type'] == 'mlp' else 1
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
        print(f"  训练损失: {train_loss:.4f} | 验证损失: {val_metrics['loss']:.4f} | 验证F1: {val_metrics['f1_score']:.4f}")

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_state = model.state_dict()
            print(f"  🎉 新的最佳验证F1分数: {best_val_f1:.4f}")

    # 测试集评估
    if best_model_state:
        model.load_state_dict(best_model_state)
    print("🧪 使用最佳模型在测试集上进行最终评估...")
    test_metrics = evaluate_model(model, test_loader, loss_fn, device, id_to_label)
    print(f"  测试集结果 -> 损失: {test_metrics['loss']:.4f}, 准确率: {test_metrics['accuracy']:.4f}, F1分数: {test_metrics['f1_score']:.4f}")

    # 添加超参数信息到结果中
    result = {
        'hyperparameters': hyperparams,
        'metrics': test_metrics,
        'best_val_f1': best_val_f1
    }

    return result

def save_experiment_results(results, config):
    """保存实验结果"""
    # 创建实验特定的输出目录
    experiment_id = config['experiment_meta']['description']
    output_dir = os.path.join(config['experiment']['base_output_dir'], experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # 保存实验信息
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
        print(f"📋 实验信息已保存到: {info_filepath}")

    if config['experiment']['save_individual_results']:
        # 保存每个实验的详细结果
        for i, result in enumerate(results):
            if result is None:
                continue
            hyperparams = result['hyperparameters']
            filename = (
                f"{hyperparams['model_name']}_{hyperparams['classifier_type']}"
            )
            if hyperparams['classifier_type'] == 'mlp':
                filename += f"_{hyperparams['mlp_layers']}layers"
            filename += (
                f"_freeze{hyperparams['freeze_encoder']}_"
                f"ep{hyperparams['epochs']}_bs{hyperparams['batch_size']}_"
                f"lr{hyperparams['learning_rate']}_"
                f"frac{int(hyperparams['data_fraction']*100)}_"
                f"seed{hyperparams['seed']}.json"
            )
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

    if config['experiment']['aggregate_results']:
        # 聚合结果分析
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
                hyperparams['mlp_layers'] if hyperparams['classifier_type'] == 'mlp' else 1,
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
                'recall': metrics['recall']
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
                        'mlp_layers': config_key[2],
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

        # 保存聚合结果
        summary_filepath = os.path.join(output_dir, 'hyperparameter_search_summary.json')
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=4)

        print(f"\n✅ 聚合结果已保存到: {summary_filepath}")

    return output_dir


# --- 6. 主函数 ---

if __name__ == '__main__':
    print("🚀 开始超参数搜索实验...")
    print(f"📝 实验名称: {CONFIG['experiment_meta']['experiment_name']}")
    print(f"🎯 实验目的: {CONFIG['experiment_meta']['purpose']}")
    print(f"📄 实验备注: {CONFIG['experiment_meta']['notes']}")

    # 生成所有超参数组合
    combinations = generate_hyperparameter_combinations(CONFIG)
    print(f"📊 总共需要运行 {len(combinations)} 个实验配置")

    # 运行所有实验
    all_results = []
    for i, hyperparams in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"实验进度: {i+1}/{len(combinations)}")
        print(f"{'='*80}")

        result = run_single_experiment(CONFIG, hyperparams)
        all_results.append(result)

        # 可选：每完成几个实验就保存一次中间结果
        if (i + 1) % 10 == 0:
            print(f"\n💾 保存中间结果... (已完成 {i+1}/{len(combinations)} 个实验)")
            experiment_dir = save_experiment_results(all_results, CONFIG)

    # 保存最终结果
    print(f"\n💾 保存最终实验结果...")
    experiment_dir = save_experiment_results(all_results, CONFIG)

    print(f"\n🎉 所有超参数搜索实验已完成！")
    print(f"📁 结果保存在: {experiment_dir}")
    print(f"📋 实验描述: {CONFIG['experiment_meta']['description']}")

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