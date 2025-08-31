import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
import random
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# --- 关键修改: 导入 AutoModel 和 AutoTokenizer ---
from modelscope import AutoModel, AutoTokenizer 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split # 用于从训练集中划分验证集
from peft import get_peft_model, LoraConfig


# 从 cl_training.py 导入必要的类
# 假设 cl_training.py 与此脚本在同一目录下
from cl_training_modelscope import ContrastiveEncoder, TextCNNModel, TextCNNTokenizer

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
    def __init__(self, base_encoder: nn.Module, num_labels: int, classifier_type: str = 'linear'):
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
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_labels)
                
            )
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

# --- 4. 主实验流程 ---

def run_experiment(config):
    """运行单次完整的实验（给定配置和种子）。"""
    
    # 设置随机种子
    set_seed(config['seed'])
    print(f"\n--- 运行实验: 模型={config['model_name']}, 方法={config['method']}, "
          f"数据比例={config['data_fraction']*100}%, 种子={config['seed']} ---")

    # 加载数据
    df_train_full, df_test = load_data(config['train_data_path'], config['test_data_path'])
    if df_train_full is None: 
        return None

    # 过滤掉 label 为 5 的样本（训练集和测试集都要过滤）
    df_train_full = df_train_full[df_train_full['label'] != 5].reset_index(drop=True)
    df_test = df_test[df_test['label'] != 5].reset_index(drop=True)

    # 1. 先从完整训练数据中划分出固定的验证集
    from sklearn.model_selection import train_test_split
    df_train_prelim, df_val = train_test_split(
        df_train_full,
        test_size=0.1, # 验证集占完整训练集的10%
        random_state=config['seed'],
        stratify=df_train_full['label']
    )

    # 2. 再从划分后的训练集中进行采样（如有需要）
    if config['data_fraction'] < 1.0:
        # 分层采样
        df_train = df_train_prelim.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=config['data_fraction'], random_state=config['seed'])
        ).reset_index(drop=True)
    else:
        df_train = df_train_prelim.reset_index(drop=True) # 当使用全部数据时，也重置索引

    # 重新生成标签映射（只包含剩下的标签）
    unique_labels = sorted(df_train_full['label'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)

    # 加载预训练的编码器和分词器
    base_encoder, tokenizer, _ = load_pretrained_encoder(config['checkpoint_path'])
    if base_encoder is None: return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    train_dataset = SupervisedTextDataset(df_train['content'].tolist(), df_train['label'].tolist(), tokenizer, label_to_id)
    val_dataset = SupervisedTextDataset(df_val['content'].tolist(), df_val['label'].tolist(), tokenizer, label_to_id)
    test_dataset = SupervisedTextDataset(df_test['content'].tolist(), df_test['label'].tolist(), tokenizer, label_to_id)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # 构建监督模型
    model = SupervisedModel(
        base_encoder=base_encoder,
        num_labels=num_labels,
        classifier_type='mlp' if config['method'] == 'fine_tune' else 'linear'
    ).to(device)

    # 根据配置冻结或解冻编码器
    if config['freeze_encoder']:
        model.freeze_encoder()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    else: # fine_tune
        model.unfreeze_encoder()
        optimizer = AdamW(model.parameters(), lr=config['lr'])

    # 准备训练
    loss_fn = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_f1 = -1
    best_model_state = None

    # 训练循环
    for epoch in range(config['epochs']):
        print(f"  Epoch {epoch + 1}/{config['epochs']}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_metrics = evaluate_model(model, val_loader, loss_fn, device, id_to_label)
        print(f"  训练损失: {train_loss:.4f} | 验证损失: {val_metrics['loss']:.4f} | 验证F1: {val_metrics['f1_score']:.4f}")

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_state = model.state_dict()
            print(f"  🎉 新的最佳验证F1分数: {best_val_f1:.4f}")

    # 使用最佳模型在测试集上评估
    if best_model_state:
        model.load_state_dict(best_model_state)
    print("🧪 使用最佳模型在测试集上进行最终评估...")
    test_metrics = evaluate_model(model, test_loader, loss_fn, device, id_to_label)
    print(f"  测试集结果 -> 损失: {test_metrics['loss']:.4f}, 准确率: {test_metrics['accuracy']:.4f}, F1分数: {test_metrics['f1_score']:.4f}")
    
    return test_metrics


# --- 5. 主函数 ---

if __name__ == '__main__':
    # --- 实验配置 ---
    # 定义所有实验的通用配置
    BASE_CONFIG = {
        'train_data_path': 'data_process/sup_train_data/trainset.csv',
        'test_data_path': 'data_process/sup_train_data/testset.csv',
        'epochs': 50,
        'batch_size': 16,
        # 'lr' is now defined in METHODS_CONFIG
    }
    
    # 定义要运行的对比学习模型
    # key: 一个描述性名称, value: checkpoint文件的路径或ModelScope模型标识符
    EXPERIMENT_MODELS = {
        # "jina_embed_none":'jinaai/jina-embeddings-v3',
        
        # "TextCNN_CL_bert_random": "model/model_random_init/best_contrastive_model.pth",
        # 'TextCNN_CL_bert':'model/my_custom_textcnn_v3_bert_pruning_paircl/best_contrastive_model.pth',
        # 推荐使用ModelScope原生支持的特征提取模型，但AutoModel也能处理bert-base-chinese
        
        'lora_bert_base_chinese_cl': 'model/google-bert_bert-base-chinese/best_contrastive_model.pth',
        "Bert_base_chinese_nocl": "google-bert/bert-base-chinese", 
        # 'TextCNN_CL_no_pruing':'model/my_custom_textcnn_v3_no_pruning_paircl/best_contrastive_model.pth'
    }

    # 定义要运行的评估方法配置
    METHODS_CONFIG = [
        {'name': 'linear_probe', 'freeze_encoder': True, 'lr': 1e-3},
        # {'name': 'fine_tune', 'freeze_encoder': False, 'lr': 2e-5}, 
    ]
    
    # 定义数据比例和随机种子
    DATA_FRACTIONS = [1, 0.5, 0.2, 0.1, 0.05]
    SEEDS = [42, 123, 456, 789, 101, 20, 30, 40, 50, 60]  # 增加更多种子以提高结果的稳定性

    # --- 实验执行 ---
    all_results = []

    for model_name, checkpoint_path in EXPERIMENT_MODELS.items():
        for method_config in METHODS_CONFIG:
            method_name = method_config['name']
            
            results_for_method = {
                "model_name": model_name,
                "method": method_name,
                "freeze_encoder": method_config['freeze_encoder'],
                "epochs": BASE_CONFIG['epochs'],
                "batch_size": BASE_CONFIG['batch_size'],
                "learning_rate": method_config['lr'],
                "results_by_fraction": {}
            }

            for fraction in DATA_FRACTIONS:
                
                fraction_key = f"{int(fraction*100)}%"
                print(f"\n======================================================================")
                print(f"开始实验系列: 模型='{model_name}', 方法='{method_name}', 冻结={method_config['freeze_encoder']}, 数据比例='{fraction_key}'")
                print(f"======================================================================\n")

                run_metrics = []
                for seed in SEEDS:
                    config = BASE_CONFIG.copy()
                    config.update({
                        "model_name": model_name,
                        "checkpoint_path": checkpoint_path,
                        "method": method_name,
                        "freeze_encoder": method_config['freeze_encoder'],
                        "data_fraction": fraction,
                        "seed": seed,
                        "lr": method_config['lr']
                    })
                    
                    metrics = run_experiment(config)
                    if metrics:
                        run_metrics.append(metrics)
                
                # 计算均值和方差
                if run_metrics:
                    # 扁平化结果以计算统计数据
                    flattened_metrics_list = []
                    for m in run_metrics:
                        flat_m = {}
                        # 复制顶级指标
                        for k, v in m.items():
                            if k != 'per_class_metrics':
                                flat_m[k] = v
                        # 扁平化每个类别的指标
                        if 'per_class_metrics' in m:
                            for class_name, class_stats in m['per_class_metrics'].items():
                                for metric_name, value in class_stats.items():
                                    flat_m[f"{class_name}_{metric_name}"] = value
                        flattened_metrics_list.append(flat_m)

                    df_metrics = pd.DataFrame(flattened_metrics_list)
                    mean_metrics = df_metrics.mean().to_dict()
                    std_metrics = df_metrics.std().to_dict()
                    
                    results_for_method["results_by_fraction"][fraction_key] = {
                        "mean": mean_metrics,
                        "std": std_metrics,
                        "runs": run_metrics # 保存每次运行的原始结构化结果
                    }

            all_results.append(results_for_method)

            # --- 保存结果 ---
            output_dir = "result"
            os.makedirs(output_dir, exist_ok=True)
            
 # 为当前模型和方法的结果创建一个单独的文件，文件名包含冻结状态和训练参数
            result_filename = (
                f"{model_name}_{method_name}_freeze_{method_config['freeze_encoder']}"
                f"_epoch{BASE_CONFIG['epochs']}_bs{BASE_CONFIG['batch_size']}_lr{method_config['lr']}_results.json"
            )
            result_filepath = os.path.join(output_dir, result_filename)
            
            with open(result_filepath, 'w', encoding='utf-8') as f:
                json.dump(results_for_method, f, ensure_ascii=False, indent=4)
            
            print(f"\n✅ 实验系列结果已保存到: {result_filepath}")

    print("\n\n🎉 所有实验已完成！")