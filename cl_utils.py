import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import json
import os
from tqdm import tqdm
from Tree_data_model import PostStorage
from cl_base_model import ContrastiveEncoder, load_model_from_modelscope, load_tokenizer_from_modelscope
from cl_loss import ContrastiveLoss
from peft import get_peft_model, LoraConfig


def build_pruned_forest(post_storage: PostStorage, similarity_threshold: float):
    """
    基于相似度阈值构建剪枝后的森林
    """
    print("🔄 构建剪枝森林...")
    post_storage.forests.clear()
    pruning_results = post_storage.prune_all_posts_by_similarity(
        similarity_threshold=similarity_threshold, show_progress=True
    )
    print(f"✅ 森林构建完成: {len(pruning_results)} 个帖子")
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

                if anchor_texts_ds1:
                    anchor_emb = model(anchor_texts_ds1)
                    positive_emb_ds1 = model(positive_texts_ds1_f)
                    
                    # 重构与过滤后样本对应的负样本
                    neg_emb = None
                    if negative_texts and num_negatives > 0:
                        current_batch_neg_texts = []
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
            print(f"💾 保存当前最佳模型，损失: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ 等待更优模型，当前计数器: {patience_counter}/{scheduler_patience}")

        if patience_counter >= scheduler_patience:
            print("🛑 早停触发，停止训练")
            break


def load_trained_model_and_tokenizer(checkpoint_path: str):
    """
    从checkpoint加载完整的训练器状态，并返回训练好的基础模型和分词器。
    """
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 文件 {checkpoint_path} 未找到。")
        return None, None, None

    print(f"正在从 {checkpoint_path} 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    model_type = checkpoint['training_model_type']
    model_identifier = checkpoint['training_model_identifier_or_path']
    proj_config = checkpoint['projection_head_config']

    # 获取PEFT配置
    use_peft = checkpoint.get('use_peft', False)
    peft_config = checkpoint.get('peft_config', None)
    
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
        # 如果使用了PEFT，重新包装模型
        if use_peft:
            print("🔧 检测到PEFT训练的checkpoint，正在重新应用LoRA配置...")
            lora_config = LoraConfig(**peft_config)
            encoder.base_model = get_peft_model(encoder.base_model, lora_config)
            print("✅ LoRA配置已重新应用。")
    else:
        print(f"错误: Checkpoint中未知的模型类型 '{model_type}'")
        return None, None, None

    encoder.load_state_dict(checkpoint['contrastive_encoder_state_dict'])
    encoder.eval()
    print(f"✅ {model_type.upper()} ContrastiveEncoder 加载完成并设置为评估模式。")
    
    return encoder.base_model, encoder.tokenizer, model_type

