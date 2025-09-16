import pandas as pd
from Tree_data_model import PostStorage
from cl_training import DynamicContrastiveTrainer

def main_training_pipeline():
    print("🚀 开始训练流程...")
    # 1. 准备数据
    try:
        comment_df = pd.read_csv('data/cl_data/train_comments_filtered.csv', encoding='utf-8')
        post_df = pd.read_csv('data/cl_data/train_posts_filtered.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误: comments_data.csv 或 contents_data.csv 未找到。请确保数据文件存在。")
        return      
    

    # 确保 note_id 和 comment_id 是字符串类型，以避免后续问题
    comment_df['note_id'] = comment_df['note_id'].astype(str)
    comment_df['comment_id'] = comment_df['comment_id'].astype(str)
    comment_df['parent_comment_id'] = comment_df['parent_comment_id'].astype(str)
    post_df['note_id'] = post_df['note_id'].astype(str)


    storage = PostStorage()
    # 确保帖子内容是字符串，如果 title 不存在，尝试 content，如果都为空，则为空字符串
    for _, row in post_df.iterrows():
        post_content = str(row.get('title', '')) or str(row.get('content', '')) # 保证是字符串
        storage.add_post(post_id=str(row['note_id']), post_content=post_content)

    for _, row in comment_df.iterrows():
        post_id_str = str(row['note_id'])
        comment_id_str = str(row['comment_id'])
        content_str = str(row.get('content', '')) # 确保内容是字符串
        parent_id_str = str(row['parent_comment_id']) if str(row['parent_comment_id']) != '0' else post_id_str
        
        try:
            storage.add_comment_to_post(post_id_str, comment_id_str, content_str, parent_id_str)
        except Exception as e:
            print(f"插入评论失败: {e}, 帖子ID: {post_id_str}, 评论ID: {comment_id_str}")

    # 2. 选择训练模型类型并配置训练器
    common_trainer_params = {
        'post_storage': storage,
        'pruning_model_path': "google-bert/bert-base-chinese", # 
        'similarity_threshold': 0.95, # 调整阈值
        'num_negatives': 8,      # 增加负样本数量
        'batch_size': 2,        # 调整批量大小
        'pruning_inference_batch_size': 16, # <--- 为剪枝模型推断设置一个合理的批大小
        'base_lr': 5e-6,         # 调整学习率
        'projection_lr': 5e-5,
        'use_weighted_loss': True,
        'loss_weights': {'dataset1': 1, 'dataset2': 0}, # 调整权重
        'adaptive_weighting': False, # 启用自适应权重
        'infonce_mode': 'bidirectional', # 双向对比
        'projection_head_config': {'hidden_dim': 768, 'output_dim': 384, 'dropout_rate': 0.15}, # 调整投影头
        'min_subtree_size_ds1': 2, 'max_samples_per_post_ds1': None,
        'min_subtree_size_ds2': 100000, 'max_samples_per_subtree_ds2': None,

        # --- 新增PEFT配置 ---
        'use_peft': True,  # 设置为 True 来启用 LoRA
        'peft_config': {
            'r': 8,              # LoRA的秩，越小参数越少，常用8, 16, 32
            'lora_alpha': 16,    # LoRA的缩放因子，通常是r的两倍
            'target_modules': ["query", "key", "value"], # 对注意力的Q,K,V应用
            'lora_dropout': 0.1, # LoRA层的dropout率
            'bias': "none",      # "none", "all", "lora_only"
        },
        
        # --- 🎯 新增：正负样本对构造策略选择 ---
        'positive_pair_strategy': 'simcse_dropout',  # 可选: 'comment_reply', 'simcse_dropout', 'hybrid'
        'simcse_temperature': 0.05,  # SimCSE的温度参数
        'simcse_remove_duplicates': True,  # 新增：是否去重相同文本
        'hybrid_ratio': 0.5,  # 混合策略中SimCSE vs 评论回复的比例 (0.0-1.0)
        # ---------------------------------------
    }

    # 根据选择的策略打印信息
    strategy = common_trainer_params['positive_pair_strategy']
    print(f"\n🎯 正样本对构造策略: {strategy}")
    if strategy == 'comment_reply':
        print("   - 使用评论-回复关系构造正样本对")
        print("   - 基于语义相似度过滤")
    elif strategy == 'simcse_dropout':
        print("   - 使用SimCSE dropout策略构造正样本对")
        print("   - 充分利用所有文本：父评论+子评论")
        print("   - 同一文本通过不同dropout生成正样本对")
        print(f"   - 温度参数: {common_trainer_params['simcse_temperature']}")
        print(f"   - 去重设置: {common_trainer_params['simcse_remove_duplicates']}")
    elif strategy == 'hybrid':
        print("   - 使用混合策略构造正样本对")
        print(f"   - SimCSE比例: {common_trainer_params['hybrid_ratio']}")
        print(f"   - 评论回复比例: {1 - common_trainer_params['hybrid_ratio']}")
    print()

    # 🎯 选项 1: ModelScope 模型
    print("\n--- 配置 ModelScope 模型训练 ---")
    trainer = DynamicContrastiveTrainer(
        training_model_type='modelscope',
        # 使用另一个ModelScope模型作为训练目标
        training_model_identifier_or_path="google-bert/bert-base-chinese",
        **common_trainer_params
    )

    # # 🎯 选项 2: 自定义 TextCNN
    # print("\n--- 配置 TextCNN 训练 ---")
    # textcnn_specific_config = {
    #     'embedding_dim': 300,       
    #     'num_filters': 128,         
    #     'filter_sizes': [2, 3, 4],  
    #     'model_dropout_rate': 0.1,  
    #     'max_seq_length': 200,      # TextCNN分词器的最大序列长度
    #     'textcnn_output_dim': 768,  # TextCNN输出维度 (与投影头输出匹配或作为其输入)
    #     'min_vocab_freq': 1         # 词汇表最小词频
    # }
    
    # 确保 TextCNN 的输出维度与投影头的输入维度匹配
    # common_trainer_params['projection_head_config']['hidden_dim'] 可以基于 textcnn_output_dim
    # 或者 textcnn_output_dim 直接作为投影头的输入
    # 这里假设 textcnn_output_dim 是投影头的输入，所以 base_dim 会是 textcnn_output_dim

    # trainer = DynamicContrastiveTrainer(
    #     training_model_type='textcnn',
    #     training_model_identifier_or_path="model/my_custom_textcnn_v4_no_pruning_paircl", # 自定义模型标识符
    #     textcnn_config=textcnn_specific_config,
    #     **common_trainer_params
    # )
    
    # 3. 开始训练
    print("\n--- 开始训练 ---")
    trainer.train(
        num_epochs=1, # 为了快速测试，减少了epoch，原为100
        rebuild_frequency=2,  # 为了快速测试，减少了频率，原为200
        scheduler_patience=7, # 原为2
        min_improvement=1e-5
    )
    
    print("🎉 训练流程完成!")
    
    # 显示实验保存信息
    model_name = trainer.training_model_identifier_or_path.replace('/', '_').replace('-', '_')
    strategy_name = trainer.positive_pair_strategy
    if trainer.positive_pair_strategy == 'hybrid':
        strategy_name = f"hybrid_ratio{trainer.hybrid_ratio}"
    similarity_str = str(trainer.similarity_threshold).replace('.', 'p')
    experiment_folder = f"{model_name}_{strategy_name}_sim{similarity_str}"
    
    print(f"💾 实验结果已保存到: model/{experiment_folder}/")
    print(f"📊 实验配置:")
    print(f"   - 模型: {trainer.training_model_identifier_or_path}")
    print(f"   - 策略: {trainer.positive_pair_strategy}")
    print(f"   - 相似度阈值: {trainer.similarity_threshold}")
    print(f"   - 文件夹: {experiment_folder}")
    print(f"🔧 基础模型位于: model/{experiment_folder}/trained_{trainer.training_model_type}_embedding_model/")


if __name__ == "__main__":
    main_training_pipeline()