import pandas as pd
from Tree_data_model import PostStorage
from cl_training import DynamicContrastiveTrainer

def main_training_pipeline():
    print("开始训练流程...")
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
        'similarity_threshold': 0.75, # 调整阈值
        'num_negatives': 8,      # 增加负样本数量
        'batch_size': 8,        # 调整批量大小
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
        
        # --- 新增：正负样本对构造策略选择 ---
        'positive_pair_strategy': 'simcse_dropout',  # 可选: 'comment_reply', 'simcse_dropout', 'hybrid'
        'simcse_temperature': 0.07,  # SimCSE的温度参数
        'simcse_remove_duplicates': True,  # 新增：是否去重相同文本
        'hybrid_ratio': 0.5,  # 混合策略中SimCSE vs 评论回复的比例 (0.0-1.0)
        # ---------------------------------------
    }

    # 根据选择的策略打印信息
    strategy = common_trainer_params['positive_pair_strategy']
    print(f"\n正样本对构造策略: {strategy}")
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

    # 选项 1: ModelScope 模型
    print("\n--- 配置 ModelScope 模型训练 ---")
    trainer = DynamicContrastiveTrainer(
        training_model_type='modelscope',
        # 使用另一个ModelScope模型作为训练目标
        training_model_identifier_or_path="google-bert/bert-base-chinese",
        **common_trainer_params
    )

    # # 选项 2: 自定义 TextCNN
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
        num_epochs=200, # 为了快速测试，减少了epoch，原为100
        rebuild_frequency=200,  # 为了快速测试，减少了频率，原为200
        scheduler_patience=7, # 原为2
        min_improvement=1e-5
    )
    
    print("训练流程完成!")
    
    # 显示实验保存信息
    model_name = trainer.training_model_identifier_or_path.replace('/', '_').replace('-', '_')
    strategy_name = trainer.positive_pair_strategy
    if trainer.positive_pair_strategy == 'hybrid':
        strategy_name = f"hybrid_ratio{trainer.hybrid_ratio}"
    similarity_str = str(trainer.similarity_threshold).replace('.', 'p')
    experiment_folder = f"{model_name}_{strategy_name}_sim{similarity_str}"
    
    print(f"实验结果已保存到: model/{experiment_folder}/")
    print(f"实验配置:")
    print(f"   - 模型: {trainer.training_model_identifier_or_path}")
    print(f"   - 策略: {trainer.positive_pair_strategy}")
    print(f"   - 相似度阈值: {trainer.similarity_threshold}")
    print(f"   - 文件夹: {experiment_folder}")
    print(f"基础模型位于: model/{experiment_folder}/trained_{trainer.training_model_type}_embedding_model/")


def run_stage1_contrastive_training(config: dict, output_dir: str) -> str:
    """
    标准化接口：运行Stage 1对比学习训练

    Args:
        config: Stage 1配置字典
        output_dir: 输出目录

    Returns:
        最佳模型路径
    """
    import sys
    import os
    import pickle
    from argparse import Namespace
    import pandas as pd
    from Tree_data_model import PostStorage

    print(f"[Stage 1] Stage 1对比学习接口调用")
    print(f"   输出目录: {output_dir}")

    # 构建PostStorage对象
    print("加载对比学习数据...")

    # 从配置获取数据路径
    comments_path = config.get('cl_comments_data', 'data/cl_data/train_comments_filtered.csv')
    posts_path = config.get('cl_posts_data', 'data/cl_data/train_posts_filtered.csv')

    try:
        comment_df = pd.read_csv(comments_path, encoding='utf-8')
        post_df = pd.read_csv(posts_path, encoding='utf-8')
        print(f"   [成功] 加载评论数据: {len(comment_df)} 条")
        print(f"   [成功] 加载帖子数据: {len(post_df)} 条")
    except FileNotFoundError as e:
        print(f"[错误] 数据文件未找到: {e}")
        raise

    # 确保ID字段是字符串类型
    comment_df['note_id'] = comment_df['note_id'].astype(str)
    comment_df['comment_id'] = comment_df['comment_id'].astype(str)
    comment_df['parent_comment_id'] = comment_df['parent_comment_id'].astype(str)
    post_df['note_id'] = post_df['note_id'].astype(str)

    # 构建PostStorage
    storage = PostStorage()

    # 添加帖子
    for _, row in post_df.iterrows():
        post_content = str(row.get('title', '')) or str(row.get('content', ''))
        storage.add_post(post_id=str(row['note_id']), post_content=post_content)

    # 添加评论
    for _, row in comment_df.iterrows():
        post_id_str = str(row['note_id'])
        comment_id_str = str(row['comment_id'])
        content_str = str(row.get('content', ''))
        parent_id_str = str(row['parent_comment_id']) if str(row['parent_comment_id']) != '0' else post_id_str

        try:
            storage.add_comment_to_post(post_id_str, comment_id_str, content_str, parent_id_str)
        except Exception as e:
            print(f"插入评论失败: {e}, 帖子ID: {post_id_str}, 评论ID: {comment_id_str}")

    print(f"   [成功] PostStorage构建完成")

    # 构造参数
    args = Namespace(
        # 数据相关（不再需要tree_data_path）
        similarity_threshold=config.get('similarity_threshold', 0.75),
        pruning_model_path=config.get('pruning_model_path', 'google-bert/bert-base-chinese'),
        pruning_inference_batch_size=config.get('pruning_inference_batch_size', 128),
        min_subtree_size_ds1=config.get('min_subtree_size_ds1', 2),
        max_samples_per_post_ds1=config.get('max_samples_per_post_ds1', None),
        min_subtree_size_ds2=config.get('min_subtree_size_ds2', 3),
        max_samples_per_subtree_ds2=config.get('max_samples_per_subtree_ds2', 30),

        # 模型相关
        model_type='modelscope',
        model_name_or_path=config.get('model_name_or_path', 'google-bert/bert-base-chinese'),

        # 训练相关
        batch_size=config.get('batch_size', 16),
        epochs=config.get('epochs', 100),
        base_lr=float(config.get('base_lr', 5e-5)),
        projection_lr=float(config.get('projection_lr', 5e-4)),
        warmup_steps=config.get('warmup_steps', 1000),

        # 对比学习相关
        positive_pair_strategy=config.get('positive_pair_strategy', 'comment_reply'),
        num_negatives=config.get('num_negatives', 2),
        infonce_mode=config.get('infonce_mode', 'in_batch'),

        # PEFT相关（包含完整的LoRA配置）
        use_peft=config.get('use_peft', True),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        lora_target_modules=config.get('lora_target_modules', ["query", "key", "value", "dense"]),
        lora_bias=config.get('lora_bias', 'none'),

        # 输出相关
        model_save_name=config.get('model_save_name', 'stage1_cl_model'),
        save_frequency=config.get('save_frequency', 20),

        # 其他
        device='auto',
        resume_from=None
    )

    try:
        # 调用原有的训练逻辑
        from cl_training import DynamicContrastiveTrainer

        # 创建训练器，使用PostStorage
        trainer = DynamicContrastiveTrainer(
            post_storage=storage,  # 使用构建好的PostStorage
            training_model_type=args.model_type,
            training_model_identifier_or_path=args.model_name_or_path,
            output_dir=output_dir,  # 传入输出目录
            # 剪枝和数据构建参数
            pruning_model_path=args.pruning_model_path,
            similarity_threshold=args.similarity_threshold,
            pruning_inference_batch_size=args.pruning_inference_batch_size,
            min_subtree_size_ds1=args.min_subtree_size_ds1,
            max_samples_per_post_ds1=args.max_samples_per_post_ds1,
            min_subtree_size_ds2=args.min_subtree_size_ds2,
            max_samples_per_subtree_ds2=args.max_samples_per_subtree_ds2,
            # 训练参数
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            projection_lr=args.projection_lr,
            # 对比学习参数
            positive_pair_strategy=args.positive_pair_strategy,
            num_negatives=args.num_negatives,
            infonce_mode=args.infonce_mode,
            # PEFT参数
            use_peft=args.use_peft,
            peft_config={
                'r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'target_modules': args.lora_target_modules,
                'lora_dropout': args.lora_dropout,
                'bias': args.lora_bias
            } if args.use_peft else None
        )

        # 训练模型
        trainer.train(
            num_epochs=args.epochs,
            rebuild_frequency=config.get('rebuild_frequency', 200),
            scheduler_patience=config.get('scheduler_patience', 15),  # 从配置读取，默认15
            min_improvement=float(config.get('min_improvement', 1e-5))  # 确保是浮点数
        )

        # 保存生成的数据集到实验目录
        if hasattr(trainer, 'dataset1') and trainer.dataset1:
            dataset_path = os.path.join(output_dir, f'dataset1_sim_{args.similarity_threshold}.pkl')
            with open(dataset_path, 'wb') as f:
                pickle.dump(trainer.dataset1, f)
            print(f"数据集1已保存: {dataset_path}")

        if hasattr(trainer, 'dataset2') and trainer.dataset2:
            dataset_path = os.path.join(output_dir, f'dataset2_sim_{args.similarity_threshold}.pkl')
            with open(dataset_path, 'wb') as f:
                pickle.dump(trainer.dataset2, f)
            print(f"数据集2已保存: {dataset_path}")

        # 查找最佳模型
        best_model_path = os.path.join(output_dir, "best_contrastive_model.pth")
        if not os.path.exists(best_model_path):
            # 查找其他可能的模型文件
            for file in os.listdir(output_dir):
                if file.endswith('.pth'):
                    best_model_path = os.path.join(output_dir, file)
                    break

        print(f"[成功] Stage 1训练完成，模型保存在: {best_model_path}")
        return best_model_path

    except Exception as e:
        print(f"[错误] Stage 1训练失败: {e}")
        raise


if __name__ == "__main__":
    main_training_pipeline()