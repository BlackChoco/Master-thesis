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


def run_stage1_contrastive_training(config: dict, output_dir: str, full_config: dict = None) -> str:
    """
    标准化接口：运行Stage 1对比学习训练（支持超参数网格搜索）

    Args:
        config: Stage 1配置字典
        output_dir: 输出目录
        full_config: 完整配置字典（包含 supervised_learning 等，用于 Grid Search 评估）

    Returns:
        最佳模型路径
    """
    import sys
    import os
    import pickle
    import shutil
    from argparse import Namespace
    import pandas as pd
    from Tree_data_model import PostStorage
    import itertools
    import json
    import torch

    print(f"[Stage 1] Stage 1对比学习接口调用")
    print(f"   输出目录: {output_dir}")

    # ========== 检测是否需要超参数网格搜索 ==========
    hyperparams_to_search = {
        'infonce_temperature': config.get('infonce_temperature', [0.07]),
        'simcse_temperature': config.get('simcse_temperature', [0.07]),
        'base_lr': config.get('base_lr', [5e-5]),
        'projection_lr': config.get('projection_lr', [5e-4]),
        'epochs': config.get('epochs', [100]),
        'batch_size': config.get('batch_size', [16])
    }

    # 确保所有参数都是列表形式，并转换类型
    for key in hyperparams_to_search:
        if not isinstance(hyperparams_to_search[key], list):
            hyperparams_to_search[key] = [hyperparams_to_search[key]]

        # 确保数值类型正确（防止 YAML 中的科学计数法被解析为字符串）
        if key in ['base_lr', 'projection_lr', 'infonce_temperature', 'simcse_temperature']:
            hyperparams_to_search[key] = [float(v) for v in hyperparams_to_search[key]]
        elif key in ['epochs', 'batch_size']:
            hyperparams_to_search[key] = [int(v) for v in hyperparams_to_search[key]]

    # 计算总搜索次数
    param_combinations = list(itertools.product(
        hyperparams_to_search['infonce_temperature'],
        hyperparams_to_search['simcse_temperature'],
        hyperparams_to_search['base_lr'],
        hyperparams_to_search['projection_lr'],
        hyperparams_to_search['epochs'],
        hyperparams_to_search['batch_size']
    ))

    total_runs = len(param_combinations)

    if total_runs > 1:
        print(f"\n🔍 检测到超参数网格搜索模式")
        print(f"   参数组合数: {total_runs}")
        print(f"   搜索参数:")
        for key, values in hyperparams_to_search.items():
            if len(values) > 1:
                print(f"     - {key}: {values}")
    else:
        print(f"\n📌 单次训练模式")

    # ========== 如果只有一次运行，使用原有逻辑 ==========
    if total_runs == 1:
        return _run_single_stage1_training(config, output_dir)

    # ========== 检查是否有 full_config（Grid Search 需要） ==========
    if full_config is None:
        print("\n⚠️  警告: Grid Search 模式但未提供 full_config")
        print("   将无法运行 Topic Pair 评估和 Supervised Training")
        print("   如需完整评估，请在调用时传入 full_config 参数")
        print("   继续运行 Grid Search（仅基于 loss 选择）...\n")

    # ========== 超参数网格搜索 ==========
    print(f"\n🚀 开始超参数网格搜索...")

    # 根据是否有 full_config 调整提示信息
    if full_config is not None:
        print(f"   每个组合将运行: CL训练 + 快速Supervised Training（仅验证集）")
        print(f"   选择标准: Val F1（验证集F1分数）")
        print(f"   预计时间: {total_runs * 1.0:.1f}-{total_runs * 1.5:.1f} 小时\n")
    else:
        print(f"   每个组合将运行: CL训练")
        print(f"   选择标准: 训练 Loss（降级模式）")
        print(f"   预计时间: {total_runs * 0.8:.1f}-{total_runs * 1.0:.1f} 小时\n")

    all_results = []
    best_auc = -1.0  # 追踪最佳 Val F1（复用变量名 best_auc）
    best_loss = float('inf')  # 追踪最佳 loss（降级模式）
    best_model_path = None
    best_result = None

    for run_idx, (infonce_temp, simcse_temp, base_lr, proj_lr, epochs, batch_size) in enumerate(param_combinations, 1):
        print(f"\n{'='*80}")
        print(f"🔬 运行 {run_idx}/{total_runs}")
        print(f"   infonce_temperature={infonce_temp}, simcse_temperature={simcse_temp}")
        print(f"   base_lr={base_lr}, projection_lr={proj_lr}")
        print(f"   epochs={epochs}, batch_size={batch_size}")
        print(f"{'='*80}\n")

        # 创建子目录
        run_name = f"run{run_idx}_infotau{infonce_temp}_simtau{simcse_temp}_blr{float(base_lr):.0e}_plr{float(proj_lr):.0e}_ep{epochs}_bs{batch_size}"
        run_output_dir = os.path.join(output_dir, 'grid_search', run_name)
        os.makedirs(run_output_dir, exist_ok=True)

        # 创建本次运行的配置
        run_config = config.copy()
        run_config['infonce_temperature'] = infonce_temp
        run_config['simcse_temperature'] = simcse_temp
        run_config['base_lr'] = base_lr
        run_config['projection_lr'] = proj_lr
        run_config['epochs'] = epochs
        run_config['batch_size'] = batch_size

        try:
            # 1. 运行对比学习训练（轻量级模式：只保存模型权重）
            print(f"[步骤 1/2] 对比学习训练（轻量级模式）...")
            model_path = _run_single_stage1_training(run_config, run_output_dir, lightweight_mode=True)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件未找到: {model_path}")

            # 2. 快速 Supervised Training 评估（仅验证集）
            print(f"\n[步骤 2/2] 快速 Supervised Training 评估（仅验证集）...")
            try:
                if full_config is not None:
                    sup_metrics = _run_quick_supervised_eval(
                        model_path,
                        run_output_dir,
                        full_config  # 传入完整配置
                    )
                    val_f1 = sup_metrics['val_f1']
                    val_acc = sup_metrics['val_acc']
                    val_precision = sup_metrics['val_precision']
                    val_recall = sup_metrics['val_recall']
                else:
                    print("   跳过（未提供 full_config）")
                    val_f1 = 0.0
                    val_acc = 0.0
                    val_precision = 0.0
                    val_recall = 0.0
            except Exception as e:
                print(f"⚠️  Supervised Training 评估失败: {e}")
                import traceback
                traceback.print_exc()
                val_f1 = 0.0
                val_acc = 0.0
                val_precision = 0.0
                val_recall = 0.0

            # ✅ 3. 清理模型文件（Grid Search 不保存模型）
            print(f"\n[清理] 删除模型文件以节省空间...")
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"   ✅ 已删除: {os.path.basename(model_path)}")

                # 删除 quick_sup_eval 中的模型文件
                quick_sup_dir = os.path.join(run_output_dir, 'quick_sup_eval', 'saved_models')
                if os.path.exists(quick_sup_dir):
                    shutil.rmtree(quick_sup_dir)
                    print(f"   ✅ 已删除: quick_sup_eval/saved_models/")

                # 删除编码器缓存（如果有）
                cache_dir = os.path.join(run_output_dir, 'quick_sup_eval', 'encoder_cache')
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"   ✅ 已删除: encoder_cache/")

            except Exception as e:
                print(f"   ⚠️  清理失败: {e}")

            # 5. 记录结果（不包含 model_path）
            result = {
                'run_idx': run_idx,
                'hyperparameters': {
                    'infonce_temperature': infonce_temp,
                    'simcse_temperature': simcse_temp,
                    'base_lr': base_lr,
                    'projection_lr': proj_lr,
                    'epochs': epochs,
                    'batch_size': batch_size
                },

                # 验证集指标（Grid Search 只用验证集）
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,

                'status': 'success'
            }

            all_results.append(result)

            # 6. 更新最佳结果（只记录超参数，不记录模型路径）
            if full_config is not None:
                # 完整模式：基于 Val F1 选择
                if val_f1 > best_auc:  # 变量名保留 best_auc，但实际存储的是 best_val_f1
                    best_auc = val_f1
                    best_result = result
                    print(f"\n✨ 新的最佳 Val F1: {best_auc:.4f}")
            else:
                # 降级模式：基于 loss 选择（从评估前的 checkpoint 读取）
                print(f"\n⚠️  降级模式不支持（需要 full_config）")

        except Exception as e:
            print(f"❌ 运行失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'run_idx': run_idx,
                'hyperparameters': {
                    'infonce_temperature': infonce_temp,
                    'simcse_temperature': simcse_temp,
                    'base_lr': base_lr,
                    'projection_lr': proj_lr,
                    'epochs': epochs,
                    'batch_size': batch_size
                },
                'val_f1': 0.0,
                'val_accuracy': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'status': 'error',
                'error': str(e)
            })

    # ========== 保存网格搜索结果 ==========
    print(f"\n{'='*80}")
    print("📊 网格搜索完成，保存结果...")
    print(f"{'='*80}\n")

    # 生成 JSON 结果
    results_summary = {
        'total_runs': total_runs,
        'successful_runs': sum(1 for r in all_results if r['status'] == 'success'),
        'failed_runs': sum(1 for r in all_results if r['status'] != 'success'),
        'evaluation_mode': 'validation_only',  # Grid Search 只使用验证集

        # ✅ 基于 Val F1 的最佳超参数
        'best_hyperparameters': {
            'val_f1': best_auc,  # best_auc 实际存储的是 best_val_f1
            'run_idx': best_result['run_idx'] if best_result else None,
            'hyperparameters': best_result['hyperparameters'] if best_result else {},
            'validation_metrics': {
                'val_f1': best_result['val_f1'] if best_result else 0.0,
                'val_accuracy': best_result['val_accuracy'] if best_result else 0.0,
                'val_precision': best_result['val_precision'] if best_result else 0.0,
                'val_recall': best_result['val_recall'] if best_result else 0.0
            }
        } if best_result and full_config is not None else None,

        'all_results': all_results,
        'hyperparameter_space': hyperparams_to_search
    }

    # 保存 JSON
    results_json_path = os.path.join(output_dir, 'grid_search_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    # 生成 CSV（方便 Excel 查看）
    import csv
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 表头（只包含验证集指标）
        writer.writerow([
            'run_idx',
            'infonce_temp', 'simcse_temp', 'base_lr', 'epochs', 'batch_size',
            'val_f1', 'val_acc', 'val_precision', 'val_recall',
            'status'
        ])

        # 数据行
        for r in all_results:
            if r['status'] == 'success':
                hp = r['hyperparameters']
                writer.writerow([
                    r['run_idx'],
                    hp['infonce_temperature'], hp['simcse_temperature'],
                    hp['base_lr'],
                    hp['epochs'], hp['batch_size'],
                    r['val_f1'],
                    r['val_accuracy'],
                    r['val_precision'],
                    r['val_recall'],
                    r['status']
                ])
            else:
                hp = r['hyperparameters']
                writer.writerow([
                    r['run_idx'],
                    hp['infonce_temperature'], hp['simcse_temperature'],
                    hp['base_lr'],
                    hp['epochs'], hp['batch_size'],
                    'FAILED', 'FAILED', 'FAILED', 'FAILED',
                    f"error: {r.get('error', 'Unknown error')}"
                ])

    print(f"✅ 结果已保存:")
    print(f"   JSON: {results_json_path}")
    print(f"   CSV:  {csv_path}")

    # 打印最佳结果
    if best_result:
        if full_config is not None:
            # 只显示验证集指标
            print(f"\n🏆 最佳超参数组合 (基于 Val F1):")
            print(f"   Run: {best_result['run_idx']}")
            print(f"   Val F1: {best_result['val_f1']:.4f}")
            print(f"   Val Accuracy: {best_result['val_accuracy']:.4f}")
            print(f"   Val Precision: {best_result['val_precision']:.4f}")
            print(f"   Val Recall: {best_result['val_recall']:.4f}")
        else:
            # 降级模式
            print(f"\n🏆 最佳超参数组合:")
            print(f"   Run: {best_result['run_idx']}")
            print(f"   ⚠️  注意：未运行验证集评估")

        print(f"   超参数:")
        for key, value in best_result['hyperparameters'].items():
            print(f"     - {key}: {value}")

    # ✅ Grid Search 完成，直接返回结果
    print(f"\n{'='*80}")
    print(f"✅ Grid Search 完成！")
    print(f"{'='*80}")
    print(f"📊 结果已保存:")
    print(f"   - {results_json_path}")
    print(f"   - {csv_path}")
    print(f"\n💡 使用扫描工具查看详细结果:")
    print(f"   python scan_grid_search_results.py {output_dir}")
    print(f"{'='*80}\n")

    # ✅ 返回结果汇总（不返回模型路径）
    return {
        'mode': 'grid_search',
        'total_runs': total_runs,
        'successful_runs': sum(1 for r in all_results if r['status'] == 'success'),
        'best_hyperparameters': best_result['hyperparameters'] if best_result else None,
        'best_metrics': {
            'val_f1': best_result['val_f1'] if best_result else 0.0,
            'val_accuracy': best_result['val_accuracy'] if best_result else 0.0,
            'val_precision': best_result['val_precision'] if best_result else 0.0,
            'val_recall': best_result['val_recall'] if best_result else 0.0
        } if best_result else None,
        'results_file': results_json_path
    }


def _run_single_stage1_training(config: dict, output_dir: str, lightweight_mode: bool = False) -> str:
    """
    运行单次 Stage 1 对比学习训练（内部函数）

    Args:
        config: Stage 1配置字典
        output_dir: 输出目录
        lightweight_mode: 轻量级模式（Grid Search用，只保存模型权重）

    Returns:
        模型路径
    """
    import sys
    import os
    import pickle
    from argparse import Namespace
    import pandas as pd
    from Tree_data_model import PostStorage

    print(f"[Stage 1] 单次训练模式")
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
        infonce_temperature=config.get('infonce_temperature', 0.07),  # InfoNCE温度系数
        simcse_temperature=config.get('simcse_temperature', 0.07),    # SimCSE温度系数
        simcse_dropout_rate=config.get('simcse_dropout_rate', 0.1),
        simcse_remove_duplicates=config.get('simcse_remove_duplicates', True),
        hybrid_ratio=config.get('hybrid_ratio', 0.5),
        bidirectional_loss=config.get('bidirectional_loss', True),  # 新增：双向损失控制

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
            seed=config.get('seed', 42),  # ✅ 新增：从配置读取随机种子
            training_model_type=args.model_type,
            training_model_identifier_or_path=args.model_name_or_path,
            output_dir=output_dir,  # 传入输出目录
            lightweight_mode=lightweight_mode,  # ✅ 新增：轻量级模式
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
            infonce_temperature=args.infonce_temperature,  # InfoNCE温度系数
            simcse_temperature=args.simcse_temperature,    # SimCSE温度系数
            simcse_dropout_rate=args.simcse_dropout_rate,
            simcse_remove_duplicates=args.simcse_remove_duplicates,
            hybrid_ratio=args.hybrid_ratio,
            bidirectional_loss=args.bidirectional_loss,  # 新增：双向损失控制
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

        # 保存生成的数据集到实验目录（仅非轻量级模式）
        if not lightweight_mode:
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
        else:
            print(f"[轻量级模式] 跳过数据集保存")

        # 查找最佳模型
        best_model_path = os.path.join(output_dir, "best_contrastive_model.pth")
        if not os.path.exists(best_model_path):
            # 查找其他可能的模型文件
            for file in os.listdir(output_dir):
                if file.endswith('.pth'):
                    best_model_path = os.path.join(output_dir, file)
                    break

        if lightweight_mode:
            print(f"[轻量级模式] Stage 1训练完成，模型已保存（仅权重）")
        else:
            print(f"[成功] Stage 1训练完成，模型保存在: {best_model_path}")

        return best_model_path

    except Exception as e:
        print(f"[错误] Stage 1训练失败: {e}")
        raise


def _evaluate_topic_pair_auc(
    encoder_path: str,
    run_output_dir: str,
    full_config: dict
) -> dict:
    """
    评估编码器的 Topic Pair AUC

    Args:
        encoder_path: 编码器模型路径
        run_output_dir: 当前运行的输出目录
        full_config: 完整配置（包含 supervised_learning）

    Returns:
        {
            'auc': float,
            'accuracy': float,
            'best_threshold': float,
            'precision_at_k': {10: float, 50: float, ...}
        }
    """
    import os
    import json
    import torch
    import torch.nn.functional as F
    import numpy as np
    from evaluate_encoder_topic_pairs import (
        load_and_split_data,
        construct_topic_pairs,
        load_encoder,
        encode_texts,
        compute_metrics
    )

    print(f"\n{'='*80}")
    print(f"📊 Topic Pair AUC 评估")
    print(f"{'='*80}")

    # 🔍 调试：打印 full_config 的结构
    print(f"\n[DEBUG] full_config 的顶层键: {list(full_config.keys()) if full_config else 'None'}")
    if full_config:
        if 'defaults' in full_config:
            print(f"[DEBUG] full_config['defaults'] 的键: {list(full_config['defaults'].keys())}")
        if 'supervised_learning' in full_config:
            print(f"[DEBUG] full_config['supervised_learning'] 存在")
        else:
            print(f"[DEBUG] ⚠️  full_config['supervised_learning'] 不存在")

    # 1. 从 supervised_learning 配置中提取数据参数
    # ✅ 修改：先尝试从 defaults.supervised_learning 读取，再尝试顶层 supervised_learning
    if 'defaults' in full_config and 'supervised_learning' in full_config['defaults']:
        sup_config = full_config['defaults']['supervised_learning']
        print(f"[INFO] 从 full_config['defaults']['supervised_learning'] 读取配置")
    elif 'supervised_learning' in full_config:
        sup_config = full_config['supervised_learning']
        print(f"[INFO] 从 full_config['supervised_learning'] 读取配置")
    else:
        sup_config = {}

    # ✅ 验证必要的配置参数是否存在
    if not sup_config:
        raise ValueError("配置错误：full_config 中缺少 'supervised_learning' 配置（既不在顶层也不在 defaults 中）")

    # ✅ 直接使用 YAML 配置，不提供硬编码默认值（强制从 YAML 读取）
    eval_config = {
        'train_csv': sup_config.get('train_data_path', 'data/sup_train_data/balanced_trainset.csv'),
        'test_csv': sup_config.get('test_data_path', 'data/sup_train_data/balanced_testset.csv'),

        # ✅ 关键修改：直接从 YAML 读取，不提供后备默认值
        'use_fixed_split': sup_config.get('use_fixed_split', True),
        'train_per_label': sup_config.get('train_samples_per_label', 500),
        'val_per_label': sup_config.get('val_samples_per_label'),  # ✅ 移除默认值 200
        'test_per_label': sup_config.get('test_samples_per_label'),
        'use_test_for_val_and_test': sup_config.get('use_test_for_val_and_test', False),
        'split_random_seed': sup_config.get('split_random_seed', 42),
        'validation_split': sup_config.get('validation_split', 0.2)
    }

    # ✅ 打印实际使用的配置（调试用）
    print(f"\n[Topic Pair AUC] 使用的数据分割配置:")
    print(f"  train_per_label: {eval_config['train_per_label']}")
    print(f"  val_per_label: {eval_config['val_per_label']}")
    print(f"  test_per_label: {eval_config['test_per_label']}")
    print(f"  use_fixed_split: {eval_config['use_fixed_split']}")
    print(f"  split_random_seed: {eval_config['split_random_seed']}\n")

    # 2. 加载数据（与 supervised training 完全一致的分割）
    train_df, dev_df, test_df = load_and_split_data(
        eval_config['train_csv'],
        eval_config['test_csv'],
        eval_config
    )
    print(f"使用验证集: {len(dev_df)} 条")

    # 3. 构造 topic pairs
    pair_indices, ground_truth, pair_stats = construct_topic_pairs(
        dev_df,
        P_pos=1,
        P_neg=1,
        random_seed=eval_config['split_random_seed'],
        shuffle_order=True
    )

    # 4. 加载编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, tokenizer, model_type = load_encoder(encoder_path, device)

    # 5. 编码文本
    texts = dev_df['content'].tolist()
    embeddings = encode_texts(
        texts, encoder, tokenizer, device,
        batch_size=64,
        max_len=256
    )

    # 6. 计算相似度
    embeddings_tensor = torch.tensor(embeddings)
    indices1 = [i for i, j in pair_indices]
    indices2 = [j for i, j in pair_indices]

    embeddings1 = embeddings_tensor[indices1]
    embeddings2 = embeddings_tensor[indices2]

    cosine_sim = F.cosine_similarity(embeddings1, embeddings2).numpy()

    # 7. 计算指标
    metrics = compute_metrics(ground_truth, cosine_sim, k_values=[10, 50, 100, 500])

    # 8. 保存结果
    result_path = os.path.join(run_output_dir, 'topic_pair_auc.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Topic Pair AUC: {metrics['auc']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision@100: {metrics['precision_at_k'].get(100, 'N/A')}")

    return metrics


def _run_quick_supervised_eval(
    encoder_path: str,
    run_output_dir: str,
    full_config: dict
) -> dict:
    """
    快速监督学习评估（用于 Grid Search）

    策略：
    - 冻结编码器
    - 只用线性分类器
    - 固定超参数（减少搜索空间）
    - 快速训练 10 epoch

    Args:
        encoder_path: 编码器模型路径
        run_output_dir: 当前运行的输出目录
        full_config: 完整配置（包含 supervised_learning）

    Returns:
        {
            'val_f1': float,
            'val_acc': float,
            'test_f1': float,
            'test_acc': float
        }
    """
    import os
    import json
    from sup_training import run_supervised_training_interface

    print(f"\n{'='*80}")
    print(f"🚀 快速 Supervised Training 评估")
    print(f"{'='*80}")

    # 🔍 调试：打印 full_config 的结构
    print(f"\n[DEBUG] full_config 的顶层键: {list(full_config.keys()) if full_config else 'None'}")
    if full_config:
        if 'defaults' in full_config:
            print(f"[DEBUG] full_config['defaults'] 的键: {list(full_config['defaults'].keys())}")
        if 'supervised_learning' in full_config:
            print(f"[DEBUG] full_config['supervised_learning'] 存在")
        else:
            print(f"[DEBUG] ⚠️  full_config['supervised_learning'] 不存在")

    # 从 supervised_learning 配置中提取参数
    # ✅ 修改：先尝试从 defaults.supervised_learning 读取，再尝试顶层 supervised_learning
    if 'defaults' in full_config and 'supervised_learning' in full_config['defaults']:
        sup_config = full_config['defaults']['supervised_learning']
        print(f"[INFO] 从 full_config['defaults']['supervised_learning'] 读取配置")
    elif 'supervised_learning' in full_config:
        sup_config = full_config['supervised_learning']
        print(f"[INFO] 从 full_config['supervised_learning'] 读取配置")
    else:
        sup_config = {}

    # ✅ 验证必要的配置参数是否存在
    if not sup_config:
        raise ValueError("配置错误：full_config 中缺少 'supervised_learning' 配置（既不在顶层也不在 defaults 中）")

    # ✅ 构造快速评估配置
    # 📌 优先从 quick_eval 子配置读取，否则使用默认值或从完整配置中取第一个值

    def _get_first_value(param):
        """提取参数的第一个值（如果是列表则取首元素，否则原样返回）"""
        if isinstance(param, list) and len(param) > 0:
            return param[0]
        return param

    # ✅ 新增：检查是否有 quick_eval 专用配置
    quick_eval_config = sup_config.get('quick_eval', {})

    if quick_eval_config:
        print(f"\n[Quick Eval] 使用 YAML 中的 quick_eval 专用配置")
    else:
        print(f"\n[Quick Eval] 未找到 quick_eval 配置，使用默认值或完整配置的第一个值")

    quick_config = {
        # 数据路径
        'train_data_path': sup_config.get('train_data_path', 'data/sup_train_data/balanced_trainset.csv'),
        'test_data_path': sup_config.get('test_data_path', 'data/sup_train_data/balanced_testset.csv'),

        # ✅ 数据分割参数（直接从 YAML 读取）
        'use_fixed_split': sup_config.get('use_fixed_split', True),
        'train_samples_per_label': sup_config.get('train_samples_per_label', 500),
        'val_samples_per_label': sup_config.get('val_samples_per_label'),
        'test_samples_per_label': sup_config.get('test_samples_per_label'),
        'use_test_for_val_and_test': sup_config.get('use_test_for_val_and_test', False),
        'split_random_seed': sup_config.get('split_random_seed', 42),
        'validation_split': sup_config.get('validation_split', 0.2),

        # ✅ 训练超参数（优先从 quick_eval 读取，否则使用默认值）
        'data_fractions': [quick_eval_config.get('data_fraction', 1.0)],
        'epochs': [quick_eval_config.get('epochs', _get_first_value(sup_config.get('epochs', [10])))],
        'batch_size': [quick_eval_config.get('batch_size', _get_first_value(sup_config.get('batch_size', [64])))],
        'learning_rate': [quick_eval_config.get('learning_rate', _get_first_value(sup_config.get('learning_rate', [1e-3])))],
        'seeds': quick_eval_config.get('seeds', [42]),
        'classifier_types': quick_eval_config.get('classifier_types', ['linear']),
        'mlp_hidden_neurons': [quick_eval_config.get('mlp_hidden_neurons', 384)],
        'freeze_encoder': quick_eval_config.get('freeze_encoder', [True]),
        'use_encoder_cache': quick_eval_config.get('use_encoder_cache', True),
        'cache_batch_size': quick_eval_config.get('cache_batch_size', 64),

        # ✅ Grid Search 专用：只在验证集上评估
        'skip_test_eval': True  # 新增参数，告诉 sup_training 跳过测试集评估
    }

    # ✅ 打印实际使用的配置（调试用）
    print(f"\n[Quick Eval] 使用的数据分割配置:")
    print(f"  train_samples_per_label: {quick_config['train_samples_per_label']}")
    print(f"  val_samples_per_label: {quick_config['val_samples_per_label']}")
    print(f"  test_samples_per_label: {quick_config['test_samples_per_label']}")
    print(f"  use_fixed_split: {quick_config['use_fixed_split']}")
    print(f"  split_random_seed: {quick_config['split_random_seed']}")

    print(f"\n[Quick Eval] 使用的训练超参数（固定单一值）:")
    print(f"  epochs: {quick_config['epochs'][0]}")
    print(f"  batch_size: {quick_config['batch_size'][0]}")
    print(f"  learning_rate: {quick_config['learning_rate'][0]}")
    print(f"  classifier_type: {quick_config['classifier_types'][0]}")
    print(f"  freeze_encoder: {quick_config['freeze_encoder'][0]}")
    print(f"  超参数组合数: 1 (固定配置，无网格搜索)\n")

    # 运行监督学习
    sup_output_dir = os.path.join(run_output_dir, 'quick_sup_eval')
    os.makedirs(sup_output_dir, exist_ok=True)

    result = run_supervised_training_interface(
        encoder_path,
        quick_config,
        sup_output_dir
    )

    # 提取关键指标（只用验证集）
    metrics = result['metrics']

    # ✅ 只提取验证集指标
    output = {
        'val_f1': metrics['dev']['f1_score'],
        'val_acc': metrics['dev']['accuracy'],
        'val_precision': metrics['dev']['precision'],
        'val_recall': metrics['dev']['recall']
    }

    print(f"✅ Val F1: {output['val_f1']:.4f}, Val Acc: {output['val_acc']:.4f}")

    # ✅ 保存结果到 quick_sup_eval/results.json（供扫描工具使用）
    results_json_path = os.path.join(sup_output_dir, 'results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"   结果已保存: {results_json_path}")

    return output


if __name__ == "__main__":
    main_training_pipeline()