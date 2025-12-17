"""
迭代实验管理系统
整合对比学习、监督学习、分类器选择、一致性评分和加权对比学习的多轮迭代流程
"""

import os
import sys
import json
import yaml
import argparse
import traceback
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path

# 导入实验管理模块
from experiment_manager import (
    ExperimentStatusDetector,
    ExperimentStateManager,
    ExperimentVariantManager,
    print_experiment_status
)

# 导入现有模块的接口
try:
    from cl_main import run_stage1_contrastive_training
except ImportError as e:
    print(f"Failed to import cl_main: {e}")
    run_stage1_contrastive_training = None
except Exception as e:
    print(f"Other error importing cl_main: {e}")
    run_stage1_contrastive_training = None

try:
    from weighted_cl_training import run_stage2_weighted_contrastive
except ImportError as e:
    print(f"Failed to import weighted_cl_training: {e}")
    run_stage2_weighted_contrastive = None
except Exception as e:
    print(f"Other error importing weighted_cl_training: {e}")
    run_stage2_weighted_contrastive = None

try:
    from sup_training import run_supervised_training_interface
except ImportError as e:
    print(f"Failed to import sup_training: {e}")
    run_supervised_training_interface = None
except Exception as e:
    print(f"Other error importing sup_training: {e}")
    run_supervised_training_interface = None

try:
    from classifier_selector import run_classifier_selection_interface
except ImportError as e:
    print(f"Failed to import classifier_selector: {e}")
    run_classifier_selection_interface = None
except Exception as e:
    print(f"Other error importing classifier_selector: {e}")
    run_classifier_selection_interface = None

try:
    from consistency_scorer import run_consistency_scoring_interface
except ImportError as e:
    print(f"Failed to import consistency_scorer: {e}")
    run_consistency_scoring_interface = None
except Exception as e:
    print(f"Other error importing consistency_scorer: {e}")
    run_consistency_scoring_interface = None


class IterativeExperimentManager:
    """迭代实验管理器（增强版，支持断点续传和多变体实验）"""

    def __init__(self, config_path: str, experiment_dir: Optional[str] = None,
                 resume: bool = False, force_restart: bool = False,
                 use_fresh_bert: Optional[bool] = None):
        """
        初始化实验管理器

        Args:
            config_path: 配置文件路径
            experiment_dir: 实验目录，如果不指定则自动创建
            resume: 是否启用断点续传
            force_restart: 是否强制重新开始
            use_fresh_bert: 是否每轮从原始BERT重新训练（命令行参数，会覆盖YAML配置）
        """
        self.config = self._load_config(config_path)
        self.experiment_name = self.config['experiment_meta']['name']
        self.resume = resume and not force_restart
        self.force_restart = force_restart

        # ✨ 处理模型基座选择参数（命令行优先级 > YAML配置）
        if use_fresh_bert is not None:
            # 命令行参数覆盖YAML配置
            if 'experiment_meta' not in self.config:
                self.config['experiment_meta'] = {}
            self.config['experiment_meta']['use_fresh_bert_each_round'] = use_fresh_bert

        if experiment_dir:
            self.experiment_dir = experiment_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = os.path.join(
                "iterative_experiments",
                f"{self.experiment_name}_{timestamp}"
            )

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.log_file = os.path.join(self.experiment_dir, "experiment_log.json")
        self.experiment_log = self._load_or_create_log()

        # 初始化状态管理器
        self.state_manager = ExperimentStateManager(self.config, self.experiment_dir)
        self._latest_supervised_output: Optional[Dict] = None
        self._latest_classifier_output: Optional[Dict] = None

        print(f"迭代实验管理器初始化")
        print(f"实验目录: {self.experiment_dir}")
        print(f"配置文件: {config_path}")
        print(f"实验名称: {self.experiment_name}")
        print(f"断点续传: {'启用' if self.resume else '禁用'}")
        print(f"强制重启: {'是' if self.force_restart else '否'}")

        # ✨ 显示模型基座模式
        use_fresh = self.config.get('experiment_meta', {}).get('use_fresh_bert_each_round', False)
        if use_fresh:
            print(f"模型基座: 每轮从原始BERT重新训练 ✨")
        else:
            print(f"模型基座: 累积训练（每轮基于上一轮模型）")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _load_or_create_log(self) -> Dict:
        """加载或创建实验日志"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'config': self.config,
                'rounds': {}
            }

    def _save_log(self):
        """保存实验日志"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)

    def _get_round_config(self, round_num: int) -> Dict:
        """获取特定轮次的配置"""
        # 从默认配置开始
        round_config = self.config.get('defaults', {}).copy()

        # 应用轮次特定的覆盖配置
        if 'round_specific' in self.config and round_num in self.config['round_specific']:
            round_overrides = self.config['round_specific'][round_num]
            for key, value in round_overrides.items():
                if key in round_config:
                    round_config[key].update(value)
                else:
                    round_config[key] = value

        return round_config

    def _find_previous_encoder(self, round_num: int) -> Optional[str]:
        """查找上一轮的最佳编码器"""
        prev_round_dir = os.path.join(self.experiment_dir, f"round{round_num-1}")

        # 首先查找best_model.pth（Stage 2+的输出）
        best_model_path = os.path.join(prev_round_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            return best_model_path

        # 如果是第一轮，查找contrastive_training目录下的模型
        contrastive_dir = os.path.join(prev_round_dir, "contrastive_training")
        if os.path.exists(contrastive_dir):
            # 查找best_contrastive_model.pth或类似文件
            for file in os.listdir(contrastive_dir):
                if 'best' in file and file.endswith('.pth'):
                    return os.path.join(contrastive_dir, file)

        return None

    def _find_previous_enhanced_dataset(self, round_num: int) -> Optional[str]:
        """查找上一轮的增强数据集"""
        prev_round_dir = os.path.join(self.experiment_dir, f"round{round_num-1}")
        consistency_dir = os.path.join(prev_round_dir, "consistency_scoring")

        if os.path.exists(consistency_dir):
            # 查找enhanced_dataset开头的pkl文件
            for root, dirs, files in os.walk(consistency_dir):
                for file in files:
                    if file.startswith('enhanced_dataset') and file.endswith('.pkl'):
                        return os.path.join(root, file)

        return None

    def _find_classifier_for_fraction(self, round_dir: str, fraction: float) -> Optional[str]:
        """查找指定数据比例的分类器"""
        selection_dir = os.path.join(round_dir, "classifier_selection", "selected_models")

        if os.path.exists(selection_dir):
            for file in os.listdir(selection_dir):
                # 查找包含指定fraction的json文件
                if f"frac{fraction}" in file and file.endswith('.json'):
                    json_path = os.path.join(selection_dir, file)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        model_path = data.get('model_path')
                        if model_path:
                            return json_path  # 返回JSON路径，让consistency_scorer处理

        return None

    def run_single_round_with_resume(self, round_num: int) -> bool:
        """
        运行单轮实验（支持断点续传）

        Args:
            round_num: 轮次编号

        Returns:
            是否成功完成
        """
        print(f"\n{'='*60}")
        print(f"开始第 {round_num} 轮实验（断点续传模式）")
        print(f"{'='*60}")

        round_dir = os.path.join(self.experiment_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        round_config = self._get_round_config(round_num)

        round_key = f'round{round_num}'
        round_entry = self.experiment_log['rounds'].setdefault(round_key, {})
        if 'start_time' not in round_entry:
            round_entry['start_time'] = datetime.now().isoformat()
        round_entry.setdefault('status', 'running')
        round_entry['config'] = round_config
        self._save_log()

        # 创建状态检测器
        detector = ExperimentStatusDetector(self.experiment_dir, round_num)

        # 记录轮次开始
        if not self.state_manager.is_round_completed(round_num):
            self.state_manager.save_stage_in_progress(round_num, 'round_start')

        try:
            # Step 1: 对比学习（Stage 1 或 Stage 2+）
            if round_num == 1:
                stage1_done, encoder_path = detector.detect_stage1_contrastive_status()
                if stage1_done and self.resume:
                    print("[跳过] ✅ Stage 1对比学习已完成")
                else:
                    print("[执行] ▶️  开始Stage 1对比学习...")
                    encoder_path = self._run_stage1_contrastive(round_config, round_dir)
                    # ✅ Grid Search 模式会返回 None
                    if encoder_path:
                        self.state_manager.save_stage_completion(round_num, 'stage1_contrastive', encoder_path)
            else:
                stage2_done, encoder_path = detector.detect_stage2_contrastive_status()
                if stage2_done and self.resume:
                    print("[跳过] ✅ Stage 2对比学习已完成")
                else:
                    print("[执行] ▶️  开始Stage 2对比学习...")
                    encoder_path = self._run_stage2_weighted_contrastive(round_num, round_config, round_dir)
                    if encoder_path:
                        self.state_manager.save_stage_completion(round_num, 'stage2_contrastive', encoder_path)

            if not encoder_path:
                # ✅ Grid Search 模式：跳过后续流程
                print(f"\n{'='*60}")
                print(f"⚠️  Grid Search 模式：Round {round_num} 仅生成评估结果")
                print(f"   请查看 Grid Search 结果，选择最佳超参数后手动配置并重新运行")
                print(f"{'='*60}\n")

                # 标记轮次为 grid_search_only
                round_entry['status'] = 'grid_search_only'
                round_entry['end_time'] = datetime.now().isoformat()
                self._save_log()

                return True  # 返回成功，但跳过后续流程

            print(f"[完成] 编码器训练完成: {encoder_path}")

            # Step 2: 监督学习
            sup_done, sup_result_dir = detector.detect_supervised_learning_status()
            if sup_done and self.resume:
                print("[跳过] ✅ 监督学习已完成")
                # 从磁盘恢复最优模型信息
                sup_output = self._load_supervised_output_from_disk(sup_result_dir)
            else:
                print("[执行] ▶️  开始监督学习...")
                sup_output = self._run_supervised_training(encoder_path, round_config, round_dir)
                self.state_manager.save_stage_completion(round_num, 'supervised_learning', sup_output['best_model_path'])

            best_model_path = sup_output['best_model_path']
            print(f"[完成] 监督学习完成: {best_model_path}")

            # ✅ 记录详细指标到实验日志
            round_entry['best_model_path'] = best_model_path
            round_entry['best_hyperparams'] = sup_output['hyperparameters']
            round_entry['best_val_f1'] = sup_output.get('best_val_f1')
            round_entry['best_epoch'] = sup_output.get('best_epoch')
            round_entry['train_loss_at_best'] = sup_output.get('train_loss_at_best')
            round_entry['metrics'] = sup_output['metrics']  # {train, dev, test}
            self._save_log()

            # ❌ Step 3: 分类器选择 - 已删除

            # Step 4: 一致性评分
            cons_done, enhanced_dataset = detector.detect_consistency_scoring_status()
            if cons_done and self.resume:
                print("[跳过] ✅ 一致性评分已完成")
                consistency_result_dir = os.path.join(round_dir, "consistency_scoring")
            else:
                print("[执行] ▶️  开始一致性评分...")
                consistency_result_dir = self._run_consistency_scoring(
                    best_model_path,  # ✅ 直接传模型路径
                    round_config,
                    round_dir
                )
                self.state_manager.save_stage_completion(round_num, 'consistency_scoring', consistency_result_dir)

            print(f"[完成] 一致性评分完成: {consistency_result_dir}")

            # 记录轮次成功
            round_entry['status'] = 'completed'
            round_entry['end_time'] = datetime.now().isoformat()
            round_entry['encoder_path'] = encoder_path
            self._save_log()

            print(f"[成功] 第 {round_num} 轮实验成功完成！")
            return True

        except Exception as e:
            print(f"[错误] 第 {round_num} 轮实验失败: {e}")
            traceback.print_exc()

            # 记录轮次失败
            round_entry['status'] = 'failed'
            round_entry['error'] = str(e)
            self._save_log()

            return False

    def run_single_round(self, round_num: int) -> bool:
        """
        运行单轮实验

        Args:
            round_num: 轮次编号

        Returns:
            是否成功完成
        """
        print(f"\n{'='*60}")
        print(f"开始第 {round_num} 轮实验")
        print(f"{'='*60}")

        round_dir = os.path.join(self.experiment_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        round_config = self._get_round_config(round_num)

        # 记录轮次开始
        self.experiment_log['rounds'][f'round{round_num}'] = {
            'start_time': datetime.now().isoformat(),
            'config': round_config,
            'status': 'running'
        }
        self._save_log()

        try:
            # Step 1: 对比学习（Stage 1 或 Stage 2+）
            if round_num == 1:
                encoder_path = self._run_stage1_contrastive(round_config, round_dir)
            else:
                encoder_path = self._run_stage2_weighted_contrastive(round_num, round_config, round_dir)

            if not encoder_path:
                # ✅ Grid Search 模式：跳过后续流程
                print(f"\n{'='*60}")
                print(f"⚠️  Grid Search 模式：Round {round_num} 仅生成评估结果")
                print(f"   请查看 Grid Search 结果，选择最佳超参数后手动配置并重新运行")
                print(f"{'='*60}\n")

                # 标记轮次为 grid_search_only
                round_entry = self.experiment_log['rounds'][f'round{round_num}']
                round_entry['status'] = 'grid_search_only'
                round_entry['end_time'] = datetime.now().isoformat()
                self._save_log()

                return True  # 返回成功，但跳过后续流程

            print(f"[完成] 编码器训练完成: {encoder_path}")

            # Step 2: 监督学习
            sup_output = self._run_supervised_training(encoder_path, round_config, round_dir)
            best_model_path = sup_output['best_model_path']
            print(f"[完成] 监督学习完成: {best_model_path}")

            # ✅ 记录详细指标到实验日志
            round_entry = self.experiment_log['rounds'][f'round{round_num}']
            round_entry['best_model_path'] = best_model_path
            round_entry['best_hyperparams'] = sup_output['hyperparameters']
            round_entry['best_val_f1'] = sup_output.get('best_val_f1')
            round_entry['best_epoch'] = sup_output.get('best_epoch')
            round_entry['train_loss_at_best'] = sup_output.get('train_loss_at_best')
            round_entry['metrics'] = sup_output['metrics']  # {train, dev, test}
            self._save_log()

            # ❌ Step 3: 分类器选择 - 已删除

            # Step 4: 一致性评分
            consistency_result_dir = self._run_consistency_scoring(
                best_model_path,  # ✅ 直接传模型路径
                round_config,
                round_dir
            )
            print(f"[完成] 一致性评分完成: {consistency_result_dir}")

            # 记录轮次成功
            self.experiment_log['rounds'][f'round{round_num}']['status'] = 'completed'
            self.experiment_log['rounds'][f'round{round_num}']['end_time'] = datetime.now().isoformat()
            self.experiment_log['rounds'][f'round{round_num}']['encoder_path'] = encoder_path
            self._save_log()

            print(f"[成功] 第 {round_num} 轮实验成功完成！")
            return True

        except Exception as e:
            print(f"[错误] 第 {round_num} 轮实验失败: {e}")
            traceback.print_exc()

            # 记录轮次失败
            self.experiment_log['rounds'][f'round{round_num}']['status'] = 'failed'
            self.experiment_log['rounds'][f'round{round_num}']['error'] = str(e)
            self._save_log()

            return False

    def _run_stage1_contrastive(self, config: Dict, round_dir: str) -> Optional[str]:
        """运行Stage 1对比学习"""
        print("[Stage 1] 运行Stage 1对比学习（BERT相似度剪枝）...")

        output_dir = os.path.join(round_dir, "contrastive_training")
        os.makedirs(output_dir, exist_ok=True)

        if not run_stage1_contrastive_training:
            raise RuntimeError("Stage 1对比学习接口未实现")

        stage1_config = config.get('stage1_contrastive', {})

        # 从全局data_paths获取数据路径
        data_paths = self.config.get('data_paths', {})
        stage1_config['cl_comments_data'] = data_paths.get('cl_comments_data', 'data/cl_data/train_comments_filtered.csv')
        stage1_config['cl_posts_data'] = data_paths.get('cl_posts_data', 'data/cl_data/train_posts_filtered.csv')

        # 设置输出目录为实验目录（确保数据集保存在实验目录内）
        stage1_config['output_dir'] = output_dir

        # ✅ 修复：传入完整配置（包含 supervised_learning 等）
        result = run_stage1_contrastive_training(
            stage1_config,
            output_dir,
            self.config  # 传入完整配置树（用于 Grid Search 评估）
        )

        # ✅ 处理 Grid Search 返回字典的情况
        if isinstance(result, dict) and result.get('mode') == 'grid_search':
            # Grid Search 模式：不返回模型，记录最佳超参数
            print(f"\n[Grid Search] 完成 {result['total_runs']} 次运行")
            print(f"[Grid Search] 成功: {result['successful_runs']} 次")

            if result['best_hyperparameters']:
                print(f"\n[Grid Search] 最佳超参数:")
                for key, value in result['best_hyperparameters'].items():
                    print(f"   - {key}: {value}")

                print(f"\n[Grid Search] 最佳验证集指标:")
                if result['best_metrics']:
                    print(f"   - Val F1: {result['best_metrics']['val_f1']:.4f}")
                    print(f"   - Val Accuracy: {result['best_metrics']['val_accuracy']:.4f}")
                    print(f"   - Val Precision: {result['best_metrics']['val_precision']:.4f}")
                    print(f"   - Val Recall: {result['best_metrics']['val_recall']:.4f}")

            # 记录 Grid Search 结果到实验日志
            self.experiment_log['rounds'][f'round1']['grid_search'] = {
                'total_runs': result['total_runs'],
                'successful_runs': result['successful_runs'],
                'best_hyperparameters': result['best_hyperparameters'],
                'best_metrics': result['best_metrics'],
                'results_file': result['results_file']
            }

            self._save_log()

            # ⚠️ Grid Search 模式：跳过后续流程（没有模型）
            print(f"\n⚠️  Grid Search 模式：仅生成评估结果，不继续后续流程")
            print(f"   结果文件: {result['results_file']}")
            print(f"\n💡 请查看 Grid Search 结果，选择最佳超参数后手动重新训练")

            return None  # 返回 None 表示没有模型

        # 单次训练模式：返回模型路径
        model_path = result

        # ✅ 从checkpoint读取并记录数据集大小到实验日志
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            dataset_sizes = checkpoint.get('training_history', {}).get('dataset_sizes', {})
            positive_pair_strategy = checkpoint.get('positive_pair_strategy', 'unknown')

            # 获取最新的数据集大小（最后一次构建）
            ds1_size = dataset_sizes.get('dataset1', [0])[-1] if dataset_sizes.get('dataset1') else 0
            ds2_size = dataset_sizes.get('dataset2', [0])[-1] if dataset_sizes.get('dataset2') else 0

            # 记录到实验日志
            self.experiment_log['rounds'][f'round1']['dataset_info'] = {
                'dataset_path': os.path.join(output_dir, 'dataset.pkl'),
                'dataset1_size': int(ds1_size),
                'dataset2_size': int(ds2_size),
                'positive_pair_strategy': positive_pair_strategy,
                'dataset_sizes_history': {
                    'dataset1': [int(x) for x in dataset_sizes.get('dataset1', [])],
                    'dataset2': [int(x) for x in dataset_sizes.get('dataset2', [])]
                }
            }

            print(f"[记录] 数据集大小已保存到实验日志:")
            print(f"   - 策略: {positive_pair_strategy}")
            print(f"   - Dataset1: {ds1_size}")
            print(f"   - Dataset2: {ds2_size}")

        except Exception as e:
            print(f"[警告] 无法从checkpoint读取数据集大小: {e}")
            # 降级处理：只记录路径
            self.experiment_log['rounds'][f'round1']['dataset_path'] = os.path.join(output_dir, 'dataset.pkl')

        self._save_log()

        return model_path

    def _run_stage2_weighted_contrastive(self, round_num: int, config: Dict, round_dir: str) -> Optional[str]:
        """运行Stage 2+加权对比学习"""

        # ✅ 检查是否每轮从原始BERT重新训练
        use_fresh_bert = self.config.get('experiment_meta', {}).get('use_fresh_bert_each_round', False)

        if use_fresh_bert:
            # 从原始BERT重新训练模式
            stage1_config = self.config.get('defaults', {}).get('stage1_contrastive', {})
            prev_encoder = stage1_config.get('model_name_or_path', 'google-bert/bert-base-chinese')
            print(f"[Stage 2+] 从原始BERT重新训练（Round {round_num}）")
            print(f"  ✨ 模式: 固定原始BERT基座")
            print(f"  使用模型: {prev_encoder}")
        else:
            # 累积训练模式（默认）
            prev_encoder = self._find_previous_encoder(round_num)
            if not prev_encoder:
                raise FileNotFoundError(f"找不到第{round_num-1}轮的编码器")
            print(f"[Stage 2+] 累积训练模式（基于第{round_num-1}轮）")
            print(f"  使用编码器: {prev_encoder}")

        # 查找上一轮的增强数据集（两种模式都需要）
        prev_enhanced_dataset = self._find_previous_enhanced_dataset(round_num)
        if not prev_enhanced_dataset:
            raise FileNotFoundError(f"找不到第{round_num-1}轮的增强数据集")

        print(f"  使用增强数据集: {prev_enhanced_dataset}")

        if not run_stage2_weighted_contrastive:
            raise RuntimeError("Stage 2加权对比学习接口未实现")

        stage2_config = config.get('stage2_contrastive', {})
        # 直接保存到round目录下的best_model.pth
        return run_stage2_weighted_contrastive(
            stage2_config, prev_encoder, prev_enhanced_dataset, round_dir, round_num
        )

    def _run_supervised_training(self, encoder_path: str, config: Dict, round_dir: str) -> Dict:
        """运行监督学习"""
        print("[监督学习] 运行监督学习超参数搜索...")

        output_dir = os.path.join(round_dir, "supervised_training")
        os.makedirs(output_dir, exist_ok=True)

        if not run_supervised_training_interface:
            raise RuntimeError("监督学习接口未实现")

        sup_config = config.get('supervised_learning', {})

        # 从round_dir提取轮次号
        round_num = int(os.path.basename(round_dir).replace('round', ''))

        sup_result = run_supervised_training_interface(encoder_path, sup_config, output_dir, round_num)

        # ✅ 直接返回完整的监督学习输出（包含best_model_path和metrics）
        if isinstance(sup_result, dict):
            sup_result.setdefault('experiment_dir', output_dir)
            self._latest_supervised_output = sup_result
            return sup_result

        # 兼容旧版本接口仅返回目录路径的情况（不应该发生）
        self._latest_supervised_output = {
            'experiment_dir': sup_result,
            'best_model_path': None,
            'metrics': {},
            'best_epoch': None,
            'train_loss_at_best': None,
            'hyperparameters': None,
            'used_cache': False
        }
        return self._latest_supervised_output

    def _load_supervised_output_from_disk(self, sup_result_dir: str) -> Dict:
        """从监督学习结果目录恢复最佳模型信息"""
        results_file = os.path.join(sup_result_dir, "all_seeds_results.json")
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"找不到监督学习结果文件: {results_file}")

        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        # ✅ 基于验证集F1选择最优模型
        best_entry = None
        best_val_f1 = float('-inf')

        for runs in all_results.values():
            for run in runs:
                val_f1 = run.get('val_f1', float('-inf'))
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_entry = run

        if not best_entry:
            raise RuntimeError(f"无法从 {results_file} 恢复监督学习最优结果")

        model_path = best_entry.get('model_path')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(sup_result_dir, model_path))

        return {
            'experiment_dir': sup_result_dir,
            'best_model_path': model_path,
            'metrics': {
                'train': best_entry.get('train_metrics'),
                'dev': best_entry.get('val_metrics'),
                'test': best_entry.get('test_metrics')
            },
            'best_epoch': best_entry.get('best_epoch'),
            'train_loss_at_best': best_entry.get('train_loss_at_best'),
            'hyperparameters': best_entry.get('hyperparameters'),
            'used_cache': best_entry.get('used_cache', False),
            'best_val_f1': best_val_f1  # ✅ 新增验证集F1
        }

    def _run_classifier_selection(self, sup_result_dir: str, config: Dict, round_dir: str) -> str:
        """运行分类器选择"""
        print("[分类器选择] 运行分类器选择...")

        output_dir = os.path.join(round_dir, "classifier_selection")
        os.makedirs(output_dir, exist_ok=True)

        if not run_classifier_selection_interface:
            raise RuntimeError("分类器选择接口未实现")

        selection_config = config.get('classifier_selection', {})
        return run_classifier_selection_interface(sup_result_dir, selection_config, output_dir)

    def _run_consistency_scoring(self, best_model_path: str, config: Dict, round_dir: str) -> str:
        """运行一致性评分"""
        print("[一致性评分] 运行一致性评分...")

        output_dir = os.path.join(round_dir, "consistency_scoring")
        os.makedirs(output_dir, exist_ok=True)

        scoring_config = config.get('consistency_scoring', {})

        # 动态获取数据集路径
        dataset_path = self._find_current_dataset(round_dir)
        if not dataset_path:
            # 获取当前轮次号
            round_num = int(os.path.basename(round_dir).replace('round', ''))
            if round_num == 1:
                # 第一轮，查找Stage 1生成的数据集
                dataset_path = os.path.join(round_dir, "contrastive_training", "dataset.pkl")
                if not os.path.exists(dataset_path):
                    # 尝试查找其他格式的数据集文件
                    contrastive_dir = os.path.join(round_dir, "contrastive_training")
                    if os.path.exists(contrastive_dir):
                        for file in os.listdir(contrastive_dir):
                            if file.endswith('.pkl') and 'dataset' in file:
                                dataset_path = os.path.join(contrastive_dir, file)
                                break
            else:
                # 后续轮次，使用上一轮的增强数据集
                dataset_path = self._find_previous_enhanced_dataset(round_num)

        if not dataset_path:
            raise FileNotFoundError(f"找不到第{round_num}轮的数据集")

        print(f"   使用数据集: {dataset_path}")
        print(f"   使用模型: {best_model_path}")
        scoring_config['dataset_path'] = dataset_path

        if not run_consistency_scoring_interface:
            raise RuntimeError("一致性评分接口未实现")

        # ✅ 直接传入模型路径，不再传classifier_path
        return run_consistency_scoring_interface(
            best_model_path,  # 直接传模型路径
            dataset_path,
            scoring_config,
            output_dir
        )

    def _find_current_dataset(self, round_dir: str) -> Optional[str]:
        """查找当前轮次的数据集"""
        # 获取轮次号
        round_num = int(os.path.basename(round_dir).replace('round', ''))

        if round_num == 1:
            # 第一轮：查找Stage 1生成的数据集
            dataset_path = os.path.join(round_dir, "contrastive_training", "dataset.pkl")
            if os.path.exists(dataset_path):
                return dataset_path

            # 查找其他可能的数据集文件
            contrastive_dir = os.path.join(round_dir, "contrastive_training")
            if os.path.exists(contrastive_dir):
                for file in os.listdir(contrastive_dir):
                    if file.endswith('.pkl') and 'dataset' in file:
                        return os.path.join(contrastive_dir, file)
        else:
            # 后续轮次：使用上一轮的增强数据集
            return self._find_previous_enhanced_dataset(round_num)

        return None

    def _get_selected_rounds(self) -> List[int]:
        """获取要运行的轮次列表"""
        return self.config['experiment_meta'].get('rounds', [1, 2, 3])

    def run_experiment(self, rounds: Optional[List[int]] = None, start_round: Optional[int] = None,
                       target_round: Optional[int] = None):
        """
        运行完整的迭代实验（支持断点续传）

        Args:
            rounds: 指定要运行的轮次列表
            start_round: 从某轮开始
            target_round: 目标轮次（运行到此轮结束）
        """
        # 如果启用断点续传，首先检查实验状态
        if self.resume and not self.force_restart:
            print("\n🔍 检测已有实验状态...")
            print_experiment_status(self.experiment_dir, target_round or 3)

            if not self.state_manager.validate_config_compatibility():
                response = input("\n⚠️  配置文件已更改，是否继续使用新配置？(y/n): ")
                if response.lower() != 'y':
                    print("实验已取消")
                    return

            print(self.state_manager.get_progress_summary())

        # 确定要运行的轮次
        if rounds:
            selected_rounds = rounds
        elif target_round:
            selected_rounds = list(range(1, target_round + 1))
        else:
            selected_rounds = self._get_selected_rounds()

        if start_round:
            selected_rounds = [r for r in selected_rounds if r >= start_round]

        print(f"\n计划运行轮次: {selected_rounds}")

        success_rounds = []
        failed_rounds = []
        skipped_rounds = []

        for round_num in selected_rounds:
            # 检查是否应该跳过
            if self._should_skip_round(round_num):
                print(f"[跳过] 跳过第 {round_num} 轮（配置中标记为skip）")
                skipped_rounds.append(round_num)
                continue

            # 如果启用断点续传，检查轮次是否已完成
            if self.resume and self.state_manager.is_round_completed(round_num):
                print(f"[跳过] ✅ 第 {round_num} 轮已完成")
                skipped_rounds.append(round_num)
                success_rounds.append(round_num)
                continue

            # 运行单轮（支持断点续传）
            if self.resume:
                success = self.run_single_round_with_resume(round_num)
            else:
                success = self.run_single_round(round_num)

            if success:
                success_rounds.append(round_num)
            else:
                failed_rounds.append(round_num)
                # 如果某轮失败，后续轮次可能依赖它，询问是否继续
                if round_num < max(selected_rounds):
                    response = input(f"第 {round_num} 轮失败，是否继续后续轮次？(y/n): ")
                    if response.lower() != 'y':
                        break

        # 实验总结
        print(f"\n{'='*60}")
        print(f"实验总结")
        print(f"{'='*60}")
        print(f"[成功] 成功轮次: {success_rounds}")
        print(f"[跳过] 跳过轮次: {skipped_rounds}")
        print(f"[失败] 失败轮次: {failed_rounds}")
        print(f"实验目录: {self.experiment_dir}")
        print(f"实验日志: {self.log_file}")

        # 更新最终状态
        self.experiment_log['end_time'] = datetime.now().isoformat()
        self.experiment_log['summary'] = {
            'success_rounds': success_rounds,
            'failed_rounds': failed_rounds,
            'skipped_rounds': skipped_rounds,
            'total_rounds': len(success_rounds) + len(failed_rounds) + len(skipped_rounds)
        }
        self._save_log()

    def _should_skip_round(self, round_num: int) -> bool:
        """检查是否应该跳过某轮"""
        if 'round_specific' in self.config:
            round_config = self.config['round_specific'].get(round_num, {})
            return round_config.get('skip', False)
        return False

    def _copy_contrastive_only(self, source_dir: str, target_dir: str):
        """
        只复制对比学习结果到目标目录
        用于加噪实验，只共享编码器，其他阶段重新训练

        Args:
            source_dir: 源实验目录
            target_dir: 目标变体目录
        """
        print(f"   [复制模式] 只复制 Round 1 对比学习结果")
        print(f"   [说明] 监督学习、分类器选择、一致性评分将重新训练")

        source_round1 = os.path.join(source_dir, 'round1')
        target_round1 = os.path.join(target_dir, 'round1')
        os.makedirs(target_round1, exist_ok=True)

        # 只复制 contrastive_training
        source_contrastive = os.path.join(source_round1, 'contrastive_training')
        if not os.path.exists(source_contrastive):
            raise FileNotFoundError(f"找不到对比学习结果: {source_contrastive}")

        target_contrastive = os.path.join(target_round1, 'contrastive_training')
        if os.path.exists(target_contrastive):
            shutil.rmtree(target_contrastive)

        shutil.copytree(source_contrastive, target_contrastive)
        print(f"    ✅ 已复制: round1/contrastive_training")
        print(f"    ⏭️  跳过: supervised_training, classifier_selection, consistency_scoring")

    def _copy_round1_results(self, source_dir: str, target_dir: str,
                             include_dirs: Optional[List[str]] = None,
                             skip_dirs: Optional[List[str]] = None):
        """
        print("[警告] run_noise_variants_legacy 已弃用，将转发到 run_noise_variants。")
        return self.run_noise_variants(
            target_round=target_round,
            noise_params=noise_params,
            variant_dir_prefix=variant_dir_prefix
        )
        复制 Round 1 结果到目标实验（允许指定需要复用的子目录，并清理需要重新生成的目录）
        """
        include_dirs = include_dirs or ['contrastive_training', 'supervised_training', 'classifier_selection']
        skip_dirs = skip_dirs or []

        source_round1 = os.path.join(source_dir, 'round1')
        target_round1 = os.path.join(target_dir, 'round1')
        os.makedirs(target_round1, exist_ok=True)

        for subdir in include_dirs:
            src = os.path.join(source_round1, subdir)
            if not os.path.exists(src):
                raise FileNotFoundError(f"未找到要复用的目录: {src}")
            dst = os.path.join(target_round1, subdir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"   ✅ 已复制: round1/{subdir}")

        for subdir in skip_dirs:
            dst = os.path.join(target_round1, subdir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
                print(f"   ♻️ 已清理: round1/{subdir}")

    def run_scoring_fraction_variants(self, target_round: int = 3,
                                      scoring_fractions: List[float] = None,
                                      variant_dir_prefix: str = None):
        """
        运行多个变体实验，只改变consistency_scoring_fraction参数

        复用策略：
        1. 复制 Round 1 的对比学习、监督学习、分类器选择结果
        2. 不复制 consistency_scoring（因为要用不同数据比例重新生成）
        3. 基于不同的数据比例重新运行一致性评分
        4. 继续运行 Round 2-N

        Args:
            target_round: 目标轮次
            scoring_fractions: 不同的数据比例列表，如 [0.05, 0.1, 0.2]
            variant_dir_prefix: 变体目录前缀，如果指定则使用该前缀而非基础目录名
        """
        print(f"\n{'='*60}")
        print(f"多变体实验：不同分类器选择比例")
        print(f"{'='*60}")

        if not scoring_fractions:
            scoring_fractions = [0.05, 0.1, 0.2]  # 默认值

        print(f"将测试以下数据比例的分类器：{scoring_fractions}")
        print(f"[复用策略] 复制Round 1的对比学习、监督学习、分类器选择")
        print(f"[重新运行] 基于不同数据比例重新运行一致性评分")

        # 检查基础实验的Round 1是否存在
        source_round1 = os.path.join(self.experiment_dir, 'round1')
        if not os.path.exists(source_round1):
            raise FileNotFoundError(f"基础实验的 Round 1 不存在: {source_round1}\n请先运行: python iterative_main.py -c config.yaml -d {self.experiment_dir} --rounds 1")

        # 为每个比例创建一个变体
        for fraction in scoring_fractions:
            print(f"\n{'='*40}")
            print(f"运行变体：数据比例 = {fraction}")
            print(f"{'='*40}")

            # 创建变体目录名
            fraction_str = str(fraction).replace('.', '_')
            if variant_dir_prefix:
                variant_dir = f"{variant_dir_prefix}_frac{fraction_str}"
            else:
                variant_dir = f"{self.experiment_dir}_frac{fraction_str}"

            # 复制配置并修改consistency_scoring_fraction
            import copy
            variant_config = copy.deepcopy(self.config)

            # 修改一致性评分使用的数据比例
            if 'defaults' not in variant_config:
                variant_config['defaults'] = {}
            if 'classifier_selection' not in variant_config['defaults']:
                variant_config['defaults']['classifier_selection'] = {}

            variant_config['defaults']['classifier_selection']['consistency_scoring_fraction'] = fraction

            # 创建新的管理器实例运行这个变体
            # 保存变体配置到临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp_config:
                yaml.dump(variant_config, tmp_config, allow_unicode=True)
                tmp_config_path = tmp_config.name

            try:
                variant_manager = IterativeExperimentManager(
                    config_path=tmp_config_path,  # 使用临时配置文件
                    experiment_dir=variant_dir,
                    resume=False,
                    force_restart=False
                )
                variant_manager.experiment_name = f"{self.experiment_name}_frac{fraction_str}"

                # 复制Round 1（排除consistency_scoring）
                print(f"\n📁 复制基础实验的 Round 1 部分结果...")
                target_round1 = os.path.join(variant_dir, 'round1')
                os.makedirs(target_round1, exist_ok=True)

                # 复制对比学习
                source_contrastive = os.path.join(source_round1, 'contrastive_training')
                if os.path.exists(source_contrastive):
                    target_contrastive = os.path.join(target_round1, 'contrastive_training')
                    if os.path.exists(target_contrastive):
                        shutil.rmtree(target_contrastive)
                    shutil.copytree(source_contrastive, target_contrastive)
                    print(f"    ✅ 已复制: round1/contrastive_training")

                # 复制监督学习
                source_supervised = os.path.join(source_round1, 'supervised_training')
                if os.path.exists(source_supervised):
                    target_supervised = os.path.join(target_round1, 'supervised_training')
                    if os.path.exists(target_supervised):
                        shutil.rmtree(target_supervised)
                    shutil.copytree(source_supervised, target_supervised)
                    print(f"    ✅ 已复制: round1/supervised_training")

                # 复制分类器选择
                source_classifier = os.path.join(source_round1, 'classifier_selection')
                if os.path.exists(source_classifier):
                    target_classifier = os.path.join(target_round1, 'classifier_selection')
                    if os.path.exists(target_classifier):
                        shutil.rmtree(target_classifier)
                    shutil.copytree(source_classifier, target_classifier)
                    print(f"    ✅ 已复制: round1/classifier_selection")

                # 不复制consistency_scoring，将重新运行
                print(f"    ⏭️  跳过: round1/consistency_scoring (将重新运行)")

                # 从Round 1的一致性评分开始运行（只运行consistency_scoring）
                print(f"\n🚀 重新运行Round 1的一致性评分（使用数据比例={fraction}）...")

                # 获取Round 1的配置
                round1_config = variant_manager._get_round_config(1)

                # 运行一致性评分
                classifier_selection_dir = os.path.join(target_round1, 'classifier_selection')
                consistency_result_dir = variant_manager._run_consistency_scoring(
                    classifier_selection_dir, round1_config, target_round1
                )
                print(f"[完成] Round 1一致性评分完成: {consistency_result_dir}")

                # 如果target_round > 1，继续运行后续轮次
                if target_round > 1:
                    print(f"\n🚀 继续运行 Round 2-{target_round}...")
                    variant_manager.run_experiment(start_round=2, target_round=target_round)

                print(f"\n✅ 变体 frac{fraction_str} 完成")
            except Exception as e:
                print(f"\n❌ 变体 frac{fraction_str} 失败: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 清理临时配置文件
                if os.path.exists(tmp_config_path):
                    os.remove(tmp_config_path)

        print(f"\n{'='*60}")
        print("所有变体实验完成！")
        print(f"基础目录: {self.experiment_dir}")
        print("变体目录:")
        for fraction in scoring_fractions:
            fraction_str = str(fraction).replace('.', '_')
            print(f"  - {self.experiment_dir}_frac{fraction_str}")
        print(f"{'='*60}")

    def run_noise_variants_legacy(self, target_round: int = 3,
                          noise_params: List[str] = None,
                          variant_dir_prefix: str = None):
        """
        运行 Round 1 监督学习加噪实验（纯命令行驱动）

        Args:
            target_round: 目标轮次
            noise_params: 加噪参数组合列表，格式 ["epoch,lr[,batch_size]", ...]
                         例如：["5,1e-5", "10,1e-4,64", "20,5e-5,128"]
                         batch_size可选，不指定则使用配置文件默认值
            variant_dir_prefix: 变体目录前缀
        """
        print(f"\n{'='*60}")
        print(f"Round 1 监督学习加噪鲁棒性实验（纯命令行）")
        print(f"{'='*60}")

        if not noise_params:
            raise ValueError("必须提供 --noise-params 参数")

        # 解析并验证参数
        parsed_params = []
        print(f"\n将创建 {len(noise_params)} 个加噪变体:")
        for param_str in noise_params:
            try:
                parts = param_str.split(',')
                if len(parts) < 2 or len(parts) > 3:
                    raise ValueError(f"格式错误，应为 'epoch,lr[,batch_size]'")

                epochs = int(parts[0].strip())
                lr = float(parts[1].strip())
                batch_size = int(parts[2].strip()) if len(parts) == 3 else None

                parsed_params.append((epochs, lr, batch_size, param_str))

                if batch_size is not None:
                    print(f"  - epoch={epochs}, lr={lr}, batch_size={batch_size}")
                else:
                    print(f"  - epoch={epochs}, lr={lr}, batch_size=<使用配置默认值>")

            except Exception as e:
                raise ValueError(f"解析参数失败 '{param_str}': {e}")

        print(f"\n实验说明:")
        print(f"  1. 共享 Round 1 对比学习编码器（来自基础实验）")
        print(f"  2. Round 1 监督学习使用加噪参数（重新训练）")
        print(f"  3. Round 2-{target_round} 使用配置文件的正常参数")

        # 创建并运行各变体
        for epochs, lr, batch_size, param_str in parsed_params:
            print(f"\n{'='*40}")
            if batch_size is not None:
                print(f"运行变体: epoch={epochs}, lr={lr}, batch_size={batch_size}")
            else:
                print(f"运行变体: epoch={epochs}, lr={lr}, batch_size=<默认>")
            print(f"{'='*40}")

            # 动态生成变体名称和目录
            if batch_size is not None:
                variant_name = f"noise_epoch{epochs}_lr{lr:.0e}_bs{batch_size}"
            else:
                variant_name = f"noise_epoch{epochs}_lr{lr:.0e}"

            if variant_dir_prefix:
                variant_dir = f"{variant_dir_prefix}_{variant_name}"
            else:
                variant_dir = f"{self.experiment_dir}_{variant_name}"

            # 动态创建变体配置（基于当前配置）
            import copy
            variant_config = copy.deepcopy(self.config)

            # 添加变体元信息
            noise_params_info = {
                'round': 1,
                'stage': 'supervised_learning',
                'epochs': epochs,
                'learning_rate': lr
            }
            if batch_size is not None:
                noise_params_info['batch_size'] = batch_size

            variant_config['variant_meta'] = {
                'type': 'noise_robustness',
                'base_experiment': self.experiment_dir,
                'noise_params': noise_params_info,
                'creation_time': datetime.now().isoformat()
            }

            # 设置 round_specific 覆盖（只影响 Round 1）
            round_specific = variant_config.setdefault('round_specific', {})
            round1_overrides = copy.deepcopy(round_specific.get(1, {}))
            sup_overrides = copy.deepcopy(round1_overrides.get('supervised_learning', {}))

            # 监督学习接口要求可迭代的超参数，封装为单元素列表确保兼容
            sup_overrides['epochs'] = [epochs]
            sup_overrides['learning_rate'] = [lr]
            if batch_size is not None:
                sup_overrides['batch_size'] = [batch_size]

            round1_overrides['supervised_learning'] = sup_overrides
            round_specific[1] = round1_overrides

            # 保存变体配置到临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp_config:
                yaml.dump(variant_config, tmp_config, allow_unicode=True)
                tmp_config_path = tmp_config.name

            try:
                # 创建变体实验管理器
                variant_manager = IterativeExperimentManager(
                    config_path=tmp_config_path,
                    experiment_dir=variant_dir,
                    resume=True,
                    force_restart=False
                )
                variant_manager.experiment_name = f"{self.experiment_name}_{variant_name}"

                # 只复制对比学习结果（关键步骤）
                print(f"\n📁 复制基础实验的 Round 1 对比学习到变体目录...")
                self._copy_contrastive_only(self.experiment_dir, variant_dir)

                # 运行实验（从 Round 1 开始，会重新训练监督学习）
                print(f"\n🚀 开始运行变体实验...")
                if batch_size is not None:
                    print(f"   Round 1: 使用加噪参数（epoch={epochs}, lr={lr}, batch_size={batch_size}）")
                else:
                    print(f"   Round 1: 使用加噪参数（epoch={epochs}, lr={lr}, batch_size=<默认>）")
                print(f"   Round 2-{target_round}: 使用正常参数")

                variant_manager.run_experiment(
                    start_round=1,
                    target_round=target_round
                )
                print(f"\n✅ 变体完成: {variant_dir}")

            except Exception as e:
                print(f"\n❌ 变体失败: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 清理临时配置文件
                if os.path.exists(tmp_config_path):
                    os.remove(tmp_config_path)

        print(f"\n{'='*60}")
        print("所有加噪变体实验完成！")
        print(f"{'='*60}")
        print(f"基础实验目录: {self.experiment_dir}")
        print("变体目录:")
        for epochs, lr, batch_size, _ in parsed_params:
            if batch_size is not None:
                variant_name = f"noise_epoch{epochs}_lr{lr:.0e}_bs{batch_size}"
            else:
                variant_name = f"noise_epoch{epochs}_lr{lr:.0e}"

            if variant_dir_prefix:
                print(f"  - {variant_dir_prefix}_{variant_name}")
            else:
                print(f"  - {self.experiment_dir}_{variant_name}")
        print(f"{'='*60}")

    def run_noise_variants(self, target_round: int = 3,
                          noise_params: List[str] = None,
                          variant_dir_prefix: str = None,
                          noise_mode: str = 'by_bins',
                          noise_threshold: float = 0.4,
                          noise_pool: str = '0.0-0.01',
                          noise_target_bins: Optional[str] = None,
                          noise_seed: Optional[int] = None):
        """
        运行 Round 1 一致性评分噪声实验（复用对比学习与监督学习结果，仅扰动一致性数据集）
        """
        print(f"\n{'='*60}")
        print("Round 1 一致性评分噪声实验（仅命令行配置）")
        print(f"{'='*60}")

        if not noise_params:
            raise ValueError("必须提供 --noise-params 参数，例如：--noise-params 0.05 0.1")

        def _parse_range(value: str, default: Tuple[float, float]) -> Tuple[float, float]:
            cleaned = value.replace('[', '').replace(']', '').strip()
            delimiter = '-' if '-' in cleaned else ','
            parts = [p.strip() for p in cleaned.split(delimiter) if p.strip()]
            if len(parts) != 2:
                return default
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return default

        def _parse_bins(value: Optional[str], default: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            if not value:
                return default
            bins = []
            for chunk in value.split(','):
                chunk = chunk.strip()
                if not chunk:
                    continue
                rng = _parse_range(chunk, None)
                if rng:
                    bins.append(rng)
            return bins or default

        base_noise_pool = _parse_range(noise_pool, (0.0, 0.01))
        default_bins = [(noise_threshold, 0.6), (0.6, 0.8), (0.8, 1.0)]
        base_target_bins = _parse_bins(noise_target_bins, default_bins)

        def _format_fraction(frac: float) -> str:
            return f"{frac:.3f}".rstrip('0').rstrip('.')

        def _sanitize_token(token: str) -> str:
            return token.replace('.', 'p').replace('-', 'to').replace(',', '_').replace(' ', '')

        def _format_range_for_suffix(rng: Tuple[float, float]) -> str:
            return _sanitize_token(f"{rng[0]:.2f}-{rng[1]:.2f}")

        def _parse_noise_param(param_str: str, base_settings: dict) -> dict:
            import copy
            settings = copy.deepcopy(base_settings)
            settings['fraction'] = None
            settings['label'] = None

            segments = [seg.strip() for seg in param_str.replace('|', ';').split(';') if seg.strip()]
            for seg in segments:
                if '=' in seg:
                    key, value = seg.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in ('fraction', 'frac', 'ratio', 'percent', 'percentage', 'p'):
                        settings['fraction'] = float(value)
                    elif key == 'mode':
                        settings['mode'] = value
                    elif key in ('threshold', 'noise_threshold'):
                        settings['threshold'] = float(value)
                    elif key in ('noise_pool', 'pool'):
                        settings['noise_pool'] = _parse_range(value, settings['noise_pool'])
                    elif key in ('target_bins', 'bins'):
                        settings['target_bins'] = _parse_bins(value, settings['target_bins'])
                    elif key == 'seed':
                        settings['seed'] = int(value)
                    elif key in ('apply', 'apply_to_dataset', 'write_back'):
                        settings['apply_to_dataset'] = value.lower() in ('true', '1', 'yes', 'y')
                    elif key in ('label', 'name', 'suffix'):
                        settings['label'] = value
                    else:
                        print(f" [警告] 未识别的噪声参数键: {key}，已忽略")
                else:
                    settings['fraction'] = float(seg)

            if settings.get('fraction') is None:
                raise ValueError(f"噪声参数 '{param_str}' 缺少 fraction 配置")

            if settings['mode'] != 'by_bins':
                settings['target_bins'] = None

            return settings

        base_settings = {
            'mode': noise_mode,
            'threshold': noise_threshold,
            'noise_pool': base_noise_pool,
            'target_bins': base_target_bins,
            'seed': noise_seed,
            'apply_to_dataset': False
        }

        parsed_variants = []
        print(f"\n将创建 {len(noise_params)} 个噪声变体：")
        for param_str in noise_params:
            settings = _parse_noise_param(param_str, base_settings)
            parsed_variants.append(settings)
            bins_desc = settings['target_bins'] if settings['target_bins'] else [(settings['threshold'], 1.0)]
            bins_text = ', '.join([f"{b[0]}-{b[1]}" for b in bins_desc])
            print(f"  - fraction={settings['fraction']}, mode={settings['mode']}, bins={bins_text}, pool={settings['noise_pool']}, apply_to_dataset={settings.get('apply_to_dataset', False)}")

        print("\n实验说明：")
        print("  1. 复用 Round 1 对比学习、监督学习和分类器选择结果")
        print("  2. 只在一致性评分阶段注入噪声，生成新的增强数据集")
        print(f"  3. Round 2-{target_round} 按原配置继续迭代")

        import copy
        import tempfile

        variant_records = []

        for settings in parsed_variants:
            fraction_tag = _format_fraction(settings['fraction']).replace('.', '_')
            if settings.get('label'):
                label_suffix = settings['label']
            else:
                auto_parts = [settings['mode'], f"frac{fraction_tag}"]
                if settings.get('target_bins'):
                    bins_token = 'bins' + '_'.join(_format_range_for_suffix(b) for b in settings['target_bins'])
                    auto_parts.append(bins_token)
                else:
                    auto_parts.append(_sanitize_token(f"thr{settings['threshold']:.2f}"))
                pool_token = 'pool' + _format_range_for_suffix(settings['noise_pool'])
                auto_parts.append(pool_token)
                if settings.get('seed') is not None:
                    auto_parts.append(f"seed{settings['seed']}")
                if settings.get('apply_to_dataset', False):
                    auto_parts.append('apply')
                label_suffix = '_'.join(auto_parts)
            variant_name = f"noise_{label_suffix}"

            if variant_dir_prefix:
                variant_dir = f"{variant_dir_prefix}_{label_suffix}"
            else:
                variant_dir = f"{self.experiment_dir}_{label_suffix}"

            print(f"\n{'='*40}")
            print(f"运行变体: {variant_name}")
            print(f"  fraction={settings['fraction']}, mode={settings['mode']}, threshold={settings['threshold']}, pool={settings['noise_pool']}")
            if settings.get('target_bins'):
                print(f"  target_bins={settings['target_bins']}")
            if settings.get('seed') is not None:
                print(f"  seed={settings['seed']}")
            print(f"  apply_to_dataset={settings.get('apply_to_dataset', False)}")
            print(f"{'='*40}")

            variant_config = copy.deepcopy(self.config)
            noise_config = {
                'enabled': True,
                'mode': settings['mode'],
                'threshold': settings['threshold'],
                'fraction': settings['fraction'],
                'noise_pool': list(settings['noise_pool']),
                'target_bins': [list(b) for b in settings.get('target_bins') or []],
                'seed': settings.get('seed'),
                'apply_to_dataset': settings.get('apply_to_dataset', False)
            }

            variant_config['variant_meta'] = {
                'type': 'noise_robustness',
                'base_experiment': self.experiment_dir,
                'noise_params': noise_config,
                'creation_time': datetime.now().isoformat()
            }

            round_specific = variant_config.setdefault('round_specific', {})
            round1_overrides = copy.deepcopy(round_specific.get(1, {}))
            consistency_overrides = copy.deepcopy(round1_overrides.get('consistency_scoring', {}))
            consistency_overrides['noise_injection'] = noise_config
            round1_overrides['consistency_scoring'] = consistency_overrides
            round_specific[1] = round1_overrides

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp_config:
                yaml.dump(variant_config, tmp_config, allow_unicode=True)
                tmp_config_path = tmp_config.name

            try:
                variant_manager = IterativeExperimentManager(
                    config_path=tmp_config_path,
                    experiment_dir=variant_dir,
                    resume=True,
                    force_restart=False
                )
                variant_manager.experiment_name = f"{self.experiment_name}_{label_suffix}"

                self._copy_round1_results(
                    self.experiment_dir,
                    variant_dir,
                    include_dirs=['contrastive_training', 'supervised_training', 'classifier_selection'],
                    skip_dirs=['consistency_scoring', 'consistency_scores']
                )

                detector = ExperimentStatusDetector(variant_dir, 1)
                stage1_done, stage1_path = detector.detect_stage1_contrastive_status()
                if stage1_done and stage1_path:
                    variant_manager.state_manager.save_stage_completion(
                        1, 'stage1_contrastive', stage1_path,
                        metadata={'reuse_from': self.experiment_dir}
                    )

                sup_done, sup_dir = detector.detect_supervised_learning_status()
                if sup_done and sup_dir:
                    variant_manager.state_manager.save_stage_completion(
                        1, 'supervised_learning', sup_dir,
                        metadata={'reuse_from': self.experiment_dir}
                    )

                cls_done, _ = detector.detect_classifier_selection_status()
                if cls_done:
                    cls_dir = os.path.join(variant_dir, 'round1', 'classifier_selection')
                    variant_manager.state_manager.save_stage_completion(
                        1, 'classifier_selection', cls_dir,
                        metadata={'reuse_from': self.experiment_dir}
                    )

                variant_manager.run_experiment(start_round=1, target_round=target_round)
                variant_records.append({'name': label_suffix, 'dir': variant_dir, 'status': 'completed'})
                print(f"✅ 变体完成: {variant_dir}")

            except Exception as exc:
                variant_records.append({'name': label_suffix, 'dir': variant_dir, 'status': 'failed', 'error': str(exc)})
                print(f"❌ 变体失败: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(tmp_config_path):
                    os.remove(tmp_config_path)

        print(f"\n{'='*60}")
        print("噪声变体实验完成")
        for record in variant_records:
            flag = '✅' if record['status'] == 'completed' else '❌'
            print(f" {flag} {record['name']}: {record['dir']}")
        print(f"{'='*60}")

    def run_multi_variant_experiment(self, target_round: int = 3,
                                    stage2_only: bool = False,
                                    custom_variants: List[dict] = None):
        """
        运行多变体实验

        Args:
            target_round: 目标轮次
            stage2_only: 是否只运行Stage2变体
            custom_variants: 自定义变体配置列表
        """
        print(f"\n{'='*60}")
        print(f"多变体实验模式")
        print(f"{'='*60}")

        # 检查第一阶段是否完成
        if stage2_only:
            detector = ExperimentStatusDetector(self.experiment_dir, 1)
            stage1_done, stage1_model = detector.detect_stage1_contrastive_status()

            if not stage1_done:
                print("❌ 第一阶段未完成，无法创建Stage2变体实验")
                response = input("是否先运行第一阶段？(y/n): ")
                if response.lower() == 'y':
                    # 只运行第一轮
                    self.run_experiment(rounds=[1])
                    # 重新检查
                    stage1_done, stage1_model = detector.detect_stage1_contrastive_status()
                    if not stage1_done:
                        print("第一阶段运行失败，无法继续")
                        return
                else:
                    return

        # 创建变体管理器
        variant_manager = ExperimentVariantManager(self.config, self.experiment_dir)

        # 从配置文件读取变体定义，如果没有则使用custom_variants或默认值
        if 'variants' in self.config:
            # 从配置文件读取变体
            config_variants = []
            for variant_config in self.config['variants']:
                stage2_params = variant_config.get('stage2_contrastive', {})
                config_variants.append({
                    'weighting_strategy': stage2_params.get('weighting_strategy', 'threshold'),
                    'weight_threshold': stage2_params.get('weight_threshold', 0.5),
                    'base_lr': stage2_params.get('base_lr', 1e-4),
                    'epochs': stage2_params.get('epochs', 30),
                    'variant_name': variant_config.get('name', 'unnamed'),
                    'description': variant_config.get('description', '')
                })
            variants = variant_manager.create_stage2_variants(config_variants)
        else:
            # 使用custom_variants或默认值
            variants = variant_manager.create_stage2_variants(custom_variants)

        print(f"\n📋 将创建 {len(variants)} 个实验变体:")
        for i, (variant_config, variant_dir) in enumerate(variants):
            variant_name = variant_config.get('variant_meta', {}).get('name', 'unknown')
            print(f"  {i+1}. {variant_name} -> {variant_dir}")

        response = input("\n确认创建这些变体？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return

        # 复制第一阶段结果到各变体目录
        variant_dirs = [variant_dir for _, variant_dir in variants]
        variant_manager.copy_stage1_results(self.experiment_dir, variant_dirs)

        print(f"\n开始运行各变体实验...")

        # 运行各变体
        variant_results = []
        for variant_config, variant_dir in variants:
            variant_name = variant_config.get('variant_meta', {}).get('name', 'unknown')
            print(f"\n{'='*40}")
            print(f"🚀 开始变体: {variant_name}")
            print(f"{'='*40}")

            try:
                # 创建变体实验管理器
                variant_experiment = IterativeExperimentManager(
                    config_path=None,  # 直接使用配置对象
                    experiment_dir=variant_dir,
                    resume=True  # 变体总是启用断点续传
                )
                # 直接设置配置
                variant_experiment.config = variant_config

                # 从第2轮开始运行（第1轮已经复制）
                if stage2_only:
                    variant_experiment.run_experiment(
                        start_round=2,
                        target_round=target_round
                    )
                else:
                    variant_experiment.run_experiment(
                        target_round=target_round
                    )

                variant_results.append({
                    'name': variant_name,
                    'dir': variant_dir,
                    'status': 'completed'
                })

            except Exception as e:
                print(f"变体 {variant_name} 运行失败: {e}")
                variant_results.append({
                    'name': variant_name,
                    'dir': variant_dir,
                    'status': 'failed',
                    'error': str(e)
                })

        # 汇总变体结果
        print(f"\n{'='*60}")
        print(f"多变体实验总结")
        print(f"{'='*60}")
        for result in variant_results:
            status_symbol = '✅' if result['status'] == 'completed' else '❌'
            print(f"{status_symbol} {result['name']}: {result['status']}")
            if result.get('error'):
                print(f"    错误: {result['error']}")

        print(f"\n基础实验目录: {self.experiment_dir}")
        print(f"变体数量: {len(variant_results)}")
        print(f"成功: {sum(1 for r in variant_results if r['status'] == 'completed')}")
        print(f"失败: {sum(1 for r in variant_results if r['status'] == 'failed')}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='迭代实验管理系统（支持断点续传和多变体实验）')

    # 基本参数
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='配置文件路径 (YAML或JSON)')

    parser.add_argument('--experiment-dir', '-d', type=str, default=None,
                       help='实验目录（可选，默认自动创建）')

    # 轮次控制
    parser.add_argument('--rounds', '-r', type=int, default=None,
                       help='目标轮次，运行到第N轮结束（如: 3 表示运行1-3轮）')

    parser.add_argument('--start-round', '-s', type=int, default=None,
                       help='从某轮开始运行')

    parser.add_argument('--specific-rounds', type=str, default=None,
                       help='指定具体运行轮次，如: "1,3,5" 或 "2-5"')

    # 断点续传相关
    parser.add_argument('--resume', action='store_true',
                       help='启用断点续传，自动跳过已完成的阶段')

    parser.add_argument('--force-restart', action='store_true',
                       help='强制重新开始，忽略已有进度')

    # 多变体实验相关
    parser.add_argument('--multi-variant', action='store_true',
                       help='运行多变体实验（基于第一阶段结果）')

    parser.add_argument('--stage2-only', action='store_true',
                       help='多变体实验只从Stage2开始（需要第一阶段已完成）')

    parser.add_argument('--scoring-fractions', type=float, nargs='+',
                       default=None,
                       help='一致性评分使用的数据比例列表（如：0.05 0.1 0.2），用于创建多个变体。'
                            '自动复用基础实验的Round 1（对比学习、监督学习、分类器选择），'
                            '只重新运行一致性评分。需要先运行基础实验的Round 1。')

    parser.add_argument('--variant-dir-prefix', type=str, default=None,
                       help='变体实验目录前缀（如：my_variants），默认使用基础实验目录名')

    parser.add_argument('--noise-round1-supervised', action='store_true',
                       help='为 Round 1 一致性评分注入噪声（复用对比学习与监督学习结果）')

    parser.add_argument('--noise-params', type=str, nargs='+',
                       default=None,
                       help='Round 1 噪声参数组合，示例："0.05" 或 '
                            '"fraction=0.1;mode=all_above_threshold;label=wide"。'
                            'batch_size可选，不指定则使用配置文件默认值。'
                            '支持键：fraction/frac、mode、threshold、noise_pool、target_bins、seed、label。'
                            '每个参数组合创建一个变体，完全通过命令行控制，无需修改配置文件。')
    parser.add_argument('--noise-mode', type=str, choices=['by_bins', 'all_above_threshold'],
                       default='by_bins',
                       help='噪声注入策略：by_bins（按区间替换）或 all_above_threshold（整体替换超过阈值的样本）。')
    parser.add_argument('--noise-threshold', type=float, default=0.4,
                       help='高置信样本阈值（默认 0.4，仅当模式为 all_above_threshold 时使用）。')
    parser.add_argument('--noise-pool', type=str, default='0.0-0.01',
                       help='噪声样本池区间（默认 0.0-0.01，对应低一致性样本）。')
    parser.add_argument('--noise-target-bins', type=str, default=None,
                       help='模式为 by_bins 时使用，指定高置信区间列表，例如："0.4-0.6,0.6-0.8,0.8-1.0"。')
    parser.add_argument('--noise-seed', type=int, default=None,
                       help='噪声注入过程的随机种子，默认随机。')

    # ✨ 新增：模型基座选择
    parser.add_argument('--use-fresh-bert', action='store_true',
                       help='每轮从原始BERT重新训练（固定原始BERT基座模式）。'
                            '默认为累积训练模式（每轮基于上一轮模型继续训练）。'
                            '此参数会覆盖YAML配置中的 use_fresh_bert_each_round 设置。')

    # 其他选项
    parser.add_argument('--check-status', action='store_true',
                       help='仅检查实验状态，不运行')

    return parser.parse_args()


def parse_rounds_string(rounds_str: str) -> List[int]:
    """解析轮次字符串"""
    if ',' in rounds_str:
        return [int(r.strip()) for r in rounds_str.split(',')]
    elif '-' in rounds_str:
        start, end = map(int, rounds_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(rounds_str)]


def main():
    """主函数"""
    print("迭代实验管理系统启动")
    print("="*60)

    args = parse_arguments()

    try:
        # 如果只是检查状态
        if args.check_status:
            if args.experiment_dir and os.path.exists(args.experiment_dir):
                print_experiment_status(args.experiment_dir, args.rounds or 3)
            else:
                print("请指定有效的实验目录")
            return 0

        # 创建实验管理器
        manager = IterativeExperimentManager(
            config_path=args.config,
            experiment_dir=args.experiment_dir,
            resume=args.resume,
            force_restart=args.force_restart,
            use_fresh_bert=args.use_fresh_bert if args.use_fresh_bert else None  # ✨ 传递模型基座参数
        )

        # 运行多变体实验
        if args.multi_variant:
            # 检查是加噪实验还是数据比例实验
            if args.noise_round1_supervised:
                # Round 1 加噪实验
                if not args.noise_params:
                    print("错误：使用 --noise-round1-supervised 需要 --noise-params")
                    print('示例：--noise-params "5,1e-5" "10,1e-4,64"')
                    print('      --noise-params "5,1e-5" "10,1e-4" "20,5e-5,128"')
                    return 1

                # 确定目标轮次：命令行参数 > 配置文件 > 默认值3
                if args.rounds:
                    target_round = args.rounds
                else:
                    config_rounds = manager.config.get('experiment_meta', {}).get('rounds', [1, 2, 3])
                    target_round = max(config_rounds) if config_rounds else 3

                print(f"目标轮次: {target_round}")
                manager.run_noise_variants(
                    target_round=target_round,
                    noise_params=args.noise_params,
                    variant_dir_prefix=args.variant_dir_prefix,
                    noise_mode=args.noise_mode,
                    noise_threshold=args.noise_threshold,
                    noise_pool=args.noise_pool,
                    noise_target_bins=args.noise_target_bins,
                    noise_seed=args.noise_seed
                )
            elif args.scoring_fractions:
                # 数据比例实验
                # 确定目标轮次：命令行参数 > 配置文件 > 默认值3
                if args.rounds:
                    target_round = args.rounds
                else:
                    # 从配置文件读取rounds设置
                    config_rounds = manager.config.get('experiment_meta', {}).get('rounds', [1, 2, 3])
                    target_round = max(config_rounds) if config_rounds else 3

                print(f"目标轮次: {target_round}")
                manager.run_scoring_fraction_variants(
                    target_round=target_round,
                    scoring_fractions=args.scoring_fractions,
                    variant_dir_prefix=args.variant_dir_prefix
                )
            else:
                print("错误：使用 --multi-variant 需要指定以下之一：")
                print("  1. 数据比例实验：--scoring-fractions 0.05 0.1 0.2")
                print('  2. Round 1 加噪实验：--noise-round1-supervised --noise-params "5,1e-5" "10,1e-4"')
                return 1
        else:
            # 解析轮次参数
            specific_rounds = None
            if args.specific_rounds:
                specific_rounds = parse_rounds_string(args.specific_rounds)

            # 运行标准实验
            manager.run_experiment(
                rounds=specific_rounds,
                start_round=args.start_round,
                target_round=args.rounds
            )

    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[中断] 用户取消实验")
        return 2
    except Exception as e:
        print(f"[错误] 实验失败: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
