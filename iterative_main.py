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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path

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
    """迭代实验管理器"""

    def __init__(self, config_path: str, experiment_dir: Optional[str] = None):
        """
        初始化实验管理器

        Args:
            config_path: 配置文件路径
            experiment_dir: 实验目录，如果不指定则自动创建
        """
        self.config = self._load_config(config_path)
        self.experiment_name = self.config['experiment_meta']['name']

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

        print(f"迭代实验管理器初始化")
        print(f"实验目录: {self.experiment_dir}")
        print(f"配置文件: {config_path}")
        print(f"实验名称: {self.experiment_name}")

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
                raise RuntimeError(f"第 {round_num} 轮对比学习失败")

            print(f"[完成] 编码器训练完成: {encoder_path}")

            # Step 2: 监督学习
            sup_result_dir = self._run_supervised_training(encoder_path, round_config, round_dir)
            print(f"[完成] 监督学习完成: {sup_result_dir}")

            # Step 3: 分类器选择
            classifier_selection_dir = self._run_classifier_selection(sup_result_dir, round_config, round_dir)
            print(f"[完成] 分类器选择完成: {classifier_selection_dir}")

            # Step 4: 一致性评分（所有轮次都执行，便于对比分析）
            consistency_result_dir = self._run_consistency_scoring(
                classifier_selection_dir, round_config, round_dir
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

        model_path = run_stage1_contrastive_training(stage1_config, output_dir)

        # 同时保存生成的数据集路径信息
        self.experiment_log['rounds'][f'round1']['dataset_path'] = os.path.join(output_dir, 'dataset.pkl')

        return model_path

    def _run_stage2_weighted_contrastive(self, round_num: int, config: Dict, round_dir: str) -> Optional[str]:
        """运行Stage 2+加权对比学习"""
        print(f"[Stage 2+] 运行Stage 2+加权对比学习（基于第{round_num-1}轮）...")

        # 查找上一轮的编码器和增强数据集
        prev_encoder = self._find_previous_encoder(round_num)
        prev_enhanced_dataset = self._find_previous_enhanced_dataset(round_num)

        if not prev_encoder:
            raise FileNotFoundError(f"找不到第{round_num-1}轮的编码器")
        if not prev_enhanced_dataset:
            raise FileNotFoundError(f"找不到第{round_num-1}轮的增强数据集")

        print(f"  使用编码器: {prev_encoder}")
        print(f"  使用增强数据集: {prev_enhanced_dataset}")

        if not run_stage2_weighted_contrastive:
            raise RuntimeError("Stage 2加权对比学习接口未实现")

        stage2_config = config.get('stage2_contrastive', {})
        # 直接保存到round目录下的best_model.pth
        return run_stage2_weighted_contrastive(
            stage2_config, prev_encoder, prev_enhanced_dataset, round_dir, round_num
        )

    def _run_supervised_training(self, encoder_path: str, config: Dict, round_dir: str) -> str:
        """运行监督学习"""
        print("[监督学习] 运行监督学习超参数搜索...")

        output_dir = os.path.join(round_dir, "supervised_training")
        os.makedirs(output_dir, exist_ok=True)

        if not run_supervised_training_interface:
            raise RuntimeError("监督学习接口未实现")

        sup_config = config.get('supervised_learning', {})

        # 从round_dir提取轮次号
        round_num = int(os.path.basename(round_dir).replace('round', ''))

        return run_supervised_training_interface(encoder_path, sup_config, output_dir, round_num)

    def _run_classifier_selection(self, sup_result_dir: str, config: Dict, round_dir: str) -> str:
        """运行分类器选择"""
        print("[分类器选择] 运行分类器选择...")

        output_dir = os.path.join(round_dir, "classifier_selection")
        os.makedirs(output_dir, exist_ok=True)

        if not run_classifier_selection_interface:
            raise RuntimeError("分类器选择接口未实现")

        selection_config = config.get('classifier_selection', {})
        return run_classifier_selection_interface(sup_result_dir, selection_config, output_dir)

    def _run_consistency_scoring(self, classifier_selection_dir: str, config: Dict, round_dir: str) -> str:
        """运行一致性评分"""
        print("[一致性评分] 运行一致性评分...")

        output_dir = os.path.join(round_dir, "consistency_scoring")
        os.makedirs(output_dir, exist_ok=True)

        # 查找选中的分类器（使用配置中指定的比例）
        scoring_config = config.get('consistency_scoring', {})
        selection_config = config.get('classifier_selection', {})

        # 使用consistency_scoring_fraction配置
        target_fraction = selection_config.get('consistency_scoring_fraction', 0.1)
        print(f"   使用数据比例 {target_fraction} 的分类器进行一致性评分")

        classifier_path = self._find_classifier_for_fraction(
            os.path.dirname(classifier_selection_dir), target_fraction
        )

        if not classifier_path:
            raise FileNotFoundError(f"找不到数据比例{target_fraction}的分类器")

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
        scoring_config['dataset_path'] = dataset_path

        if not run_consistency_scoring_interface:
            raise RuntimeError("一致性评分接口未实现")

        return run_consistency_scoring_interface(
            classifier_path, dataset_path, scoring_config, output_dir
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

    def run_experiment(self, rounds: Optional[List[int]] = None, start_round: Optional[int] = None):
        """
        运行完整的迭代实验

        Args:
            rounds: 指定要运行的轮次列表
            start_round: 从某轮开始
        """
        if rounds:
            selected_rounds = rounds
        else:
            selected_rounds = self._get_selected_rounds()

        if start_round:
            selected_rounds = [r for r in selected_rounds if r >= start_round]

        print(f"计划运行轮次: {selected_rounds}")

        success_rounds = []
        failed_rounds = []

        for round_num in selected_rounds:
            # 检查是否应该跳过
            if self._should_skip_round(round_num):
                print(f"[跳过] 跳过第 {round_num} 轮（配置中标记为skip）")
                continue

            # 运行单轮
            if self.run_single_round(round_num):
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
        print(f"[失败] 失败轮次: {failed_rounds}")
        print(f"实验目录: {self.experiment_dir}")
        print(f"实验日志: {self.log_file}")

        # 更新最终状态
        self.experiment_log['end_time'] = datetime.now().isoformat()
        self.experiment_log['summary'] = {
            'success_rounds': success_rounds,
            'failed_rounds': failed_rounds,
            'total_rounds': len(success_rounds) + len(failed_rounds)
        }
        self._save_log()

    def _should_skip_round(self, round_num: int) -> bool:
        """检查是否应该跳过某轮"""
        if 'round_specific' in self.config:
            round_config = self.config['round_specific'].get(round_num, {})
            return round_config.get('skip', False)
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='迭代实验管理系统')

    parser.add_argument('--config', '-c', type=str, required=True,
                       help='配置文件路径 (YAML或JSON)')

    parser.add_argument('--experiment-dir', '-d', type=str, default=None,
                       help='实验目录（可选，默认自动创建）')

    parser.add_argument('--rounds', '-r', type=str, default=None,
                       help='指定运行轮次，如: "1,3,5" 或 "2-5"')

    parser.add_argument('--start-round', '-s', type=int, default=None,
                       help='从某轮开始运行')

    parser.add_argument('--resume', action='store_true',
                       help='从上次中断处继续')

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

    # 解析轮次参数
    rounds = None
    if args.rounds:
        rounds = parse_rounds_string(args.rounds)

    try:
        # 创建实验管理器
        manager = IterativeExperimentManager(args.config, args.experiment_dir)

        # 运行实验
        manager.run_experiment(rounds=rounds, start_round=args.start_round)

    except Exception as e:
        print(f"[错误] 实验失败: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())