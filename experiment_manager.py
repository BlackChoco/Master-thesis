"""
实验管理模块
提供断点续传、状态检测和多变体实验功能
"""

import os
import json
import shutil
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pickle
import yaml
import glob
import copy


class ExperimentStatusDetector:
    """检测实验各阶段的完成状态"""

    def __init__(self, output_dir: str, round_num: int):
        self.output_dir = output_dir
        self.round_num = round_num
        self.round_dir = os.path.join(output_dir, f"round{round_num}")

    def detect_stage1_contrastive_status(self) -> Tuple[bool, Optional[str]]:
        """检测Stage1对比学习是否完成

        Returns:
            (是否完成, 模型路径)
        """
        # 首先检查round1目录下的contrastive_training
        round1_contrastive_dir = os.path.join(self.output_dir, "round1", "contrastive_training")
        if os.path.exists(round1_contrastive_dir):
            for file in os.listdir(round1_contrastive_dir):
                if file.endswith('.pth') and 'best' in file.lower():
                    model_path = os.path.join(round1_contrastive_dir, file)
                    print(f"  [OK] 找到Stage1模型: {model_path}")
                    return True, model_path

        # 检查 iter_model/ 下是否有第一阶段的模型
        iter_model_dir = os.path.join(self.output_dir, "iter_model")

        if not os.path.exists(iter_model_dir):
            return False, None

        # 查找包含 best_model.pth 的目录
        for root, dirs, files in os.walk(iter_model_dir):
            if "best_model.pth" in files:
                model_path = os.path.join(root, "best_model.pth")

                # 验证是否是第一阶段的模型
                try:
                    checkpoint = self._load_checkpoint(model_path)
                    # 检查是否有Stage2的标记
                    if checkpoint.get('training_stage') == 'weighted_contrastive':
                        continue  # 这是Stage2的模型，跳过

                    # 检查是否有第一阶段的标记
                    if checkpoint.get('training_history'):
                        if not checkpoint['training_history'].get('second_stage_training', False):
                            print(f"  [OK] 找到Stage1模型: {model_path}")
                            return True, model_path
                    else:
                        # 没有training_history，可能是旧版本的第一阶段模型
                        print(f"  [OK] 找到Stage1模型 (legacy): {model_path}")
                        return True, model_path

                except Exception as e:
                    print(f"  [警告] 无法加载模型 {model_path}: {e}")

        return False, None

    def detect_supervised_learning_status(self) -> Tuple[bool, Optional[str]]:
        """检测监督学习是否完成

        Returns:
            (是否完成, 结果目录路径)
        """
        supervised_dir = os.path.join(self.round_dir, "supervised_training")

        if not os.path.exists(supervised_dir):
            return False, None

        # 检查关键结果文件
        required_files = [
            os.path.join(supervised_dir, "best_results_comparison", "model_comparison_by_data_fraction.json"),
            os.path.join(supervised_dir, "all_seeds_results.json")
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"  [缺少] 缺少监督学习结果文件: {os.path.basename(file_path)}")
                return False, None

        # 验证saved_models目录中是否有模型
        saved_models_dir = os.path.join(supervised_dir, "saved_models")
        if os.path.exists(saved_models_dir):
            model_count = len([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
            if model_count > 0:
                print(f"  [OK] 找到监督学习结果: {model_count} 个模型")
                return True, supervised_dir

        return False, None

    def detect_classifier_selection_status(self) -> Tuple[bool, Dict[str, str]]:
        """检测分类器选择是否完成

        Returns:
            (是否完成, {数据比例: 分类器路径})
        """
        # 先尝试新路径格式
        selection_dir = os.path.join(self.round_dir, "classifier_selection")

        if not os.path.exists(selection_dir):
            # 尝试旧路径格式
            selection_dir = os.path.join(self.round_dir, "classifier_selections")

        if not os.path.exists(selection_dir):
            return False, {}

        # 查找所有 *_selected.json 文件
        selected_files = glob.glob(os.path.join(selection_dir, "*", "*_selected.json"))

        if not selected_files:
            # 尝试在selected_models子目录下查找
            selected_files = glob.glob(os.path.join(selection_dir, "selected_models", "*_selected.json"))

        if not selected_files:
            return False, {}

        selected_classifiers = {}
        for json_path in selected_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                data_fraction = str(data.get('data_fraction', ''))
                model_path = data.get('model_path', '')

                if data_fraction and model_path:
                    # 验证模型文件是否存在
                    if self._find_model_file(model_path):
                        selected_classifiers[data_fraction] = json_path

            except Exception as e:
                print(f"  [警告] 无法读取分类器选择文件 {json_path}: {e}")

        if selected_classifiers:
            print(f"  [OK] 找到 {len(selected_classifiers)} 个选定的分类器")
            return True, selected_classifiers

        return False, {}

    def detect_consistency_scoring_status(self) -> Tuple[bool, Optional[str]]:
        """检测一致性评分是否完成

        Returns:
            (是否完成, 增强数据集路径)
        """
        # 先尝试新路径格式
        scoring_dir = os.path.join(self.round_dir, "consistency_scoring")

        if not os.path.exists(scoring_dir):
            # 尝试旧路径格式
            scoring_dir = os.path.join(self.round_dir, "consistency_scores")

        if not os.path.exists(scoring_dir):
            return False, None

        # 查找增强数据集文件
        enhanced_dataset_patterns = [
            "enhanced_dataset_*.pkl",
            "*_round*_history.pkl"
        ]

        for pattern in enhanced_dataset_patterns:
            files = glob.glob(os.path.join(scoring_dir, "**", pattern), recursive=True)
            if files:
                # 使用最新的文件
                latest_file = max(files, key=os.path.getmtime)

                # 验证文件完整性
                try:
                    with open(latest_file, 'rb') as f:
                        data = pickle.load(f)

                    if isinstance(data, dict) and 'samples' in data:
                        sample_count = len(data['samples'])
                        print(f"  [OK] 找到增强数据集: {os.path.basename(latest_file)} ({sample_count} 样本)")
                        return True, latest_file

                except Exception as e:
                    print(f"  [警告] 无法加载增强数据集 {latest_file}: {e}")

        return False, None

    def detect_stage2_contrastive_status(self) -> Tuple[bool, Optional[str]]:
        """检测Stage2对比学习是否完成

        Returns:
            (是否完成, 模型路径)
        """
        # 首先检查round目录下的best_model.pth
        round_model_path = os.path.join(self.round_dir, "best_model.pth")
        if os.path.exists(round_model_path):
            print(f"  [OK] 找到Stage2模型: {round_model_path}")
            return True, round_model_path

        # Stage2模型可能保存在 round{n}/contrastive_model/ 或 iter_model/
        possible_dirs = [
            os.path.join(self.round_dir, "contrastive_model"),
            os.path.join(self.output_dir, "iter_model")
        ]

        for search_dir in possible_dirs:
            if not os.path.exists(search_dir):
                continue

            for root, dirs, files in os.walk(search_dir):
                if "best_model.pth" in files:
                    model_path = os.path.join(root, "best_model.pth")

                    try:
                        checkpoint = self._load_checkpoint(model_path)

                        # 检查是否是Stage2的模型
                        if checkpoint.get('training_stage') == 'weighted_contrastive':
                            print(f"  [OK] 找到Stage2模型: {model_path}")
                            return True, model_path

                        # 或者检查training_history中的标记
                        if checkpoint.get('training_history', {}).get('second_stage_training', False):
                            print(f"  [OK] 找到Stage2模型: {model_path}")
                            return True, model_path

                    except Exception as e:
                        print(f"  [警告] 无法加载模型 {model_path}: {e}")

        return False, None

    def _load_checkpoint(self, path: str) -> dict:
        """加载模型检查点"""
        import torch
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except:
            return {}

    def _find_model_file(self, path: str) -> bool:
        """智能查找模型文件（处理相对路径问题）"""
        if os.path.exists(path):
            return True

        # 尝试在output_dir下查找
        relative_path = os.path.basename(path)
        for root, dirs, files in os.walk(self.output_dir):
            if relative_path in files:
                return True

        return False

    def get_round_status_summary(self) -> Dict[str, bool]:
        """获取当前轮次的完整状态摘要"""
        summary = {}

        if self.round_num == 1:
            # 第一轮：检查Stage 1对比学习
            stage1_done, _ = self.detect_stage1_contrastive_status()
            summary['stage1_contrastive'] = stage1_done
        else:
            # 第二轮及以后：检查Stage 2对比学习
            stage2_done, _ = self.detect_stage2_contrastive_status()
            summary['stage2_contrastive'] = stage2_done

        # 所有轮次都需要的阶段
        sup_done, _ = self.detect_supervised_learning_status()
        summary['supervised_learning'] = sup_done

        cls_done, _ = self.detect_classifier_selection_status()
        summary['classifier_selection'] = cls_done

        cons_done, _ = self.detect_consistency_scoring_status()
        summary['consistency_scoring'] = cons_done

        return summary


class ExperimentStateManager:
    """管理实验状态的持久化和恢复"""

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, 'experiment_state.json')
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict:
        """加载或创建实验状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[警告] 无法加载状态文件: {e}")

        # 创建新状态
        return {
            'experiment_meta': {
                'config_hash': self._compute_config_hash(),
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'config_name': self.config.get('experiment_meta', {}).get('name', 'unknown')
            },
            'rounds': {}
        }

    def _compute_config_hash(self) -> str:
        """计算配置文件的哈希值"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save_stage_completion(self, round_num: int, stage: str,
                             result_path: str, metadata: dict = None):
        """保存阶段完成状态"""
        if str(round_num) not in self.state['rounds']:
            self.state['rounds'][str(round_num)] = {}

        self.state['rounds'][str(round_num)][stage] = {
            'status': 'completed',
            'result_path': result_path,
            'completion_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.state['experiment_meta']['last_update'] = datetime.now().isoformat()
        self._save_state()

    def save_stage_in_progress(self, round_num: int, stage: str):
        """标记阶段正在进行"""
        if str(round_num) not in self.state['rounds']:
            self.state['rounds'][str(round_num)] = {}

        self.state['rounds'][str(round_num)][stage] = {
            'status': 'in_progress',
            'start_time': datetime.now().isoformat()
        }

        self._save_state()

    def _save_state(self):
        """保存状态到文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def get_stage_status(self, round_num: int, stage: str) -> Optional[dict]:
        """获取特定阶段的状态"""
        return self.state.get('rounds', {}).get(str(round_num), {}).get(stage)

    def is_stage_completed(self, round_num: int, stage: str) -> bool:
        """检查阶段是否完成"""
        status = self.get_stage_status(round_num, stage)
        return status and status.get('status') == 'completed'

    def get_next_pending_stage(self, round_num: int) -> Optional[str]:
        """获取下一个需要执行的阶段"""
        stages = ['stage1_contrastive'] if round_num == 1 else ['stage2_contrastive']
        stages.extend(['supervised_learning', 'classifier_selection',
                      'consistency_scoring'])

        for stage in stages:
            if not self.is_stage_completed(round_num, stage):
                return stage

        return None

    def is_round_completed(self, round_num: int) -> bool:
        """检查整轮是否完成"""
        required_stages = ['supervised_learning', 'classifier_selection',
                          'consistency_scoring']

        if round_num == 1:
            required_stages.insert(0, 'stage1_contrastive')
        else:
            required_stages.insert(0, 'stage2_contrastive')

        for stage in required_stages:
            if not self.is_stage_completed(round_num, stage):
                return False

        return True

    def validate_config_compatibility(self) -> bool:
        """验证当前配置与保存状态的兼容性"""
        saved_hash = self.state.get('experiment_meta', {}).get('config_hash', '')
        current_hash = self._compute_config_hash()

        if saved_hash and saved_hash != current_hash:
            print(f"[警告] 配置文件已更改 (saved: {saved_hash}, current: {current_hash})")
            return False

        return True

    def get_progress_summary(self) -> str:
        """获取进度摘要"""
        lines = []
        lines.append(" 实验进度摘要:")
        lines.append(f"  开始时间: {self.state['experiment_meta'].get('start_time', 'unknown')}")
        lines.append(f"  最后更新: {self.state['experiment_meta'].get('last_update', 'unknown')}")
        lines.append(f"  配置名称: {self.state['experiment_meta'].get('config_name', 'unknown')}")
        lines.append("")

        for round_num in sorted(self.state.get('rounds', {}).keys()):
            lines.append(f"  Round {round_num}:")
            round_data = self.state['rounds'][round_num]

            for stage, info in round_data.items():
                status = info.get('status', 'unknown')
                symbol = '[完成]' if status == 'completed' else '[进行中]' if status == 'in_progress' else '[未完成]'
                lines.append(f"    {symbol} {stage}: {status}")

                if status == 'completed':
                    lines.append(f"       完成时间: {info.get('completion_time', 'unknown')}")
                elif status == 'in_progress':
                    lines.append(f"       开始时间: {info.get('start_time', 'unknown')}")

        return '\n'.join(lines)


class ExperimentVariantManager:
    """管理基于第一阶段结果的多个实验变体"""

    def __init__(self, base_config: dict, base_output_dir: str):
        self.base_config = base_config
        self.base_output_dir = base_output_dir

    def create_stage2_variants(self, custom_variants: List[dict] = None) -> List[Tuple[dict, str]]:
        """基于第一阶段结果创建Stage2的不同超参数变体

        Args:
            custom_variants: 自定义的变体配置列表

        Returns:
            [(变体配置, 变体目录), ...]
        """
        variants = []

        # 使用自定义变体或默认变体
        if custom_variants:
            stage2_variants = custom_variants
        else:
            # 默认的Stage2超参数组合
            stage2_variants = [
                {
                    'weighting_strategy': 'linear',
                    'weight_threshold': 0.0,
                    'variant_name': 'linear_00',
                    'description': 'Linear权重策略，无阈值'
                },
                {
                    'weighting_strategy': 'threshold',
                    'weight_threshold': 0.3,
                    'variant_name': 'threshold_03',
                    'description': 'Threshold策略，阈值0.3'
                },
                {
                    'weighting_strategy': 'threshold',
                    'weight_threshold': 0.5,
                    'variant_name': 'threshold_05',
                    'description': 'Threshold策略，阈值0.5'
                },
                {
                    'weighting_strategy': 'threshold',
                    'weight_threshold': 0.7,
                    'variant_name': 'threshold_07',
                    'description': 'Threshold策略，阈值0.7'
                }
            ]

        for variant_params in stage2_variants:
            variant_config = self._create_variant_config(variant_params)
            # 支持自定义变体目录名
            if 'output_dir' in variant_params and variant_params['output_dir']:
                variant_dir = variant_params['output_dir']
            else:
                variant_dir = f"{self.base_output_dir}_{variant_params['variant_name']}"
            variants.append((variant_config, variant_dir))

        return variants

    def _create_variant_config(self, variant_params: dict) -> dict:
        """创建变体配置"""
        variant_config = copy.deepcopy(self.base_config)

        # 添加变体元信息
        variant_config['variant_meta'] = {
            'name': variant_params['variant_name'],
            'description': variant_params.get('description', ''),
            'base_experiment': self.base_output_dir,
            'creation_time': datetime.now().isoformat()
        }

        # 更新Stage2参数
        if 'defaults' not in variant_config:
            variant_config['defaults'] = {}
        if 'stage2_contrastive' not in variant_config['defaults']:
            variant_config['defaults']['stage2_contrastive'] = {}

        # 更新具体参数
        stage2_config = variant_config['defaults']['stage2_contrastive']
        stage2_config['weighting_strategy'] = variant_params['weighting_strategy']
        stage2_config['weight_threshold'] = variant_params['weight_threshold']

        # 可以添加其他参数的变化
        if 'base_lr' in variant_params:
            stage2_config['base_lr'] = variant_params['base_lr']
        if 'projection_lr' in variant_params:
            stage2_config['projection_lr'] = variant_params['projection_lr']
        if 'epochs' in variant_params:
            stage2_config['epochs'] = variant_params['epochs']

        return variant_config

    def copy_stage1_results(self, source_dir: str, target_dirs: List[str],
                           regenerate_consistency: bool = True):
        """将第一阶段结果复制到各个变体目录

        Args:
            source_dir: 源目录（包含Stage1结果）
            target_dirs: 目标目录列表
            regenerate_consistency: 是否删除一致性评分，让变体重新生成
        """
        for target_dir in target_dirs:
            print(f"   复制Stage1结果到: {target_dir}")
            os.makedirs(target_dir, exist_ok=True)

            # 复制 round1 目录
            source_round1 = os.path.join(source_dir, 'round1')
            target_round1 = os.path.join(target_dir, 'round1')

            if os.path.exists(source_round1):
                os.makedirs(target_round1, exist_ok=True)

                for item in os.listdir(source_round1):
                    source_item = os.path.join(source_round1, item)
                    target_item = os.path.join(target_round1, item)

                    # 复制除了 consistency_scoring 以外的所有内容
                    if item == 'consistency_scoring' and regenerate_consistency:
                        print(f"    [跳过] 不复制 consistency_scoring（变体将基于自己的配置重新生成）")
                        continue

                    if os.path.isdir(source_item):
                        if os.path.exists(target_item):
                            shutil.rmtree(target_item)
                        shutil.copytree(source_item, target_item)
                        print(f"    [OK] 复制目录: round1/{item}")
                    else:
                        shutil.copy2(source_item, target_item)
                        print(f"    [OK] 复制文件: round1/{item}")

            # 复制实验状态文件
            state_file = os.path.join(source_dir, 'experiment_state.json')
            if os.path.exists(state_file):
                shutil.copy2(state_file, os.path.join(target_dir, 'experiment_state.json'))
                print(f"    [OK] 复制文件: experiment_state.json")

            # 更新复制后的experiment_state.json，移除 consistency_scoring 状态
            self._update_variant_state(target_dir, regenerate_consistency)

    def _update_variant_state(self, variant_dir: str, regenerate_consistency: bool = True):
        """更新变体的实验状态文件

        Args:
            variant_dir: 变体目录
            regenerate_consistency: 是否需要重新生成一致性评分
        """
        state_file = os.path.join(variant_dir, 'experiment_state.json')

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # 添加变体信息
                state['variant_info'] = {
                    'is_variant': True,
                    'base_experiment': self.base_output_dir,
                    'variant_dir': variant_dir,
                    'creation_time': datetime.now().isoformat()
                }

                # 如果需要重新生成一致性评分，移除 Round 1 的 consistency_scoring 状态
                if regenerate_consistency and '1' in state.get('rounds', {}):
                    round1_stages = state['rounds']['1'].get('stages', {})
                    if 'consistency_scoring' in round1_stages:
                        del round1_stages['consistency_scoring']
                        print(f"    [状态] 已移除 Round 1 一致性评分状态，变体将重新生成")

                # 标记Round 2尚未开始
                if '2' in state.get('rounds', {}):
                    del state['rounds']['2']

                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"    [警告] 无法更新状态文件: {e}")


def print_experiment_status(output_dir: str, target_round: int):
    """打印实验状态的详细信息"""
    print("\n" + "="*60)
    print("实验状态检测")
    print("="*60)

    for round_num in range(1, target_round + 1):
        print(f"\nRound {round_num}:")
        detector = ExperimentStatusDetector(output_dir, round_num)
        summary = detector.get_round_status_summary()

        for stage, completed in summary.items():
            symbol = '[完成]' if completed else '[未完成]'
            print(f"  {symbol} {stage}")

    state_manager = ExperimentStateManager({}, output_dir)
    if os.path.exists(state_manager.state_file):
        print("\n" + state_manager.get_progress_summary())