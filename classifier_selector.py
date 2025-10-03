import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import glob

def load_all_seeds_results(results_path: str) -> Dict:
    """加载所有种子的完整结果"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"结果文件未找到: {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    print(f" 已加载 {len(all_results)} 个模型-数据比例组合的结果")
    return all_results

def extract_fraction_from_key(key: str) -> float:
    """从键中提取数据比例"""
    # 例如: "lora_bert_base_chinese_cl_frac1.0" -> 1.0
    parts = key.split('_frac')
    if len(parts) == 2:
        return float(parts[1])
    return 0.0

def calculate_statistics(seeds_results: List[Dict]) -> Dict:
    """计算统计信息"""
    if not seeds_results:
        return {}

    metrics = ['test_f1', 'test_accuracy', 'test_precision', 'test_recall', 'val_f1']
    stats = {}

    for metric in metrics:
        values = [r[metric] for r in seeds_results if metric in r]
        if values:
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
            stats[f'{metric}_values'] = values

    stats['num_seeds'] = len(seeds_results)
    return stats

def select_closest_to_mean_classifier(seeds_results: List[Dict], metric: str = 'test_f1') -> Tuple[Dict, Dict]:
    """选择指定指标最接近均值的分类器"""
    if not seeds_results:
        return None, {}

    values = [r[metric] for r in seeds_results if metric in r]
    if not values:
        return None, {}

    mean_value = np.mean(values)

    # 找到最接近均值的种子
    distances = [abs(v - mean_value) for v in values]
    closest_idx = np.argmin(distances)

    selected_classifier = seeds_results[closest_idx]

    selection_info = {
        'selection_metric': metric,
        'mean_value': mean_value,
        'selected_value': values[closest_idx],
        'distance_to_mean': distances[closest_idx],
        'all_values': values,
        'selected_seed': selected_classifier['seed'],
        'selection_strategy': 'closest_to_mean'
    }

    return selected_classifier, selection_info

def select_classifiers_for_fractions(all_results: Dict,
                                   target_fractions: List[float] = [1.0, 0.5, 0.2, 0.1, 0.05],
                                   selection_metric: str = 'test_f1') -> Dict:
    """为指定的数据比例选择最优分类器"""

    selected_classifiers = {}

    for model_fraction_key, seeds_results in all_results.items():
        fraction = extract_fraction_from_key(model_fraction_key)

        # 只处理目标数据比例
        if fraction not in target_fractions:
            continue

        print(f"\n 处理 {model_fraction_key} (数据比例: {fraction})")
        print(f"   可用种子数: {len(seeds_results)}")

        # 计算统计信息
        stats = calculate_statistics(seeds_results)

        # 选择最接近均值的分类器
        selected, selection_info = select_closest_to_mean_classifier(seeds_results, selection_metric)

        if selected:
            selected_classifiers[model_fraction_key] = {
                'selected_classifier': selected,
                'selection_info': selection_info,
                'statistics': stats,
                'model_fraction_key': model_fraction_key,
                'data_fraction': fraction
            }

            print(f"    选中种子: {selected['seed']}")
            print(f"    {selection_metric}: {selected[selection_metric]:.4f} (均值: {selection_info['mean_value']:.4f})")
            print(f"    模型路径: {selected['model_path']}")
        else:
            print(f"    无法选择分类器")

    return selected_classifiers

def create_selection_visualization(selected_classifiers: Dict, output_dir: str):
    """创建选择结果的可视化图表"""

    # 准备数据
    data = []
    for key, info in selected_classifiers.items():
        stats = info['statistics']
        selection = info['selection_info']

        data.append({
            'model_fraction': key,
            'data_fraction': info['data_fraction'],
            'mean_f1': stats.get('test_f1_mean', 0),
            'std_f1': stats.get('test_f1_std', 0),
            'selected_f1': selection['selected_value'],
            'distance_to_mean': selection['distance_to_mean'],
            'num_seeds': stats['num_seeds']
        })

    df = pd.DataFrame(data)

    if df.empty:
        print(" 没有数据可供可视化")
        return

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. F1分数对比 (均值 vs 选中值)
    ax1 = axes[0, 0]
    x_pos = np.arange(len(df))
    width = 0.35

    ax1.bar(x_pos - width/2, df['mean_f1'], width, label='均值', alpha=0.8, color='skyblue')
    ax1.bar(x_pos + width/2, df['selected_f1'], width, label='选中值', alpha=0.8, color='orange')
    ax1.errorbar(x_pos - width/2, df['mean_f1'], yerr=df['std_f1'], fmt='none', color='black', capsize=3)

    ax1.set_xlabel('模型-数据比例组合')
    ax1.set_ylabel('F1分数')
    ax1.set_title('F1分数对比: 均值 vs 选中值')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"frac{f:.1f}" for f in df['data_fraction']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 距离均值的偏差
    ax2 = axes[0, 1]
    ax2.bar(x_pos, df['distance_to_mean'], color='lightcoral', alpha=0.8)
    ax2.set_xlabel('模型-数据比例组合')
    ax2.set_ylabel('距离均值的偏差')
    ax2.set_title('选中分类器距离均值的偏差')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"frac{f:.1f}" for f in df['data_fraction']], rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. 标准差分布
    ax3 = axes[1, 0]
    ax3.bar(x_pos, df['std_f1'], color='lightgreen', alpha=0.8)
    ax3.set_xlabel('模型-数据比例组合')
    ax3.set_ylabel('F1分数标准差')
    ax3.set_title('不同数据比例下的性能稳定性')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"frac{f:.1f}" for f in df['data_fraction']], rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. 数据比例 vs 性能
    ax4 = axes[1, 1]
    ax4.plot(df['data_fraction'], df['mean_f1'], 'o-', label='平均F1', linewidth=2, markersize=8)
    ax4.plot(df['data_fraction'], df['selected_f1'], 's-', label='选中F1', linewidth=2, markersize=8)
    ax4.fill_between(df['data_fraction'],
                     df['mean_f1'] - df['std_f1'],
                     df['mean_f1'] + df['std_f1'],
                     alpha=0.3, label='±1σ')
    ax4.set_xlabel('数据比例')
    ax4.set_ylabel('F1分数')
    ax4.set_title('性能随数据比例的变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(output_dir, 'classifier_selection_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" 可视化图表已保存到: {plot_path}")

def save_selection_results(selected_classifiers: Dict, output_dir: str, experiment_name: str = "classifier_selection"):
    """保存分类器选择结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存选择配置
    selection_config = {
        'experiment_name': experiment_name,
        'selection_strategy': 'closest_to_mean',
        'selection_metric': 'test_f1',
        'timestamp': datetime.now().isoformat(),
        'total_selected': len(selected_classifiers),
        'selected_fractions': sorted(list(set(info['data_fraction'] for info in selected_classifiers.values())))
    }

    config_path = os.path.join(output_dir, 'selection_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(selection_config, f, ensure_ascii=False, indent=4)
    print(f" 选择配置已保存到: {config_path}")

    # 2. 保存统计分析
    statistical_analysis = {}
    for key, info in selected_classifiers.items():
        statistical_analysis[key] = {
            'data_fraction': info['data_fraction'],
            'statistics': info['statistics'],
            'selection_info': info['selection_info']
        }

    stats_path = os.path.join(output_dir, 'statistical_analysis.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistical_analysis, f, ensure_ascii=False, indent=4)
    print(f" 统计分析已保存到: {stats_path}")

    # 3. 保存选中的模型信息
    selected_models_dir = os.path.join(output_dir, 'selected_models')
    os.makedirs(selected_models_dir, exist_ok=True)

    for key, info in selected_classifiers.items():
        model_info = {
            'model_fraction_key': key,
            'data_fraction': info['data_fraction'],
            'selected_classifier': info['selected_classifier'],
            'selection_reason': f"最接近{info['selection_info']['selection_metric']}均值",
            'model_path': info['selected_classifier']['model_path'],
            'hyperparameters': info['selected_classifier']['hyperparameters']
        }

        model_filename = f"{key}_selected.json"
        model_path = os.path.join(selected_models_dir, model_filename)

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)

    print(f" 选中模型信息已保存到: {selected_models_dir}")

    # 4. 创建可视化
    create_selection_visualization(selected_classifiers, output_dir)

    return output_dir

def generate_selection_report(selected_classifiers: Dict) -> str:
    """生成选择结果报告"""
    report = []
    report.append(" 分类器选择报告")
    report.append("=" * 50)
    report.append(f"选择时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"选择策略: 最接近均值")
    report.append(f"选择指标: test_f1")
    report.append(f"总计选中: {len(selected_classifiers)} 个分类器")
    report.append("")

    for key, info in selected_classifiers.items():
        report.append(f" {key}")
        report.append(f"   数据比例: {info['data_fraction']}")
        report.append(f"   选中种子: {info['selected_classifier']['seed']}")
        report.append(f"   F1分数: {info['selected_classifier']['test_f1']:.4f}")
        report.append(f"   距离均值: {info['selection_info']['distance_to_mean']:.4f}")
        report.append(f"   模型路径: {info['selected_classifier']['model_path']}")
        report.append("")

    return "\n".join(report)

def create_directory_index(base_output_dir: str):
    """创建目录管理索引文件"""
    index_path = os.path.join(base_output_dir, "README.md")

    # 扫描已有的选择结果
    existing_experiments = []
    if os.path.exists(base_output_dir):
        for item in os.listdir(base_output_dir):
            exp_dir = os.path.join(base_output_dir, item)
            if os.path.isdir(exp_dir) and item != "__pycache__":
                config_file = os.path.join(exp_dir, "selection_config.json")
                if os.path.exists(config_file):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    existing_experiments.append((item, config))

    # 生成README内容
    readme_content = f"""# 分类器选择结果目录

 **目录说明**: 此目录包含从不同有监督学习实验中选择的最优分类器结果

⏰ **最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

##  目录结构

```
{base_output_dir}/
├── README.md                    # 本说明文件
├── [实验名1]/                   # 第一个实验的分类器选择结果
│   ├── selection_config.json   # 选择配置
│   ├── statistical_analysis.json # 统计分析
│   ├── selection_report.txt    # 选择报告
│   ├── classifier_selection_analysis.png # 可视化图表
│   └── selected_models/        # 选中的模型信息
├── [实验名2]/                   # 第二个实验的分类器选择结果
└── ...
```

##  已有的分类器选择结果

| 实验名称 | 选择时间 | 选择策略 | 选择指标 | 选中数量 |
|---------|---------|---------|---------|---------|
"""

    for exp_name, config in existing_experiments:
        timestamp = config.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass

        readme_content += f"| {exp_name} | {timestamp} | {config.get('selection_strategy', 'N/A')} | {config.get('selection_metric', 'N/A')} | {config.get('total_selected', 'N/A')} |\n"

    if not existing_experiments:
        readme_content += "| (暂无) | - | - | - | - |\n"

    readme_content += f"""

##  使用方法

### 查看可用实验
```bash
python classifier_selector.py --list
```

### 为特定实验选择分类器
```bash
python classifier_selector.py --experiment [实验名]
```

### 自定义输出目录
```bash
python classifier_selector.py --experiment [实验名] --output-dir custom_output
```

##  文件说明

- **selection_config.json**: 记录选择配置和元信息
- **statistical_analysis.json**: 包含所有种子的统计分析结果
- **selection_report.txt**: 人类可读的选择报告
- **classifier_selection_analysis.png**: 选择结果的可视化分析
- **selected_models/**: 每个选中分类器的详细信息

##  下一步

选择好的分类器可以用于:
1. **一致性评分**: 使用 `consistency_scorer.py` 计算pair数据的一致性得分
2. **加权对比学习**: 使用 `weighted_cl_training.py` 进行加权训练
3. **迭代改进**: 使用 `iterative_main.py` 进行完整的迭代流程
"""

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f" 目录索引已更新: {index_path}")

def discover_experiments(base_dir: str = "sup_result_hyperparams") -> List[str]:
    """自动发现可用的实验"""
    if not os.path.exists(base_dir):
        print(f" 基础目录不存在: {base_dir}")
        return []

    experiments = []
    for item in os.listdir(base_dir):
        experiment_path = os.path.join(base_dir, item)
        if os.path.isdir(experiment_path):
            # 检查是否包含all_seeds_results.json
            results_file = os.path.join(experiment_path, "all_seeds_results.json")
            if os.path.exists(results_file):
                experiments.append(item)
            else:
                print(f" 实验 '{item}' 缺少 all_seeds_results.json 文件")

    return sorted(experiments)

def get_experiment_info(experiment_path: str) -> Dict:
    """获取实验信息"""
    info_file = os.path.join(experiment_path, "experiment_info.json")
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def select_experiment_interactive(available_experiments: List[str]) -> str:
    """交互式选择实验"""
    if not available_experiments:
        print(" 没有找到可用的实验")
        return None

    print(" 发现以下可用的实验:")
    for i, exp in enumerate(available_experiments, 1):
        print(f"  {i}. {exp}")

    while True:
        try:
            choice = input(f"\n请选择实验 (1-{len(available_experiments)}) 或输入实验名称: ").strip()

            # 尝试按数字选择
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_experiments):
                    return available_experiments[idx]
                else:
                    print(f" 请输入 1-{len(available_experiments)} 之间的数字")
                    continue

            # 尝试按名称选择
            if choice in available_experiments:
                return choice

            print(f" 找不到实验 '{choice}'，请重新选择")
        except KeyboardInterrupt:
            print("\n 用户取消选择")
            return None
        except Exception as e:
            print(f" 输入错误: {e}")

def save_selection_results(selected_classifiers: Dict, output_dir: str, experiment_name: str):
    """保存分类器选择结果 - 修改版，支持实验名称"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存选择配置
    selection_config = {
        'source_experiment': experiment_name,
        'selection_strategy': 'closest_to_mean',
        'selection_metric': 'test_f1',
        'timestamp': datetime.now().isoformat(),
        'total_selected': len(selected_classifiers),
        'selected_fractions': sorted(list(set(info['data_fraction'] for info in selected_classifiers.values())))
    }

    config_path = os.path.join(output_dir, 'selection_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(selection_config, f, ensure_ascii=False, indent=4)
    print(f" 选择配置已保存到: {config_path}")

    # 2. 保存统计分析
    statistical_analysis = {}
    for key, info in selected_classifiers.items():
        statistical_analysis[key] = {
            'data_fraction': info['data_fraction'],
            'statistics': info['statistics'],
            'selection_info': info['selection_info']
        }

    stats_path = os.path.join(output_dir, 'statistical_analysis.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistical_analysis, f, ensure_ascii=False, indent=4)
    print(f" 统计分析已保存到: {stats_path}")

    # 3. 保存选中的模型信息
    selected_models_dir = os.path.join(output_dir, 'selected_models')
    os.makedirs(selected_models_dir, exist_ok=True)

    for key, info in selected_classifiers.items():
        # 修正模型路径 - 确保使用正确的路径
        original_model_path = info['selected_classifier']['model_path']

        # 智能路径处理：提取关键部分
        path_parts = original_model_path.replace('\\', '/').split('/')

        # 查找关键起始点（从后向前查找，避免重复）
        key_indices = []
        for i in range(len(path_parts) - 1, -1, -1):
            if path_parts[i] == 'supervised_training' or path_parts[i] == 'saved_models':
                key_indices.append(i)

        if key_indices:
            # 使用最后一个 supervised_training 或 saved_models 作为起点
            # 但要确保包含完整的有意义路径
            start_idx = key_indices[0]

            # 如果找到 saved_models，向前查找 supervised_training
            if path_parts[start_idx] == 'saved_models' and start_idx > 0:
                for j in range(start_idx - 1, -1, -1):
                    if path_parts[j] == 'supervised_training':
                        start_idx = j
                        break

            # 如果找到 supervised_training，检查前面是否有 round 目录
            if start_idx > 0 and path_parts[start_idx - 1].startswith('round'):
                # 包含 round 目录
                corrected_path = '/'.join(path_parts[start_idx - 1:])
            else:
                corrected_path = '/'.join(path_parts[start_idx:])

        elif original_model_path.startswith(('autodl-tmp', 'test_exp', '/')):
            # 对于自定义路径，找到 round 目录开始的部分
            round_idx = -1
            for i, part in enumerate(path_parts):
                if part.startswith('round'):
                    round_idx = i
                    break

            if round_idx >= 0:
                corrected_path = '/'.join(path_parts[round_idx:])
            else:
                # 保留完整路径
                corrected_path = original_model_path
        else:
            # 其他情况，使用原始路径
            corrected_path = original_model_path

        # 规范化路径分隔符
        corrected_path = corrected_path.replace('\\', '/')

        model_info = {
            'source_experiment': experiment_name,
            'model_fraction_key': key,
            'data_fraction': info['data_fraction'],
            'selected_classifier': info['selected_classifier'],
            'selection_reason': f"最接近{info['selection_info']['selection_metric']}均值",
            'model_path': corrected_path,
            'hyperparameters': info['selected_classifier']['hyperparameters']
        }

        model_filename = f"{key}_selected.json"
        model_path = os.path.join(selected_models_dir, model_filename)

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)

    print(f" 选中模型信息已保存到: {selected_models_dir}")

    # 4. 创建可视化
    create_selection_visualization(selected_classifiers, output_dir)

    return output_dir

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分类器选择器 - 从有监督学习结果中选择最优分类器')

    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='指定实验名称 (如果不指定，将交互式选择)')

    parser.add_argument('--base-dir', '-b', type=str, default='sup_result_hyperparams',
                       help='实验结果基础目录 (默认: sup_result_hyperparams)')

    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='输出目录 (默认: classifier_selections/[实验名])')

    parser.add_argument('--base-output-dir', type=str, default='classifier_selections',
                       help='输出基础目录 (默认: classifier_selections)')

    parser.add_argument('--fractions', '-f', type=float, nargs='+',
                       default=[0.2, 0.1],
                       help='目标数据比例列表 (默认: 1.0 0.5 0.2 0.1 0.05)')

    parser.add_argument('--metric', '-m', type=str, default='test_f1',
                       choices=['test_f1', 'test_accuracy', 'test_precision', 'test_recall', 'val_f1'],
                       help='选择指标 (默认: test_f1)')

    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有可用实验并退出')

    return parser.parse_args()

def main():
    """主函数 - 支持多实验"""
    print(" 分类器选择器启动...")

    # 解析命令行参数
    args = parse_arguments()

    # 发现可用实验
    available_experiments = discover_experiments(args.base_dir)

    if args.list:
        print(f" 在 {args.base_dir} 中发现 {len(available_experiments)} 个可用实验:")
        for exp in available_experiments:
            exp_path = os.path.join(args.base_dir, exp)
            exp_info = get_experiment_info(exp_path)
            description = exp_info.get('experiment_meta', {}).get('experiment_name', '无描述')
            print(f"  • {exp}: {description}")
        return

    # 选择实验
    if args.experiment:
        if args.experiment in available_experiments:
            selected_experiment = args.experiment
            print(f" 使用指定实验: {selected_experiment}")
        else:
            print(f" 指定的实验 '{args.experiment}' 不存在")
            print("可用实验:", available_experiments)
            return
    else:
        selected_experiment = select_experiment_interactive(available_experiments)
        if not selected_experiment:
            return

    # 设置路径
    experiment_path = os.path.join(args.base_dir, selected_experiment)
    results_file = os.path.join(experiment_path, "all_seeds_results.json")

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 使用统一的基础目录管理
        output_dir = os.path.join(args.base_output_dir, selected_experiment)

    # 确保基础输出目录存在
    os.makedirs(args.base_output_dir, exist_ok=True)

    # 创建目录管理说明文件
    create_directory_index(args.base_output_dir)

    print(f" 实验路径: {experiment_path}")
    print(f" 输出目录: {output_dir}")
    print(f" 目标数据比例: {args.fractions}")
    print(f" 选择指标: {args.metric}")

    try:
        # 1. 加载所有种子结果
        all_results = load_all_seeds_results(results_file)

        # 2. 选择分类器
        selected_classifiers = select_classifiers_for_fractions(
            all_results,
            target_fractions=args.fractions,
            selection_metric=args.metric
        )

        if not selected_classifiers:
            print(" 没有选择到任何分类器")
            return

        # 3. 保存结果
        final_output_dir = save_selection_results(
            selected_classifiers,
            output_dir,
            experiment_name=selected_experiment
        )

        # 4. 生成报告
        report = generate_selection_report(selected_classifiers)
        print("\n" + report)

        # 保存报告
        report_path = os.path.join(final_output_dir, 'selection_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f" 选择报告已保存到: {report_path}")

        print(f"\n 分类器选择完成!")
        print(f" 结果保存在: {final_output_dir}")
        print(f" 共选择了 {len(selected_classifiers)} 个分类器")

    except Exception as e:
        print(f" 分类器选择过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def run_classifier_selection_interface(sup_result_dir: str, config: dict, output_dir: str) -> str:
    """
    标准化接口：运行分类器选择

    Args:
        sup_result_dir: 监督学习结果目录
        config: 分类器选择配置字典
        output_dir: 输出目录

    Returns:
        选择结果目录路径
    """
    import os
    import json

    print(f" 分类器选择接口调用")
    print(f"   监督学习结果: {sup_result_dir}")
    print(f"   输出目录: {output_dir}")

    try:
        # 查找all_seeds_results.json
        results_file = os.path.join(sup_result_dir, "all_seeds_results.json")
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"找不到结果文件: {results_file}")

        # 加载所有种子结果
        all_results = load_all_seeds_results(results_file)

        # 选择分类器
        selected_classifiers = select_classifiers_for_fractions(
            all_results,
            target_fractions=config.get('target_fractions', [0.1]),
            selection_metric=config.get('selection_metric', 'test_f1')
        )

        if not selected_classifiers:
            raise RuntimeError("没有选择到任何分类器")

        # 获取实验名称
        experiment_name = os.path.basename(sup_result_dir)

        # 保存结果
        final_output_dir = save_selection_results(
            selected_classifiers,
            output_dir,
            experiment_name=experiment_name
        )

        # 生成报告
        report = generate_selection_report(selected_classifiers)
        report_path = os.path.join(final_output_dir, 'selection_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f" 分类器选择完成，结果保存在: {final_output_dir}")
        return final_output_dir

    except Exception as e:
        print(f" 分类器选择失败: {e}")
        raise


if __name__ == "__main__":
    main()