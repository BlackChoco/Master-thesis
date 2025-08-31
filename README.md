"# CL_Training 重构项目

## 项目概述

这是一个完全重构的三阶段训练管道项目，用于处理论坛讨论数据的对比学习和有监督学习。该项目将原本高度耦合的代码重构为模块化、可扩展的架构。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整训练管道

```python
from example_three_stage_training_new import ThreeStageTrainingPipeline
from config import get_config

# 创建训练管道
pipeline = ThreeStageTrainingPipeline(
    experiment_name="my_first_experiment",
    base_output_dir="./experiments"
)

# 使用预定义配置运行
config = get_config('debug')  # 快速调试配置
results = pipeline.run_complete_pipeline(
    raw_data_path="./data/raw_data/discussion_data.json",
    labeled_data_path="./data/sup_train_data/trainset.csv",
    config=config.to_dict()
)
```

### 3. 分阶段运行

```python
# 阶段1：数据剪枝
from pipelines.stage1_pruning import Stage1PruningPipeline
stage1 = Stage1PruningPipeline(output_dir="./output/stage1")
pruned_data = stage1.run(data_path="./data/raw_data.json")

# 阶段2：对比学习
from pipelines.stage2_contrastive import Stage2ContrastivePipeline  
stage2 = Stage2ContrastivePipeline(output_dir="./output/stage2")
contrastive_model = stage2.run(data_path=pruned_data)

# 阶段3：有监督学习
from pipelines.stage3_supervised import Stage3SupervisedPipeline
stage3 = Stage3SupervisedPipeline(output_dir="./output/stage3")
final_model = stage3.run(
    data_path="./data/labeled_data.csv",
    contrastive_encoder_path="./output/stage2/contrastive_encoder"
)
```

## 📁 项目结构

```
CL_training/
├── core/                    # 核心数据结构
│   ├── data_structures.py   # 评论树、森林管理器等
│   └── similarity.py        # 相似度计算函数
├── preprocessing/           # 数据预处理
│   ├── text_utils.py       # 文本处理工具
│   ├── pruning.py          # 剪枝算法
│   └── analysis.py         # 数据分析工具
├── models/                  # 模型定义
│   ├── base_encoder.py     # 对比学习编码器
│   ├── textcnn.py          # TextCNN模型
│   ├── supervised.py       # 有监督学习模型
│   └── modelscope_utils.py # ModelScope集成
├── training/                # 训练器
│   ├── contrastive.py      # 对比学习训练器
│   ├── supervised.py       # 有监督学习训练器
│   └── losses.py           # 损失函数
├── utils/                   # 工具模块
│   ├── data_loaders.py     # 数据加载器
│   ├── evaluation.py       # 模型评估工具
│   └── io_utils.py         # 输入输出工具
├── pipelines/               # 训练管道
│   ├── stage1_pruning.py   # 第一阶段：剪枝
│   ├── stage2_contrastive.py # 第二阶段：对比学习
│   └── stage3_supervised.py # 第三阶段：有监督学习
├── config.py                # 配置管理
├── example_three_stage_training_new.py # 使用示例
└── test_refactoring.py      # 重构验证测试
```

## 🎯 三阶段训练流程

### 阶段1：数据剪枝
- **目标**: 基于语义相似度剪枝，提高数据质量
- **输入**: 原始论坛讨论数据
- **输出**: 剪枝后的高质量数据
- **特性**: 智能阈值推荐、多种剪枝策略、详细分析报告

### 阶段2：对比学习
- **目标**: 学习文本的语义表示
- **输入**: 剪枝后的数据
- **输出**: 预训练的文本编码器
- **特性**: 父子评论正样本对、负样本采样、InfoNCE损失

### 阶段3：有监督学习
- **目标**: 基于预训练编码器进行下游任务学习
- **输入**: 标注数据 + 预训练编码器
- **输出**: 任务特定的分类模型
- **特性**: 编码器微调、多种损失函数、全面评估

## ⚙️ 配置管理

项目提供了灵活的配置管理系统：

```python
from config import get_config, create_custom_config

# 使用预定义配置
config = get_config('production')  # default, debug, production, small

# 创建自定义配置
config = create_custom_config(
    model_name='bert-base-chinese',
    batch_size=16,
    num_epochs_stage2=5,
    num_epochs_stage3=3,
    experiment_name="custom_experiment"
)

# 保存和加载配置
config.save_to_file("./my_config.json")
config = TrainingConfig.load_from_file("./my_config.json")
```

## 🔧 预定义配置

- **default**: 标准配置，平衡性能和速度
- **debug**: 快速调试配置，减少训练时间
- **production**: 生产配置，最大化模型性能
- **small**: 小模型配置，适用于资源受限环境

## 📊 模型评估

项目提供了全面的评估工具：

```python
from utils.evaluation import ModelEvaluator, evaluate_pipeline_results

# 评估整个管道
evaluate_pipeline_results(
    stage1_path="./experiments/my_experiment/stage1_pruning",
    stage2_path="./experiments/my_experiment/stage2_contrastive",
    stage3_path="./experiments/my_experiment/stage3_supervised",
    output_dir="./evaluation_results"
)

# 单独评估模型
evaluator = ModelEvaluator(output_dir="./model_evaluation")
results = evaluator.evaluate_classification(
    y_true=true_labels,
    y_pred=predictions,
    y_proba=probabilities,
    class_names=["positive", "negative", "neutral"]
)
```

## 🧪 测试验证

运行重构验证测试：

```bash
python test_refactoring.py
```

该脚本会测试：
- 所有模块的导入
- 基本功能的正确性
- 管道初始化
- 生成详细的测试报告

## 📖 详细文档

更多详细信息请参考：
- `REFACTORING_SUMMARY.md`: 完整的重构总结和说明
- 各模块的docstring文档
- 示例代码和使用说明

## 💡 使用建议

### 新用户
1. 从 `example_three_stage_training_new.py` 开始
2. 使用 `debug` 配置进行快速验证
3. 逐步了解各个模块的功能

### 高级用户
1. 自定义配置文件
2. 扩展模型和训练策略
3. 添加新的评估指标

### 生产环境
1. 使用 `production` 配置
2. 启用检查点保存
3. 监控训练过程和性能

## 🔄 从原代码迁移

如果你之前使用的是重构前的代码：

1. **数据结构**: `Tree_data_model.py` → `core/data_structures.py`
2. **对比学习**: `cl_training.py` → `training/contrastive.py` + `models/base_encoder.py`
3. **有监督学习**: `sup_training.py` → `training/supervised.py` + `models/supervised.py`
4. **配置和使用**: 使用新的管道API和配置系统

## 🤝 贡献指南

1. 代码风格：遵循PEP 8规范
2. 文档：为新功能添加详细的docstring
3. 测试：为新模块添加相应的测试
4. 类型注解：使用类型提示提高代码可读性

## 📄 许可证

[在此添加许可证信息]

## 📞 联系方式

[在此添加联系方式]

---

🎉 **重构完成！** 这个新的模块化架构大大提高了代码的可维护性、可扩展性和可复用性。享受更好的开发体验吧！" 
