# TCM-Lung-Nodule-Benchmark (中医肺结节 QA 基准测试集)

这是一个用于评估大型语言模型（LLMs）在中医肺结节（肺积、痰核、息积等）诊疗与辨证论治方面能力的专业基准测试仓库。

---

## 目录结构

```text
TCM-Lung-Nodule-Benchmark/
├── data/                  # 标准化后的测试集和答案文件
│   ├── questions.json       # 仅包含问题和选项
│   ├── answers.json         # 仅包含问题 ID 和答案
│   └── full_test_set.json   # 包含完整的问题、选项、答案、题型、难度、方向及解析
├── scripts/               # 测评和推理脚本
│   ├── evaluate.py              # 单个模型评估脚本
│   ├── evaluate_multiple.py     # 多个模型对比评估脚本
│   ├── creat_openai_eval_async.py # 使用 OpenAI / DeepSeek API 进行异步推理的脚本
│   └── generate_lung_nodule_1k.py # 批量生成与校验 1000 条数据的自动化工具
├── model_results/         # 存放模型的预测结果与评估对比报告/可视化图表
└── README.md              # 本说明文件
```

---

## 数据说明

`data/` 目录包含了用于评估的中医肺结节问答数据：
- `questions.json`：直接作为模型推理输入。
- `answers.json`：标准答案文件。
- `full_test_set.json`：包含所有元数据（题型、难度、辨证方向、详细解析），便于深入评估。

### 题型与难度分布配额 (全量 1000 条规划)

* **题型划分**：
  - 单选题 (权重 1.0): 500 条 (50%)
  - 多选题 (权重 2.0): 300 条 (30%)
  - 连续选择题 (权重 3.0): 200 条 (20%)
* **难度划分**：
  - 低难度 (权重 1.0): 270 条 (27%)
  - 中难度 (权重 1.5): 494 条 (49.4%)
  - 高难度 (权重 2.0): 236 条 (23.6%)

---

## 脚本说明

1. `scripts/evaluate.py`: 用于评估单个模型预测结果的准确率和加权得分。
2. `scripts/evaluate_multiple.py`: 用于横向对比多个模型的评估结果，并生成高清对比报告与图表。
3. `scripts/creat_openai_eval_async.py`: 使用 DeepSeek 或 OpenAI API 对 `data/questions.json` 进行并发推理。
4. `scripts/generate_lung_nodule_1k.py`: 批量生成与校验 1,000 条测试数据的准备工具。

---

## 使用方法

### 1. 验证 Demo 数据集评估
```bash
python scripts/evaluate_multiple.py data/full_test_set.json --output_dir model_results
```

### 2. 批量推理模型预测 (以 DeepSeek 为例)
```bash
python scripts/creat_openai_eval_async.py \
  --input_file data/questions.json \
  --output_file model_results/deepseek_chat/predictions.json \
  --model_name deepseek-chat \
  --api_key "<YOUR_API_KEY>" \
  --base_url "https://api.deepseek.com/v1" \
  --temperature 0.1 \
  --concurrency 10
```

### 3. 生成模型对比评估报告
```bash
python scripts/evaluate_multiple.py data/full_test_set.json model_results/deepseek_chat/predictions.json --output_dir model_results
```
