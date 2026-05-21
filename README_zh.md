# Benchmark + MUSE

这是一个面向 **sample-wise machine unlearning** 的 benchmark 项目，统一接入了多种隐私攻击评估方法，并实现了新的遗忘方法 **MUSE**。

项目当前的重点是：

- 统一 sample-wise forget index
- 用统一入口跑多种攻击方法
- 统一输出 utility 和 privacy 指标
- 在尽量少改原框架的前提下接入新遗忘方法

## 主要特性

- 统一 benchmark 入口：[`run_benchmark.py`](./run_benchmark.py)
- 已接入攻击方法：
  - `lira`
  - `rea`
  - `ruli`
  - `unlearningleaks`
- 已整理的数据集/骨干网络预设：
  - `Cifar10` -> `ResNet18`
  - `Cifar100` -> `ResNet50`
  - `TinyImageNet` -> `ViT`
- 新增遗忘方法：
  - `MUSE`（Membership-Undistinguishable Sample Erasure）
- 支持多轮实验脚本，并保证每轮内部共享同一组 `seed + sample`

## MUSE 是什么

MUSE 是在现有 sample-wise 遗忘主线上的一个新方法。它保留 retain set 上的基础目标，同时加入两个正则项，让 forget 样本的输出行为更接近 reference non-member 样本，从而降低 membership inference 风险。

每个 step 同时采样：

- forget batch `D_f`
- retain batch `D_r`
- reference non-member batch `D_t`

总损失为：

```text
L_total = L_base + lambda_align * L_align + lambda_stat * L_stat
```

其中：

- `L_base = CE(model(x_r), y_r)`
- `L_align`：forget batch 与 non-member batch 的 batch-mean softmax 分布对齐
- `L_stat`：forget batch 与 non-member batch 的攻击敏感统计量对齐
  - 最大 softmax confidence
  - top1-top2 logit margin

主要实现文件：

- [`muse/trainer.py`](./muse/trainer.py)
- [`forget_random_strategies.py`](./forget_random_strategies.py)
- [`forget_sample_main.py`](./forget_sample_main.py)

## 仓库结构

- [`run_benchmark.py`](./run_benchmark.py)：主 benchmark 入口
- [`benchmark/samplewise.py`](./benchmark/samplewise.py)：数据集预设、stage 编排、攻击分发
- [`forget_sample_main.py`](./forget_sample_main.py)：sample-wise 遗忘主入口
- [`rea_reminiscence_random.py`](./rea_reminiscence_random.py)：REA 专用 reminiscence 阶段
- [`attacks/`](./attacks)：攻击方法适配层
- [`muse/`](./muse)：MUSE 实现
- [`Ruli/`](./Ruli)：RULI 攻击代码
- [`scripts/`](./scripts)：批量实验脚本

## 环境

当前项目主要在名为 `exp` 的 Conda 环境中使用。

当前使用的一套已验证运行环境为：

- Python: `3.10.0`
- PyTorch: `2.7.1+cu128`
- TorchVision: `0.22.1+cu128`
- GPU: `NVIDIA GeForce RTX 5090`

典型安装方式：

```bash
conda create -n exp python=3.10 -y
conda activate exp
pip install -r requirements.txt
```

## 数据集

当前 benchmark 使用的数据集：

- `Cifar10`
- `Cifar100`
- `TinyImageNet`

默认数据目录：

```text
./data
```

## Benchmark 流程

支持的 stages：

- `pretrain`
- `shadow`
- `unlearn`
- `reminiscence`
- `attack`

各阶段含义：

- `pretrain`：训练 target model
- `shadow`：准备可复用的 shadow models
- `unlearn`：运行指定遗忘方法
- `reminiscence`：REA 专用附加阶段
- `attack`：运行攻击方法

重要行为：

- `shadow` 如果已有 checkpoint，会直接复用
- `pretrain`、`unlearn`、`reminiscence`、`attack` 可以针对不同 seed/sample 反复重跑
- `REA` 需要额外的 `reminiscence`，其他攻击不需要

## 当前已接入的遗忘方法

主线常用方法包括：

- `retrain`
- `finetune`
- `negative_grad`
- `scrub`
- `muse`

## 当前已接入的攻击方法

- `lira`
- `rea`
- `ruli`
- `unlearningleaks`


## 快速开始

### 1. 在 CIFAR-10 上运行 `finetune`，并用 REA 攻击测试

完整流程：

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages pretrain shadow unlearn reminiscence attack
```

如果 `pretrain` 和 `shadow` 已经有了，可以只跑后半段：

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages unlearn reminiscence attack
```

### 2. 用一个遗忘方法跑四种攻击

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn attack
```

### 3. 先做 dry-run 检查命令链路

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn reminiscence attack \
  --dry-run
```

## 多轮实验脚本示例

如果你要在 CIFAR-10 上跑五轮，每轮内部共享同一组 `seed + sample`，并同时比较四种攻击和五种遗忘方法（含 `muse`）：

```bash
PYTHON_BIN=/root/miniconda3/envs/exp/bin/python \
bash ./scripts/run_cifar10_five_rounds_muse.sh
```

这个脚本会：

- 每轮生成一个随机 `seed`
- 每轮生成一份共享的 forget sample
- 如果已有 `shadow`，就直接复用
- 每轮重新跑 `pretrain / unlearn / attack`
- 中途某个实验失败时自动跳过并继续
- 最后输出多轮 `mean/std`

## 常用参数

`run_benchmark.py` 常用参数：

```bash
--dataset Cifar10|Cifar100|TinyImageNet|Cinic10
--methods method 或 method:para1:para2
--stages pretrain shadow unlearn reminiscence attack
--attacks lira rea ruli unlearningleaks
--seed 1
--forget-perc 0.1
--num-shadow 8
--num-aug 10
--dry-run
```

攻击方法相关参数示例：

- RULI：
  - `--ruli-task selective`
  - `--ruli-shadow-num 8`
  - `--ruli-train-shadow-mode auto`
- UnlearningLeaks：
  - `--unlearningleaks-feature direct_diff`
  - `--unlearningleaks-attack-model lr`
  - `--unlearningleaks-num-shadow 8`

完整参数列表：

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py --help
```

## MUSE 默认超参

当前 `MUSE` 的数据集默认超参定义在 [`config.py`](./config.py)：

- `Cifar10`: `lr=4e-4`, `epochs=7`, `lambda_align=1.0`, `lambda_stat=0.5`
- `Cifar100`: `lr=3e-4`, `epochs=8`, `lambda_align=1.0`, `lambda_stat=0.75`
- `Cinic10`: `lr=4e-4`, `epochs=8`, `lambda_align=1.0`, `lambda_stat=0.5`
- `TinyImageNet`: `lr=3e-4`, `epochs=10`, `lambda_align=0.5`, `lambda_stat=0.25`

也可以手动覆盖：

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods muse:4e-4:7 \
  --attacks rea \
  --stages unlearn reminiscence attack
```

## 输出结果

### 遗忘阶段指标

每个方法的 `unlearning` 结果存放在：

```text
log_files\model/forget_random_main/<experiment>/unlearning/<method_key>/
```

其中 `tsv` 文件会记录：

- `clean_acc` -> `TA`
- `forgetting_acc` -> `UA`
- `remaining_acc` -> `RA`
- `zrf`
- `mia`
- `time`

注意：

- 这些 `tsv` 是工作目录下的当前结果
- 同一个方法目录重复运行时，`tsv` 会被覆盖

### 攻击阶段指标

攻击结果存放在：

```text
benchmark_results/samplewise/<experiment>/<method_key>/<attack>/seed_<seed>/
```

常见输出包括：

- `attack_result.json`
- `ruli_summary.json`
- 攻击日志
- LiRA/REA 的 ROC 相关数组

### 顶层运行摘要

每次 benchmark 运行还会保存：

```text
benchmark_results/samplewise/<experiment>/run_summary_seed_<seed>.json
```

## 指标说明

遗忘效能指标：

- `TA`：测试集准确率
- `UA`：forget set 上的准确率
- `RA`：retain set 上的准确率

隐私攻击指标统一输出为：

- `auc`
- `accuracy`
- `TPR@0.1%FPR`
- `TPR@1%FPR`
- `TPR@10%FPR`

如果你采用较严格的 legacy 定义，把 `0.1 FPR` 固定下来，那么最关键的隐私指标通常是：

- `TPR@10%FPR`

## 备注

- `REA` 需要额外的 `reminiscence` 阶段
- `RULI` 除了复用 benchmark 主线信息，还会维护它自己的一套 shadow attack 资产
- `UnlearningLeaks` 会复用主线 shadow，同时生成自己的 shadow-unlearned assets
- 仓库里有多个第三方子模块；如果只是想做统一实验，建议优先使用 benchmark 入口
