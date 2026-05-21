# Benchmark + MUSE

A sample-wise machine unlearning benchmark with unified evaluation across multiple privacy attacks and an integrated unlearning method, **MUSE**.

This repository focuses on:

- sample-wise forgetting with a shared forget index
- unified benchmarking across multiple attacks
- reproducible outputs for utility and privacy metrics
- lightweight integration of new unlearning methods into the existing framework

## Highlights

- Unified benchmark entry: [`run_benchmark.py`](./run_benchmark.py)
- Supported attacks:
  - `lira`
  - `rea`
  - `ruli`
  - `unlearningleaks`
- Supported datasets/backbones:
  - `Cifar10` -> `ResNet18`
  - `Cifar100` -> `ResNet50`
  - `TinyImageNet` -> `ViT`
- New unlearning method:
  - `MUSE` (Membership-Undistinguishable Sample Erasure)
- Multi-round experiment scripts with shared seed/sample per round

## What MUSE Is

MUSE is implemented as a new unlearning method on top of the existing sample-wise framework. It keeps the base retain-set objective and adds two regularizers so that forgotten samples behave more like reference non-member samples.

For each training step, MUSE samples:

- forget batch `D_f`
- retain batch `D_r`
- reference non-member batch `D_t`

The total loss is:

```text
L_total = L_base + lambda_align * L_align + lambda_stat * L_stat
```

with:

- `L_base = CE(model(x_r), y_r)`
- `L_align`: MSE between the batch-mean softmax distributions of forget and non-member batches
- `L_stat`: MSE between batch-mean attack-sensitive statistics:
  - maximum softmax confidence
  - top1-top2 logit margin

Main implementation files:

- [`muse/trainer.py`](./muse/trainer.py)
- [`forget_random_strategies.py`](./forget_random_strategies.py)
- [`forget_sample_main.py`](./forget_sample_main.py)

## Repository Structure

- [`run_benchmark.py`](./run_benchmark.py): main benchmark entry
- [`benchmark/samplewise.py`](./benchmark/samplewise.py): dataset presets, stage orchestration, attack dispatch
- [`forget_sample_main.py`](./forget_sample_main.py): main sample-wise unlearning entry
- [`rea_reminiscence_random.py`](./rea_reminiscence_random.py): REA-specific reminiscence stage
- [`attacks/`](./attacks): benchmark attack adapters
- [`muse/`](./muse): MUSE implementation
- [`Ruli/`](./Ruli): RULI attack code
- [`scripts/`](./scripts): convenience scripts for repeated experiments

## Environment

The repository has been used with a Conda environment named `exp`.

A validated runtime setup used in this project is:

- Python: `3.10.0`
- PyTorch: `2.7.1+cu128`
- TorchVision: `0.22.1+cu128`
- GPU: `NVIDIA GeForce RTX 5090`

Typical setup:

```bash
conda create -n exp python=3.10 -y
conda activate exp
pip install -r requirements.txt
```

## Datasets

The benchmark currently uses:

- `Cifar10`
- `Cifar100`
- `TinyImageNet`

Expected default root:

```text
./data
```

## Benchmark Pipeline

Available stages:

- `pretrain`
- `shadow`
- `unlearn`
- `reminiscence`
- `attack`

Typical stage usage:

- `pretrain`: train the target model
- `shadow`: prepare reusable shadow models
- `unlearn`: run the chosen unlearning method
- `reminiscence`: extra REA-only stage
- `attack`: run selected attacks

Important behavior:

- `shadow` is reusable if checkpoints already exist
- `pretrain`, `unlearn`, `reminiscence`, and `attack` can be rerun per seed/sample setup
- REA requires `reminiscence`; other attacks do not

## Supported Unlearning Methods

Common methods currently wired into the benchmark:

- `retrain`
- `finetune`
- `negative_grad`
- `scrub`
- `muse`

## Supported Attacks

Current benchmark attack adapters:

- `lira`
- `rea`
- `ruli`
- `unlearningleaks`

All four are wired into the benchmark runner. `MUSE` is also connected to these attack pipelines.

## Quick Start

### 1. Run `finetune` on CIFAR-10 and evaluate with REA

Full run:

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages pretrain shadow unlearn reminiscence attack
```

If `pretrain` and `shadow` already exist:

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages unlearn reminiscence attack
```

### 2. Run all four attacks on one method

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn attack
```

### 3. Dry-run a command

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn reminiscence attack \
  --dry-run
```

## Multi-Round Script Example

For CIFAR-10 only, five rounds, four attacks, and five unlearning methods including MUSE:

```bash
PYTHON_BIN=/root/miniconda3/envs/exp/bin/python \
bash ./scripts/run_cifar10_five_rounds_muse.sh
```

This script:

- uses one random seed per round
- uses one shared forget sample per round
- reuses existing shadow checkpoints if available
- reruns pretrain/unlearn/attack each round
- skips failed sub-experiments and continues
- aggregates mean/std across rounds

## Main CLI Parameters

Common `run_benchmark.py` options:

```bash
--dataset Cifar10|Cifar100|TinyImageNet|Cinic10
--methods method or method:para1:para2
--stages pretrain shadow unlearn reminiscence attack
--attacks lira rea ruli unlearningleaks
--seed 1
--forget-perc 0.1
--num-shadow 8
--num-aug 10
--dry-run
```

Attack-specific examples:

- RULI:
  - `--ruli-task selective`
  - `--ruli-shadow-num 8`
  - `--ruli-train-shadow-mode auto`
- UnlearningLeaks:
  - `--unlearningleaks-feature direct_diff`
  - `--unlearningleaks-attack-model lr`
  - `--unlearningleaks-num-shadow 8`

For the full parameter list:

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py --help
```

## MUSE Defaults

Current dataset-specific MUSE defaults are defined in [`config.py`](./config.py):

- `Cifar10`: `lr=4e-4`, `epochs=7`, `lambda_align=1.0`, `lambda_stat=0.5`
- `Cifar100`: `lr=3e-4`, `epochs=8`, `lambda_align=1.0`, `lambda_stat=0.75`
- `Cinic10`: `lr=4e-4`, `epochs=8`, `lambda_align=1.0`, `lambda_stat=0.5`
- `TinyImageNet`: `lr=3e-4`, `epochs=10`, `lambda_align=0.5`, `lambda_stat=0.25`

You can also override parameters explicitly:

```bash
/root/miniconda3/envs/exp/bin/python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods muse:4e-4:7 \
  --attacks rea \
  --stages unlearn reminiscence attack
```

## Outputs

### Unlearning metrics

Per-method unlearning results are stored under:

```text
log_files\model/forget_random_main/<experiment>/unlearning/<method_key>/
```

The TSV file contains:

- `clean_acc` -> `TA`
- `forgetting_acc` -> `UA`
- `remaining_acc` -> `RA`
- `zrf`
- `mia`
- `time`

Note:

- these TSV files are per-method working outputs
- rerunning the same method directory overwrites the TSV

### Attack metrics

Attack results are stored under:

```text
benchmark_results/samplewise/<experiment>/<method_key>/<attack>/seed_<seed>/
```

Typical outputs include:

- `attack_result.json`
- `ruli_summary.json`
- attack logs
- summary arrays such as ROC-related files for LiRA/REA

### Run summary

Top-level benchmark summaries:

```text
benchmark_results/samplewise/<experiment>/run_summary_seed_<seed>.json
```

## Metrics

Legacy unlearning utility metrics:

- `TA`: test accuracy
- `UA`: accuracy on the forget set
- `RA`: accuracy on the retain set

Privacy metrics are attack-specific. The benchmark standardizes outputs such as:

- `auc`
- `accuracy`
- `TPR@0.1%FPR`
- `TPR@1%FPR`
- `TPR@10%FPR`

If you follow the stricter legacy definition of MIA efficacy at fixed `0.1` FPR, the most relevant metric is:

- `TPR@10%FPR`

## Notes

- `REA` requires the additional `reminiscence` stage.
- `RULI` maintains its own internal shadow attack assets, in addition to the shared benchmark shadows.
- `UnlearningLeaks` uses shared benchmark shadows and also builds its own shadow-unlearned assets.
- The project contains several third-party components; the benchmark wrapper is the main entry recommended for new experiments.

## Recommended Citation / Description

If you upload this repository to GitHub, a short description can be:

> A sample-wise machine unlearning benchmark with unified LiRA/REA/RULI/UnlearningLeaks evaluation and MUSE integration.
