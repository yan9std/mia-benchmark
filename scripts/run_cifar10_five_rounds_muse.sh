#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/exp/bin/python}"
FORGET_PERC="${FORGET_PERC:-0.1}"
ROUNDS="${ROUNDS:-5}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/experiment_runs/cifar10_five_rounds_muse}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/log_files\\model}"
SHADOW_SEED="${SHADOW_SEED:-1}"
BENCHMARK_DRY_RUN="${BENCHMARK_DRY_RUN:-0}"

DATASET="Cifar10"
EXPERIMENT_NAME="ResNet18-Cifar10-10"
ATTACKS=("lira" "rea" "ruli" "unlearningleaks")
METHOD_SPECS=(
  "retrain:0.1:150"
  "finetune:0.11:10"
  "negative_grad:0.04:8"
  "scrub:0.0004:7"
  "muse:4e-4:7"
)

mkdir -p "$RESULT_ROOT"

benchmark_dry_run_args() {
  if [[ "$BENCHMARK_DRY_RUN" == "1" ]]; then
    printf '%s\n' "--dry-run"
  fi
}

methods_csv() {
  local IFS=,
  echo "${METHOD_SPECS[*]}"
}

cleanup_round_artifacts() {
  local seed="$1"

  rm -rf "$CHECKPOINT_ROOT/pretrain/$EXPERIMENT_NAME"
  for method_spec in "${METHOD_SPECS[@]}"; do
    local method_name para1 para2 method_key
    IFS=':' read -r method_name para1 para2 <<< "$method_spec"
    method_key="${method_name}_${para1}_${para2}"
    rm -rf "$CHECKPOINT_ROOT/forget_random_main/$EXPERIMENT_NAME/unlearning/$method_key"
    rm -rf "$CHECKPOINT_ROOT/forget_random_main/$EXPERIMENT_NAME/reminisence/$method_key"
    for attack in "${ATTACKS[@]}"; do
      rm -rf "$ROOT_DIR/benchmark_results/samplewise/$EXPERIMENT_NAME/$method_key/$attack/seed_$seed"
    done
  done
}

ensure_shadow_once() {
  local log_file="$RESULT_ROOT/shadow_${DATASET}.log"
  if ! "$PYTHON_BIN" "$ROOT_DIR/run_benchmark.py" \
    --dataset "$DATASET" \
    --seed "$SHADOW_SEED" \
    --stages shadow \
    $(benchmark_dry_run_args) \
    2>&1 | tee "$log_file"; then
    echo "[warn] shadow stage failed or partially completed: dataset=$DATASET" | tee -a "$log_file"
  fi
}

generate_round_seed() {
  "$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.randbelow(10**9))
PY
}

generate_forget_index() {
  local seed="$1"
  local round_dir="$2"
  ROUND_SEED="$seed" FORGET_PERC="$FORGET_PERC" ROOT_DIR="$ROOT_DIR" ROUND_DIR="$round_dir" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np

seed = int(os.environ["ROUND_SEED"])
forget_perc = float(os.environ["FORGET_PERC"])
root_dir = Path(os.environ["ROOT_DIR"])
round_dir = Path(os.environ["ROUND_DIR"])

sys.path.insert(0, str(root_dir))

import config
import datasets

dataset_name = "Cifar10"
spec = {"net": "ResNet18", "classes": 10, "img_size": 32}

trainset = getattr(datasets, dataset_name)(
    root=str(root_dir / "data"),
    download=True,
    train=True,
    unlearning=True,
    img_size=spec["img_size"],
)

forget_size = int(len(trainset) * forget_perc)
rng = np.random.default_rng(seed)
indices = np.sort(rng.choice(len(trainset), size=forget_size, replace=False))

index_dir = root_dir / Path(config.CHECKPOINT_PATH) / "forget_random_main" / f'{spec["net"]}-{dataset_name}-{spec["classes"]}' / "random_index_set"
index_dir.mkdir(parents=True, exist_ok=True)
index_path = index_dir / f"forgetting_dataset_index_{forget_perc}.npy"
np.save(index_path, indices)

round_index_dir = round_dir / "samples"
round_index_dir.mkdir(parents=True, exist_ok=True)
round_index_path = round_index_dir / f"{dataset_name}_forget_indices.npy"
np.save(round_index_path, indices)

meta = {
    "dataset": dataset_name,
    "seed": seed,
    "forget_perc": forget_perc,
    "train_size": len(trainset),
    "forget_size": forget_size,
    "checkpoint_index_path": str(index_path),
    "round_copy_path": str(round_index_path),
    "first_20_indices": indices[:20].tolist(),
}
meta_path = round_index_dir / f"{dataset_name}_sample_meta.json"
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False))
PY
}

run_round() {
  local seed="$1"
  local round_dir="$2"
  local pretrain_unlearn_log="$round_dir/logs/${DATASET}_pretrain_unlearn.log"
  local reminiscence_log="$round_dir/logs/${DATASET}_reminiscence.log"
  mkdir -p "$(dirname "$pretrain_unlearn_log")"

  {
    echo "[prepare] dataset=$DATASET seed=$seed forget_perc=$FORGET_PERC"
    echo "[prepare] methods=$(methods_csv)"
  } >"$pretrain_unlearn_log"

  cleanup_round_artifacts "$seed"

  if ! "$PYTHON_BIN" "$ROOT_DIR/run_benchmark.py" \
    --dataset "$DATASET" \
    --seed "$seed" \
    --forget-perc "$FORGET_PERC" \
    --methods "${METHOD_SPECS[@]}" \
    --stages pretrain unlearn \
    $(benchmark_dry_run_args) \
    2>&1 | tee -a "$pretrain_unlearn_log"; then
    echo "[warn] pretrain/unlearn stage failed: dataset=$DATASET round_dir=$round_dir" | tee -a "$pretrain_unlearn_log"
  fi

  if ! METHODS_CSV="$(methods_csv)" ROOT_DIR="$ROOT_DIR" ROUND_DIR="$round_dir" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

methods_csv = os.environ["METHODS_CSV"].split(",")
round_dir = Path(os.environ["ROUND_DIR"])
checkpoint_root = Path(os.environ["CHECKPOINT_ROOT"])
dataset_name = "Cifar10"
experiment_name = "ResNet18-Cifar10-10"
log_filename = "log_ResNet18-Cifar10-10.tsv"

for method_spec in methods_csv:
    method_name, para1, para2 = method_spec.split(":")
    method_key = f"{method_name}_{para1}_{para2}"
    src = checkpoint_root / "forget_random_main" / experiment_name / "unlearning" / method_key / log_filename
    if not src.exists():
        print(f"[warn] missing unlearning metrics tsv: {src}")
        continue
    dst_dir = round_dir / "results" / dataset_name / "unlearning_metrics" / method_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / log_filename)

manifest_path = round_dir / "results" / dataset_name / "unlearning_metrics" / "manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "dataset": dataset_name,
    "type": "unlearning_metrics",
    "methods": methods_csv,
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
  then
    echo "[warn] unlearning metrics collection failed: dataset=$DATASET round_dir=$round_dir" | tee -a "$pretrain_unlearn_log"
  fi

  for attack in "${ATTACKS[@]}"; do
    local attack_log="$round_dir/logs/${DATASET}_${attack}.log"

    if [[ "$attack" == "rea" ]]; then
      if ! "$PYTHON_BIN" "$ROOT_DIR/run_benchmark.py" \
        --dataset "$DATASET" \
        --seed "$seed" \
        --forget-perc "$FORGET_PERC" \
        --methods "${METHOD_SPECS[@]}" \
        --stages reminiscence \
        $(benchmark_dry_run_args) \
        2>&1 | tee -a "$reminiscence_log"; then
        echo "[warn] reminiscence stage failed: dataset=$DATASET round_dir=$round_dir" | tee -a "$reminiscence_log"
      fi
    fi

    if ! "$PYTHON_BIN" "$ROOT_DIR/run_benchmark.py" \
      --dataset "$DATASET" \
      --seed "$seed" \
      --forget-perc "$FORGET_PERC" \
      --methods "${METHOD_SPECS[@]}" \
      --attacks "$attack" \
      --stages attack \
      $(benchmark_dry_run_args) \
      2>&1 | tee "$attack_log"; then
      echo "[warn] attack stage failed: dataset=$DATASET attack=$attack round_dir=$round_dir" | tee -a "$attack_log"
      printf '%s\n' "{\"dataset\":\"$DATASET\",\"attack\":\"$attack\",\"seed\":$seed,\"status\":\"failed\"}" > "$round_dir/results/${DATASET}_${attack}_failed.json"
      continue
    fi

    if ! DATASET_NAME="$DATASET" ATTACK_NAME="$attack" ROUND_SEED="$seed" METHODS_CSV="$(methods_csv)" ROOT_DIR="$ROOT_DIR" ROUND_DIR="$round_dir" "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

dataset_name = os.environ["DATASET_NAME"]
attack_name = os.environ["ATTACK_NAME"]
seed = int(os.environ["ROUND_SEED"])
methods_csv = os.environ["METHODS_CSV"].split(",")
root_dir = Path(os.environ["ROOT_DIR"])
round_dir = Path(os.environ["ROUND_DIR"])
experiment_name = "ResNet18-Cifar10-10"

for method_spec in methods_csv:
    method_name, para1, para2 = method_spec.split(":")
    method_key = f"{method_name}_{para1}_{para2}"
    seed_dir = root_dir / "benchmark_results" / "samplewise" / experiment_name / method_key / attack_name / f"seed_{seed}"
    result_name = "ruli_summary.json" if attack_name == "ruli" else "attack_result.json"
    src = seed_dir / result_name
    if not src.exists():
        print(f"[warn] missing attack result: {src}")
        continue
    dst_dir = round_dir / "results" / dataset_name / attack_name / method_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / result_name)

manifest_path = round_dir / "results" / dataset_name / attack_name / "manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "dataset": dataset_name,
    "attack": attack_name,
    "seed": seed,
    "methods": methods_csv,
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
    then
      echo "[warn] result collection failed: dataset=$DATASET attack=$attack round_dir=$round_dir" | tee -a "$attack_log"
    fi
  done
}

aggregate_results() {
  if ! RESULT_ROOT="$RESULT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
records: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
unlearning_records: dict[tuple[str, str], list[dict]] = defaultdict(list)

for round_dir in sorted(result_root.glob("round_*")):
    for pattern in ("results/*/*/*/attack_result.json", "results/*/*/*/ruli_summary.json"):
        for result_path in sorted(round_dir.glob(pattern)):
            dataset = result_path.parts[-4]
            attack = result_path.parts[-3]
            method_key = result_path.parts[-2]
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            records[(attack, method_key, dataset)].append(payload)
    for tsv_path in sorted(round_dir.glob("results/*/unlearning_metrics/*/log_*.tsv")):
        dataset = tsv_path.parts[-4]
        method_key = tsv_path.parts[-2]
        with open(tsv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if rows:
            unlearning_records[(method_key, dataset)].append(rows[-1])

summary = {}
unlearning_summary = {}
lines = []
csv_lines = ["attack,method,dataset,metric,mean,std,n"]
unlearning_csv_lines = ["method,dataset,metric,mean,std,n"]

for key in sorted(records):
    attack, method_key, dataset = key
    payloads = records[key]
    metric_values: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        for metric_name, value in payload.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_values[metric_name].append(float(value))

    combo_summary = {}
    for metric_name, values in sorted(metric_values.items()):
        n = len(values)
        mean = sum(values) / n
        std = 0.0
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        combo_summary[metric_name] = {"mean": mean, "std": std, "n": n}
        csv_lines.append(f"{attack},{method_key},{dataset},{metric_name},{mean:.10f},{std:.10f},{n}")

    summary.setdefault(attack, {}).setdefault(method_key, {})[dataset] = combo_summary

    headline_metrics = []
    for metric_name in ["auc", "accuracy", "TPR@0.1%FPR", "TPR@1%FPR", "TPR@10%FPR"]:
        if metric_name in combo_summary:
            item = combo_summary[metric_name]
            headline_metrics.append(f"{metric_name}={item['mean']:.6f}±{item['std']:.6f} (n={item['n']})")
    if headline_metrics:
        lines.append(f"{attack}-{method_key}-{dataset}: " + ", ".join(headline_metrics))
    else:
        lines.append(f"{attack}-{method_key}-{dataset}: no numeric metrics collected")

for key in sorted(unlearning_records):
    method_key, dataset = key
    rows = unlearning_records[key]
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mapping = {
            "TA": row.get("clean_acc"),
            "UA": row.get("forgetting_acc"),
            "RA": row.get("remaining_acc"),
            "time": row.get("time"),
        }
        for metric_name, value in mapping.items():
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                metric_values[metric_name].append(parsed)

    combo_summary = {}
    for metric_name, values in sorted(metric_values.items()):
        n = len(values)
        mean = sum(values) / n
        std = 0.0
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        combo_summary[metric_name] = {"mean": mean, "std": std, "n": n}
        unlearning_csv_lines.append(f"{method_key},{dataset},{metric_name},{mean:.10f},{std:.10f},{n}")

    unlearning_summary.setdefault(method_key, {})[dataset] = combo_summary
    headline = []
    for metric_name in ["TA", "UA", "RA", "time"]:
        if metric_name in combo_summary:
            item = combo_summary[metric_name]
            headline.append(f"{metric_name}={item['mean']:.6f}±{item['std']:.6f} (n={item['n']})")
    if headline:
        lines.append(f"unlearning-{method_key}-{dataset}: " + ", ".join(headline))
    else:
        lines.append(f"unlearning-{method_key}-{dataset}: no numeric metrics collected")

summary_dir = result_root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
(summary_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(summary_dir / "aggregate_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
(summary_dir / "aggregate_unlearning_metrics.json").write_text(json.dumps(unlearning_summary, indent=2, ensure_ascii=False), encoding="utf-8")
(summary_dir / "aggregate_unlearning_metrics.csv").write_text("\n".join(unlearning_csv_lines) + "\n", encoding="utf-8")
(summary_dir / "aggregate_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
print(f"\n[summary] json={summary_dir / 'aggregate_metrics.json'}")
print(f"[summary] csv={summary_dir / 'aggregate_metrics.csv'}")
print(f"[summary] unlearning_json={summary_dir / 'aggregate_unlearning_metrics.json'}")
print(f"[summary] unlearning_csv={summary_dir / 'aggregate_unlearning_metrics.csv'}")
print(f"[summary] txt={summary_dir / 'aggregate_metrics.txt'}")
PY
  then
    echo "[warn] aggregate_results failed"
  fi
}

main() {
  echo "[config] ROOT_DIR=$ROOT_DIR"
  echo "[config] PYTHON_BIN=$PYTHON_BIN"
  echo "[config] DATASET=$DATASET"
  echo "[config] ROUNDS=$ROUNDS"
  echo "[config] FORGET_PERC=$FORGET_PERC"
  echo "[config] RESULT_ROOT=$RESULT_ROOT"

  echo "[shadow] dataset=$DATASET seed=$SHADOW_SEED"
  ensure_shadow_once

  for ((round_idx = 1; round_idx <= ROUNDS; round_idx++)); do
    local round_dir="$RESULT_ROOT/round_${round_idx}"
    mkdir -p "$round_dir/logs"
    local seed
    seed="$(generate_round_seed)"
    echo "[round] round=${round_idx} seed=${seed}"
    printf '%s\n' "$seed" > "$round_dir/seed.txt"

    echo "[sample] round=${round_idx} dataset=${DATASET} seed=${seed}"
    if ! generate_forget_index "$seed" "$round_dir" | tee "$round_dir/logs/${DATASET}_sample_generation.log"; then
      echo "[warn] sample generation failed: round=${round_idx} dataset=${DATASET} seed=${seed}" | tee -a "$round_dir/logs/${DATASET}_sample_generation.log"
      continue
    fi
    if ! run_round "$seed" "$round_dir"; then
      echo "[warn] round execution failed unexpectedly: round=${round_idx} dataset=${DATASET} seed=${seed}" | tee -a "$round_dir/logs/${DATASET}_pretrain_unlearn.log"
      continue
    fi
  done

  aggregate_results
}

main "$@"
