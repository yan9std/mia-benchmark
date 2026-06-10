import json
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

plt.rcParams["font.sans-serif"] = ["SimHei"]   # Windows 常用黑体
plt.rcParams["axes.unicode_minus"] = False     # 解决负号显示问题

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_shared_resnet18():
    resnet_path = REPO_ROOT / "models" / "resnet.py"
    spec = importlib.util.spec_from_file_location("shared_main_resnet", resnet_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    def build(num_classes=10):
        return module.ResNet(module.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return build


def repo_path(maybe_windows_path: str) -> Path:
    candidates = [
        REPO_ROOT / Path(maybe_windows_path.replace("\\", "/")),
        REPO_ROOT / Path(maybe_windows_path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_lira_or_rea_curve(result_json: Path):
    payload = load_json(result_json)
    fpr = np.load(repo_path(payload["artifacts"]["fpr_npy"]))
    tpr = np.load(repo_path(payload["artifacts"]["tpr_npy"]))
    label = f'{payload["attack_name"].upper()} (AUC={payload["metrics"]["auc"]:.3f})'
    return fpr, tpr, label


def load_unlearningleaks_curve(result_dir: Path):
    arrays = np.load(result_dir / "unlearningleaks_arrays.npz")
    labels = arrays["test_y"]
    scores = arrays["test_scores"]
    fpr, tpr, _ = roc_curve(labels, scores)
    curve_auc = auc(fpr, tpr)
    return fpr, tpr, f"UnlearningLeaks (AUC={curve_auc:.3f})"


def load_ruli_curve(result_json: Path):
    sys.path.insert(0, str(REPO_ROOT / "Ruli/core"))
    from attack.unlearn_attack import TargetModelEvaluator
    from utils.loader import mul_loader
    import torch
    from types import SimpleNamespace

    payload = load_json(result_json)
    saved_results = payload["artifacts"]["ruli_shadow_results"]
    seed = int(result_json.parent.name.split("_")[-1])

    args = SimpleNamespace(
        dataset="cifar10",
        task="selective",
        forget_size=5000,
        forget_index_path=None,
        output_type="logit",
        device="cuda" if torch.cuda.is_available() else "cpu",
        trained_model_path=None,
        arch="resnet18",
        seed=seed,
        attack_size=24000,
        train_epochs=50,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        num_workers=2,
        batch_size=128,
        checkpoint_dir="./checkpoint/cifar10",
        forget_label=0,
        shadow_num=8,
        result_path="../attack/attack_inferences",
        train_shadow=False,
        saved_results=saved_results,
        target_model_path=None,
        vulnerable_path="./attack/cifar10",
        privacy_path="./data/cifar10",
        return_accuracy=False,
        test_batch_size=128,
        config_path="./unlearn_config.json",
    )

    split_data = mul_loader.load_mul_data(
        args.dataset,
        args.task,
        f_label=5,
        forget_size=args.forget_size,
        forget_indices_path=args.forget_index_path,
    )
    target_data = split_data["forget"]
    with open(saved_results, "rb") as f:
        shadow_results = torch.load(f, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed)
    total_indices = torch.randperm(len(target_data))
    split_point = len(target_data) // 3
    in_indices = total_indices[:split_point].tolist()
    out_indices = total_indices[split_point : 2 * split_point].tolist()
    unlearned_indices = total_indices[2 * split_point :].tolist()

    SharedResNet18 = load_shared_resnet18()
    target_model = SharedResNet18(num_classes=10).to(args.device)
    target_unlearned_model = SharedResNet18(num_classes=10).to(args.device)
    target_model.load_state_dict(
        torch.load(
            REPO_ROOT / r"log_files\model/pretrain/ResNet18-Cifar10-10/best.pth",
            map_location=args.device,
        )
    )
    target_unlearned_model.load_state_dict(
        torch.load(
            REPO_ROOT / r"log_files\model/forget_random_main/ResNet18-Cifar10-10/unlearning/finetune_0.11_10/1-last.pth",
            map_location=args.device,
        )
    )

    evaluator = TargetModelEvaluator(
        target_model,
        target_unlearned_model,
        target_data,
        shadow_results,
        in_indices,
        unlearned_indices,
        out_indices,
        args,
    )
    sample_likelihoods, _, _, _ = evaluator.evaluate_sample_likelihood()
    scores = []
    labels = []
    for idx in out_indices + unlearned_indices:
        rec = sample_likelihoods[idx]
        score = rec["unl_likelihood"] / (rec["unl_likelihood"] + rec["unl_out_likelihood"])
        scores.append(float(score))
        labels.append(1 if idx in unlearned_indices else 0)
    fpr, tpr, _ = roc_curve(labels, scores)
    curve_auc = auc(fpr, tpr)
    return fpr, tpr, f"RULI (AUC={curve_auc:.3f})"


def main():
    base_dir = (
        REPO_ROOT
        / "benchmark_results/samplewise/ResNet18-Cifar10-10/finetune_0.11_10"
    )
    seed = "seed_1"

    curves = [
        load_lira_or_rea_curve(base_dir / "lira" / seed / "attack_result.json"),
        load_lira_or_rea_curve(base_dir / "rea" / seed / "attack_result.json"),
        load_unlearningleaks_curve(base_dir / "unlearningleaks" / seed),
        load_ruli_curve(base_dir / "ruli" / seed / "ruli_summary.json"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=180)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for (fpr, tpr, label), color in zip(curves, colors):
        ax.plot(fpr, tpr, linewidth=2.2, color=color, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#888888")
    ax.set_title("CIFAR-10 | Finetune | Seed 1", fontsize=14, pad=10)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    note = (
        "Preview ROC using stored benchmark artifacts.\n"
        "Attacks shown: LiRA, REA, UnlearningLeaks, RULI."
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    out_dir = REPO_ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cifar10_finetune_seed1_roc_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
