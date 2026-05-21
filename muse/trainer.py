from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import torch
import torch.nn.functional as F

from utils.overall_utils import evaluate


@dataclass
class MUSEConfig:
    epochs: int
    lr: float
    lambda_align: float
    lambda_stat: float
    device: str
    model_name: str


def _unpack_batch(batch, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    images, _, labels = batch
    return images.to(device), labels.to(device)


def compute_alignment_loss(
    forget_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    forget_probs = F.softmax(forget_logits, dim=1)
    reference_probs = F.softmax(reference_logits, dim=1)
    return F.mse_loss(forget_probs.mean(dim=0), reference_probs.mean(dim=0))


def _compute_batch_stats(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    max_confidence = probs.max(dim=1).values
    top2_logits = torch.topk(logits, k=2, dim=1).values
    margins = top2_logits[:, 0] - top2_logits[:, 1]
    return torch.stack([max_confidence.mean(), margins.mean()])


def compute_stat_loss(
    forget_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    forget_stats = _compute_batch_stats(forget_logits)
    reference_stats = _compute_batch_stats(reference_logits)
    return F.mse_loss(forget_stats, reference_stats)


def _build_optimizer(model: torch.nn.Module, cfg: MUSEConfig) -> torch.optim.Optimizer:
    if cfg.model_name == "ViT":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)


def train_muse_epoch(
    model: torch.nn.Module,
    retain_loader,
    forget_loader,
    reference_loader,
    optimizer: torch.optim.Optimizer,
    cfg: MUSEConfig,
) -> dict[str, float]:
    model.train()
    running = {
        "total": 0.0,
        "base": 0.0,
        "align": 0.0,
        "stat": 0.0,
        "steps": 0,
    }

    forget_iter = cycle(forget_loader)
    reference_iter = cycle(reference_loader)

    for retain_batch in retain_loader:
        forget_batch = next(forget_iter)
        reference_batch = next(reference_iter)

        x_r, y_r = _unpack_batch(retain_batch, cfg.device)
        x_f, _ = _unpack_batch(forget_batch, cfg.device)
        x_t, _ = _unpack_batch(reference_batch, cfg.device)

        retain_logits = model(x_r)
        forget_logits = model(x_f)
        reference_logits = model(x_t)

        loss_base = F.cross_entropy(retain_logits, y_r)
        loss_align = compute_alignment_loss(forget_logits, reference_logits)
        loss_stat = compute_stat_loss(forget_logits, reference_logits)
        loss_total = (
            loss_base
            + cfg.lambda_align * loss_align
            + cfg.lambda_stat * loss_stat
        )

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        running["total"] += float(loss_total.detach().item())
        running["base"] += float(loss_base.detach().item())
        running["align"] += float(loss_align.detach().item())
        running["stat"] += float(loss_stat.detach().item())
        running["steps"] += 1

    steps = max(1, running["steps"])
    return {
        "train_loss": running["total"] / steps,
        "base_loss": running["base"] / steps,
        "align_loss": running["align"] / steps,
        "stat_loss": running["stat"] / steps,
    }


def run_muse_unlearning(
    model: torch.nn.Module,
    retain_train_dataloader,
    forget_train_dataloader,
    reference_nonmember_dataloader,
    valid_dataloader,
    *,
    epochs: int,
    lr: float,
    lambda_align: float,
    lambda_stat: float,
    device: str,
    model_name: str,
) -> torch.nn.Module:
    cfg = MUSEConfig(
        epochs=epochs,
        lr=lr,
        lambda_align=lambda_align,
        lambda_stat=lambda_stat,
        device=device,
        model_name=model_name,
    )
    optimizer = _build_optimizer(model, cfg)

    for epoch in range(epochs):
        stats = train_muse_epoch(
            model,
            retain_train_dataloader,
            forget_train_dataloader,
            reference_nonmember_dataloader,
            optimizer,
            cfg,
        )
        val_stats = evaluate(model, valid_dataloader, device)
        print(
            "Epoch [{}], train_loss: {:.4f}, base_loss: {:.4f}, align_loss: {:.4f}, stat_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                stats["train_loss"],
                stats["base_loss"],
                stats["align_loss"],
                stats["stat_loss"],
                val_stats["Loss"],
                val_stats["Acc"],
            )
        )

    return model
