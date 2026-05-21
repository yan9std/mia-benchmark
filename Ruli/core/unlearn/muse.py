import time
from itertools import cycle

import torch
import torch.nn.functional as F
from torch import nn

from utils.average_meter import AverageMeter


def _top1_top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def compute_alignment_loss(forget_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    forget_probs = F.softmax(forget_logits, dim=1)
    reference_probs = F.softmax(reference_logits, dim=1)
    return F.mse_loss(forget_probs.mean(dim=0), reference_probs.mean(dim=0))


def compute_stat_loss(forget_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    forget_probs = F.softmax(forget_logits, dim=1)
    reference_probs = F.softmax(reference_logits, dim=1)

    forget_stats = torch.stack(
        [
            forget_probs.max(dim=1).values.mean(),
            _top1_top2_margin(forget_logits).mean(),
        ]
    )
    reference_stats = torch.stack(
        [
            reference_probs.max(dim=1).values.mean(),
            _top1_top2_margin(reference_logits).mean(),
        ]
    )
    return F.mse_loss(forget_stats, reference_stats)


class MUSE:
    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.forget_loader = LOADER_DICT["forget"]
        self.retain_loader = LOADER_DICT["remain"]
        self.reference_loader = LOADER_DICT["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.lambda_align = float(getattr(args, "lambda_align", 1.0))
        self.lambda_stat = float(getattr(args, "lambda_stat", 0.5))
        self.args = args

        optimizer_name = str(getattr(args, "optimizer", "sgd")).lower()
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif optimizer_name == "adamw" or getattr(args, "arch", "") == "vit":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.num_epochs)
        )

    def _run_epoch(self, epoch: int):
        self.model.train()
        total_meter = AverageMeter()
        base_meter = AverageMeter()
        align_meter = AverageMeter()
        stat_meter = AverageMeter()
        correct = 0
        total = 0
        start = time.time()

        forget_iter = cycle(self.forget_loader)
        reference_iter = cycle(self.reference_loader)

        for retain_batch in self.retain_loader:
            x_r, y_r = retain_batch
            x_f, y_f = next(forget_iter)
            x_t, y_t = next(reference_iter)
            del y_f, y_t

            x_r = x_r.to(self.device)
            y_r = y_r.to(self.device)
            x_f = x_f.to(self.device)
            x_t = x_t.to(self.device)

            logits_r = self.model(x_r)
            logits_f = self.model(x_f)
            logits_t = self.model(x_t)

            loss_base = self.criterion(logits_r, y_r)
            loss_align = compute_alignment_loss(logits_f, logits_t)
            loss_stat = compute_stat_loss(logits_f, logits_t)
            loss = loss_base + self.lambda_align * loss_align + self.lambda_stat * loss_stat

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = y_r.size(0)
            total_meter.update(loss.item(), batch_size)
            base_meter.update(loss_base.item(), batch_size)
            align_meter.update(loss_align.item(), batch_size)
            stat_meter.update(loss_stat.item(), batch_size)

            pred = logits_r.argmax(dim=1)
            total += batch_size
            correct += pred.eq(y_r).sum().item()

        self.scheduler.step()
        elapsed = time.time() - start
        train_acc = 100.0 * correct / max(1, total)
        print(
            "Epoch [{}], train_loss: {:.4f}, base_loss: {:.4f}, align_loss: {:.4f}, "
            "stat_loss: {:.4f}, train_acc: {:.4f}, time: {:.2f}s".format(
                epoch,
                total_meter.avg,
                base_meter.avg,
                align_meter.avg,
                stat_meter.avg,
                train_acc,
                elapsed,
            )
        )

    def unlearn(self):
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch)
        return self.model
