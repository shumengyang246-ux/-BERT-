# 训练模型脚本
from __future__ import annotations
import os
import math
import json
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from config import load_config_from_args, Config
from dataset import SoftMaskedJsonlDataset, collate_softmasked_batch, IGNORE_INDEX
from model import SoftMaskedBertForCSC
from functools import partial

# -------------------------
# 1) 辅助函数
# -------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


# -------------------------
# 2) 损失计算
# -------------------------
def compute_losses(
    corr_logits: torch.Tensor,        # [B, L, V]
    detect_logits: torch.Tensor,      # [B, L]
    labels_correct: torch.Tensor,     # [B, L]
    labels_detect: torch.Tensor,      # [B, L]
    attention_mask: torch.Tensor,     # [B, L]
    lambda_corr: float,
    detect_pos_weight: float = 20.0,
    corr_error_weight: float = 12.0,  # 新增：方案B权重，>=1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lc: correction CE (ignore_index=-100)
    Ld: detection BCEWithLogits (manual masking + optional pos_weight)
    L : lambda_corr * Lc + (1-lambda_corr) * Ld
    """
    # 纠错损失
    ce_none = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    per_tok = ce_none(
        corr_logits.view(-1, corr_logits.size(-1)),
        labels_correct.view(-1)
    ).view(labels_correct.size(0), labels_correct.size(1))  # [B, L]

    valid_corr = (labels_correct != IGNORE_INDEX) & (attention_mask == 1)
    if valid_corr.any():
        w = torch.ones_like(per_tok)
        w = w + (corr_error_weight - 1.0) * (labels_detect == 1).float()  # 错误位放大
        # 建议用“加权平均”，保持loss量纲稳定
        Lc = (per_tok[valid_corr] * w[valid_corr]).sum() / (w[valid_corr].sum() + 1e-12)
    else:
        Lc = corr_logits.new_tensor(0.0)

    # 检测损失
    valid = (labels_detect != IGNORE_INDEX) & (attention_mask == 1)
    if valid.any():
        pos_weight = torch.tensor([detect_pos_weight], device=detect_logits.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        Ld = bce(detect_logits[valid], labels_detect[valid].float())
    else:
        Ld = detect_logits.new_tensor(0.0)

    loss = lambda_corr * Lc + (1.0 - lambda_corr) * Ld
    return loss, Lc, Ld


# -------------------------
# 3) 用于快速调试的指标
# -------------------------
@torch.no_grad()
def compute_det_metrics(detect_logits: torch.Tensor, labels_detect: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
    """
    token-level detection precision/recall/f1 with threshold 0.5
    """
    valid = (labels_detect != IGNORE_INDEX) & (attention_mask == 1)
    if not valid.any():
        return {"det_p": 0.0, "det_r": 0.0, "det_f1": 0.0}

    pred = (torch.sigmoid(detect_logits[valid]) > 0.5).long()
    gold = labels_detect[valid].long()

    tp = int(((pred == 1) & (gold == 1)).sum().item())
    fp = int(((pred == 1) & (gold == 0)).sum().item())
    fn = int(((pred == 0) & (gold == 1)).sum().item())

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    return {"det_p": float(p), "det_r": float(r), "det_f1": float(f1)}


@torch.no_grad()
def compute_corr_acc(corr_logits: torch.Tensor, labels_correct: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """
    token-level correction accuracy (excluding IGNORE_INDEX)
    """
    valid = (labels_correct != IGNORE_INDEX) & (attention_mask == 1)
    if not valid.any():
        return 0.0
    pred = corr_logits.argmax(dim=-1)
    acc = (pred[valid] == labels_correct[valid]).float().mean().item()
    return float(acc)

@torch.no_grad() # 计算在错误位置的纠正准确率
def compute_corr_acc_on_errors(
    corr_logits: torch.Tensor,
    labels_correct: torch.Tensor,
    labels_detect: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    valid_err = (
        (labels_correct != IGNORE_INDEX)
        & (attention_mask == 1)
        & (labels_detect == 1)
    )
    if not valid_err.any():
        return 0.0
    pred = corr_logits.argmax(dim=-1)
    acc = (pred[valid_err] == labels_correct[valid_err]).float().mean().item()
    return float(acc)

# -------------------------
# 4) 评估循环
# -------------------------
@torch.no_grad()
def evaluate(model: SoftMaskedBertForCSC, dl: DataLoader, cfg: Config, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_lc, total_ld = 0.0, 0.0, 0.0
    total_corr_acc = 0.0
    # detection counts aggregate for F1
    tp = fp = fn = 0
    n_batches = 0
    # 新增计数器
    total_corr_acc_on_errors = 0.0
    valid_error_batches = 0  # 记录有多少个batch里确实包含错误样本

    for batch in dl:
        batch = to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        loss, lc, ld = compute_losses(
            corr_logits=out.corr_logits,
            detect_logits=out.detect_logits,
            labels_correct=batch["labels_correct"],
            labels_detect=batch["labels_detect"],
            attention_mask=batch["attention_mask"],
            lambda_corr=cfg.lambda_corr,
            detect_pos_weight=cfg.detect_pos_weight,
            corr_error_weight=cfg.corr_error_weight,  # 新增：方案B权重
        )

        total_loss += float(loss.item())
        total_lc += float(lc.item())
        total_ld += float(ld.item())
        total_corr_acc += compute_corr_acc(out.corr_logits, batch["labels_correct"], batch["attention_mask"])
        acc_on_err = compute_corr_acc_on_errors(
            out.corr_logits, 
            batch["labels_correct"], 
            batch["labels_detect"], 
            batch["attention_mask"]
        )
        
        valid_err_mask = (batch["labels_detect"] == 1) & (batch["labels_correct"] != IGNORE_INDEX)
        if valid_err_mask.any():
            total_corr_acc_on_errors += acc_on_err
            valid_error_batches += 1

        # 累积检测混淆
        valid = (batch["labels_detect"] != IGNORE_INDEX) & (batch["attention_mask"] == 1)
        if valid.any():
            pred = (torch.sigmoid(out.detect_logits[valid]) > 0.5).long()
            gold = batch["labels_detect"][valid].long()
            tp += int(((pred == 1) & (gold == 1)).sum().item())
            fp += int(((pred == 1) & (gold == 0)).sum().item())
            fn += int(((pred == 0) & (gold == 1)).sum().item())

        n_batches += 1

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    model.train()
    return {
        "dev_loss": total_loss / max(n_batches, 1),
        "dev_lc": total_lc / max(n_batches, 1),
        "dev_ld": total_ld / max(n_batches, 1),
        "dev_corr_acc": total_corr_acc / max(n_batches, 1),
        "dev_det_f1": float(f1),
        "dev_det_p": float(p),
        "dev_det_r": float(r),
        "dev_corr_acc_on_errors":total_corr_acc_on_errors / max(valid_error_batches, 1),
    }

# -------------------------
# 5) 训练循环
# -------------------------
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset / dataloader ----
    train_ds = SoftMaskedJsonlDataset(
        jsonl_path=cfg.train_jsonl,
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
        drop_length_mismatch=cfg.drop_length_mismatch,
        return_text=False,
    )
    pad_id = train_ds.tokenizer.pad_token_id
    
    collate_fn = partial(collate_softmasked_batch, pad_token_id=pad_id)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    dev_dl = None
    if cfg.dev_jsonl:
        dev_ds = SoftMaskedJsonlDataset(
            jsonl_path=cfg.dev_jsonl,
            tokenizer_name=cfg.tokenizer_name,
            max_length=cfg.max_length,
            drop_length_mismatch=cfg.drop_length_mismatch,
            return_text=False,
        )
        dev_dl = DataLoader(
            dev_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )

    # ---- model ----
    model = SoftMaskedBertForCSC(
        bert_name_or_path=cfg.bert_name,
        gru_hidden=cfg.gru_hidden,
        mask_token_id=cfg.mask_token_id,
        dropout=cfg.dropout,
    ).to(device)
    
    # ---- 加载预训练的模型权重 ----
    if getattr(cfg, "init_ckpt", None):
        ckpt_path = cfg.init_ckpt
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"init_ckpt not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init_ckpt] loaded weights from: {ckpt_path}")
        if missing:
            print(f"[init_ckpt] missing keys: {len(missing)}")
        if unexpected:
            print(f"[init_ckpt] unexpected keys: {len(unexpected)}")

    # 你的模型 forward 结构：embedding -> detector -> softmask -> corrector 
    # ---- optimizer / scheduler ----
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(train_dl) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- fp16 ----
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.fp16))

    # ---- logging / checkpointing ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")

    best_score = None
    global_step = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{cfg.epochs}")
        running = {"loss": 0.0, "lc": 0.0, "ld": 0.0}
        for step, batch in enumerate(pbar, start=1):
            batch = to_device(batch, device)

            with torch.amp.autocast("cuda", enabled=bool(cfg.fp16)):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )
                loss, lc, ld = compute_losses(
                    corr_logits=out.corr_logits,
                    detect_logits=out.detect_logits,
                    labels_correct=batch["labels_correct"],
                    labels_detect=batch["labels_detect"],
                    attention_mask=batch["attention_mask"],
                    lambda_corr=cfg.lambda_corr,
                    detect_pos_weight=cfg.detect_pos_weight,
                    corr_error_weight=cfg.corr_error_weight,
                )
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running["loss"] += float(loss.item()) * cfg.grad_accum_steps
            running["lc"] += float(lc.item())
            running["ld"] += float(ld.item())

            if global_step > 0 and global_step % cfg.log_every == 0:
                corr_acc = compute_corr_acc(out.corr_logits, batch["labels_correct"], batch["attention_mask"])
                corr_acc_only = compute_corr_acc_on_errors(
    out.corr_logits, batch["labels_correct"], batch["labels_detect"], batch["attention_mask"]
)
                det_m = compute_det_metrics(out.detect_logits, batch["labels_detect"], batch["attention_mask"])
                pbar.set_postfix({
                    "loss": running["loss"] / max(1, cfg.log_every),
                    "Lc": running["lc"] / max(1, cfg.log_every),
                    "Ld": running["ld"] / max(1, cfg.log_every),
                    "corr_acc": corr_acc,
                    "det_f1": det_m["det_f1"],
                })

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": running["loss"] / max(1, cfg.log_every),
                        "train_lc": running["lc"] / max(1, cfg.log_every),
                        "train_ld": running["ld"] / max(1, cfg.log_every),
                        "train_corr_acc": corr_acc,
                        "train_corr_acc_on_errors": corr_acc_only,
                        **{f"train_{k}": v for k, v in det_m.items()},
                        "lr": float(scheduler.get_last_lr()[0]),
                    }, ensure_ascii=False) + "\n")

                running = {"loss": 0.0, "lc": 0.0, "ld": 0.0}

        # ---- epoch-end eval ----
        dev_metrics = {}
        if dev_dl is not None:
            dev_metrics = evaluate(model, dev_dl, cfg, device)
            print(f"[dev] {dev_metrics}")

        # ---- save checkpoint ----
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        }
        last_path = os.path.join(cfg.output_dir, "ckpt_last.pt")
        torch.save(ckpt, last_path)

        # ---- save best ----
        if dev_dl is not None and cfg.save_best:
            key = cfg.metric_for_best
            score = dev_metrics.get(key)
            if score is None:
                raise KeyError(
                 f"[save_best] '{key}' not found in dev_metrics. "
                 f"Available keys: {sorted(dev_metrics.keys())}"
                 )

            is_better = (best_score is None) or (score > best_score)

            if is_better:
                best_score = score
                best_path = os.path.join(cfg.output_dir, "ckpt_best.pt")
                torch.save(ckpt, best_path)
                print(f"[save_best] {key}={score:.6f} -> {best_path}")

    print("Training finished.")


if __name__ == "__main__":
    cfg = load_config_from_args()
    train(cfg)


