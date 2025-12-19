# 超参数配置
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

@dataclass
class Config:
    # 数据
    train_jsonl: str = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\processed\train.jsonl" # 训练集
    dev_jsonl: Optional[str] = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\processed\dev.jsonl" # 验证集
    tokenizer_name: str = "bert-base-chinese"
    max_length: int = 128
    drop_length_mismatch: bool = False

    # 模型
    bert_name: str = "bert-base-chinese"
    gru_hidden: int = 256
    mask_token_id: int = 103
    dropout: float = 0.1
    init_ckpt: Optional[str] = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\checkpoints\ckpt_best.pt"

    # 优化器
    epochs: int = 15
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1  # or set warmup_steps explicitly if you prefer
    num_workers: int = 2

    # loss:
    lambda_corr: float = 0.8  # L = λ Lc + (1-λ) Ld 总损失权重
    # detection 的正负样本通常极不平衡，可选 pos_weight（>1 提升正类权重）
    detect_pos_weight: float = 20.0
    corr_error_weight: float = 12.0  # 方案B中，纠正错误位置的权重，>=1

    # 训练
    seed: int = 42
    fp16: bool = True
    log_every: int = 50
    eval_every: int = 0  # 0 表示每个 epoch 结束评估；>0 表示每 N steps 评估

    # 输出
    output_dir: str = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\final_model" 
    save_best: bool = True
    metric_for_best: str = "dev_corr_acc_on_errors"  # dev_loss / dev_corr_acc / dev_det_f1

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Soft-Masked BERT training")
    paired_bools = {"fp16"}
    for k, v in asdict(Config()).items():
        if k in paired_bools:
           # 先给默认值
           p.set_defaults(**{k: v})
           # 开启
           p.add_argument(f"--{k}", dest=k, action="store_true", help=f"Enable {k}")
           # 关闭
           p.add_argument(f"--no_{k}", dest=k, action="store_false", help=f"Disable {k}")
           continue
        arg_type = type(v) if v is not None else str
        if isinstance(v, bool):
            # bool 用 flag 处理
            p.add_argument(f"--{k}", action="store_true" if v is False else "store_false")
        else:
            p.add_argument(f"--{k}", type=arg_type, default=v)
    return p

def load_config_from_args() -> Config:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = Config(**vars(args))
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    return cfg

