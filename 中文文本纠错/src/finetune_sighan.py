# 使用SIGHAN数据集进行第二阶段的训练
from __future__ import annotations
import os
import json
from dataclasses import asdict

from config import Config, build_argparser
from train import train

def main():
    parser = build_argparser()

    # 额外加两个“更直观”的参数名（也可以不用，直接用 --train_jsonl / --dev_jsonl）
    parser.add_argument("--sighan_train_jsonl", type=str, default=r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\processed\train.jsonl", help="SIGHAN train jsonl")
    parser.add_argument("--sighan_dev_jsonl", type=str, default=r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\processed\dev.jsonl",help="SIGHAN dev jsonl")
    #parser.add_argument("--init_ckpt", type=str, default=r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\checkpoints\ckpt_best.pt",help="Checkpoint path to initialize model weights")

    args = parser.parse_args()

    # 只取 Config 字段构造 cfg（避免额外参数塞进 Config）
    cfg_kwargs = {}
    for k in Config.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if hasattr(args, k):
            cfg_kwargs[k] = getattr(args, k)
    cfg = Config(**cfg_kwargs)

    # 覆盖为 SIGHAN 数据
    cfg.train_jsonl = args.sighan_train_jsonl
    cfg.dev_jsonl = args.sighan_dev_jsonl

    # 初始化 ckpt
    cfg.init_ckpt = args.init_ckpt

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config_finetune.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    train(cfg)

if __name__ == "__main__":
    main()
