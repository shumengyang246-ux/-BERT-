# 对模型先进行预训练
from __future__ import annotations
import os
import json
import argparse
from dataclasses import asdict
from typing import Optional

from config import Config, build_argparser
from train import train as train_one_stage


def estimate_detect_pos_weight(jsonl_path: str, max_lines: Optional[int] = None) -> float:
    """
    估计 pos_weight = #neg / #pos，用于 BCEWithLogitsLoss(pos_weight=...)。
    """
    pos = 0
    neg = 0
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            det = obj.get("detection_labels")
            src = obj.get("src")
            tgt = obj.get("tgt")
            if not isinstance(det, list) or not isinstance(src, str) or not isinstance(tgt, str):
                continue
            # 要求严格对齐（你的 dataset 里也会这样过滤）
            if len(src) != len(tgt) or len(det) != len(src):
                continue

            p = sum(1 for x in det if int(x) == 1)
            pos += p
            neg += (len(det) - p)
            n += 1
            if max_lines is not None and n >= max_lines:
                break

    if pos <= 0:
        return 1.0
    return float(neg) / float(pos)


def build_cfg_from_args(args: argparse.Namespace) -> Config:
    """
    只把 Config dataclass 里定义过的字段灌进 Config，
    避免把脚本新增参数（如 auto_pos_weight 等）塞进去导致 TypeError。
    """
    cfg_kwargs = {}
    for k in Config.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if hasattr(args, k):
            cfg_kwargs[k] = getattr(args, k)
    return Config(**cfg_kwargs)


def main():
    # 1) 先拿到 Config 的训练参数（build_argparser 会暴露 Config 的字段）
    parser = build_argparser()

    # 2) 预训练脚本额外参数（不进入 Config）
    parser.add_argument(
    "--train_jsonl_path",
    type=str,
    default=r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\final_pretrain_train.jsonl",
    help="Pretrain train jsonl path"
)
    parser.add_argument(
    "--dev_jsonl_path",
    type=str,
    default=r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\final_pretrain_dev.jsonl",
    help="Pretrain dev jsonl path"
)

    parser.add_argument("--auto_pos_weight", action="store_true", help="Auto estimate detect_pos_weight from train jsonl")
    parser.add_argument("--pos_weight_cap", type=float, default=100.0, help="Cap for auto pos_weight")
    parser.add_argument("--pos_weight_max_lines", type=int, default=50000, help="Max lines to scan for pos_weight estimation")

    args = parser.parse_args()

    # 3) 构造 cfg（只取 Config 字段）
    cfg = build_cfg_from_args(args)

    # 4) 覆盖 train/dev 路径为你已经划分好的预训练集
    cfg.train_jsonl = args.train_jsonl_path
    cfg.dev_jsonl = args.dev_jsonl_path

    # 5) pos_weight：默认用 config.py 里的 cfg.detect_pos_weight（你设为 10.0）
    #    只有显式传了 --auto_pos_weight 才会用估计值覆盖
    if args.auto_pos_weight:
        w = estimate_detect_pos_weight(cfg.train_jsonl, max_lines=args.pos_weight_max_lines)
        w = max(1.0, min(float(args.pos_weight_cap), w))
        print(f"[auto_pos_weight] estimated detect_pos_weight={w:.3f} (override config value {cfg.detect_pos_weight})")
        cfg.detect_pos_weight = w
    else:
        print(f"[pos_weight] use detect_pos_weight from config: {cfg.detect_pos_weight}")

    # 6) 将最终运行配置落盘（确保记录的 train/dev 路径与 pos_weight 都是真实生效的）
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # 7) 开始预训练
    train_one_stage(cfg)


if __name__ == "__main__":
    main()
