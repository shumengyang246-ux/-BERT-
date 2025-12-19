"""
合并 SIGHAN13/14/15 简体中文拼写纠错（CSC）训练集。
预期的行格式（稳健型）：
错误文本 \t 正确文本
或者
错误文本 <空格> 正确文本
输出：
merged_train.txt（TSV 格式：错误文本 \t 正确文本）
可选的包含检测标签的 merged_train.jsonl
本脚本与 Soft-Masked BERT 训练数据设计保持一致：
训练对（X，Y）
可选的字符级检测标签 g_i = 1（x_i ≠ y_i）
可选移除未更改的样本对以提高效率
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

def smart_split_line(line: str) -> Optional[Tuple[str, str]]:
    """
    把一行文本拆分为错误文本和正确文本。
    规则：
    1. 按制表符分割，如果有多个制表符，则合并到第二个制表符后面
    2. 按空格分割，如果有多个空格，则合并到第二个空格后面
    """
    line = line.strip()
    if not line:
        return None

    # 1) tab
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) >= 2:
            wrong = parts[0].strip()
            correct = "\t".join(parts[1:]).strip()
            if wrong and correct:
                return wrong, correct

    # 2) 空格
    parts = re.split(r"\s+", line, maxsplit=1)
    if len(parts) == 2:
        wrong, correct = parts[0].strip(), parts[1].strip()
        if wrong and correct:
            return wrong, correct
    return None


def build_detection_labels(src: str, tgt: str) -> Optional[List[int]]:
    """
    检测网络的字符级标签：
    若长度不匹配，则返回 None。
    """
    if len(src) != len(tgt):
        return None
    return [1 if a != b else 0 for a, b in zip(src, tgt)]


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            parsed = smart_split_line(line)
            if parsed is None:
                continue
            wrong, correct = parsed
            pairs.append((wrong, correct))
    return pairs


def main():
    # 设置默认数据目录
    default_data_dir = r"C:\Users\dell\Desktop\chinese-correct\data"
    
    # 检查目录是否存在
    if not os.path.exists(default_data_dir):
        print(f"警告: 数据目录不存在: {default_data_dir}")
        print("请确保目录路径正确，或使用 --data_dir 参数指定正确路径")
        default_data_dir = "."  # 使用当前目录作为备选
    
    parser = argparse.ArgumentParser(description="合并 SIGHAN13/14/15 训练集")
    parser.add_argument(
        "--data_dir",
        default=default_data_dir,
        help=f"数据目录路径，默认为 {default_data_dir}",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["train_13.txt", "train_14.txt", "train_15.txt"],
        help="输入文件名（在data_dir目录下），默认: train_13.txt train_14.txt train_15.txt",
    )
    parser.add_argument(
        "--out_dir",
        default=default_data_dir,
        help="输出文件目录，默认为data_dir",
    )
    parser.add_argument(
        "--out_txt",
        default="merged_train.txt",
        help="输出合并的TSV文件名称",
    )
    parser.add_argument(
        "--out_jsonl",
        default="merged_train.jsonl",
        help="输出jsonl文件名称（可选）",
    )
    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="是否输出包含检测标签的jsonl文件",
    )
    parser.add_argument(
        "--drop_unchanged",
        action="store_true",
        help="删除错误文本和正确文本相同的样本对（论文中提到为提升效率移除未更改文本）",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="去重相同的（错误，正确）样本对（默认启用）",
    )
    parser.add_argument(
        "--strict_length",
        action="store_true",
        help="如果设置，删除长度不匹配的样本对；否则保留在txt中但jsonl中的标签为None",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="输出数据集的详细分析报告",
    )

    args = parser.parse_args()

    # 构建完整文件路径
    input_paths = [Path(args.data_dir) / p for p in args.inputs]
    
    # 检查文件是否存在
    missing_files = [str(p) for p in input_paths if not p.exists()]
    if missing_files:
        print("错误: 以下文件不存在:")
        for f in missing_files:
            print(f"  {f}")
        print(f"\n请检查 --data_dir 参数是否正确 (当前为: {args.data_dir})")
        return

    all_pairs: List[Tuple[str, str]] = []
    per_file_stats = {}

    for p in input_paths:
        print(f"正在加载: {p}")
        pairs = load_pairs(p)
        per_file_stats[str(p)] = {
            "loaded_pairs": len(pairs),
        }
        all_pairs.extend(pairs)
        print(f"  已加载 {len(pairs)} 条数据")

    total_before = len(all_pairs)
    print(f"\n合并前总数据量: {total_before}")

    # 删除未更改的样本
    unchanged_count = sum(1 for w, c in all_pairs if w == c)
    print(f"未更改的样本对数量: {unchanged_count} ({unchanged_count/total_before*100:.1f}%)")
    
    if args.drop_unchanged:
        all_pairs = [(w, c) for (w, c) in all_pairs if w != c]
        print(f"删除未更改样本后: {len(all_pairs)} 条数据")

    total_after_drop = len(all_pairs)

    # 去重
    if args.dedup:
        # 保留顺序
        seen = set()
        deduped = []
        for w, c in all_pairs:
            key = (w, c)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        all_pairs = deduped
        print(f"去重后: {len(all_pairs)} 条数据")

    total_after_dedup = len(all_pairs)

    # 确保输出目录存在
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入合并的txt文件（始终使用TSV格式）
    out_txt_path = out_dir / args.out_txt
    with out_txt_path.open("w", encoding="utf-8") as f:
        for w, c in all_pairs:
            f.write(f"{w}\t{c}\n")

    print(f"\n已保存合并数据到: {out_txt_path}")
    print(f"最终数据量: {len(all_pairs)} 条")

    # 可选：写入带有检测标签的jsonl文件
    if args.write_jsonl:
        out_jsonl_path = out_dir / args.out_jsonl
        kept = 0
        dropped_len_mismatch = 0
        length_mismatch_count = 0

        with out_jsonl_path.open("w", encoding="utf-8") as f:
            for w, c in all_pairs:
                labels = build_detection_labels(w, c)
                if labels is None:
                    length_mismatch_count += 1
                    if args.strict_length:
                        dropped_len_mismatch += 1
                        continue

                obj = {
                    "src": w,
                    "tgt": c,
                    "length_src": len(w),
                    "length_tgt": len(c),
                    "is_changed": w != c,
                    "detection_labels": labels,  # None if length mismatch (unless strict_length)
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

        print(f"\n[JSONL] 已保存到: {out_jsonl_path}")
        print(f"[JSONL] 长度不匹配的样本对数量: {length_mismatch_count}")
        if args.strict_length:
            print(f"[JSONL] 因长度不匹配删除: {dropped_len_mismatch}")
        print(f"[JSONL] 保留的记录数量: {kept}")

    # 数据分析
    if args.analysis:
        print("\n=== 数据详细分析 ===")
        
        # 句子长度分布
        src_lengths = [len(w) for w, _ in all_pairs]
        tgt_lengths = [len(c) for _, c in all_pairs]
        
        print(f"源句子平均长度: {sum(src_lengths)/len(src_lengths):.1f}")
        print(f"目标句子平均长度: {sum(tgt_lengths)/len(tgt_lengths):.1f}")
        print(f"源句子最小长度: {min(src_lengths) if src_lengths else 0}")
        print(f"源句子最大长度: {max(src_lengths) if src_lengths else 0}")
        
        # 错误数量统计
        error_counts = []
        changed_pairs = [(w, c) for w, c in all_pairs if w != c]
        
        for w, c in changed_pairs:
            if len(w) == len(c):
                error_count = sum(1 for a, b in zip(w, c) if a != b)
                error_counts.append(error_count)
        
        if error_counts:
            avg_errors = sum(error_counts) / len(error_counts)
            print(f"\n有错误的样本对数量: {len(changed_pairs)}")
            print(f"平均每个错误句子的错误数: {avg_errors:.2f}")
            print(f"最多错误的句子错误数: {max(error_counts) if error_counts else 0}")
            print(f"最少错误的句子错误数: {min(error_counts) if error_counts else 0}")
        
        # 显示一些示例
        print(f"\n=== 数据示例（前5条）===")
        for i, (w, c) in enumerate(all_pairs[:5]):
            print(f"\n示例 {i+1}:")
            print(f"  错误: {w}")
            print(f"  正确: {c}")
            if w != c:
                # 找出具体差异
                if len(w) == len(c):
                    diffs = []
                    for j, (char_w, char_c) in enumerate(zip(w, c)):
                        if char_w != char_c:
                            diffs.append(f"位置{j}: '{char_w}'→'{char_c}'")
                    if diffs:
                        print(f"  差异: {', '.join(diffs[:3])}" + ("..." if len(diffs) > 3 else ""))
                else:
                    print(f"  长度不同: 错误文本长度={len(w)}, 正确文本长度={len(c)}")


if __name__ == "__main__":
    main()