# 使用SIGHAN训练集估算合理的detect_pos_weight
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def analyze_sighan_labels(data_path: str) -> Dict:
    """
    分析SIGHAN数据集的字符级标签分布
    """
    print(f"正在分析数据集: {data_path}")
    
    all_labels = []
    sample_count = 0
    changed_samples = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 获取标签
                detection_labels = data.get('detection_labels', [])
                src = data.get('src', '')
                
                # 验证标签长度与文本长度是否一致
                if len(detection_labels) != len(src):
                    print(f"警告: 第{line_num}行标签长度({len(detection_labels)})与文本长度({len(src)})不一致")
                    continue
                
                all_labels.extend(detection_labels)
                sample_count += 1
                
                if data.get('is_changed', False):
                    changed_samples += 1
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析错误在第{line_num}行: {e}")
                continue
    
    # 统计标签分布
    label_counts = Counter(all_labels)
    total_tokens = len(all_labels)
    positive_tokens = label_counts.get(1, 0)
    negative_tokens = label_counts.get(0, 0)
    
    print(f"\n=== SIGHAN数据集分析结果 ===")
    print(f"总样本数: {sample_count}")
    print(f"有错误的样本数: {changed_samples}")
    print(f"总字符数: {total_tokens}")
    print(f"错误字符数: {positive_tokens} ({positive_tokens/total_tokens*100:.2f}%)")
    print(f"正确字符数: {negative_tokens} ({negative_tokens/total_tokens*100:.2f}%)")
    
    # 计算pos_weight
    if positive_tokens > 0:
        pos_weight = negative_tokens / positive_tokens
        print(f"\npos_weight计算:")
        print(f"  负样本数: {negative_tokens}")
        print(f"  正样本数: {positive_tokens}")
        print(f"  pos_weight = 负样本数 / 正样本数 = {pos_weight:.4f}")
    else:
        pos_weight = 1.0
        print(f"\n警告: 没有找到正样本，pos_weight设为1.0")
    
    return {
        'total_samples': sample_count,
        'changed_samples': changed_samples,
        'total_tokens': total_tokens,
        'positive_tokens': positive_tokens,
        'negative_tokens': negative_tokens,
        'pos_weight': pos_weight,
        'label_distribution': dict(label_counts)
    }

def visualize_label_distribution(stats: Dict):
    """
    可视化标签分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 标签分布饼图
    labels = ['Correct (0)', 'Error (1)']
    sizes = [stats['negative_tokens'], stats['positive_tokens']]
    colors = ['#66b3ff', '#ff9999']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Character-Level Label Distribution')
    
    # 标签数量柱状图
    axes[1].bar(['Correct', 'Error'], sizes, color=colors)
    axes[1].set_ylabel('Token Count')
    axes[1].set_title('Token Count by Label Type')
    
    # 添加数值标签
    for i, v in enumerate(sizes):
        axes[1].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def estimate_pos_weight_detailed(data_path: str, max_lines: int = None) -> float:
    """
    详细估算pos_weight的函数
    """
    print(f"\n开始详细估算pos_weight...")
    
    pos_count = 0
    neg_count = 0
    sample_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and line_num > max_lines:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                detection_labels = data.get('detection_labels', [])
                
                for label in detection_labels:
                    if label == 1:
                        pos_count += 1
                    elif label == 0:
                        neg_count += 1
                    else:
                        print(f"警告: 发现非0/1标签值: {label}")
                        
                sample_count += 1
                
                if sample_count % 1000 == 0:
                    print(f"已处理 {sample_count} 个样本，当前pos_weight估计: {neg_count/max(pos_count,1):.4f}")
                    
            except json.JSONDecodeError:
                continue
    
    if pos_count > 0:
        pos_weight = neg_count / pos_count
        print(f"\n=== 最终pos_weight估算结果 ===")
        print(f"正样本数 (错误字符): {pos_count}")
        print(f"负样本数 (正确字符): {neg_count}")
        print(f"pos_weight = {neg_count}/{pos_count} = {pos_weight:.4f}")
        return pos_weight
    else:
        print("\n错误: 没有找到正样本，返回默认值1.0")
        return 1.0

def recommend_pos_weight_range(pos_weight: float) -> Dict:
    """
    根据估算的pos_weight推荐使用范围
    """
    print(f"\n=== pos_weight参数建议 ===")
    print(f"估算得到的pos_weight: {pos_weight:.4f}")
    
    recommendations = {
        'recommended': pos_weight,
        'conservative': min(pos_weight, 10.0),
        'aggressive': max(pos_weight, 20.0),
        'balanced': max(1.0, min(pos_weight, 50.0))
    }
    
    print(f"推荐值: {recommendations['recommended']:.4f}")
    print(f"保守值 (不超过10): {recommendations['conservative']:.4f}")
    print(f"激进值 (增强正样本影响): {recommendations['aggressive']:.4f}")
    print(f"平衡值 (1-50范围内): {recommendations['balanced']:.4f}")
    
    return recommendations

def create_sample_analysis_report(data_path: str, sample_size: int = 5):
    """
    创建样本分析报告
    """
    print(f"\n=== 样本分析报告 (前{sample_size}个样本) ===")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
                
            try:
                data = json.loads(line.strip())
                src = data.get('src', '')
                tgt = data.get('tgt', '')
                labels = data.get('detection_labels', [])
                is_changed = data.get('is_changed', False)
                
                print(f"\n样本 {i+1}:")
                print(f"  原文: {src}")
                print(f"  目标: {tgt}")
                print(f"  是否有错误: {is_changed}")
                print(f"  标签: {labels}")
                
                # 高亮错误位置
                errors = []
                for j, label in enumerate(labels):
                    if label == 1:
                        errors.append((j, src[j]))
                if errors:
                    print(f"  错误位置: {errors}")
                    
            except Exception as e:
                print(f"解析样本{i+1}失败: {e}")

def main():
    """
    主函数
    """
    data_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\processed\train.jsonl"
    
    # 1. 分析数据集
    stats = analyze_sighan_labels(data_path)
    
    # 2. 详细估算pos_weight
    pos_weight = estimate_pos_weight_detailed(data_path)
    
    # 3. 推荐参数范围
    recommendations = recommend_pos_weight_range(pos_weight)
    
    # 4. 创建样本分析报告
    create_sample_analysis_report(data_path)
    
    # 5. 可视化（如果可用）
    try:
        visualize_label_distribution(stats)
    except ImportError:
        print("\n注意: matplotlib/seaborn未安装，跳过可视化")
    
    print(f"\n=== 最终建议 ===")
    print(f"对于您的SIGHAN数据集，建议的detect_pos_weight值为: {recommendations['recommended']:.2f}")
    print(f"如果训练不稳定，可以尝试较小的值，如: {recommendations['balanced']:.2f}")
    print(f"如果检测性能不佳，可以尝试较大的值，如: {recommendations['aggressive']:.2f}")
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()



