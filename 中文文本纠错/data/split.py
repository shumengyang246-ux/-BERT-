# 划分验证集和训练集
import json
import random
import os
from pathlib import Path

def split_train_dev_pure(input_file, train_output, dev_output, dev_ratio=0.1, seed=42):
    print("正在读取数据...")
    
    # 设置随机种子
    random.seed(seed)
    
    # 读取所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"总数据量: {len(data)} 条")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算验证集大小
    dev_size = int(len(data) * dev_ratio)
    train_size = len(data) - dev_size
    
    # 划分数据集
    train_data = data[:train_size]
    dev_data = data[train_size:]
    
    print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
    print(f"验证集: {len(dev_data)} 条 ({len(dev_data)/len(data)*100:.1f}%)")
    
    # 保存训练集
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"训练集已保存到: {train_output}")
    
    # 保存验证集
    with open(dev_output, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"验证集已保存到: {dev_output}")
    
    return train_data, dev_data

def validate_split(train_file, dev_file):
    print("\n=== 验证划分结果 ===")
    
    # 读取训练集和验证集
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = [json.loads(line) for line in f]
    
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_lines = [json.loads(line) for line in f]
    
    # 检查是否有重复数据
    train_texts = set(item['src'] for item in train_lines)
    dev_texts = set(item['src'] for item in dev_lines)
    
    duplicates = train_texts.intersection(dev_texts)
    
    if duplicates:
        print(f"警告: 发现 {len(duplicates)} 条重复数据！")
        for i, text in enumerate(list(duplicates)[:5]):  # 只显示前5个
            print(f"  重复 {i+1}: {text[:50]}...")
    else:
        print("✓ 无数据泄漏: 训练集和验证集没有重复数据")
    
    return len(duplicates) == 0

if __name__ == "__main__":
    # 设置路径
    base_dir = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train"
    input_file = os.path.join(base_dir, "final_pretrain.jsonl")
    train_output = os.path.join(base_dir, "final_pretrain_train.jsonl")
    dev_output = os.path.join(base_dir, "fonal_pretrain_dev.jsonl")
    
    # 创建输出目录（如果不存在）
    os.makedirs(base_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        exit(1)
    
    # 划分数据集
    train_data, dev_data = split_train_dev_pure(
        input_file=input_file,
        train_output=train_output,
        dev_output=dev_output,
        dev_ratio=0.1,
        seed=42
    )
    
    # 输出统计信息
    print("\n=== 详细统计 ===")
    print(f"原始文件: {input_file}")
    print(f"训练集文件: {train_output}")
    print(f"验证集文件: {dev_output}")
    
    # 验证划分结果
    is_valid = validate_split(train_output, dev_output)
    
    if is_valid:
        print("\n✓ 数据集划分完成！")
        print(f"   训练集: {len(train_data)} 条")
        print(f"   验证集: {len(dev_data)} 条")
    else:
        print("\n⚠️ 数据集划分存在数据泄漏问题，请检查！")