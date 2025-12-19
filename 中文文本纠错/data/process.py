# 构建混淆表进行数据增强
import json
import random
import re
from collections import defaultdict
from pypinyin import pinyin, Style
import os

def build_homophone_map(texts, min_freq=2):
    """
    构建同音字映射表
    :param texts: 文本列表
    :param min_freq: 最小频率阈值，用于过滤低频字符
    :return: 同音字映射表 {pinyin: [char1, char2, ...]}
    """
    print("正在构建同音字映射表...")
    
    # 统计字符频率
    char_freq = defaultdict(int)
    for text in texts:
        for char in text:
            if re.match(r'[\u4e00-\u9fff]', char):  # 中文字符
                char_freq[char] += 1
    
    # 过滤低频字符
    valid_chars = set()
    for char, freq in char_freq.items():
        if freq >= min_freq:
            valid_chars.add(char)
    
    # 构建同音字映射
    homophone_map = defaultdict(set)
    processed_chars = set()
    
    for text in texts:
        for char in text:
            if re.match(r'[\u4e00-\u9fff]', char) and char in valid_chars and char not in processed_chars:
                py = pinyin(char, style=Style.NORMAL, strict=False)[0][0]
                homophone_map[py].add(char)
                processed_chars.add(char)
    
    # 转换为普通字典
    result = {}
    for py, chars in homophone_map.items():
        if len(chars) > 1:  # 至少有两个不同的字符
            result[py] = list(chars)
    
    print(f"构建完成，共找到 {len(result)} 个同音字组合")
    return result

def generate_error_text(text, homophone_map, error_rate=0.15, homophone_ratio=0.8):
    """
    生成带错误的文本
    :param text: 原始文本
    :param homophone_map: 同音字映射表
    :param error_rate: 错误率，默认15%
    :param homophone_ratio: 同音字替换比例，默认80%
    :return: 带错误的文本
    """
    chars = list(text)
    num_errors = int(len([c for c in text if re.match(r'[\u4e00-\u9fff]', c)]) * error_rate)
    
    # 获取所有中文字符的位置
    chinese_positions = []
    for i, char in enumerate(chars):
        if re.match(r'[\u4e00-\u9fff]', char):
            chinese_positions.append(i)
    
    if not chinese_positions:
        return text  # 没有中文字符则直接返回
    
    # 随机选择要替换的位置
    selected_positions = random.sample(chinese_positions, min(num_errors, len(chinese_positions)))
    
    for pos in selected_positions:
        original_char = chars[pos]
        
        # 决定使用同音字替换还是随机替换
        if random.random() < homophone_ratio:
            # 同音字替换
            py = pinyin(original_char, style=Style.NORMAL, strict=False)[0][0]
            if py in homophone_map and len(homophone_map[py]) > 1:
                # 选择一个不同于原字符的同音字
                candidates = [c for c in homophone_map[py] if c != original_char]
                if candidates:
                    chars[pos] = random.choice(candidates)
        else:
            # 随机替换 - 从所有同音字中随机选择
            all_chars = []
            for chars_list in homophone_map.values():
                all_chars.extend(chars_list)
            
            if all_chars:
                # 随机选择一个字符替换
                chars[pos] = random.choice(all_chars)
    
    return ''.join(chars)

def process_jsonl_file(input_path, output_path, error_rate=0.15, homophone_ratio=0.8):
    """
    处理JSONL文件，为每条记录添加src字段
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param error_rate: 错误率
    :param homophone_ratio: 同音字替换比例
    """
    print(f"正在读取文件: {input_path}")
    
    # 读取所有文本
    texts = []
    original_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['tgt'])
            original_data.append(data)
    
    print(f"读取到 {len(texts)} 条记录")
    
    # 构建同音字映射表
    homophone_map = build_homophone_map(texts)
    
    print(f"开始处理数据...")
    processed_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data in original_data:
            tgt_text = data['tgt']
            
            # 生成错误文本作为src
            src_text = generate_error_text(tgt_text, homophone_map, error_rate, homophone_ratio)
            
            # 创建新的数据对象，src在前，tgt在后
            new_data = {
                'src': src_text,
                'tgt': tgt_text
            }
            
            # 写入文件
            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"已处理 {processed_count} 条记录")
    
    print(f"处理完成，结果保存到: {output_path}")

def main():
    # 文件路径
    input_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\extracted_texts.jsonl"
    output_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\extracted_texts_with.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 处理文件
    process_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        error_rate=0.15,  # 15%的字符被替换
        homophone_ratio=0.8  # 80%是同音字替换，20%是随机替换
    )
    
    # 显示一些示例
    print("\n示例输出:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只显示前3个示例
                break
            data = json.loads(line.strip())
            print(f"Src: {data['src']}")
            print(f"Tgt: {data['tgt']}")
            print("-" * 50)

if __name__ == "__main__":
    main()



