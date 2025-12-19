# 把预训练数据格式统一成训练数据格式
import json
import os

def convert_to_training_format(input_path, output_path):
    """
    将预训练数据格式转换为训练集格式
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    """
    print(f"正在读取文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        total_lines = 0
        processed_lines = 0
        
        # 先计算总行数
        with open(input_path, 'r', encoding='utf-8') as temp_f:
            total_lines = sum(1 for _ in temp_f)
        
        print(f"总共 {total_lines} 行数据")
        
        for line_num, line in enumerate(f_in, 1):
            try:
                # 解析JSON
                data = json.loads(line.strip())
                
                src = data['src']
                tgt = data['tgt']
                
                # 计算长度
                length_src = len(src)
                length_tgt = len(tgt)
                
                # 确保src和tgt长度一致
                if length_src != length_tgt:
                    # 如果长度不一致，取较短的长度，多余的字符在检测标签中设为0
                    min_len = min(length_src, length_tgt)
                    max_len = max(length_src, length_tgt)
                    
                    if length_src > length_tgt:
                        src = src[:min_len]
                        length_src = min_len
                    elif length_tgt > length_src:
                        tgt = tgt[:min_len]
                        length_tgt = min_len
                
                # 生成检测标签
                detection_labels = []
                is_changed = False
                
                for i in range(min(length_src, length_tgt)):
                    if src[i] != tgt[i]:
                        detection_labels.append(1)
                        is_changed = True
                    else:
                        detection_labels.append(0)
                
                # 创建新的数据对象
                new_data = {
                    'src': src,
                    'tgt': tgt,
                    'length_src': length_src,
                    'length_tgt': length_tgt,
                    'is_changed': is_changed,
                    'detection_labels': detection_labels
                }
                
                # 写入输出文件
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
                processed_lines += 1
                
                # 显示进度
                if processed_lines % 1000 == 0:
                    progress = (processed_lines / total_lines) * 100
                    print(f"已处理: {processed_lines}/{total_lines} ({progress:.2f}%)")
                    
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {str(e)}")
                continue
    
    print(f"转换完成，结果保存到: {output_path}")
    print(f"总共处理了 {processed_lines} 条记录")

def main():
    # 输入输出路径
    input_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\extracted_texts_with.jsonl"
    output_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\pretrain1.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    # 执行转换
    convert_to_training_format(input_path, output_path)
    
    # 显示示例
    print("\n转换后数据格式示例:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i in range(3):  # 显示前3条记录
            try:
                line = f.readline().strip()
                if not line:
                    break
                data = json.loads(line)
                print(f"\n示例 {i+1}:")
                print(f"src: {data['src']}")
                print(f"tgt: {data['tgt']}")
                print(f"length_src: {data['length_src']}, length_tgt: {data['length_tgt']}")
                print(f"is_changed: {data['is_changed']}")
                print(f"detection_labels: {data['detection_labels']}")
            except Exception as e:
                print(f"读取示例时出错: {str(e)}")
                break

if __name__ == "__main__":
    main()



