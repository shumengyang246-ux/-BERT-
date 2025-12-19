# 把两个jsonl数据集合并成一个
import json
import os

def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    合并两个JSONL文件
    :param file1_path: 第一个JSONL文件路径
    :param file2_path: 第二个JSONL文件路径
    :param output_path: 输出文件路径
    """
    print(f"正在合并文件:")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")
    print(f"  输出: {output_path}")
    
    total_lines = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 处理第一个文件
        print("正在处理第一个文件...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            count1 = 0
            for line in f1:
                line = line.strip()
                if line:  # 确保不是空行
                    f_out.write(line + '\n')
                    count1 += 1
                    total_lines += 1
                    
                    # 显示进度
                    if count1 % 10000 == 0:
                        print(f"  已处理第一个文件: {count1} 行")
        
        print(f"第一个文件处理完成，共 {count1} 行")
        
        # 处理第二个文件
        print("正在处理第二个文件...")
        with open(file2_path, 'r', encoding='utf-8') as f2:
            count2 = 0
            for line in f2:
                line = line.strip()
                if line:  # 确保不是空行
                    f_out.write(line + '\n')
                    count2 += 1
                    total_lines += 1
                    
                    # 显示进度
                    if count2 % 10000 == 0:
                        print(f"  已处理第二个文件: {count2} 行")
        
        print(f"第二个文件处理完成，共 {count2} 行")
    
    print(f"合并完成！总共有 {total_lines} 行数据")
    print(f"结果已保存到: {output_path}")

def verify_merged_file(file_path, sample_size=3):
    """
    验证合并后的文件
    :param file_path: 文件路径
    :param sample_size: 验证样本数量
    """
    print(f"\n验证合并后的文件，显示前 {sample_size} 行:")
    print("-" * 80)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                data = json.loads(line.strip())
                print(f"样本 {i+1}:")
                print(f"  src: {data.get('src', '')[:50]}...")
                print(f"  tgt: {data.get('tgt', '')[:50]}...")
                print(f"  length_src: {data.get('length_src', 'N/A')}")
                print(f"  length_tgt: {data.get('length_tgt', 'N/A')}")
                print(f"  is_changed: {data.get('is_changed', 'N/A')}")
                print(f"  detection_labels长度: {len(data.get('detection_labels', []))}")
                print("-" * 80)
            except Exception as e:
                print(f"验证第 {i+1} 行时出错: {e}")

def main():
    # 文件路径
    file1_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\pretrain.jsonl"
    file2_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\pretrain1.jsonl"
    output_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\final_pretrain.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(file1_path):
        print(f"错误: 第一个输入文件不存在: {file1_path}")
        return
    
    if not os.path.exists(file2_path):
        print(f"错误: 第二个输入文件不存在: {file2_path}")
        return
    
    # 合并文件
    merge_jsonl_files(file1_path, file2_path, output_path)
    
    # 验证合并结果
    verify_merged_file(output_path)

if __name__ == "__main__":
    main()



