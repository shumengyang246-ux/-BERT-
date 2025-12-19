# 处理原始数据，把数据转换成tgt字段的纯文本jsonl
import pandas as pd
import json

def extract_chinese_text_from_csv(train_file_path, output_file_path):
   
    # 读取csv文件
    train_df = pd.read_csv(train_file_path)
    
    # 检查review列是否存在
    if 'review' not in train_df.columns:
        raise ValueError("CSV文件中没有'review'列")
    
     # 修正：确保将review列转换为字符串列表
    text_list = train_df['review'].astype(str).tolist()  # 关键修改
    
    # 将每个文本保存为jsonl格式，每个句子隶属于tgt字段
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in text_list:
            line_dict = {"tgt": text}
            f.write(json.dumps(line_dict, ensure_ascii=False) + '\n')
    
    print(f"成功提取了{len(text_list)}条文本记录")
    print(f"结果已保存至: {output_file_path}")

# 文件路径配置
train_file_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\pre_train\online_shopping_10_cats.csv"
output_file_path = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\data\extracted_texts.jsonl"

# 执行提取函数
if __name__ == "__main__":
    extract_chinese_text_from_csv(train_file_path, output_file_path)