import pandas as pd
import os
import re

video_path = '/nfs/xtjin/benchmark/datas/outputs'
prompt_path = '/nfs/xtjin/benchmark/datas/prompts/TI2AV_ref_remin1.csv'

def extract_words(text):
    """提取 [WORDS] 和 [END_WORDS] 之间的所有内容"""
    if pd.isna(text):  # 处理空值
        return []
    
    # 使用正则表达式查找所有匹配
    pattern = r'\[WORDS\](.*?)\[/?END_WORDS\]'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    return matches

def extract_number(filename):

    # 匹配 row_ 或 row- 后面的第一个数字
    match = re.search(r'row[_-](\d+)', filename, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    
    # 如果没有找到 row_ 模式，返回0
    print(f"警告: {filename} 中没有找到 row_ 模式")
    return 0

def video_info(video_path,prompt_path):
    video_list = os.listdir(video_path)
    video_list.sort(key=extract_number)
    idx_prompt = pd.read_csv(prompt_path,dtype=str)
    col_name = idx_prompt.columns
    #idx_prompt['dialogue'] = idx_prompt['text_prompt'].apply(extract_words)

    video_dict = {}
    for file in video_list:
        video_dict[file] = idx_prompt.iloc[extract_number(file)-1]

    return video_list, video_dict


