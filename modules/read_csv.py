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


def video_info(video_path,prompt_path):
    video_list = os.listdir(video_path)
    idx_prompt = pd.read_csv(prompt_path)
    col_name = idx_prompt.columns
    idx_prompt['dialogue'] = idx_prompt['text_prompt'].apply(extract_words)

    video_dict = {}
    for file in video_list:
        video_dict[file] = idx_prompt.iloc[int(file[4])-1]

    return video_list, video_dict


