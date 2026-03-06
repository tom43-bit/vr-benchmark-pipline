import pandas as pd
import os

video_path = '/nfs/xtjin/benchmark/datas/outputs'
prompt_path = '/nfs/xtjin/benchmark/datas/prompts/TI2AV_ref_remin1.csv'

def video_info(video_path,prompt_path):
    video_list = os.listdir(video_path)
    idx_prompt = pd.read_csv(prompt_path)
    col_name = idx_prompt.columns

    video_dict = {}
    for file in video_list:
        video_dict[file] = idx_prompt.iloc[int(file[4])-1]

    return video_list, video_dict
