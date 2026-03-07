import modules.read_csv as my_csv
import modules.video_eval as video_eval
import modules.args as my_args
import modules.wer as wer
import os
import torch
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json
import pandas as pd
from typing import OrderedDict
import numpy as np
import gc
import psutil
import multiprocessing
import pickle
import queue

# 设置显示选项，显示所有行和列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动检测宽度
pd.set_option('display.max_colwidth', None)  # 显示完整单元格内容

torch.serialization.add_safe_globals([OrderedDict])

def calculate_wer_average_exclude_total(df, metrics, total_identifier='total_datas', video_col='video_name'):
    
    filtered_df = df[df[video_col] != total_identifier]
    average_wer = filtered_df[metrics].mean()
    average_wer = round(average_wer, 3)
    
    return average_wer

def print_memory():
    process = psutil.Process()
    mem = process.memory_info()
    print(f"RSS: {mem.rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def run_video_eval_in_subprocess(args):
    """
    在子进程中运行 video_eval.video_eval(args)
    返回结果文件名或错误信息
    """
    try:
        
        # 执行评估
        result_name = video_eval.video_eval(args)
        
        # 通过队列返回结果
        return result_name
        
    except Exception as e:
        import traceback
        error_msg = f"子进程错误: {str(e)}\n{traceback.format_exc()}"
        return error_msg

def target_function(q, args):
    """目标函数 - 必须在模块顶层"""
    try:
        result = run_video_eval_in_subprocess(args)
        q.put(result)
    except Exception as e:
        q.put(f"ERROR:{str(e)}")

def run_with_subprocess(args, timeout=None):
    """带超时控制的子进程执行"""
    import multiprocessing
    
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    
    # 使用普通函数，不用 lambda
    p = ctx.Process(target=target_function, args=(result_queue, args))
    p.start()
    
    # 等待进程结束（可选超时）
    p.join(timeout=timeout)
    
    if p.is_alive():
        print(f"进程超时，终止...")
        p.terminate()
        p.join()
        return None
    
    # 获取结果
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = None
    
    # 清理
    p.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result

if __name__ == '__main__':
    video_path = '/nfs/xtjin/benchmark/datas/outputs'
    prompt_path = '/nfs/xtjin/benchmark/datas/prompts/TI2AV_ref_remin1.csv'
    mode_list = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
    metrics_list = ['wer','subs','dele','inse','FD_VGG','FD_PANN','FD_PASST','KL-PANNS-softmax','KL-PASST-softmax','ISC-PANNS-mean/std',
                    'ISC-PASST-mean/std','IB-Score','DeSync','LAION-CLAP-Score','MS-CLAP-Score']

    video_list, video_dict = my_csv.video_info(video_path,prompt_path)
    result_csv = pd.DataFrame(columns=(['video_name'] + mode_list + metrics_list))
    cols = result_csv.columns.tolist()
    video_col_index = cols.index('video_name')

    #定义结果表的各个视频元组
    for video_n in video_list:
       all_nan_tuple = tuple([np.nan] * len(cols))
       new_row_list = list(all_nan_tuple)
       new_row_list[video_col_index] = video_n  
       new_row_tuple = tuple(new_row_list)
       result_csv.loc[len(result_csv)] = new_row_tuple

    #定义测试集元组
    all_nan_tuple = tuple([np.nan] * len(cols))
    new_row_list = list(all_nan_tuple)
    new_row_list[video_col_index] = 'total_datas'
    new_row_tuple = tuple(new_row_list)
    result_csv.loc[len(result_csv)] = new_row_tuple

    parser_eval = my_args.parse_args()
    print_memory()
    for mode in mode_list:
        args_list = [
                '--dimension', mode,
                '--videos_path', video_path,
                '--video_mode', 'custom_input'
        ]

        args = parser_eval.parse_args(args_list)
        result_name = run_with_subprocess(args)

        with open(f'/nfs/xtjin/benchmark/evaluation_results/{result_name}', 'r') as f:
            data = json.load(f)

        video_mode_result = data[mode][1]
        df = pd.DataFrame(video_mode_result)
        df['video_name'] = df['video_path'].apply(lambda x: x.split('/')[-1])

        result_csv = result_csv.merge(
        df[['video_name', 'video_results']],  # 只取需要的列
        on='video_name',  # 匹配的列名
        how='left',
        suffixes=('', '_new')  # 避免列名重复
           )

        result_csv[mode] = result_csv['video_results'] # 替换原wer列
        result_csv = result_csv.drop(columns=['video_results']) # 删除临时列
        result_csv.loc[result_csv['video_name'] == 'total_datas',mode] = data[mode][0]
        """
        print("result_csv0")
        print(result_csv.iloc[0])
        print("result_csv-1")
        print(result_csv.iloc[-1])  
        print(df)
        """

    print_memory()
    gc.collect()
    torch.cuda.empty_cache()
    print_memory()

    wer.calculate_wer(video_path,video_list,video_dict,result_csv)

    print(result_csv)


