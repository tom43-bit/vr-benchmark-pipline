import modules.read_csv as my_csv
import modules.video_eval as video_eval
import modules.args as my_args
import modules.wer as wer

import subprocess
import os,sys
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

if __name__ == '__main__':
   total_args=my_args.total_args()

   mode_list = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
   metrics_list = ['wer','subs','dele','inse','FD_VGG','FD_PANN','FD_PASST','KL-PANNS-softmax','KL-PASST-softmax','ISC-PANNS-mean','ISC-PANNS-std',
                    'ISC-PASST-mean','ISC-PASST-std','IB-Score','DeSync','LAION-CLAP-Score','MS-CLAP-Score']
   video_path=total_args.video_path
   prompt_path=total_args.prompt_path
   output_cache_path=total_args.output_cache_path
   splits=total_args.splits
   if_ref_video=total_args.if_ref_video
   if_ref_audio=total_args.if_ref_audio
   _syncformer_ckpt_path = total_args._syncformer_ckpt_path
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

   all_nan_tuple = tuple([np.nan] * len(cols))
   new_row_list = list(all_nan_tuple)
   new_row_list[video_col_index] = 'total_datas'
   new_row_tuple = tuple(new_row_list)
   result_csv.loc[len(result_csv)] = new_row_tuple

   print(video_dict[video_list[-1]])

   
   #wer.calculate_wer(video_path,video_list,video_dict,result_csv)
   wer.calculate_wer_tts(video_path,video_list,video_dict,result_csv)
   wer_means = result_csv.iloc[:-1][['wer','subs','dele','inse']].mean()
   result_csv.loc[result_csv.index[-1],['wer','subs','dele','inse']] = wer_means

   result_csv.to_csv(total_args.eval_name, index=False)
   