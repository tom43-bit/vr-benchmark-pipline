import modules.read_csv as my_csv
import modules.video_eval as video_eval
import modules.args as my_args
import os
import torch
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json
import pandas as pd
from typing import OrderedDict

torch.serialization.add_safe_globals([OrderedDict])

video_path = '/nfs/xtjin/benchmark/datas/outputs'
prompt_path = '/nfs/xtjin/benchmark/datas/prompts/TI2AV_ref_remin1.csv'
mode_list = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']

file_list, video_dict = my_csv.video_info(video_path,prompt_path)

parser_eval = my_args.parse_args()
for mode in mode_list:
    args_list = [
            '--dimension', mode,
            '--videos_path', video_path,
            '--video_mode', 'custom_input'
    ]

    args = parser_eval.parse_args(args_list)
    result_name = video_eval.video_eval(args)

    with open(f'/nfs/xtjin/benchmark/evaluation_results/{result_name}', 'r') as f:
        data = json.load(f)

    video_list = data[mode][1]
    df = pd.DataFrame(video_list)

