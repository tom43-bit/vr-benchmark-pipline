import torch
import os
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

def get_eval_args():
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('')

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench/VBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--video_load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--video_mode",
        choices=['custom_input', 'vbench_standard', 'vbench_category'],
        default='vbench_standard',
        help="""This flags determine the mode of evaluations, choose one of the following:
        1. "custom_input": receive input prompt from either --prompt/--prompt_file flags or the filename
        2. "vbench_standard": evaluate on standard prompt suite of VBench
        3. "vbench_category": evaluate on specific category
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="None",
        help="""Specify the input prompt
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt_file.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        help="""This is for mode=='vbench_category'
        The category to evaluate on, usage: --category=animal.
        """,
    )

    ## for dimension specific params ###
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        type=str,
        required=False,
        default='longer',
        help="""This is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
        """,
    )
    
    return parser

def total_args():

    parser1 = argparse.ArgumentParser(description='eval_pipline', formatter_class=argparse.RawTextHelpFormatter)
    parser1.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="eval video path the evaluation results",
    )

    parser1.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="prompt path the evaluation results",
    )

    parser1.add_argument(
        "--output_cache_path",
        type=str,
        required=True,
        help="output cache path the evaluation cache",
    )

    parser1.add_argument(
        "--_syncformer_ckpt_path",
        type=str,
        required=False,
        default="./weights",
        help="flag of ref audio",
    )

    parser1.add_argument(
        "--if_ref_video",
        action='store_true',
        required=False,
        help="flag of ref video",
    )

    parser1.add_argument(
        "--if_ref_audio",
        action='store_true',
        required=False,
        help="flag of ref audio",
    )

    parser1.add_argument(
        "--splits",
        type=int,
        required=False,
        help="flag of ref audio",
    )

    return parser1.parse_args()