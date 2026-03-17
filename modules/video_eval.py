import torch
import sys, os
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json
import torch.distributed as dist
import gc, psutil


def print_memory():
    process = psutil.Process()
    mem = process.memory_info()
    print(f"RSS: {mem.rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def cleanup_vbench_completely():
    """彻底清理 VBench 相关的所有缓存和模块"""
    
    # 1. 查找并清理所有 vbench 模块的全局缓存
    vbench_modules = [name for name in sys.modules if 'vbench' in name]
    for module_name in vbench_modules:
        module = sys.modules[module_name]
        
        # 常见的缓存变量名
        cache_names = ['_MODEL_CACHE', '_CACHE', 'model_cache', 'MODELS', 'model_zoo']
        for cache_name in cache_names:
            if hasattr(module, cache_name):
                print(f"清理 {module_name}.{cache_name}")
                getattr(module, cache_name).clear()
                delattr(module, cache_name)
    
    # 2. 强制垃圾回收
    for _ in range(3):
        gc.collect()
    
    # 3. 清空 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def cleanup_vbench(vbench_obj):
    """递归删除 VBench 对象的所有属性"""
    if vbench_obj is None:
        return
    
    # 获取所有属性
    for attr_name in dir(vbench_obj):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(vbench_obj, attr_name)
            # 如果是模型对象，尝试删除
            if 'model' in attr_name.lower() or 'net' in attr_name.lower():
                if hasattr(attr, 'cpu'):
                    attr.cpu()  # 确保在 CPU
                del attr
        except:
            pass
    
    # 删除对象本身
    del vbench_obj

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
    args = parser.parse_args()
    return args


def video_eval(args):

    """
    if dist.is_initialized():
        pass
    else:
        dist_init()
    """

    print0(f'args: {args}')
    device = torch.device("cuda")
    
    my_VBench = VBench(device, args.full_json_dir, args.output_path)
    
    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    prompt = []

    if (args.prompt_file is not None) and (args.prompt != "None"):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "None") and (args.video_mode!='custom_input'):
        raise Exception("must set --mode=custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
        assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"
    elif args.prompt != "None":
        prompt = [args.prompt]

    if args.category != "":
        kwargs['category'] = args.category

    kwargs['imaging_quality_preprocessing_mode'] = args.imaging_quality_preprocessing_mode

    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = f'results_{args.dimension}_{args.model_name}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.video_load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.video_mode,
        **kwargs
    )
    print0('done')

    return f'results_{args.dimension}_{args.model_name}_eval_results.json'
