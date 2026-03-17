#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python metrics_total.py --video_path "/nfs/xtjin/Wan2.2/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/wan_csv.csv" \
--output_cache_path "/nfs/xtjin/eval_pipline/wan_feature" --splits 10 --eval_name "wan_eval.csv" --no_audio --model_name "wan"