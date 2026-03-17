#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

#python metrics_total.py --video_path "/nfs/xtjin/Wan2.2/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/wan_csv.csv" \
#--output_cache_path "/nfs/xtjin/eval_pipline/wan_feature" --splits 10

python metrics_total.py --video_path "/nfs/xtjin/eval_pipline/datas/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/datas/prompts/TI2AV_ref_remin1.csv" \
--output_cache_path "/nfs/xtjin/eval_pipline/feature_cache" --splits 1 --eval_name "test.csv" --model_name "test"