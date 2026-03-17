#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

cd /nfs/xtjin/eval_pipline

/home/xtjin/miniconda3/envs/metrics/bin/python metrics_total.py --video_path "/nfs/xtjin/UniAV/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ours.csv" \
--output_cache_path "/nfs/xtjin/uniav_feature" --splits 10 --eval_name "uniav_eval.csv" --model_name "uniav"