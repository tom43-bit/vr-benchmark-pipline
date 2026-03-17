#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

cd /nfs/xtjin/eval_pipline

/home/xtjin/miniconda3/envs/metrics/bin/python metrics_total.py --video_path "/nfs/xtjin/LTX-2/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ltx2.csv" \
--output_cache_path "/nfs/xtjin/LTX-2/ltx_feature" --splits 10 --eval_name "ltx_eval.csv" --model_name "ltx"