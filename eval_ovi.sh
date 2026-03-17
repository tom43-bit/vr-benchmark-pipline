#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

cd /nfs/xtjin/eval_pipline

/home/xtjin/miniconda3/envs/metrics/bin/python metrics_total.py --video_path "/nfs/xtjin/Ovi/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ovi.csv" \
--output_cache_path "/nfs/xtjin/eval_pipline/ovi_feature" --splits 10 --eval_name "ovi_eval.csv" --model_name "ovi"