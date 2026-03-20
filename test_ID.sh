#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cd /nfs/xtjin/eval_pipline

#/home/xtjin/miniconda3/envs/metrics/bin/python test_ID.py --video_path "/nfs/xtjin/UniAV/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ours.csv" \
#--output_cache_path "/nfs/xtjin/uniav_feature" --splits 10 --eval_name "test_ID_uniav.csv" --model_name "uniav"

/home/xtjin/miniconda3/envs/metrics/bin/python test_ID.py --video_path "/nfs/xtjin/LTX-2/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ltx2.csv" \
--output_cache_path "/nfs/xtjin/ltx_feature" --splits 10 --eval_name "test_ID_ltx2.csv" --model_name "ltx2"

/home/xtjin/miniconda3/envs/metrics/bin/python test_ID.py --video_path "/nfs/xtjin/Ovi/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ovi.csv" \
--output_cache_path "/nfs/xtjin/ovi_feature" --splits 10 --eval_name "test_ID_ovi.csv" --model_name "ovi"