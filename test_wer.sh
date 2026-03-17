#!/bin/bash

python test_wer.py --video_path "/nfs/xtjin/LTX-2/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ltx2.csv" \
--output_cache_path "/nfs/xtjin/eval_pipline/ltx_feature" --splits 10 --eval_name "ltx_wer_eval.csv" --model_name "ltx"

#python test_wer.py --video_path "/nfs/xtjin/UniAV/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ours.csv" \
#--output_cache_path "/nfs/xtjin/eval_pipline/uniav_feature" --splits 10 --eval_name "uniav_wer_eval.csv" --model_name "uniav"

#python test_wer.py --video_path "/nfs/xtjin/Ovi/outputs"  --prompt_path "/nfs/xtjin/eval_pipline/csvs/set1_ovi.csv" \
#--output_cache_path "/nfs/xtjin/eval_pipline/ovi_feature" --splits 10 --eval_name "ovi_wer_eval.csv" --model_name "ovi"