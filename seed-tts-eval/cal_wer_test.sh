#!/bin/bash
set -x

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

echo "Step 1: 生成 wav_res_ref_text 文件"
python3 get_wav_res_ref_text.py $meta_lst $output_dir $wav_wav_text

echo "Step 2: 准备模型"
python3 prepare_ckpt.py

echo "Step 3: 直接计算WER（单线程）"
echo "输入文件: $wav_wav_text"
echo "任务数量: $(wc -l < $wav_wav_text)"

# 直接用cuda0处理，输出到终端
CUDA_VISIBLE_DEVICES=0 python3 run_wer.py $wav_wav_text /dev/stdout $lang

# 清理
rm -f $wav_wav_text

echo "=== 计算完成 ==="