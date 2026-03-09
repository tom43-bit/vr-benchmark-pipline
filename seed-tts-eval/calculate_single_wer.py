#!/usr/bin/env python
# 文件名: calculate_single_wer.py

import sys
import os
import soundfile as sf
import scipy
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string

# 设置参数
wav_path = "/nfs/xtjin/benchmark/seed-tts-eval/seedtts_testset/test/wavs/row4.wav"
reference_text = "You trust me. I trust you, right? This is the guy. THIS IS THE GUY!"
lang = "en"
device = "cuda:0"

# 标点符号处理
punctuation_all = punctuation + string.punctuation

def process_one(hypo, truth):
    """计算WER和各项指标"""
    raw_truth = truth
    raw_hypo = hypo

    # 去除标点符号
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    # 处理多余空格
    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    # 语言特定处理
    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()

    # 计算指标
    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list) if len(ref_list) > 0 else 0
    dele = measures["deletions"] / len(ref_list) if len(ref_list) > 0 else 0
    inse = measures["insertions"] / len(ref_list) if len(ref_list) > 0 else 0
    
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def load_en_model():
    """加载英文模型"""
    print("加载 Whisper 模型...")
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def calculate_single_wer():
    """计算单个WAV文件的WER"""
    
    # 1. 检查文件是否存在
    if not os.path.exists(wav_path):
        print(f"错误: 文件不存在 - {wav_path}")
        return
    
    print(f"=" * 60)
    print(f"音频文件: {wav_path}")
    print(f"参考文本: {reference_text}")
    print(f"=" * 60)
    
    # 2. 加载模型
    processor, model = load_en_model()
    model.eval()
    
    # 3. 读取音频
    print("\n读取音频文件...")
    wav, sr = sf.read(wav_path)
    print(f"音频信息: 形状={wav.shape}, 采样率={sr}, 范围=[{wav.min():.3f}, {wav.max():.3f}]")
    
    # 4. 重采样（如果需要）
    if sr != 16000:
        print(f"重采样: {sr} -> 16000")
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
    
    # 5. 提取特征
    print("提取音频特征...")
    input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(dtype=torch.float16, device=device)
    
    # 6. 生成转录
    print("运行ASR模型...")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids
        )
    
    # 7. 解码结果
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"\n识别结果: {transcription}")
    
    # 8. 计算WER
    print("\n计算WER...")
    raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, reference_text)
    
    # 9. 输出详细结果
    print(f"\n" + "=" * 60)
    print(f"详细结果:")
    print(f"=" * 60)
    print(f"参考文本: {raw_truth}")
    print(f"识别文本: {raw_hypo}")
    print(f"-" * 60)
    print(f"词错误率 (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"替换错误率 (Substitutions): {subs:.4f} ({subs*100:.2f}%)")
    print(f"删除错误率 (Deletions): {dele:.4f} ({dele*100:.2f}%)")
    print(f"插入错误率 (Insertions): {inse:.4f} ({inse*100:.2f}%)")
    print(f"=" * 60)
    
    # 10. 保存结果到文件
    result_file = os.path.join(os.path.dirname(wav_path), "single_wer_result.txt")
    with open(result_file, "w") as f:
        f.write(f"音频文件: {wav_path}\n")
        f.write(f"参考文本: {raw_truth}\n")
        f.write(f"识别文本: {raw_hypo}\n")
        f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
        f.write(f"Substitutions: {subs:.4f}\n")
        f.write(f"Deletions: {dele:.4f}\n")
        f.write(f"Insertions: {inse:.4f}\n")
    
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    calculate_single_wer()