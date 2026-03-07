import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import torch
import subprocess
import tempfile
from tqdm import tqdm
import psutil

punctuation_all = punctuation + string.punctuation
"""
wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en
"""
device = "cuda:0"

def print_memory():
    process = psutil.Process()
    mem = process.memory_info()
    print(f"RSS: {mem.rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def load_en_model():
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def extract_audio_with_tempfile(mp4_path, target_sr=16000):
    """从MP4提取音频为WAV文件，用sf.read()读取后删除临时文件"""
    
    if not os.path.exists(mp4_path):
        print(f"文件不存在: {mp4_path}")
        return None, None
    
    # 创建临时WAV文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_wav_path = tmp_file.name
    
    try:
        # 使用ffmpeg提取音频并保存为WAV文件
        cmd = [
            'ffmpeg',
            '-i', mp4_path,           # 输入文件
            '-ac', '1',                # 强制单声道（模型通常需要单声道）
            '-ar', str(target_sr),      # 设置采样率
            '-acodec', 'pcm_f32le',     # 使用32位浮点编码
            '-f', 'wav',                # 输出格式为WAV
            '-vn',                      # 不处理视频
            '-y',                       # 覆盖输出文件
            tmp_wav_path
        ]
        
        print(f"提取音频到WAV文件: {tmp_wav_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg 错误: {result.stderr}")
            return None, None
        
        # 使用soundfile读取WAV文件（这正是参考代码的方式）
        audio_data, sr = sf.read(tmp_wav_path)
        
        print(f"音频数据形状: {audio_data.shape}, 采样率: {sr}, 范围: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
        return audio_data, sr
        
    except Exception as e:
        print(f"处理音频时出错: {e}")
        return None, None
        
    finally:
        # 删除临时WAV文件
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)
            print(f"已删除临时文件: {tmp_wav_path}")

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    """
    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError
    """
    truth = truth.lower()
    hypo = hypo.lower()

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def calculate_wer(video_path,video_list,dialogue_dict,result_csv):
    processor, model = load_en_model()
    for video in tqdm(video_list):
        dialogue = ' '.join(dialogue_dict[video]['dialogue'])
        print(dialogue)
        total_path = os.path.join(video_path,video)
        #wav, sr = extract_audio_from_mp4(mp4_path = total_path)
        wav, sr = extract_audio_with_tempfile(mp4_path = total_path)
        print("wav is extracted")
        print_memory()
        if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(dtype=torch.float16,device=device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, dialogue)
        result_csv.loc[result_csv['video_name'] == video,'wer'] = wer 
        result_csv.loc[result_csv['video_name'] == video,'subs'] = subs
        result_csv.loc[result_csv['video_name'] == video,'dele'] = dele
        result_csv.loc[result_csv['video_name'] == video,'inse'] = inse  