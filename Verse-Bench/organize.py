import json
import pandas as pd
import os
import numpy as np
import subprocess

bench_path = "/nfs/xtjin/eval_pipline/Verse-Bench"
#set_names = ['set1','set2','set3']
set_names = ['set1']

if not os.path.isfile('/nfs/xtjin/eval_pipline/wan_csv.csv'):
    set_dict={}
    for set_i in set_names:
        completed_path = os.path.join(bench_path,set_i)
        set_content = [json_file for json_file in os.listdir(completed_path) if json_file.endswith(".json") ]
        set_dict[set_i] = set_content

    info_csv = pd.DataFrame()
    info_csv[['video_name','text_prompt','dialogue','image_path']] = np.nan
    start = "[WORDS]"
    end = "[WORDS_END]"
    audio_caption = "[AUDIO_CAPTION]"
    audio_caption_end = "[END_AUDIO_CAPTION]"

    for sets in set_names:
        completed_path = os.path.join(bench_path,set_i)
        for json_file in set_dict[sets]:
            json_path = os.path.join(completed_path,json_file)
            jpg = json_file[:-4] + "jpg"
            jpg_path = os.path.join(completed_path,jpg)

            with open(json_path, "r", encoding="utf-8") as f:
                df = json.load(f)

            if df["speech_prompt"]["text"]:
                text_prompt = df["video_prompt"] + start + df["speech_prompt"]["text"] + end + audio_caption + df["speech_prompt"]["speaker"] + (",".join(df["audio_prompt"])) + audio_caption_end
            else:
                text_prompt = df["video_prompt"] + audio_caption + (",".join(df["audio_prompt"])) + audio_caption_end

            new_row = pd.DataFrame(
                {'video_name': json_file[:-5], 'text_prompt': text_prompt, 'dialogue': df["speech_prompt"]["text"], 'image_path': jpg_path},index = [0]
            )

            info_csv = pd.concat([info_csv,new_row], ignore_index = True)

    info_csv.to_csv('wan_csv.csv', index=False)

else:
    info_csv = pd.read_csv('/nfs/xtjin/eval_pipline/wan_csv.csv',dtype={'video_name': str})

work_dir = '/nfs/xtjin/Wan2.2'

for i in range(172,len(info_csv)):
    # 获取当前行的数据
    row = info_csv.iloc[i]
    image_path = row['image_path']
    prompt = row['text_prompt']
    
    # 构建命令
    cmd = [
        'conda', 'run', '-n', 'wan', '--no-capture-output',
        'torchrun',
        '--nproc_per_node=8',
        'generate.py',
        '--task', 'ti2v-5B',
        '--size', '1280*704',
        '--ckpt_dir', './Wan2.2-TI2V-5B',
        '--dit_fsdp',
        '--t5_fsdp',
        '--ulysses_size', '8',
        '--image', image_path,
        '--prompt', prompt,
        '--row_index',str(i+1),
        '--original_filename',row['video_name']
    ]

    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

    if result.stdout:
        print(f"标准输出:\n{result.stdout}")
    
    # 如果有错误输出，打印（最重要的！）
    if result.stderr:
        print(f"错误输出:\n{result.stderr}")