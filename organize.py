import json
import pandas as pd
import os
import numpy as np
import subprocess
import sys

bench_path = "/nfs/xtjin/eval_pipline/Verse-Bench/set1"
json_paths = ["/nfs/zzzhong/codes/joint_gen/vr-benchmark-pipline/Verse-Bench/set1_ours","/nfs/zzzhong/codes/joint_gen/vr-benchmark-pipline/Verse-Bench/set1_ovi",
                "/nfs/zzzhong/codes/joint_gen/vr-benchmark-pipline/Verse-Bench/set1_ltx2"]
output_path = "/nfs/xtjin/eval_pipline/csvs"

for model_n in json_paths:
    info_csv = pd.DataFrame()
    info_csv[['video_name','text_prompt','dialogue','image_path']] = np.nan
    json_list = [jsonfile for jsonfile in os.listdir(model_n) if jsonfile.endswith(".json")]
    for json_n in json_list:
        complete_path = os.path.join(model_n,json_n)
        try:
            with open(complete_path, "r", encoding="utf-8") as f:
                df = json.load(f)
        except:
            print(json_n)
            exit(1)
        
        image_name = os.path.basename(df["image_path"])
        video_name = os.path.splitext(image_name)[0]
        new_row = pd.DataFrame([{
                'video_name': video_name,
                'text_prompt': df["text_prompt"],
                'dialogue': df["original"]["speech_prompt"]["text"],
                'image_path': os.path.join(bench_path, image_name)
            }])
        info_csv = pd.concat([info_csv,new_row],ignore_index=True)

    model_name = os.path.basename(model_n)
    outputs = os.path.join(output_path,f'{model_name}.csv')
    info_csv.to_csv(outputs,index=False)


