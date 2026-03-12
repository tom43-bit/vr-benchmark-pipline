import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
#wav_dir = sys.argv[2]
#wav_res_ref_text = sys.argv[3]

f = open(metalst)
lines = f.readlines()
f.close()

print(lines)