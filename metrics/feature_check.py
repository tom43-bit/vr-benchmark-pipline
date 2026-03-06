# debug_pann_keys.py
import torch
"""
gt_pann = torch.load('/nfs/xtjin/benchmark/metrics/gt_cache/vggish_features.pth', weights_only=True)
pred_pann = torch.load('/nfs/xtjin/benchmark/metrics/pred_cache/vggish_features.pth', weights_only=True)

print("GT PANNs keys:", list(gt_pann.keys()))
print("Pred PANNs keys:", list(pred_pann.keys()))

# 找出共同的键
common_keys = set(gt_pann.keys()) & set(pred_pann.keys())
print(f"Common keys: {common_keys}")
print(f"Number of common keys: {len(common_keys)}")
"""
video = torch.load('/nfs/xtjin/benchmark/metrics/pred_cache/synchformer_audio.pth', weights_only=True)
print(video['BR2049_a_16k'].shape)