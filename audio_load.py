import torchaudio

print("="*60)
print("TorchAudio MP4 直接读取验证")
print("="*60)

# 1. 检查版本
print(f"TorchAudio版本: {torchaudio.__version__}")

# 2. 检查是否支持MP4
print("\n检查MP4支持情况...")
try:
    # 尝试加载MP4文件
    audio_path = "/nfs/xtjin/benchmark/metrics/test_data/vedio_16k/BR2049_a_16k.mp4"
    
    waveform, sample_rate = torchaudio.load(audio_path)
    print("✅ 成功！torchaudio可以直接读取MP4")
    print(f"   波形形状: {waveform.shape}")
    print(f"   采样率: {sample_rate}")
    
except Exception as e:
    print("❌ 失败！torchaudio不能直接读取MP4")
    print(f"   错误信息: {e}")
    raise  # 直接抛出异常