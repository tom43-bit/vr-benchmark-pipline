def inspect_pth_keys(pth_path: str):
    """
    查看 .pth 文件中的所有 key
    
    Args:
        pth_path: .pth 文件的路径
    """
    import torch
    
    # 加载 .pth 文件
    data = torch.load(pth_path, weights_only=True)
    
    print(f"\n{'='*60}")
    print(f"文件: {pth_path}")
    print(f"{'='*60}")
    
    # 检查是否是字典
    if isinstance(data, dict):
        print(f"数据类型: dict")
        print(f"总键数: {len(data)}")
        print("\n所有 keys:")
        for i, key in enumerate(data.keys(), 1):
            # 获取值的类型和大小（如果是张量）
            value = data[key]
            value_type = type(value).__name__
            if torch.is_tensor(value):
                value_info = f"torch.Tensor, shape={tuple(value.shape)}, dtype={value.dtype}"
            else:
                value_info = str(value_type)
            print(f"  {i:3d}. {key}: {value_info}")
    else:
        print(f"数据类型: {type(data).__name__}")
        print("文件内容不是字典，无法显示 keys")
    
    return data

# 使用示例
if __name__ == '__main__':
    # 假设你的文件路径
    #text_pth = '/nfs/xtjin/benchmark/feature_cache/gt_cache/clap_ms_text.pth'
    #audio_pth = '/nfs/xtjin/benchmark/feature_cache/pred_cache/clap_ms_audio.pth'
    #key_t = inspect_pth_keys(text_pth)
    #key_a = inspect_pth_keys(audio_pth)
    pann_pth = '/nfs/xtjin/benchmark/feature_cache/pred_cache/pann_features.pth'
    passt_pth = '/nfs/xtjin/benchmark/feature_cache/pred_cache/passt_logits.pth'
    key_nn = inspect_pth_keys(pann_pth)
    key_ss = inspect_pth_keys(passt_pth)
    print(key_nn)
    print('-'*20)
    print(key_ss)