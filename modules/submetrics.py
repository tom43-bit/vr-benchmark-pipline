def av_bench_eval(queue, video_list, video_dict, video_path,output_cache_path, unpaired, if_ref_video=False, if_ref_audio=False):
    import modules.extract_modality as extract
    from av_bench.evaluate import evaluate
    print(f"[子进程 {multiprocessing.current_process().name}] 开始执行...")
    extract.extract_all_features(video_list=video_list,video_dict=video_dict,video_path=video_path,output_cache_path=output_cache_path,
                                    if_ref_video=False,if_ref_audio=False)
    num_samples = 1
    gt_cache = os.path.join(output_cache_path,'gt_cache')
    pred_cache = os.path.join(output_cache_path,'pred_cache')
    output_metrics = evaluate(gt_audio_cache=gt_cache,
                              pred_audio_cache=pred_cache,
                              num_samples=num_samples,
                              is_paired=not unpaired)
