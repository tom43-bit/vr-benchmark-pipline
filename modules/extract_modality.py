import logging
from pathlib import Path

import laion_clap
import pandas as pd
import torch
from colorlog import ColoredFormatter
from msclap import CLAP
from tqdm import tqdm
import psutil
import os
from einops import rearrange
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader

from av_bench.args import get_eval_parser
from av_bench.data.video_dataset import VideoDataset, error_avoidance_collate
from av_bench.synchformer.synchformer import Synchformer

import torchaudio

from av_bench.data.audio_dataset import (AudioDataset, ImageBindAudioDataset,
                                         SynchformerAudioDataset, pad_or_truncate)
from av_bench.extraction_models import ExtractionModels

import sys

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"
log = logging.getLogger()
device = 'cuda'

def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)

setup_eval_logging()

def encode_video_with_sync(synchformer: Synchformer, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, C, H, W) H/W: 224

    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    # partition the video
    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
    x = synchformer.extract_vfeats(x)
    x = rearrange(x, '(b s) 1 t d -> b s t d', b=b)
    return x

def encode_video_with_imagebind(imagebind: imagebind_model, x: torch.Tensor) -> torch.Tensor:
    # x: B * NUM_CROPS * T * C * H * W
    clips = []
    b, num_crops, t, c, h, w = x.shape
    for i in range(t - 1):
        clips.append(x[:, :, i:i + 2])
    clips = torch.cat(clips, dim=1)

    # clips: B * (NUM_CROPS * NUM_CLIPS) * 2 * C * H * W
    clips = rearrange(clips, 'b n t c h w -> b n c t h w')

    emb = imagebind({ModalityType.VISION: clips})
    return emb[ModalityType.VISION]

def encode_audio_with_sync(synchformer: Synchformer, x: torch.Tensor,
                           mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    b, t = x.shape

    # partition the video
    segment_size = 10240
    step_size = 10240 // 2
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    x = mel(x)
    x = torch.log(x + 1e-6)
    x = pad_or_truncate(x, 66)

    mean = -4.2677393
    std = 4.5689974
    x = (x - mean) / (2 * std)
    # x: B * S * 128 * 66
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x


@torch.inference_mode()
def extract_all_features(video_list,video_dict,video_path,output_cache_path,
                            skip_video_related = False,skip_clap = False,
                            if_ref_video=False,if_ref_audio=False,ref_video_path=None,ref_audio_path=None,_syncformer_ckpt_path = "./weight",
                            no_audio = False):
    _clap_ckpt_path = os.path.join(_syncformer_ckpt_path,"music_speech_audioset_epoch_15_esc_89.98.pt")
    _syncformer_ckpt_path = os.path.join(_syncformer_ckpt_path,'synchformer_state_dict.pth')
    gt_cache = os.path.join(output_cache_path,'gt_cache')
    pred_cache = os.path.join(output_cache_path,'pred_cache')
    video_paths = [Path(os.path.join(video_path,f)) for f in video_list if f.endswith('.mp4')]

    #定义！！！
    audio_length: float = 3 #args.audio_length
    num_workers: int = 32 #args.num_workers
    batch_size: int = 1 #args.gt_batch_size

    log.info(f'{len(video_paths)} videos found.')

    v_dataset = VideoDataset(video_paths, duration_sec=audio_length)
    v_loader = DataLoader(v_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=error_avoidance_collate)

    #video model 初始化
    sync_model = Synchformer().to(device).eval()
    sd = torch.load(_syncformer_ckpt_path, weights_only=True)
    sync_model.load_state_dict(sd)

    cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)
    cmp_encode_video_with_imagebind = torch.compile(encode_video_with_imagebind)

    imagebind = imagebind_model.imagebind_huge(pretrained=True).to(device).eval() #视频提取模型

    log.info('Extracting features...')
    output_sync_features = {}
    output_ib_features = {}
    for data in tqdm(v_loader):

        name = data['name']
        ib_video = data['ib_video'].to(device)
        sync_video = data['sync_video'].to(device)

        sync_features = cmp_encode_video_with_sync(sync_model, sync_video)
        ib_features = cmp_encode_video_with_imagebind(imagebind, ib_video)

        sync_features = sync_features.cpu().detach()
        ib_features = ib_features.cpu().detach()

        for i, n in enumerate(name):
            # saving a view will save the entire tensor so don't
            output_sync_features[n] = sync_features[i].clone()
            output_ib_features[n] = ib_features[i].clone()

    torch.save(output_sync_features, os.path.join(gt_cache , 'synchformer_video.pth'))
    torch.save(output_ib_features, os.path.join(gt_cache , 'imagebind_video.pth'))


    #释放变量
    del output_sync_features
    del output_ib_features
    del sync_model
    del imagebind
    del v_loader
    torch.cuda.empty_cache()


    #text model 初始化
    laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False,
                                              amodel='HTSAT-base').cuda().eval()
    laion_clap_model.load_ckpt(_clap_ckpt_path, verbose=False)
    ms_clap_model = CLAP(version='2023', use_cuda=True)

    all_laion_clap = {}
    all_ms_clap = {}
    for video in tqdm(video_list):

        caption = video_dict[video]['text_prompt']
        text_data = [caption]
        text_embed = laion_clap_model.get_text_embedding(caption, use_tensor=True)
        all_laion_clap[video.replace('.mp4', '')] = text_embed.cpu().squeeze()
        text_embed = ms_clap_model.get_text_embeddings(text_data)
        all_ms_clap[video.replace('.mp4', '')] = text_embed.cpu().squeeze()

    torch.save(all_laion_clap, os.path.join(gt_cache , 'clap_laion_text.pth'))
    torch.save(all_ms_clap, os.path.join(gt_cache,'clap_ms_text.pth') )

    #释放变量
    del all_laion_clap
    del all_ms_clap
    del laion_clap_model
    del ms_clap_model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()



    if not no_audio:
        models = ExtractionModels().to(device).eval() #音频特征提取模型

        a_dataset = AudioDataset(video_paths, audio_length=audio_length, sr=16000)
        a_loader = DataLoader(a_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        # extract features with PANN
        out_dict = {}
        for wav, filename in tqdm(a_loader):
            wav = wav.squeeze(1).float().to(device)

            features = models.panns(wav)
            features = {k: v.cpu() for k, v in features.items()}
            for i, f_name in enumerate(filename):
                out_dict[f_name] = {k: v[i] for k, v in features.items()}

    
        pann_feature_path = os.path.join(pred_cache, 'pann_features.pth')
        log.info(f'Saving {len(out_dict)} features to {pann_feature_path}')
        torch.save(out_dict, pann_feature_path)
        del out_dict

        # extract features with VGGish
        out_dict = {}
        for wav, filename in tqdm(a_loader):
            wav = wav.squeeze(1).float()
            features = models.vggish(wav).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]

    
        vggish_feature_path = os.path.join(pred_cache, 'vggish_features.pth')
        log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
        torch.save(out_dict, vggish_feature_path)
        del out_dict

        video_path = Path(video_path)
        audios = sorted(list(video_path.glob('*.wav')) + list(video_path.glob('*.flac')) + list(video_path.glob('*.mp4')),
                    key=lambda x: x.stem)

        if not skip_video_related and not no_audio:
            # extract features with ImageBind
            dataset = ImageBindAudioDataset(audios)
            loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
            out_dict = {}
            for wav, filename in tqdm(loader):
                wav = wav.squeeze(1).to(device)
                features = models.imagebind({ModalityType.AUDIO: wav})[ModalityType.AUDIO].cpu()
                for i, f_name in enumerate(filename):
                    out_dict[f_name] = features[i]
        
            imagebind_feature_path = os.path.join(pred_cache, 'imagebind_audio.pth')
            log.info(f'Saving {len(out_dict)} features to {imagebind_feature_path}')
            torch.save(out_dict, imagebind_feature_path)

            # extract features with Synchformer
            dataset = SynchformerAudioDataset(audios, duration=audio_length)
            loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
            out_dict = {}
            for wav, filename in tqdm(loader):
                wav = wav.to(device)
                features = encode_audio_with_sync(models.synchformer, wav,
                                              models.sync_mel_spectrogram).cpu()
                for i, f_name in enumerate(filename):
                    out_dict[f_name] = features[i]
        
            synchformer_feature_path = os.path.join(pred_cache, 'synchformer_audio.pth')
            log.info(f'Saving {len(out_dict)} features to {synchformer_feature_path}')
            torch.save(out_dict, synchformer_feature_path)

    if not skip_clap and not no_audio:
        # extract features with CLAP
        dataset = AudioDataset(audios, audio_length=audio_length, sr=48_000)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.squeeze(1).to(device)
            clap_features = models.laion_clap.get_audio_embedding_from_data(wav,
                                                                            use_tensor=True).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = clap_features[i].squeeze()
        
        clap_feature_path = os.path.join(pred_cache, 'clap_laion_audio.pth')
        log.info(f'Saving {len(out_dict)} features to {clap_feature_path}')
        torch.save(out_dict, clap_feature_path)

        out_dict = {}
        for i in range(0, len(audios), batch_size):
            audio_paths = audios[i:i + batch_size]
            embeddings = models.ms_clap.get_audio_embeddings(audio_paths)
            print(f"原始形状: {embeddings.shape}")  # 查看形状
            ms_features = embeddings.cpu()
            print(f"squeeze前形状: {ms_features.shape}")
            for f_name, f in zip(audio_paths, ms_features):
                out_dict[f_name.stem] = f
        ms_feature_path = os.path.join(pred_cache, 'clap_ms_audio.pth')
        log.info(f'Saving {len(out_dict)} features to {ms_feature_path}')
        torch.save(out_dict, ms_feature_path)

    if not no_audio:
        # PassT, need 32K sampling rate
        dataset = AudioDataset(audios, audio_length=audio_length, sr=32_000)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        out_features = {}
        out_logits = {}
        for wav, filename in tqdm(loader):
            wav = wav.squeeze(1).float().to(device)

            if (wav.size(-1) >= 320000):
                wav = wav[..., :320000]
            else:
                wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

            features = models.passt_model(wav).cpu()
            # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
            # logits is 527 dim; features is 768
            logits = features[:, :527]
            features = features[:, 527:]
            for i, f_name in enumerate(filename):
                out_features[f_name] = features[i]
                out_logits[f_name] = logits[i]
    
        passt_feature_path = os.path.join(pred_cache, 'passt_features_embed.pth')
        log.info(f'Saving {len(out_features)} features to {passt_feature_path}')
        torch.save(out_features, passt_feature_path)

        passt_feature_path = os.path.join(pred_cache, 'passt_logits.pth')
        log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
        torch.save(out_logits, passt_feature_path)


    if if_ref_video:
        rv_dataset = VideoDataset(ref_video_path, duration_sec=audio_length)
        rv_loader = DataLoader(rv_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=error_avoidance_collate)

        output_sync_features = {}
        output_ib_features = {}
        for data in tqdm(rv_loader):
            name = data['name']
            ib_video = data['ib_video'].to(device)
            sync_video = data['sync_video'].to(device)

            sync_features = cmp_encode_video_with_sync(sync_model, sync_video)
            ib_features = cmp_encode_video_with_imagebind(imagebind, ib_video)

            sync_features = sync_features.cpu().detach()
            ib_features = ib_features.cpu().detach()

            for i, n in enumerate(name):
                # saving a view will save the entire tensor so don't
                output_sync_features[n] = sync_features[i].clone()
                output_ib_features[n] = ib_features[i].clone()

        torch.save(output_sync_features, pred_cache / 'synchformer_video.pth')
        torch.save(output_ib_features, pred_cache / 'imagebind_video.pth')

    if if_ref_audio:
        # extract features with PANN
        out_dict = {}
        for wav, filename in tqdm(a_loader):
            wav = wav.squeeze(1).float().to(device)

            features = models.panns(wav)
            features = {k: v.cpu() for k, v in features.items()}
            for i, f_name in enumerate(filename):
                out_dict[f_name] = {k: v[i] for k, v in features.items()}

    
        pann_feature_path = gt_cache / 'pann_features.pth'
        log.info(f'Saving {len(out_dict)} features to {pann_feature_path}')
        torch.save(out_dict, pann_feature_path)
        del out_dict

        # extract features with VGGish
        out_dict = {}
        for wav, filename in tqdm(a_loader):
            wav = wav.squeeze(1).float()
            features = models.vggish(wav).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]

    
        vggish_feature_path = gt_cache / 'vggish_features.pth'
        log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
        torch.save(out_dict, vggish_feature_path)
        del out_dict

        if not skip_video_related:
            # extract features with ImageBind
            dataset = ImageBindAudioDataset(audios)
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True)
            out_dict = {}
            for wav, filename in tqdm(loader):
                wav = wav.squeeze(1).to(device)
                features = models.imagebind({ModalityType.AUDIO: wav})[ModalityType.AUDIO].cpu()
                for i, f_name in enumerate(filename):
                    out_dict[f_name] = features[i]
        
            imagebind_feature_path = gt_cache / 'imagebind_audio.pth'
            log.info(f'Saving {len(out_dict)} features to {imagebind_feature_path}')
            torch.save(out_dict, imagebind_feature_path)

            # extract features with Synchformer
            dataset = SynchformerAudioDataset(audios, duration=audio_length)
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True)
            out_dict = {}
            for wav, filename in tqdm(loader):
                wav = wav.to(device)
                features = encode_audio_with_sync(models.synchformer, wav,
                                              models.sync_mel_spectrogram).cpu()
                for i, f_name in enumerate(filename):
                    out_dict[f_name] = features[i]
        
            synchformer_feature_path = gt_cache / 'synchformer_audio.pth'
            log.info(f'Saving {len(out_dict)} features to {synchformer_feature_path}')
            torch.save(out_dict, synchformer_feature_path)

        if not skip_clap:
            # extract features with CLAP
            dataset = AudioDataset(audios, audio_length=audio_length, sr=48_000)
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True)
            out_dict = {}
            for wav, filename in tqdm(loader):
                wav = wav.squeeze(1).to(device)
                clap_features = models.laion_clap.get_audio_embedding_from_data(wav,
                                                                            use_tensor=True).cpu()
                for i, f_name in enumerate(filename):
                    out_dict[f_name] = clap_features[i].squeeze()
        
            clap_feature_path = gt_cache / 'clap_laion_audio.pth'
            log.info(f'Saving {len(out_dict)} features to {clap_feature_path}')
            torch.save(out_dict, clap_feature_path)

            out_dict = {}
            for i in range(0, len(audios), batch_size):
                audio_paths = audios[i:i + batch_size]
                ms_features = models.ms_clap.get_audio_embeddings(audio_paths).cpu().squeeze()
                for f_name, f in zip(audio_paths, ms_features):
                    out_dict[f_name.stem] = f
            ms_feature_path = gt_cache / 'clap_ms_audio.pth'
            log.info(f'Saving {len(out_dict)} features to {ms_feature_path}')
            torch.save(out_dict, ms_feature_path)

        # PassT, need 32K sampling rate
        dataset = AudioDataset(audios, audio_length=audio_length, sr=32_000)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        out_features = {}
        out_logits = {}
        for wav, filename in tqdm(loader):
            wav = wav.squeeze(1).float().to(device)

            if (wav.size(-1) >= 320000):
                wav = wav[..., :320000]
            else:
                wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

            features = models.passt_model(wav).cpu()
            # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
            # logits is 527 dim; features is 768
            logits = features[:, :527]
            features = features[:, 527:]
            for i, f_name in enumerate(filename):
                out_features[f_name] = features[i]
                out_logits[f_name] = logits[i]
    
        passt_feature_path = gt_cache / 'passt_features_embed.pth'
        log.info(f'Saving {len(out_features)} features to {passt_feature_path}')
        torch.save(out_features, passt_feature_path)

        passt_feature_path = gt_cache / 'passt_logits.pth'
        log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
        torch.save(out_logits, passt_feature_path)    
