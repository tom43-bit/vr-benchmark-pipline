from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

from decord import VideoReader

def dino_transform(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.
    
    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.
    
    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.
    
    Raises:
    - NotImplementedError: If the video format is not supported.
    
    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
            num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
        num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def identity_consistency(model, video_dict, video_list, base_path, result_csv, device, read_frame=False):
    sim = 0.0
    cnt = 0
    #video_results = []
    if read_frame:
        image_transform = dino_transform_Image(224)
    else:
        image_transform = dino_transform(224)
    for video_n in tqdm(video_list, False):
        video_path = os.path.join(base_path,video_n)
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
        else:
            images = load_video(video_path)
            images = image_transform(images)
        ref_image_transform = dino_transform_Image(224)
        ref_image = ref_image_transform(Image.open(video_dict[video_n]['image_path']))
        ref_image = ref_image.unsqueeze(0).to(device)
        output = model(ref_image)
        ref_features = output.pooler_output
        ref_features = F.normalize(ref_features, dim=-1, p=2)
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                output = model(image)
                image_features = output.pooler_output
                image_features = F.normalize(image_features, dim=-1, p=2)
                sim_pre = max(0.0, F.cosine_similarity(ref_features, image_features).item())
                video_sim += sim_pre
                cnt += 1
        sim_per_images = video_sim / len(images)
        result_csv.loc[result_csv['video_name'] == video_n, 'ID'] = sim_per_images
        sim += video_sim
        #video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    print(sim_per_frame)
    result_csv.loc[result_csv.index[-1], 'ID'] = sim_per_frame

def compute_ID(video_list,video_dict, base_path, result_csv, device):

    os.environ['HF_TOKEN'] = "your_hf_token" 

    model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m' 
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    identity_consistency(model, video_dict, video_list, base_path, result_csv, device)
