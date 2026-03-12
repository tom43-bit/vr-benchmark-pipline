import os
import json
import subprocess
from tqdm import tqdm

def download_video(url: str, data_source: str, output_path: str) -> bool:
    """
    Downloads a video from a given URL using yt-dlp with multiple fallback strategies.
    It automatically detects the source (Bilibili or YouTube) to apply the best strategy.

    Args:
        url (str): The URL of the video to download.
        data_source (str): The source of url.
        output_path (str): The full path where the video should be saved, including filename and extension.

    Returns:
        bool: True if download was successful, False otherwise.
    """

    cmd = [
        "yt-dlp",
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio',
        '--skip-unavailable-fragments',
        '--merge-output-format', 'mp4',
        url, "--output",
        output_path, "--external-downloader", "aria2c",
        "--external-downloader-args", '"-x 16 -k 1M"'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True

    except Exception as e:
        print(e)

    return False

def main():
    
    meta_path = "meta.json"
    output_dir = "videos_raw/"
    os.makedirs(output_dir, exist_ok=True)

    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    for key in tqdm(meta_dict):
        url = meta_dict[key]['url']
        data_source = meta_dict[key]['source']
        download_video(
            url = url,
            data_source = data_source,
            output_path = os.path.join(output_dir, key.replace("set3/", "")),
        )

if __name__ == "__main__":
    main()