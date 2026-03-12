import os
import json
import subprocess

def ffmpeg_process(
    video_path: str,
    start_time: str,
    duration: str,
    reference_frame_idx: int,
    output_video_path: str,
    output_audio_path: str,
    output_ref_path: str,
) -> bool:
    """
    Processes a video file to extract a clip, its audio, and a specific reference frame.

    Args:
        video_path (str): Path to the source video file.
        start_time (str): The start time in seconds for the clip.
        duration (str): The duration in seconds of the clip.
        reference_frame_idx (int): The index of the frame to extract from the final clip.
        output_video_path (str): Path to save the output video clip.
        output_audio_path (str): Path to save the extracted audio from the clip.
        output_ref_path (str): Path to save the extracted reference frame image.

    Returns:
        bool: True if all operations were successful, False otherwise.
    """
    try:
        # Step 1: Cut the video clip and set its FPS to 25.
        # The audio is copied directly ('-c:a copy') in this step to maintain quality
        # before the final, separate audio extraction.
        clip_cmd = [
            'ffmpeg',
            '-y',                           # Overwrite output file if it exists
            '-i', video_path,      # Input file
            '-ss', start_time,         # Seek to the start time
            '-t', duration,            # Specify the duration of the clip
            '-r', '25',                     # Set the frame rate to a fixed 25 FPS
            '-c:a', 'copy',                 # Copy the audio stream without re-encoding
            output_video_path                 # Output file path for the video clip
        ]
        subprocess.run(clip_cmd, check=True, capture_output=True, text=True)

        # Step 2: Extract the audio from the newly created clip.
        # This ensures the audio perfectly matches the clipped video.
        # '-c:a copy' preserves the original codec and sampling rate.
        audio_cmd = [
            'ffmpeg',
            '-y',                           # Overwrite output file if it exists
            '-i', output_video_path,          # Input is the clip we just created
            '-vn',                          # Disable video recording
            '-c:a', 'copy',                 # Copy the audio stream directly
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            '-async', '1',
            output_audio_path                 # Output file path for the audio
        ]
        subprocess.run(audio_cmd, check=True, capture_output=True, text=True)

        # Step 3: Extract the specific reference frame from the clip.
        # The frame index is relative to the newly created clip.
        frame_cmd = [
            'ffmpeg',
            '-y',                           # Overwrite output file if it exists
            '-i', output_video_path,          # Input is the clip we just created
            '-vf', f"select='eq(n,{reference_frame_idx})'", # Video filter to select a frame by its index 'n'
            '-vframes', '1',                # Tell FFmpeg to output only one frame
            output_ref_path            # Output file path for the image
        ]
        subprocess.run(frame_cmd, check=True, capture_output=True, text=True)

        return True
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # Catch errors if ffmpeg is not found or if a command fails.
        print(f"An error occurred during FFmpeg processing: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg stderr: {e.stderr}")
        return False

def main():
    
    meta_path = "meta.json"
    output_dir = "clips/"
    raw_video_dir = "videos_raw/"
    os.makedirs(output_dir, exist_ok=True)

    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    for key in meta_dict:
        start_time = meta_dict[key]['start_time']
        duration = meta_dict[key]['duration']
        reference_frame_idx = meta_dict[key]['reference_frame_idx']

        video_path = os.path.join(raw_video_dir, key.replace("set2/", ""))

        ffmpeg_process(
            video_path = video_path,
            start_time = start_time,
            duration = duration,
            reference_frame_idx = reference_frame_idx,
            output_video_path = os.path.join(output_dir, key.replace("set2/", "")),
            output_audio_path = os.path.join(output_dir, key.replace("set2/", "").replace(".mp4", ".wav")),
            output_ref_path = os.path.join(output_dir, key.replace("set2/", "").replace(".mp4", ".jpg")),
        )

if __name__ == "__main__":
    main()