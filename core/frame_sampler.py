import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Clip:
    clip_id: int
    frames: List[np.ndarray]
    start_sec: float
    end_sec: float


def sample_clips(video_path: str, fps: float = 2.0, clip_duration: float = 1.0, stride: float = 0.5) -> List[Clip]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    all_frames = []
    all_timestamps = []

    frame_interval = int(video_fps / fps)
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
        all_timestamps.append(frame_idx / video_fps)
        frame_idx += frame_interval

    cap.release()

    frames_per_clip = max(1, int(clip_duration * fps))
    stride_frames = max(1, int(stride * fps))

    clips = []
    clip_id = 0
    i = 0
    while i < len(all_frames):
        end = min(i + frames_per_clip, len(all_frames))
        clip_frames = all_frames[i:end]
        start_t = all_timestamps[i]
        end_t = all_timestamps[end - 1]
        clips.append(Clip(clip_id=clip_id, frames=clip_frames, start_sec=start_t, end_sec=end_t))
        clip_id += 1
        i += stride_frames

    print(f"[frame_sampler] {video_path}: {len(clips)} clips from {duration_sec:.1f}s video")
    return clips
