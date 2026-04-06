import cv2
import numpy as np
from typing import List, Tuple
from core.frame_sampler import Clip


def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    # frames are HxWx3 uint8 RGB
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow  # HxWx2


def clip_motion_score(clip: Clip) -> Tuple[float, np.ndarray]:
    if len(clip.frames) < 2:
        return 0.0, np.zeros((224, 224, 2), dtype=np.float32)

    flow_magnitudes = []
    last_flow = None
    for i in range(len(clip.frames) - 1):
        flow = compute_optical_flow(clip.frames[i], clip.frames[i + 1])
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(mag)
        last_flow = flow

    mean_mag = float(np.mean([m.mean() for m in flow_magnitudes]))
    avg_flow = last_flow if last_flow is not None else np.zeros((224, 224, 2), dtype=np.float32)
    return mean_mag, avg_flow


def compute_motion_scores(clips: List[Clip]) -> Tuple[np.ndarray, List[np.ndarray]]:
    raw_scores = []
    flow_fields = []
    for clip in clips:
        score, flow = clip_motion_score(clip)
        raw_scores.append(score)
        flow_fields.append(flow)

    raw = np.array(raw_scores, dtype=np.float32)
    max_val = raw.max() if raw.max() > 0 else 1.0
    normalized = raw / max_val
    return normalized, flow_fields
