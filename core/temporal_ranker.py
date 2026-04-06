import numpy as np
from typing import List, Tuple


def sliding_window_smooth(scores: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth scores with a sliding window average to reward temporally consistent regions."""
    if len(scores) < window:
        return scores
    smoothed = np.convolve(scores, np.ones(window) / window, mode="same")
    return smoothed


def rank_clips(fused_scores: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    smoothed = sliding_window_smooth(fused_scores)
    indices = np.argsort(smoothed)[::-1][:top_k]
    return [(int(i), float(smoothed[i])) for i in indices]
