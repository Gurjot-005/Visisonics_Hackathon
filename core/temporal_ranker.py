import numpy as np
from typing import List, Tuple


def weighted_smooth(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 3:
        return scores

    weights = np.array([0.2, 0.6, 0.2])
    return np.convolve(scores, weights, mode="same")


def rank_clips(fused_scores: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    smoothed = weighted_smooth(fused_scores)
    indices = np.argsort(smoothed)[::-1][:top_k]
    return [(int(i), float(smoothed[i])) for i in indices]