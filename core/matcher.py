import numpy as np
from typing import List, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_query_to_clips(text_emb: np.ndarray, clip_embs: np.ndarray) -> np.ndarray:
    # text_emb: D,  clip_embs: N x D
    # returns scores: N,  values in [-1, 1]
    norms = np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-8
    clip_embs_norm = clip_embs / norms
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    scores = clip_embs_norm @ text_norm  # N,
    # normalize to [0, 1]
    scores = (scores + 1.0) / 2.0
    return scores


def top_k_clips(scores: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in indices]
