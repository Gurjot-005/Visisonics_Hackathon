import numpy as np
from typing import List


def rotation_score(flow_field: np.ndarray) -> float:
    """
    Detect rotational motion from an optical flow field.
    A rotation has vectors that are tangential to concentric circles around the center.

    flow_field: HxWx2 (dx, dy per pixel)
    returns: float [0, 1] — 1.0 = pure rotation, 0.0 = no rotation
    """
    H, W = flow_field.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # pixel grid
    xs = np.arange(W, dtype=np.float32) - cx
    ys = np.arange(H, dtype=np.float32) - cy
    grid_x, grid_y = np.meshgrid(xs, ys)

    # radial distance from center
    radius = np.sqrt(grid_x ** 2 + grid_y ** 2) + 1e-6

    dx = flow_field[..., 0]
    dy = flow_field[..., 1]
    flow_mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

    # tangential unit vectors for CCW rotation: (-y/r, x/r)
    tan_x = -grid_y / radius
    tan_y =  grid_x / radius

    # normalized flow vectors
    dx_norm = dx / flow_mag
    dy_norm = dy / flow_mag

    # dot product with tangential direction
    dot_ccw = dx_norm * tan_x + dy_norm * tan_y
    # dot product for CW rotation
    dot_cw  = dx_norm * (-tan_x) + dy_norm * (-tan_y)

    # only consider pixels with meaningful motion (top 50% by magnitude)
    threshold = np.percentile(flow_mag, 50)
    mask = flow_mag > threshold

    if mask.sum() < 10:
        return 0.0

    score_ccw = float(np.mean(dot_ccw[mask]))
    score_cw  = float(np.mean(dot_cw[mask]))
    score = max(score_ccw, score_cw)

    # clamp to [0, 1]
    return float(np.clip(score, 0.0, 1.0))


def compute_rotation_scores(flow_fields: List[np.ndarray]) -> np.ndarray:
    scores = [rotation_score(f) for f in flow_fields]
    return np.array(scores, dtype=np.float32)
