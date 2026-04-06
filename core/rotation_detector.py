import numpy as np


def rotation_score(flow_field):
    # 🔥 HANDLE NONE (CRITICAL FIX)
    if flow_field is None:
        return 0.0

    H, W = flow_field.shape[:2]

    center_x = W / 2
    center_y = H / 2

    y, x = np.mgrid[0:H, 0:W]

    dx = x - center_x
    dy = y - center_y

    fx = flow_field[..., 0]
    fy = flow_field[..., 1]

    # perpendicular component (rotation indicator)
    dot = fx * (-dy) + fy * dx

    mag_flow = np.sqrt(fx**2 + fy**2) + 1e-6
    mag_pos = np.sqrt(dx**2 + dy**2) + 1e-6

    score = np.abs(dot) / (mag_flow * mag_pos)

    return float(np.mean(score))


def compute_rotation_scores(flow_fields):
    scores = []

    for f in flow_fields:
        if f is None:
            scores.append(0.0)  # 🔥 FIX
        else:
            scores.append(rotation_score(f))

    scores = np.array(scores)

    # normalize
    if np.max(scores) > 0:
        scores = scores / (np.max(scores) + 1e-6)

    return scores