import numpy as np
import cv2


MOTION_SIZE = (224, 224)


def compute_motion_scores(clips):
    motion_scores = []
    flow_fields = []

    for clip in clips:
        if len(clip.frames) < 2:
            motion_scores.append(0.0)
            flow_fields.append(None)
            continue

        f1_small = cv2.resize(clip.frames[0], MOTION_SIZE, interpolation=cv2.INTER_AREA)
        f2_small = cv2.resize(clip.frames[-1], MOTION_SIZE, interpolation=cv2.INTER_AREA)
        f1 = cv2.cvtColor(f1_small, cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(f2_small, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        score = float(np.mean(mag))

        motion_scores.append(score)
        flow_fields.append(flow)

    motion_scores = np.array(motion_scores)

    # 🔥 NORMALIZE (CRITICAL)
    if np.max(motion_scores) > 0:
        motion_scores = motion_scores / (np.max(motion_scores) + 1e-6)

    return motion_scores, flow_fields
