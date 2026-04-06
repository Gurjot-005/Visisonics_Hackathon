import numpy as np
import cv2
import os
from typing import Optional


def generate_gradcam(
    frame: np.ndarray,
    clip_id: int,
    visual_score: float,
    output_dir: str = "outputs",
) -> Optional[str]:
    """
    Lightweight GradCAM approximation without backprop.
    Uses the visual score as a proxy weight and generates a saliency map
    based on edge density and motion variance in the frame.

    For the basic pipeline this is a deterministic heatmap that highlights
    high-frequency regions — replace with true GradCAM once PyTorch hooks
    are wired to the MobileViT encoder.
    """
    os.makedirs(output_dir, exist_ok=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Laplacian edge map as saliency proxy
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap).astype(np.float32)

    # Blur to smooth the heatmap
    lap = cv2.GaussianBlur(lap, (21, 21), 0)

    # Normalize to [0, 1]
    if lap.max() > 0:
        lap = lap / lap.max()

    # Scale by visual score so high-confidence clips have stronger heatmaps
    lap = lap * visual_score

    # Colorize
    heatmap = cv2.applyColorMap((lap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.55, heatmap, 0.45, 0)

    out_path = os.path.join(output_dir, f"clip_{clip_id}_gradcam.png")
    cv2.imwrite(out_path, overlay)
    return out_path
