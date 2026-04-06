import numpy as np
import os
from typing import List
import cv2

from core.frame_sampler import sample_clips
from core.visual_encoder import VisualEncoder
from core.text_encoder import TextEncoder
from core.matcher import match_query_to_clips
from core.motion_filter import compute_motion_scores
from core.rotation_detector import compute_rotation_scores
from core.object_scorer import ObjectScorer
from core.temporal_ranker import rank_clips
from core.segmentor import Segmentor

from explainability.confidence_fuser import fuse
from explainability.constraint_checker import check
from explainability.reason_builder import build_reason, is_rotation_query
from explainability.gradcam import generate_gradcam
from explainability.schemas import FIBAResult

MODELS_DIR = "models"


# =========================
# MODEL LOADER
# =========================
def load_models():
    print("[run] Loading lightweight models...")

    visual_enc = VisualEncoder()
    text_enc = TextEncoder()
    obj_scorer = ObjectScorer()
    segmentor = Segmentor(None)

    return visual_enc, text_enc, obj_scorer, segmentor


# =========================
# BBOX DRAW FUNCTION
# =========================
def save_bbox_image(frame, bbox, clip_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    x1, y1, x2, y2 = bbox
    img = frame.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    path = os.path.join(output_dir, f"clip_{clip_id}_bbox.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return path


# =========================
# PIPELINE
# =========================
def run_pipeline(
    video_path: str,
    query: str,
    top_k: int = 3,
    fps: float = 1.0,
    output_dir: str = "outputs",
    visual_enc=None,
    text_enc=None,
    obj_scorer=None,
    segmentor=None,
) -> List[FIBAResult]:

    print(f"\n{'='*60}")
    print(f"FIBA Pipeline | video: {video_path} | query: '{query}'")
    print(f"{'='*60}")

    # Step 1: sample clips
    clips = sample_clips(video_path, fps=fps)

    if not clips:
        print("[pipeline] No clips extracted.")
        return []

    # Step 2: visual embeddings
    print(f"[pipeline] Encoding {len(clips)} clips visually...")
    clip_embs = visual_enc.encode_all_clips(clips)

    # Step 3: text embedding
    text_emb = text_enc.encode(query)

    # Step 4: similarity scores
    visual_scores = match_query_to_clips(text_emb, clip_embs)

    # Step 5: motion
    print("[pipeline] Computing motion scores...")
    motion_scores, flow_fields = compute_motion_scores(clips)

    # Step 6: rotation
    rotation_scores = compute_rotation_scores(flow_fields)

    # Step 7: object scoring
    print("[pipeline] Running object detection...")
    object_scores = obj_scorer.score_all_clips(clips, query)

    # Step 8: fuse scores
    fused_scores = []
    score_bundles = []

    for i in range(len(clips)):
        bundle = fuse(
            visual=float(visual_scores[i]),
            motion=float(motion_scores[i]),
            rotation=float(rotation_scores[i]),
            obj=float(object_scores[i]),
        )
        score_bundles.append(bundle)
        fused_scores.append(bundle.confidence)

    fused_arr = np.array(fused_scores, dtype=np.float32)

    # Step 9: ranking
    top_clips = rank_clips(fused_arr, top_k=top_k)

    # Step 10: results
    rotation_query = is_rotation_query(query)
    results = []

    for rank, (clip_idx, smoothed_conf) in enumerate(top_clips):
        clip = clips[clip_idx]
        scores = score_bundles[clip_idx]
        constraints = check(scores)

        rotation_inferred = rotation_query and scores.rotation_score > 0.35
        reason = build_reason(scores, constraints, query, rotation_inferred)

        # ✅ segmentation + bbox
        bbox = segmentor.segment(clip)

        bbox_path = save_bbox_image(
            frame=clip.frames[0],
            bbox=bbox,
            clip_id=clip_idx,
            output_dir=output_dir
        )

        # GradCAM
        gradcam_path = generate_gradcam(
            frame=clip.frames[0],
            clip_id=clip_idx,
            visual_score=scores.visual_score,
            output_dir=output_dir,
        )

        result = FIBAResult(
            rank=rank + 1,
            clip_id=clip_idx,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            scores=scores,
            constraints=constraints,
            reason=reason,
            rotation_inferred=rotation_inferred,
            bbox=bbox,
            gradcam_path=gradcam_path,
        )

        results.append(result)

    return results