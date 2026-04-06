import os
import time

import cv2
import numpy as np

from core.frame_sampler import sample_clips
from core.motion_filter import compute_motion_scores
from core.object_scorer import ObjectScorer
from core.rotation_detector import compute_rotation_scores
from core.segmentor import Segmentor
from core.temporal_ranker import rank_clips
from explainability.confidence_fuser import fuse
from explainability.constraint_checker import check
from explainability.gradcam import generate_gradcam
from explainability.reason_builder import build_reason
from explainability.schemas import FIBAResult

try:
    from core.text_encoder import TextEncoder
    from core.visual_encoder import VisualEncoder
except Exception:
    TextEncoder = None
    VisualEncoder = None


MOTION_THRESHOLD = 0.15
SHORTLIST_MIN = 72
SHORTLIST_MULTIPLIER = 24
INTERACTION_SHORTLIST = 8
ACTION_ONLY_SHORTLIST = 16
FAST_SHORTLIST_MIN = 24
FAST_SHORTLIST_MULTIPLIER = 8
RESULT_TIME_GAP_SEC = 1.0


# -------------------------------
# LOAD MODELS
# -------------------------------
def load_models():
    print("[run] Loading lightweight models...")

    visual_enc = None
    text_enc = None

    if VisualEncoder is not None and TextEncoder is not None:
        try:
            visual_enc = VisualEncoder()
            text_enc = TextEncoder()
        except Exception as exc:
            print(f"[run] Semantic encoders unavailable, continuing without visual scoring: {exc}")
    else:
        print("[run] Semantic encoders unavailable, continuing without visual scoring")

    return visual_enc, text_enc, ObjectScorer(), Segmentor(None)


# -------------------------------
# SAVE BBOX IMAGE
# -------------------------------
def save_bbox_image(frame, bbox, clip_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    x1, y1, x2, y2 = bbox
    img = frame.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    path = os.path.join(output_dir, f"clip_{clip_id}_bbox.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return path


def save_frame_image(frame, clip_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"clip_{clip_id}_frame.png")
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def compute_visual_scores(clips, query, visual_enc, text_enc, obj_scorer):
    if visual_enc is None or text_enc is None:
        return np.zeros(len(clips), dtype=np.float32)

    expanded_query = obj_scorer.expand_query(query)
    clip_embeddings = visual_enc.encode_all_clips(clips)
    query_embedding = text_enc.encode(expanded_query)

    scores = clip_embeddings @ query_embedding
    scores = np.clip(scores, -1.0, 1.0)
    scores = (scores + 1.0) / 2.0
    return scores.astype(np.float32)


def shortlist_candidate_indices(motion_scores, rotation_scores, top_k):
    if len(motion_scores) == 0:
        return np.array([], dtype=np.int32)

    stage1_scores = 0.7 * motion_scores + 0.3 * rotation_scores
    shortlist_size = min(len(stage1_scores), max(SHORTLIST_MIN, top_k * SHORTLIST_MULTIPLIER))
    shortlist = np.argsort(stage1_scores)[::-1][:shortlist_size]
    return np.sort(shortlist.astype(np.int32))


def refine_shortlist_indices(motion_scores, rotation_scores, object_scores, top_k):
    if len(motion_scores) == 0:
        return np.array([], dtype=np.int32)

    stage2_scores = 0.45 * motion_scores + 0.15 * rotation_scores + 0.40 * object_scores
    shortlist_size = min(len(stage2_scores), max(SHORTLIST_MIN, top_k * SHORTLIST_MULTIPLIER))
    shortlist = np.argsort(stage2_scores)[::-1][:shortlist_size]
    return np.sort(shortlist.astype(np.int32))


def fast_shortlist_indices(motion_scores, rotation_scores, object_scores, top_k):
    if len(motion_scores) == 0:
        return np.array([], dtype=np.int32)

    stage_scores = 0.40 * motion_scores + 0.20 * rotation_scores + 0.40 * object_scores
    shortlist_size = min(len(stage_scores), max(FAST_SHORTLIST_MIN, top_k * FAST_SHORTLIST_MULTIPLIER))
    shortlist = np.argsort(stage_scores)[::-1][:shortlist_size]
    return np.sort(shortlist.astype(np.int32))


def is_interaction_query(query):
    query_text = query.lower()
    return any(term in query_text for term in ["holding", "hold", "taking", "take", "grabbing", "grab", "carrying", "carry", "using", "cutting", "cut", "slicing", "slice", "chopping", "chop"])


def is_fridge_interaction_query(query):
    query_text = query.lower()
    has_fridge = ("fridge" in query_text) or ("refrigerator" in query_text)
    has_transfer = any(term in query_text for term in ["taking", "take", "grabbing", "grab", "getting", "get", "pulling", "pull", "removing", "remove", "from"])
    return has_fridge and has_transfer


def select_bbox(frame, object_detection, segment_bbox, object_score=0.0, query=""):
    frame_h, frame_w = frame.shape[:2]
    object_bbox = None
    object_label = None

    if isinstance(object_detection, tuple):
        if len(object_detection) >= 2:
            object_bbox, object_label = object_detection[0], object_detection[1]
        else:
            object_bbox = object_detection
    else:
        object_bbox = object_detection

    def normalize(box):
        if box is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))
        return (x1, y1, x2, y2)

    object_bbox = normalize(object_bbox)
    segment_bbox = normalize(segment_bbox)

    if object_bbox is None:
        return segment_bbox
    if segment_bbox is None:
        return object_bbox

    obj_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])
    seg_area = (segment_bbox[2] - segment_bbox[0]) * (segment_bbox[3] - segment_bbox[1])
    frame_area = frame_h * frame_w
    is_default_segment = segment_bbox == (50, 50, 180, 180)
    large_object_labels = {
        "refrigerator", "oven", "microwave", "tv", "bed", "couch", "chair",
        "dining table", "sink", "toilet",
    }
    prefers_large_object = (
        object_score >= 0.80 and (
            (object_label in large_object_labels) or
            ("fridge" in query.lower()) or
            ("refrigerator" in query.lower())
        )
    )
    food_prep_query = any(term in query.lower() for term in ["making", "make", "preparing", "prepare", "food", "sandwich"])

    if prefers_large_object and obj_area < 0.95 * frame_area:
        return object_bbox

    if food_prep_query:
        object_touches_edge = (
            object_bbox[0] <= 2 or object_bbox[1] <= 2 or
            object_bbox[2] >= frame_w - 2 or object_bbox[3] >= frame_h - 2
        )
        small_object = obj_area < 0.06 * frame_area
        object_misaligned = segment_bbox is not None and box_intersection(object_bbox, segment_bbox) <= 0
        if segment_bbox is not None and (object_touches_edge or small_object or object_misaligned):
            return segment_bbox

    if is_default_segment and object_score >= 0.70:
        return object_bbox

    if obj_area > 0.45 * frame_area or seg_area < obj_area:
        return segment_bbox

    return object_bbox


def detection_parts(object_detection):
    if isinstance(object_detection, tuple):
        if len(object_detection) == 3:
            return object_detection[0], object_detection[1]
        if len(object_detection) == 2:
            return object_detection
    return object_detection, None


def detection_frame_index(object_detection):
    if isinstance(object_detection, tuple) and len(object_detection) == 3:
        return int(object_detection[2])
    return 0


def box_area(box):
    if box is None:
        return 0.0
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_center(box):
    return np.array([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], dtype=np.float32)


def hand_object_proximity(object_box, hand_boxes):
    if object_box is None or not hand_boxes:
        return 0.0

    obj_center = box_center(object_box)
    obj_scale = max(20.0, np.sqrt(box_area(object_box)))
    best = 0.0

    for hand_box in hand_boxes:
        hand_center = box_center(hand_box)
        distance = float(np.linalg.norm(obj_center - hand_center))
        proximity = max(0.0, 1.0 - (distance / (2.0 * obj_scale)))
        best = max(best, proximity)

    return best


def box_intersection(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def hand_object_overlap(object_box, hand_boxes):
    if object_box is None or not hand_boxes:
        return 0.0
    obj_area = max(1.0, box_area(object_box))
    best = 0.0
    for hand_box in hand_boxes:
        inter = box_intersection(object_box, hand_box)
        hand_area = max(1.0, box_area(hand_box))
        overlap = max(inter / obj_area, inter / hand_area)
        best = max(best, overlap)
    return best


def merge_boxes(boxes):
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return None
    x1 = min(int(box[0]) for box in valid_boxes)
    y1 = min(int(box[1]) for box in valid_boxes)
    x2 = max(int(box[2]) for box in valid_boxes)
    y2 = max(int(box[3]) for box in valid_boxes)
    return (x1, y1, x2, y2)


def expand_box(box, frame_shape, pad_x=0.08, pad_y=0.08):
    if box is None:
        return None
    frame_h, frame_w = frame_shape[:2]
    width = max(1, int(box[2] - box[0]))
    height = max(1, int(box[3] - box[1]))
    dx = max(8, int(width * pad_x))
    dy = max(8, int(height * pad_y))
    return (
        max(0, int(box[0]) - dx),
        max(0, int(box[1]) - dy),
        min(frame_w, int(box[2]) + dx),
        min(frame_h, int(box[3]) + dy),
    )


def nearest_hand_box(object_box, hand_boxes):
    if object_box is None or not hand_boxes:
        return None
    obj_center = box_center(object_box)
    best_box = None
    best_distance = None
    for hand_box in hand_boxes:
        hand_center = box_center(hand_box)
        distance = float(np.linalg.norm(obj_center - hand_center))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_box = hand_box
    return best_box


def box_touches_edge(box, frame_shape, margin=4):
    if box is None or frame_shape is None:
        return False
    frame_h, frame_w = frame_shape[:2]
    return (
        int(box[0]) <= margin or int(box[1]) <= margin or
        int(box[2]) >= frame_w - margin or int(box[3]) >= frame_h - margin
    )


def boxes_close(box_a, box_b, scale=1.2):
    if box_a is None or box_b is None:
        return False
    center_a = box_center(box_a)
    center_b = box_center(box_b)
    distance = float(np.linalg.norm(center_a - center_b))
    size_a = max(20.0, np.sqrt(box_area(box_a)))
    size_b = max(20.0, np.sqrt(box_area(box_b)))
    return distance <= scale * max(size_a, size_b)


def hand_region_box(hand_boxes, frame_shape, max_hands=2):
    if not hand_boxes:
        return None
    ranked = sorted(hand_boxes, key=box_area, reverse=True)[:max_hands]
    merged = merge_boxes(ranked)
    return expand_box(merged, frame_shape, pad_x=0.12, pad_y=0.12)


def action_aware_bbox(query, object_detection, selected_bbox, hand_boxes, frame_shape=None):
    object_box, object_label = detection_parts(object_detection)
    if not hand_boxes:
        return selected_bbox

    query_text = query.lower()
    cutting_terms = {"cutting", "cut", "slicing", "slice", "chopping", "chop"}
    direct_hand_terms = {"holding", "hold", "taking", "take", "grabbing", "grab", "carrying", "carry", "using"}
    relevant_tools = {"knife", "scissors", "bottle", "cup", "fork", "spoon", "book", "cell phone", "remote"}
    food_prep_terms = {"making", "make", "preparing", "prepare", "food", "sandwich", "cooking", "cook"}

    if is_fridge_interaction_query(query):
        if object_box is None:
            return selected_bbox
        if object_label != "refrigerator":
            return selected_bbox
        if hand_object_proximity(object_box, hand_boxes) < 0.15:
            return selected_bbox
        hand_box = nearest_hand_box(object_box, hand_boxes)
        merged = merge_boxes([selected_bbox, object_box, hand_box])
        return merged if merged is not None else selected_bbox

    if any(term in query_text for term in food_prep_terms):
        prep_region = merge_boxes(sorted(hand_boxes, key=box_area, reverse=True)[:2])
        if prep_region is None:
            return selected_bbox
        prep_region = (
            max(0, prep_region[0] - 20),
            max(0, prep_region[1] - 20),
            prep_region[2] + 20 if frame_shape is None else min(frame_shape[1], prep_region[2] + 20),
            prep_region[3] + 20 if frame_shape is None else min(frame_shape[0], prep_region[3] + 20),
        )
        if selected_bbox is not None:
            if box_intersection(prep_region, selected_bbox) <= 0:
                return selected_bbox
        if object_box is None or object_label not in {"knife", "bottle", "cup", "fork", "spoon", "sandwich"}:
            return prep_region

        overlap = hand_object_overlap(object_box, hand_boxes)
        proximity = hand_object_proximity(object_box, hand_boxes)
        object_small = frame_shape is not None and box_area(object_box) < 0.06 * float(frame_shape[0] * frame_shape[1])
        object_on_edge = box_touches_edge(object_box, frame_shape)
        if (overlap >= 0.01 or proximity >= 0.35) and not object_small and not object_on_edge:
            merged = merge_boxes([prep_region, object_box])
            return merged if merged is not None else prep_region
        return prep_region

    if not (
        any(term in query_text for term in cutting_terms) or
        any(term in query_text for term in direct_hand_terms)
    ):
        return selected_bbox

    if object_box is None:
        return selected_bbox

    if object_label not in relevant_tools:
        return selected_bbox

    overlap = hand_object_overlap(object_box, hand_boxes)
    proximity = hand_object_proximity(object_box, hand_boxes)
    if overlap < 0.02 and proximity < 0.55:
        return selected_bbox

    hand_box = nearest_hand_box(object_box, hand_boxes)
    merged = merge_boxes([selected_bbox, object_box, hand_box])
    return merged if merged is not None else selected_bbox


def verify_result(query, scores, object_detection, bbox, frame_shape, obj_scorer, hand_boxes=None):
    object_bbox, object_label = detection_parts(object_detection)
    targets = obj_scorer.get_query_objects(query)
    explicit_targets = obj_scorer.get_explicit_query_objects(query)
    frame_h, frame_w = frame_shape[:2]
    frame_area = float(frame_h * frame_w)
    query_text = query.lower()

    if bbox is None:
        return False

    bbox_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    bbox_ratio = bbox_area / frame_area if frame_area else 0.0

    required_targets = explicit_targets or targets
    action_only_query = obj_scorer.is_action_only_query(query)
    fridge_interaction_query = is_fridge_interaction_query(query)
    food_prep_query = obj_scorer.is_food_prep_query(query)

    if required_targets and not action_only_query:
        if object_label not in required_targets:
            return False

    if targets:
        if scores.object_score < 0.70:
            return False
        if scores.visual_score < 0.44:
            return False

        large_targets = {"refrigerator", "oven", "microwave", "tv", "bed", "couch", "chair", "dining table"}
        small_targets = {"knife", "bottle", "cup", "fork", "spoon", "cell phone", "book"}

        if object_label in large_targets and bbox_ratio < 0.08:
            return False
        if object_label in small_targets and bbox_ratio > 0.40:
            return False
        if object_label == "refrigerator" and bbox_ratio < 0.15:
            return False
        if object_label == "knife" and bbox_ratio > 0.12:
            return False

    if food_prep_query:
        if object_label in {"refrigerator", "oven", "microwave"}:
            return False
        if scores.motion_score < 0.20:
            return False
        if scores.visual_score < 0.46:
            return False
        if scores.object_score < 0.18:
            return False
        if object_bbox is not None and scores.object_score >= 0.70:
            if box_intersection(bbox, object_bbox) <= 0 and not boxes_close(bbox, object_bbox, scale=0.9):
                return False

    if action_only_query:
        action_tools = {"knife", "scissors"}
        if object_label in action_tools:
            if scores.motion_score < 0.28:
                return False
            if scores.object_score < 0.45:
                return False
        else:
            if scores.motion_score < 0.45:
                return False
            if scores.rotation_score < 0.60:
                return False
            if scores.visual_score < 0.48:
                return False
            if not hand_boxes:
                return False

    interaction_terms = {"holding", "opening", "cutting", "slicing", "picking", "taking", "grabbing", "using"}
    if any(term in query_text for term in interaction_terms):
        if scores.motion_score < 0.18:
            return False

    if fridge_interaction_query:
        target_box = object_bbox if object_bbox is not None else bbox
        if object_label != "refrigerator":
            return False
        if scores.object_score < 0.70:
            return False
        if scores.motion_score < 0.25:
            return False
        if not hand_boxes:
            return False
        proximity = hand_object_proximity(target_box, hand_boxes or [])
        if proximity < 0.15:
            return False

    direct_hand_terms = {"holding", "hold", "taking", "take", "grabbing", "grab", "carrying", "carry"}
    if (not fridge_interaction_query) and any(term in query_text for term in direct_hand_terms) and object_label in {"knife", "bottle", "cup", "book", "cell phone", "remote"}:
        target_box = object_bbox if object_bbox is not None else bbox
        proximity = hand_object_proximity(target_box, hand_boxes or [])
        overlap = hand_object_overlap(target_box, hand_boxes or [])
        if overlap < 0.03:
            return False

        object_area_ratio = box_area(target_box) / frame_area if frame_area else 0.0
        if object_label == "knife" and object_area_ratio > 0.08:
            return False
        if object_label == "bottle" and object_area_ratio < 0.003:
            return False
        if object_label == "knife" and overlap < 0.035:
            return False
        if object_label == "bottle" and overlap < 0.02:
            return False

    cutting_terms = {"cutting", "cut", "slicing", "slice", "chopping", "chop"}
    if any(term in query_text for term in cutting_terms):
        target_box = object_bbox if object_bbox is not None else bbox
        proximity = hand_object_proximity(target_box, hand_boxes or [])
        overlap = hand_object_overlap(target_box, hand_boxes or [])
        if object_label in {"knife", "scissors"}:
            if overlap < 0.025:
                return False
        elif overlap < 0.015 and proximity < 0.55:
            return False

    evidence_count = sum([
        scores.visual_score >= 0.48,
        scores.motion_score >= 0.35,
        scores.rotation_score >= 0.55,
        scores.object_score >= 0.70,
    ])
    if evidence_count < 2:
        return False

    return scores.confidence >= 0.50


def is_temporally_distinct(candidate_clip, accepted_results, min_gap_sec=RESULT_TIME_GAP_SEC):
    candidate_center = (float(candidate_clip.start_sec) + float(candidate_clip.end_sec)) / 2.0
    for result in accepted_results:
        accepted_center = (float(result.start_sec) + float(result.end_sec)) / 2.0
        if abs(candidate_center - accepted_center) < min_gap_sec:
            return False
    return True


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def run_pipeline(video_path, query, top_k=3, fps=1.5, output_dir="outputs",
                 visual_enc=None, text_enc=None, obj_scorer=None, segmentor=None):

    start_time = time.time()
    obj_scorer = obj_scorer or ObjectScorer()
    segmentor = segmentor or Segmentor(None)

    clips = sample_clips(video_path, fps=fps)

    motion_scores, flow_fields = compute_motion_scores(clips)

    idx = [i for i, m in enumerate(motion_scores) if m > MOTION_THRESHOLD]

    if idx:
        clips = [clips[i] for i in idx]
        motion_scores = motion_scores[idx]
        flow_fields = [flow_fields[i] for i in idx]

    print(f"[pipeline] After prefilter: {len(clips)} clips")

    if not clips:
        return []

    interaction_query = is_interaction_query(query)
    coarse_imgsz = 640 if interaction_query else 416
    refine_imgsz = 640 if interaction_query else 512

    rotation_scores = compute_rotation_scores(flow_fields)
    visual_scores = np.zeros(len(clips), dtype=np.float32)
    object_scores, all_boxes = obj_scorer.score_all_clips(clips, query, imgsz=coarse_imgsz)
    object_scores = np.asarray(object_scores, dtype=np.float32)

    shortlist_idx = (
        refine_shortlist_indices(motion_scores, rotation_scores, object_scores, top_k)
        if interaction_query
        else fast_shortlist_indices(motion_scores, rotation_scores, object_scores, top_k)
    )
    print(f"[pipeline] Shortlist for semantic rerank: {len(shortlist_idx)} clips")

    if len(shortlist_idx) > 0:
        shortlist_clips = [clips[i] for i in shortlist_idx]
        if (not interaction_query) or obj_scorer.is_action_only_query(query):
            shortlist_visual = compute_visual_scores(shortlist_clips, query, visual_enc, text_enc, obj_scorer)
            visual_scores[shortlist_idx] = shortlist_visual
        refined_object_scores, refined_boxes = obj_scorer.refine_shortlist_clips(shortlist_clips, query, imgsz=refine_imgsz)
        object_scores[shortlist_idx] = np.asarray(refined_object_scores, dtype=np.float32)
        for local_idx, clip_index in enumerate(shortlist_idx):
            all_boxes[clip_index] = refined_boxes[local_idx]

    fused_scores = []
    score_bundles = []

    for i in range(len(clips)):
        bundle = fuse(
            visual=float(visual_scores[i]),
            motion=float(motion_scores[i]),
            rotation=float(rotation_scores[i]),
            obj=float(object_scores[i]),
        )

        fused_scores.append(bundle.confidence)
        score_bundles.append(bundle)

    fused_arr = np.array(fused_scores, dtype=np.float32)
    ranked_candidates = rank_clips(fused_arr, top_k=len(fused_arr))
    if interaction_query:
        ranked_candidates = [candidate for candidate in ranked_candidates if candidate[0] in set(shortlist_idx)]
        shortlist_cap = ACTION_ONLY_SHORTLIST if obj_scorer.is_action_only_query(query) else INTERACTION_SHORTLIST
        ranked_candidates = ranked_candidates[:shortlist_cap]

    results = []
    verified_count = 0
    dropped_count = 0

    for clip_idx, _ in ranked_candidates:
        if len(results) >= top_k:
            break

        clip = clips[clip_idx]
        if not is_temporally_distinct(clip, results):
            dropped_count += 1
            continue

        scores = score_bundles[clip_idx]
        constraints = check(scores)
        frame_idx = detection_frame_index(all_boxes[clip_idx])
        frame_idx = max(0, min(frame_idx, len(clip.frames) - 1))
        frame = clip.frames[frame_idx]
        if segmentor is not None:
            hand_boxes = (
                segmentor.hand_detector.detect_precise(frame)
                if interaction_query
                else segmentor.hand_detector.detect(frame)
            )
        else:
            hand_boxes = []

        bbox = select_bbox(
            frame=frame,
            object_detection=all_boxes[clip_idx],
            segment_bbox=segmentor.segment(clip) if segmentor is not None else None,
            object_score=scores.object_score,
            query=query,
        )
        bbox = action_aware_bbox(query, all_boxes[clip_idx], bbox, hand_boxes, frame_shape=frame.shape)

        if bbox is None:
            bbox = (0, 0, frame.shape[1], frame.shape[0])

        if not verify_result(query, scores, all_boxes[clip_idx], bbox, frame.shape, obj_scorer, hand_boxes):
            dropped_count += 1
            continue

        verified_count += 1
        frame_path = save_frame_image(frame, clip_idx, output_dir)
        bbox_path = save_bbox_image(frame, bbox, clip_idx, output_dir)

        gradcam_path = generate_gradcam(
            frame=frame,
            clip_id=clip_idx,
            visual_score=scores.visual_score,
            output_dir=output_dir,
        )

        reason = build_reason(scores, constraints, query, False)

        results.append(FIBAResult(
            rank=len(results) + 1,
            clip_id=clip_idx,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            frame_sec=clip.start_sec if len(clip.frames) <= 1 else clip.start_sec + (frame_idx / max(1, len(clip.frames) - 1)) * (clip.end_sec - clip.start_sec),
            scores=scores,
            constraints=constraints,
            reason=reason,
            rotation_inferred=False,
            bbox=bbox,
            frame_path=frame_path,
            gradcam_path=gradcam_path,
        ))

        print(f"  Frame Image: {frame_path}")
        print(f"  BBox Image: {bbox_path}")

    total_time = time.time() - start_time

    print("\n=== PERFORMANCE ===")
    print(f"Time: {total_time:.2f}s")
    print(f"Clips processed: {len(clips)}")
    avg_result_confidence = float(np.mean([result.scores.confidence for result in results])) if results else 0.0
    print(f"Avg Top-{len(results)} Confidence: {avg_result_confidence:.3f}")
    print(f"Verified results: {verified_count}")
    print(f"Dropped candidates: {dropped_count}")

    return results
