"""
Unit tests for confidence fusion and bbox selection.
Usage:
    python tests/test_confidence.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from explainability.confidence_fuser import fuse
from explainability.constraint_checker import check
from pipeline import action_aware_bbox, is_temporally_distinct, select_bbox, verify_result


def test_fuse_weights():
    bundle = fuse(visual=1.0, motion=1.0, rotation=1.0, obj=1.0)
    assert abs(bundle.confidence - 1.0) < 1e-4, f"Expected 1.0, got {bundle.confidence}"
    print("[ok] fuse: all-ones gives confidence 1.0")

    bundle = fuse(visual=0.0, motion=0.0, rotation=0.0, obj=0.0)
    assert bundle.confidence == 0.0, f"Expected 0.0, got {bundle.confidence}"
    print("[ok] fuse: all-zeros gives confidence 0.0")

    bundle = fuse(visual=0.5, motion=0.5, rotation=0.5, obj=0.5)
    assert abs(bundle.confidence - 0.53) < 1e-4, f"Expected 0.53, got {bundle.confidence}"
    print("[ok] fuse: all-half gives calibrated confidence 0.53")

    bundle = fuse(visual=0.9, motion=0.0, rotation=0.0, obj=0.0)
    assert bundle.visual_score == 0.9, f"Expected visual score 0.9, got {bundle.visual_score}"
    assert bundle.confidence > 0.0, "Visual score should contribute to confidence"
    print("[ok] fuse: visual score is preserved and contributes")


def test_constraints_pass():
    bundle = fuse(visual=0.8, motion=0.6, rotation=0.3, obj=0.7)
    result = check(bundle)
    assert result.passed, f"Expected pass, got failed: {result.failed_rules}"
    print("[ok] constraint: high scores pass all rules")


def test_constraints_fail_motion():
    bundle = fuse(visual=0.8, motion=0.05, rotation=0.3, obj=0.7)
    result = check(bundle)
    assert not result.passed, "Expected fail on low motion"
    assert "motion_floor" in result.failed_rules
    print("[ok] constraint: low motion triggers motion_floor failure")


def test_select_bbox_prefers_tighter_segment_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    object_bbox = ((0, 0, 180, 180), "person")
    segment_bbox = (60, 60, 120, 120)
    selected = select_bbox(frame, object_bbox, segment_bbox, object_score=0.2, query="walking")
    assert selected == segment_bbox, f"Expected tighter segment box, got {selected}"
    print("[ok] bbox: tighter segment box is preferred over oversized object box")


def test_select_bbox_keeps_large_query_object():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    object_bbox = ((160, 60, 640, 480), "refrigerator")
    segment_bbox = (50, 50, 180, 180)
    selected = select_bbox(frame, object_bbox, segment_bbox, object_score=0.95, query="opening a refrigerator")
    assert selected == (160, 60, 640, 480), f"Expected refrigerator box, got {selected}"
    print("[ok] bbox: strong large-object detection is preserved for refrigerator queries")


def test_select_bbox_prefers_segment_for_food_prep_edge_object():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    object_bbox = ((424, 0, 536, 184), "bottle")
    segment_bbox = (180, 140, 480, 420)
    selected = select_bbox(frame, object_bbox, segment_bbox, object_score=0.95, query="preparing food")
    assert selected == segment_bbox, f"Expected food-prep segment box, got {selected}"
    print("[ok] bbox: food prep prefers action region over tiny edge object")


def test_action_aware_bbox_merges_hand_and_knife_for_cutting():
    merged = action_aware_bbox(
        query="cutting",
        object_detection=((290, 180, 340, 235), "knife"),
        selected_bbox=(290, 180, 340, 235),
        hand_boxes=[(235, 170, 310, 260)],
        frame_shape=(480, 640, 3),
    )
    assert merged == (235, 170, 340, 260), f"Expected merged action box, got {merged}"
    print("[ok] bbox: cutting merges knife and nearby hand into one action box")


def test_action_aware_bbox_merges_fridge_and_hand_for_taking():
    merged = action_aware_bbox(
        query="taking something from the fridge",
        object_detection=((140, 20, 620, 470), "refrigerator"),
        selected_bbox=(140, 20, 620, 470),
        hand_boxes=[(420, 180, 505, 320)],
        frame_shape=(480, 640, 3),
    )
    assert merged == (140, 20, 620, 470), f"Expected fridge action box to stay anchored on fridge, got {merged}"
    print("[ok] bbox: fridge interaction keeps the refrigerator action region")


def test_action_aware_bbox_prefers_hands_for_food_prep():
    merged = action_aware_bbox(
        query="preparing food",
        object_detection=((37, 254, 213, 360), "bottle"),
        selected_bbox=(37, 254, 213, 360),
        hand_boxes=[(150, 290, 245, 430), (255, 290, 360, 440)],
        frame_shape=(480, 640, 3),
    )
    x1, y1, x2, y2 = merged
    assert x1 >= 37 and x2 >= 360 and y1 >= 254, f"Expected food prep box to shift toward hands, got {merged}"
    print("[ok] bbox: food prep prefers hand-centered work region")


def test_action_aware_bbox_keeps_selected_region_when_hands_are_far_for_food_prep():
    merged = action_aware_bbox(
        query="preparing food",
        object_detection=((37, 254, 213, 360), "bottle"),
        selected_bbox=(30, 172, 402, 480),
        hand_boxes=[(403, 0, 640, 123)],
        frame_shape=(480, 640, 3),
    )
    assert merged == (30, 172, 402, 480), f"Expected distant false hand box to be ignored, got {merged}"
    print("[ok] bbox: distant false hand detections are ignored for food prep")


def test_warnings():
    bundle = fuse(visual=0.5, motion=0.20, rotation=0.05, obj=0.5)
    result = check(bundle)
    assert "low_motion" in result.warnings or "low_rotation" in result.warnings
    print("[ok] constraint: low motion/rotation triggers warnings")


class DummyScorer:
    def get_query_objects(self, query):
        if "refrigerator" in query:
            return {"refrigerator"}
        if "knife" in query:
            return {"knife"}
        return set()

    def get_explicit_query_objects(self, query):
        if "refrigerator" in query:
            return {"refrigerator"}
        if "knife" in query:
            return {"knife"}
        return set()

    def is_action_only_query(self, query):
        return query in {"cutting", "slicing", "chopping"}

    def is_food_prep_query(self, query):
        return query in {"making a sandwich", "preparing food"}


def test_verify_result_rejects_wrong_object():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.51, motion=0.55, rotation=0.60, obj=0.95)
    passed = verify_result(
        query="opening a refrigerator",
        scores=scores,
        object_detection=((430, 0, 500, 55), "bottle"),
        bbox=(430, 0, 500, 55),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
    )
    assert not passed, "Expected wrong-object refrigerator result to be rejected"
    print("[ok] verifier: wrong object is rejected")


def test_verify_result_rejects_refrigerator_for_food_prep():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.52, motion=0.60, rotation=0.72, obj=0.82)
    passed = verify_result(
        query="making a sandwich",
        scores=scores,
        object_detection=((140, 20, 620, 470), "refrigerator"),
        bbox=(140, 20, 620, 470),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(420, 180, 505, 320)],
    )
    assert not passed, "Expected refrigerator result to be rejected for food prep query"
    print("[ok] verifier: refrigerator is rejected for food prep queries")


def test_verify_result_accepts_food_prep_object():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.54, motion=0.46, rotation=0.62, obj=0.32)
    passed = verify_result(
        query="preparing food",
        scores=scores,
        object_detection=((260, 180, 330, 235), "knife"),
        bbox=(250, 160, 360, 280),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(235, 170, 310, 260)],
    )
    assert passed, "Expected countertop prep object to be accepted for food prep query"
    print("[ok] verifier: food prep object is accepted")


def test_verify_result_rejects_food_prep_when_bbox_disagrees_with_object():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.54, motion=0.46, rotation=0.62, obj=0.95)
    passed = verify_result(
        query="preparing food",
        scores=scores,
        object_detection=((440, 149, 514, 368), "knife"),
        bbox=(37, 254, 213, 360),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(403, 0, 640, 123)],
    )
    assert not passed, "Expected mismatched food prep bbox/object result to be rejected"
    print("[ok] verifier: food prep bbox/object mismatch is rejected")


def test_verify_result_accepts_taking_something_from_fridge():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.50, motion=0.52, rotation=0.64, obj=0.90)
    passed = verify_result(
        query="taking something from the fridge",
        scores=scores,
        object_detection=((140, 20, 620, 470), "refrigerator"),
        bbox=(140, 20, 620, 470),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(420, 180, 505, 320)],
    )
    assert passed, "Expected fridge interaction result to be accepted"
    print("[ok] verifier: taking something from the fridge is accepted with fridge + hand evidence")


def test_verify_result_rejects_fridge_without_hand():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.50, motion=0.52, rotation=0.64, obj=0.90)
    passed = verify_result(
        query="taking something from the fridge",
        scores=scores,
        object_detection=((140, 20, 620, 470), "refrigerator"),
        bbox=(140, 20, 620, 470),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[],
    )
    assert not passed, "Expected fridge interaction without hand evidence to be rejected"
    print("[ok] verifier: fridge interaction without hand evidence is rejected")


def test_verify_result_accepts_knife_match():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.52, motion=0.46, rotation=0.58, obj=0.92)
    passed = verify_result(
        query="holding a knife",
        scores=scores,
        object_detection=((280, 210, 330, 245), "knife"),
        bbox=(280, 210, 330, 245),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(275, 205, 340, 255)],
    )
    assert passed, "Expected strong knife result to be accepted"
    print("[ok] verifier: correct knife result is accepted")


def test_verify_result_rejects_unheld_knife():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.50, motion=0.50, rotation=0.70, obj=0.90)
    passed = verify_result(
        query="holding a knife",
        scores=scores,
        object_detection=((260, 120, 360, 240), "knife"),
        bbox=(260, 120, 360, 240),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(40, 300, 120, 390)],
    )
    assert not passed, "Expected distant hand knife result to be rejected"
    print("[ok] verifier: unheld knife is rejected")


def test_verify_result_rejects_low_overlap_knife():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.55, motion=0.55, rotation=0.72, obj=0.93)
    passed = verify_result(
        query="holding a knife",
        scores=scores,
        object_detection=((280, 210, 330, 245), "knife"),
        bbox=(280, 210, 330, 245),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(332, 205, 380, 255)],
    )
    assert not passed, "Expected near-but-not-overlapping knife result to be rejected"
    print("[ok] verifier: low-overlap knife is rejected")


def test_verify_result_rejects_tiny_unheld_bottle():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scores = fuse(visual=0.50, motion=0.40, rotation=0.70, obj=0.95)
    passed = verify_result(
        query="taking a bottle",
        scores=scores,
        object_detection=((500, 10, 530, 45), "bottle"),
        bbox=(500, 10, 530, 45),
        frame_shape=frame.shape,
        obj_scorer=DummyScorer(),
        hand_boxes=[(250, 250, 340, 360)],
    )
    assert not passed, "Expected tiny distant bottle result to be rejected"
    print("[ok] verifier: tiny or unheld bottle is rejected")


class DummyResult:
    def __init__(self, start_sec, end_sec):
        self.start_sec = start_sec
        self.end_sec = end_sec


class DummyClip:
    def __init__(self, start_sec, end_sec):
        self.start_sec = start_sec
        self.end_sec = end_sec


def test_temporal_distinctness_rejects_near_duplicate():
    accepted = [DummyResult(350.0, 350.5)]
    assert not is_temporally_distinct(DummyClip(349.5, 350.0), accepted), "Expected near-duplicate clip to be rejected"
    assert is_temporally_distinct(DummyClip(352.0, 352.5), accepted), "Expected distant clip to be accepted"
    print("[ok] ranker: near-duplicate timestamps are filtered out")


def main():
    print("=== test_confidence.py ===\n")
    test_fuse_weights()
    test_constraints_pass()
    test_constraints_fail_motion()
    test_select_bbox_prefers_tighter_segment_box()
    test_select_bbox_keeps_large_query_object()
    test_select_bbox_prefers_segment_for_food_prep_edge_object()
    test_action_aware_bbox_merges_hand_and_knife_for_cutting()
    test_action_aware_bbox_merges_fridge_and_hand_for_taking()
    test_action_aware_bbox_prefers_hands_for_food_prep()
    test_action_aware_bbox_keeps_selected_region_when_hands_are_far_for_food_prep()
    test_warnings()
    test_verify_result_rejects_wrong_object()
    test_verify_result_rejects_refrigerator_for_food_prep()
    test_verify_result_accepts_food_prep_object()
    test_verify_result_rejects_food_prep_when_bbox_disagrees_with_object()
    test_verify_result_accepts_taking_something_from_fridge()
    test_verify_result_rejects_fridge_without_hand()
    test_verify_result_accepts_knife_match()
    test_verify_result_rejects_unheld_knife()
    test_verify_result_rejects_low_overlap_knife()
    test_verify_result_rejects_tiny_unheld_bottle()
    test_temporal_distinctness_rejects_near_duplicate()
    print("\n[all tests passed]")


if __name__ == "__main__":
    main()
