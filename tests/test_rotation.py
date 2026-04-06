"""
Unit tests for the rotation detector.
Usage:
    python tests/test_rotation.py
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.rotation_detector import rotation_score
from explainability.reason_builder import is_rotation_query


def make_ccw_flow(H=224, W=224) -> np.ndarray:
    """Generate a synthetic pure counter-clockwise rotation flow field."""
    cx, cy = W / 2.0, H / 2.0
    xs = np.arange(W, dtype=np.float32) - cx
    ys = np.arange(H, dtype=np.float32) - cy
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = np.sqrt(grid_x**2 + grid_y**2) + 1e-6
    dx = -grid_y / radius * 5.0  # tangential CCW, scaled
    dy =  grid_x / radius * 5.0
    return np.stack([dx, dy], axis=-1)


def make_linear_flow(H=224, W=224) -> np.ndarray:
    """Generate a pure horizontal translation flow (no rotation)."""
    dx = np.ones((H, W), dtype=np.float32) * 5.0
    dy = np.zeros((H, W), dtype=np.float32)
    return np.stack([dx, dy], axis=-1)


def make_zero_flow(H=224, W=224) -> np.ndarray:
    return np.zeros((H, W, 2), dtype=np.float32)


def test_pure_rotation_high():
    flow = make_ccw_flow()
    score = rotation_score(flow)
    assert score > 0.60, f"Expected >0.60 for pure CCW rotation, got {score:.4f}"
    print(f"[ok] pure CCW rotation score: {score:.4f} (expected >0.60)")


def test_linear_flow_low():
    flow = make_linear_flow()
    score = rotation_score(flow)
    assert score < 0.40, f"Expected <0.40 for linear flow, got {score:.4f}"
    print(f"[ok] linear translation score: {score:.4f} (expected <0.40)")


def test_zero_flow():
    flow = make_zero_flow()
    score = rotation_score(flow)
    assert score == 0.0, f"Expected 0.0 for zero flow, got {score:.4f}"
    print(f"[ok] zero flow score: {score:.4f} (expected 0.0)")


def test_rotation_query_detection():
    assert is_rotation_query("opening a jar") == True
    assert is_rotation_query("unscrewing the lid") == True
    assert is_rotation_query("turning the knob") == True
    assert is_rotation_query("cutting onion") == False
    assert is_rotation_query("picking up book") == False
    print("[ok] rotation query keyword detection works correctly")


def main():
    print("=== test_rotation.py ===\n")
    test_pure_rotation_high()
    test_linear_flow_low()
    test_zero_flow()
    test_rotation_query_detection()
    print("\n[all tests passed]")


if __name__ == "__main__":
    main()
