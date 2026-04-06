"""
Unit tests for confidence fuser and constraint checker.
Usage:
    python tests/test_confidence.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from explainability.confidence_fuser import fuse
from explainability.constraint_checker import check


def test_fuse_weights():
    bundle = fuse(visual=1.0, motion=1.0, rotation=1.0, obj=1.0)
    assert abs(bundle.confidence - 1.0) < 1e-4, f"Expected 1.0, got {bundle.confidence}"
    print("[ok] fuse: all-ones gives confidence 1.0")

    bundle = fuse(visual=0.0, motion=0.0, rotation=0.0, obj=0.0)
    assert bundle.confidence == 0.0, f"Expected 0.0, got {bundle.confidence}"
    print("[ok] fuse: all-zeros gives confidence 0.0")

    bundle = fuse(visual=0.5, motion=0.5, rotation=0.5, obj=0.5)
    assert abs(bundle.confidence - 0.5) < 1e-4, f"Expected 0.5, got {bundle.confidence}"
    print("[ok] fuse: all-half gives confidence 0.5")


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


def test_constraints_fail_visual():
    bundle = fuse(visual=0.10, motion=0.6, rotation=0.3, obj=0.7)
    result = check(bundle)
    assert not result.passed, "Expected fail on low visual"
    assert "visual_floor" in result.failed_rules
    print("[ok] constraint: low visual triggers visual_floor failure")


def test_warnings():
    bundle = fuse(visual=0.5, motion=0.20, rotation=0.05, obj=0.5)
    result = check(bundle)
    assert "low_motion" in result.warnings or "low_rotation" in result.warnings
    print("[ok] constraint: low motion/rotation triggers warnings")


def main():
    print("=== test_confidence.py ===\n")
    test_fuse_weights()
    test_constraints_pass()
    test_constraints_fail_motion()
    test_constraints_fail_visual()
    test_warnings()
    print("\n[all tests passed]")


if __name__ == "__main__":
    main()
