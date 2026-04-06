"""
End-to-end smoke test.
Usage:
    python tests/test_pipeline.py --video data/sample_videos/video1.mp4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import run_pipeline, load_models

TEST_QUERIES = [
    "cutting onion",
    "opening box",
    "picking up object",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a sample video")
    args = parser.parse_args()

    assert os.path.exists(args.video), f"Video not found: {args.video}"

    print("[test] Loading models...")
    visual_enc, text_enc, obj_scorer, segmentor = load_models()

    passed = 0
    failed = 0

    for query in TEST_QUERIES:
        print(f"\n[test] Query: '{query}'")
        results = run_pipeline(
            video_path=args.video,
            query=query,
            top_k=3,
            fps=2.0,
            output_dir="outputs/test",
            visual_enc=visual_enc,
            text_enc=text_enc,
            obj_scorer=obj_scorer,
            segmentor=segmentor,
        )

        if not results:
            print(f"  [FAIL] No results returned")
            failed += 1
            continue

        top = results[0]
        has_reason      = len(top.reason) > 10
        has_scores      = top.scores.confidence > 0.0
        has_timestamps  = top.end_sec > top.start_sec
        has_gradcam     = top.gradcam_path is not None and os.path.exists(top.gradcam_path)

        checks = {
            "has results":     True,
            "has reason":      has_reason,
            "has scores":      has_scores,
            "valid timestamps": has_timestamps,
            "gradcam saved":   has_gradcam,
        }

        all_pass = all(checks.values())
        status = "PASS" if all_pass else "FAIL"
        print(f"  [{status}] confidence={top.scores.confidence:.3f} | "
              f"clips={len(results)} | rotation={top.rotation_inferred}")

        for check_name, result in checks.items():
            mark = "ok" if result else "FAIL"
            print(f"    [{mark}] {check_name}")

        if all_pass:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*40}")
    print(f"Tests: {passed} passed, {failed} failed out of {len(TEST_QUERIES)}")
    print(f"{'='*40}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
