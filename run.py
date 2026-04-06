import argparse
import json
import os
import sys

from explainability.schemas import confidence_band, format_hhmmss
from pipeline import run_pipeline, load_models


def parse_args():
    parser = argparse.ArgumentParser(description="FIBA - Find it by Action")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--query", required=True, help="Action query string e.g. 'cutting onion'")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--output", default="outputs", help="Directory to write outputs")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    return parser.parse_args()


def print_results(results, as_json=False):
    if as_json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    print(f"\n{'=' * 60}")
    print(f"FIBA Results - {len(results)} clip(s) found")
    print(f"{'=' * 60}")
    for r in results:
        passed = "PASS" if r.constraints.passed else "FAIL"
        print(
            f"\nRank #{r.rank}  |  Clip {r.clip_id}  |  "
            f"{format_hhmmss(r.start_sec)} - {format_hhmmss(r.end_sec)}  |  "
            f"Confidence: {r.scores.confidence:.3f}  |  Constraints: {passed}"
        )
        print(f"  Confidence band: {confidence_band(r.scores.confidence)}")
        print(f"  Selected frame time: {format_hhmmss(r.frame_sec)}")
        print(
            f"  visual={r.scores.visual_score:.3f}  "
            f"motion={r.scores.motion_score:.3f}  "
            f"rotation={r.scores.rotation_score:.3f}  "
            f"object={r.scores.object_score:.3f}"
        )
        if r.rotation_inferred:
            print("  [ROTATION INFERRED]")
        if r.bbox:
            x1, y1, x2, y2 = r.bbox
            print(f"  BBox: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
        if r.frame_path:
            print(f"  Frame: {r.frame_path}")
        if r.gradcam_path:
            print(f"  GradCAM: {r.gradcam_path}")
        if r.constraints.failed_rules:
            print(f"  Failed rules: {', '.join(r.constraints.failed_rules)}")
        if r.constraints.warnings:
            print(f"  Warnings: {', '.join(r.constraints.warnings)}")
        print(f"  Reason: {r.reason}")
    print(f"\n{'=' * 60}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[error] Video not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("[run] Loading models...")
    visual_enc, text_enc, obj_scorer, segmentor = load_models()

    results = run_pipeline(
        video_path=args.video,
        query=args.query,
        top_k=args.top_k,
        fps=args.fps,
        output_dir=args.output,
        visual_enc=visual_enc,
        text_enc=text_enc,
        obj_scorer=obj_scorer,
        segmentor=segmentor,
    )

    if not results:
        print("[run] No results found.")
        sys.exit(0)

    out_file = os.path.join(args.output, "results.json")
    with open(out_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"[run] Results written to {out_file}")

    print_results(results, as_json=args.json)


if __name__ == "__main__":
    main()
