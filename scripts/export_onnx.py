"""
Export RAFT-small and SAM2-tiny to ONNX format.
Run once after installing torch:
    python scripts/export_onnx.py --model raft
    python scripts/export_onnx.py --model sam2
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def export_raft():
    print("[export] Exporting RAFT-small to ONNX...")
    try:
        # torchvision RAFT
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        model.eval()

        H, W = 224, 224
        dummy_frame1 = torch.randn(1, 3, H, W)
        dummy_frame2 = torch.randn(1, 3, H, W)

        out_path = os.path.join(MODELS_DIR, "raft_small.onnx")

        torch.onnx.export(
            model,
            (dummy_frame1, dummy_frame2),
            out_path,
            input_names=["frame1", "frame2"],
            output_names=["flow"],
            dynamic_axes={
                "frame1": {0: "batch", 2: "height", 3: "width"},
                "frame2": {0: "batch", 2: "height", 3: "width"},
                "flow":   {0: "batch"},
            },
            opset_version=16,
        )
        print(f"[export] RAFT-small saved to {out_path}")
    except Exception as e:
        print(f"[export] RAFT export failed: {e}")
        print("  Farneback optical flow (OpenCV) will be used as fallback — no action needed.")


def export_sam2():
    print("[export] SAM2-tiny ONNX export...")
    print("  SAM2 ONNX export requires the official Meta SAM2 repo.")
    print("  Steps:")
    print("    1. git clone https://github.com/facebookresearch/sam2")
    print("    2. cd sam2 && pip install -e .")
    print("    3. Download checkpoint: sam2_hiera_tiny.pt from Meta's model zoo")
    print("    4. Use their export script or convert with torch.onnx.export")
    print("  The FIBA pipeline uses motion-bbox fallback if sam2_tiny.onnx is missing.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["raft", "sam2", "all"], default="all")
    args = parser.parse_args()

    if args.model in ("raft", "all"):
        export_raft()
    if args.model in ("sam2", "all"):
        export_sam2()


if __name__ == "__main__":
    main()
