"""
One-shot script to download all required ONNX model weights.
Run once before using the pipeline:
    python scripts/download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# ✅ FIXED MODEL URLS
# =========================
MODELS = [
    {
        "name": "mobilevit_small.onnx",  # renamed to match actual file
        "url": "https://huggingface.co/apple/mobilevit-small/resolve/main/mobilevit_small.onnx",
        "note": "MobileViT-S visual encoder (~22MB)",
    },
    {
        "name": "minilm_l6_int8.onnx",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "note": "MiniLM-L6 text encoder (~23MB)",
    },
    {
        "name": "yolov8n.onnx",
        "url": "https://github.com/ultralytics/assets/releases/latest/download/yolov8n.onnx",
        "note": "YOLOv8 nano object detector (~6MB)",
    },
]

TOKENIZER_FILES = [
    {
        "name": "minilm_tokenizer/vocab.txt",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
        "note": "MiniLM tokenizer vocab",
    },
    {
        "name": "minilm_tokenizer/tokenizer_config.json",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json",
        "note": "MiniLM tokenizer config",
    },
]

MANUAL_MODELS = [
    {
        "name": "raft_small.onnx",
        "note": "RAFT-small optical flow — export manually (see scripts/export_onnx.py)",
    },
    {
        "name": "sam2_tiny.onnx",
        "note": "SAM2-tiny — export manually (see scripts/export_onnx.py). "
                "Pipeline uses motion bbox fallback if missing.",
    },
]


# =========================
# ✅ FIXED DOWNLOAD FUNCTION
# =========================
def download(url: str, dest: str, label: str):
    if os.path.exists(dest):
        print(f"  [skip] {label} already exists")
        return

    print(f"  [download] {label} ...")
    try:
        # Fix for HuggingFace 401/403
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())

        size_mb = os.path.getsize(dest) / 1e6
        print(f"  [ok] {label} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"  [error] {label} failed: {e}")


# =========================
# MAIN
# =========================
def main():
    print("=== FIBA Model Downloader (Fixed) ===\n")

    # Download models
    for m in MODELS:
        dest = os.path.join(MODELS_DIR, m["name"])
        print(f"{m['note']}")
        download(m["url"], dest, m["name"])
        print()

    # Download tokenizer files
    for t in TOKENIZER_FILES:
        dest = os.path.join(MODELS_DIR, t["name"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"{t['note']}")
        download(t["url"], dest, t["name"])
        print()

    # Manual models check
    print("=== Manual exports required ===")
    for m in MANUAL_MODELS:
        dest = os.path.join(MODELS_DIR, m["name"])
        status = "EXISTS" if os.path.exists(dest) else "MISSING"
        print(f"  [{status}] {m['name']} — {m['note']}")

    print("\n✅ Done.")
    print("👉 Run:")
    print("python run.py --video data/sample_videos/your_video.mp4 --query 'cutting onion'")


if __name__ == "__main__":
    main()