# Visisonics Hackathon

Lightweight video retrieval pipeline for finding query-relevant clips, frames, timestamps, explanations, and visual grounding from kitchen activity videos.

## What It Does

Given a video and a natural-language query, the pipeline:

- samples short clips from the video
- ranks candidate moments using motion, rotation, object evidence, and lightweight semantic similarity
- returns the top verified results with timestamps
- saves frame images, bbox images, GradCAM images, and a JSON summary

The current pipeline is optimized around two practical modes:

- `Fast mode`
  Best for broader, object-heavy queries like `opening a refrigerator` or `preparing food`
- `Interaction mode`
  Stricter verification for hand-object interaction and action queries like `cutting`

## Repository Outputs

Each run writes assets into [outputs](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/outputs):

- `clip_<id>_frame.png`
- `clip_<id>_bbox.png`
- `clip_<id>_gradcam.png`
- `results.json`

`results.json` is generated automatically on every run and includes:

- clip start and end time
- selected frame time
- raw second values and `hh:mm:ss` timestamps
- score breakdown
- confidence band
- explanation text
- bbox coordinates
- saved frame and GradCAM paths

## Setup

Recommended Python version:

- `Python 3.10+`

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you prefer editable install:

```powershell
pip install -e .
```

## Main Run Command

Run from the project root:

```powershell
python run.py --video <path-to-video> --query "<your query>" --top_k 3 --fps 2.0
```

Example:

```powershell
python run.py --video data\sample_videos\OP01-R02-TurkeySandwich.mp4 --query "preparing food" --top_k 3 --fps 2.0
```

If you also want the JSON printed in the terminal:

```powershell
python run.py --video data\sample_videos\OP01-R02-TurkeySandwich.mp4 --query "preparing food" --top_k 3 --fps 2.0 --json
```

## Recommended Demo Queries

These are the safest demo queries based on the current tuned behavior.

### 1. Opening A Refrigerator

Video:
[OP01-R04-ContinentalBreakfast.mp4](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/data/sample_videos/OP01-R04-ContinentalBreakfast.mp4)

Command:

```powershell
python run.py --video data\sample_videos\OP01-R04-ContinentalBreakfast.mp4 --query "opening a refrigerator" --top_k 3 --fps 2.0
```

### 2. Preparing Food

Video:
[OP01-R02-TurkeySandwich.mp4](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/data/sample_videos/OP01-R02-TurkeySandwich.mp4)

Command:

```powershell
python run.py --video data\sample_videos\OP01-R02-TurkeySandwich.mp4 --query "preparing food" --top_k 3 --fps 2.0
```

### 3. Cutting

Video:
[OP01-R02-TurkeySandwich.mp4](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/data/sample_videos/OP01-R02-TurkeySandwich.mp4)

Command:

```powershell
python run.py --video data\sample_videos\OP01-R02-TurkeySandwich.mp4 --query "cutting" --top_k 3 --fps 2.0
```

Note:
`cutting` is more fragile than the first two demo queries. For judge demos, prefer:

- `opening a refrigerator`
- `preparing food`

## Current Behavior Notes

- The pipeline always writes `outputs\results.json`
- Terminal output includes:
  - `hh:mm:ss` timestamps
  - score breakdown
  - confidence band
  - verification summary
  - average confidence of the returned top results
- Duplicate nearby clips are filtered from final results
- The system may return fewer than `top_k` results if verification rejects weak candidates

## Lightweight Design Choices

To stay relatively edge-friendly:

- MobileCLIP uses representative frames and batched inference
- fast mode uses a smaller shortlist for semantic reranking
- hand verification is used more selectively for interaction-heavy queries
- final verification rejects inconsistent object/bbox pairings instead of forcing bad outputs

## Limitations

- Broad kitchen activity queries work better than highly specific interaction phrases
- `holding a knife` is still harder than `opening a refrigerator`
- Some food-prep bboxes are still broader than ideal
- Accuracy depends on the visible objects and motions actually being present in the sampled frames

## Tests

Run the focused test suite:

```powershell
python tests\test_confidence.py
```

## Project Entry Points

- [run.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/run.py): CLI entry point
- [pipeline.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/pipeline.py): main ranking and verification pipeline
- [core/object_scorer.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/core/object_scorer.py): YOLO-based object scoring and query expansion
- [core/hand_detector.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/core/hand_detector.py): lightweight hand detection with optional MediaPipe verifier
- [explainability/confidence_fuser.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/explainability/confidence_fuser.py): confidence fusion
- [tests/test_confidence.py](/C:/Users/HP/OneDrive/Desktop/Visisonics_Hackathon/tests/test_confidence.py): focused regression tests
