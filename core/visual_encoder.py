import numpy as np
import open_clip
import torch
from PIL import Image
from pathlib import Path


LOCAL_PRETRAINED = Path(
    "C:/Users/HP/.cache/huggingface/hub/models--apple--MobileCLIP-S1-OpenCLIP/"
    "snapshots/59d35241939f6942255489b83c9068e48ebf57f8/open_clip_model.safetensors"
)


class VisualEncoder:
    BATCH_SIZE = 32

    def __init__(self):
        print("[visual_encoder] Loading MobileCLIP-S1 (lightweight)...")

        self.device = "cpu"
        pretrained = str(LOCAL_PRETRAINED) if LOCAL_PRETRAINED.exists() else "datacompdr"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S1",
            pretrained=pretrained,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_frame(self, frame):
        image = Image.fromarray(frame)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()

    def encode_frames_batch(self, frames):
        if not frames:
            return np.empty((0, 0), dtype=np.float32)

        batches = []
        for start in range(0, len(frames), self.BATCH_SIZE):
            batch_frames = frames[start:start + self.BATCH_SIZE]
            images = [
                self.preprocess(Image.fromarray(frame))
                for frame in batch_frames
            ]
            image_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)

            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            batches.append(embedding.cpu().numpy())

        return np.concatenate(batches, axis=0).astype(np.float32)

    @staticmethod
    def representative_frame(clip):
        return clip.frames[len(clip.frames) // 2]

    def encode_all_clips(self, clips):
        frames = [self.representative_frame(clip) for clip in clips]
        return self.encode_frames_batch(frames)
