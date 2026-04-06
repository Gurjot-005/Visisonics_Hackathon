import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List
from core.frame_sampler import Clip


class VisualEncoder:
    def __init__(self):
        # ✅ updated weights API (fix warning)
        self.model = models.mobilenet_v3_small(weights="DEFAULT")
        self.model.classifier = torch.nn.Identity()
        self.model.eval()

        # ✅ projection to match MiniLM (384)
        self.proj = torch.nn.Linear(576, 384)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("[visual_encoder] MobileNetV3 + projection loaded")

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        img = self.transform(frame).unsqueeze(0)

        with torch.no_grad():
            emb = self.model(img)      # (1, 576)
            emb = self.proj(emb)       # (1, 384)

        emb = emb.squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def encode_clip(self, clip: Clip) -> np.ndarray:
        # ⚡ speed optimization (use first frame only)
        return self.encode_frame(clip.frames[0])

    def encode_all_clips(self, clips: List[Clip]) -> np.ndarray:
        return np.stack([self.encode_clip(c) for c in clips], axis=0)