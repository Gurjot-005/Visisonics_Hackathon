import open_clip
import torch
from pathlib import Path


LOCAL_PRETRAINED = Path(
    "C:/Users/HP/.cache/huggingface/hub/models--apple--MobileCLIP-S1-OpenCLIP/"
    "snapshots/59d35241939f6942255489b83c9068e48ebf57f8/open_clip_model.safetensors"
)


class TextEncoder:
    def __init__(self):
        print("[text_encoder] MobileCLIP text encoder")

        self.device = "cpu"
        pretrained = str(LOCAL_PRETRAINED) if LOCAL_PRETRAINED.exists() else "datacompdr"
        self.model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S1",
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()
