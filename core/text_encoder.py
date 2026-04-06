from sentence_transformers import SentenceTransformer
import numpy as np


class TextEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[text_encoder] SentenceTransformer loaded")

    def encode(self, text: str) -> np.ndarray:
        emb = self.model.encode(text)
        return emb / (np.linalg.norm(emb) + 1e-8)