import numpy as np
import cv2
from typing import List
from core.frame_sampler import Clip
from core.hand_detector import HandDetector


class ObjectScorer:
    def __init__(self):
        self.hand_detector = HandDetector()
        print("[object_scorer] smart scoring + hands")

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, 50, 150)

    def detect_motion(self, f1, f2):
        diff = cv2.absdiff(f1, f2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        return thresh

    def score_clip(self, clip: Clip, query: str) -> float:
        score = 0.0
        frame = clip.frames[0]

        # hands
        hand_boxes = self.hand_detector.detect(frame)
        if hand_boxes:
            score += 0.4

        # motion
        if len(clip.frames) > 1:
            motion = self.detect_motion(clip.frames[0], clip.frames[-1])
            score += 0.3 * (np.mean(motion) / 255.0)

        # edges (knife hint)
        edges = self.detect_edges(frame)
        if "knife" in query:
            score += 0.2 * (np.mean(edges) / 255.0)

        # action keyword
        if "cut" in query:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def score_all_clips(self, clips: List[Clip], query: str):
        return np.array([self.score_clip(c, query) for c in clips], dtype=np.float32)