import numpy as np
import cv2
from typing import List, Optional
from core.frame_sampler import Clip
from core.hand_detector import HandDetector


class Segmentor:
    def __init__(self, model_path: Optional[str] = None):
        self.hand_detector = HandDetector()

    def segment(self, clip: Clip):
        frame1 = clip.frames[0]
        frame2 = clip.frames[-1]

        # motion detection
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_box = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            motion_box = [x, y, x + w, y + h]

        # hands
        hand_boxes = self.hand_detector.detect(frame1)

        all_boxes = []

        # filter motion noise
        if motion_box:
            x1, y1, x2, y2 = motion_box
            area = (x2 - x1) * (y2 - y1)
            if area > 500:
                all_boxes.append(motion_box)

        all_boxes.extend(hand_boxes)

        # fallback (center crop instead of full frame)
        if not all_boxes:
            h, w = frame1.shape[:2]
            return [w//4, h//4, 3*w//4, 3*h//4]

        x1 = min(b[0] for b in all_boxes)
        y1 = min(b[1] for b in all_boxes)
        x2 = max(b[2] for b in all_boxes)
        y2 = max(b[3] for b in all_boxes)

        # tighten box
        margin = 10
        x1 = max(0, x1 + margin)
        y1 = max(0, y1 + margin)
        x2 = min(224, x2 - margin)
        y2 = min(224, y2 - margin)

        return [int(x1), int(y1), int(x2), int(y2)]