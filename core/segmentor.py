import numpy as np
import cv2
from typing import Optional
from core.frame_sampler import Clip
from core.hand_detector import HandDetector


class Segmentor:
    def __init__(self, model_path: Optional[str] = None):
        self.hand_detector = HandDetector()

    @staticmethod
    def _area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def _intersection(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    @staticmethod
    def _union(box_a, box_b):
        return [
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        ]

    def segment(self, clip: Clip):
        frame1 = clip.frames[0]
        frame2 = clip.frames[-1]

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_box = None
        if contours:
            frame_area = frame1.shape[0] * frame1.shape[1]
            significant = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.002 * frame_area]
            if significant:
                largest = max(significant, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                motion_box = [x, y, x + w, y + h]

        hand_boxes = self.hand_detector.detect(frame1)
        candidate = None

        if motion_box:
            nearby_hands = [
                box for box in hand_boxes
                if self._intersection(box, motion_box) > 0 or self._area(box) <= 1.5 * self._area(motion_box)
            ]
            if nearby_hands:
                best_hand = max(nearby_hands, key=lambda box: self._intersection(box, motion_box))
                candidate = self._union(motion_box, best_hand)
            else:
                candidate = motion_box
        elif hand_boxes:
            candidate = max(hand_boxes, key=self._area)

        if candidate is None:
            return [50, 50, 180, 180]

        x1, y1, x2, y2 = candidate
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame1.shape[0] * frame1.shape[1]

        if area > 0.7 * frame_area:
            return [50, 50, 180, 180]

        return [int(x1), int(y1), int(x2), int(y2)]
