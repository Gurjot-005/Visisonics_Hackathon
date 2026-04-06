import cv2
import numpy as np
from pathlib import Path

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp_python = None
    mp_vision = None


HAND_LANDMARKER_PATH = Path("models/hand_landmarker.task")


class HandDetector:
    def __init__(self):
        self._hand_landmarker = None
        print("[hand_detector] lightweight skin detector + optional MediaPipe verifier")

    def _skin_detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 150, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h])

        return boxes

    def _ensure_mediapipe(self):
        if mp is None or mp_python is None or mp_vision is None:
            return False
        if not HAND_LANDMARKER_PATH.exists():
            return False
        if self._hand_landmarker is None:
            base_options = mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        return True

    def detect(self, frame):
        return self._skin_detect(frame)

    def detect_precise(self, frame):
        if not self._ensure_mediapipe():
            return self._skin_detect(frame)

        image_h, image_w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self._hand_landmarker.detect(mp_image)
        boxes = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                xs = [lm.x for lm in hand_landmarks]
                ys = [lm.y for lm in hand_landmarks]
                x1 = max(0, int(min(xs) * image_w) - 8)
                y1 = max(0, int(min(ys) * image_h) - 8)
                x2 = min(image_w, int(max(xs) * image_w) + 8)
                y2 = min(image_h, int(max(ys) * image_h) + 8)
                if (x2 - x1) * (y2 - y1) > 200:
                    boxes.append([x1, y1, x2, y2])

        if boxes:
            return boxes

        return self._skin_detect(frame)
