import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        print("[hand_detector] lightweight (no mediapipe)")

    def detect(self, frame):
        # convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # skin color range (works decently for most lighting)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 150, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # filter small noise
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h])

        return boxes