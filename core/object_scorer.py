import re

import numpy as np

from core.hand_detector import HandDetector
from ultralytics import YOLO


class ObjectScorer:
    BATCH_SIZE = 24
    STOPWORDS = {
        "a", "an", "the", "with", "and", "or", "to", "of", "in", "on", "at", "for",
        "from", "into", "up", "down", "over", "under", "someone", "something",
    }
    HUMAN_TERMS = {"person", "people", "human", "man", "woman", "boy", "girl", "hand", "hands"}
    SYNONYMS = {
        "fridge": {"refrigerator"},
        "refrigerator": {"refrigerator"},
        "tv": {"tv", "monitor"},
        "phone": {"cell phone"},
        "sandwich": {"sandwich", "knife", "bowl", "plate"},
        "making": {"preparing"},
        "prepare": {"preparing", "food"},
        "preparing": {"food", "sandwich", "bowl", "knife"},
        "food": {"sandwich", "bowl", "cup", "knife", "fork", "spoon"},
        "holding": {"hold", "grasp", "grabbing", "carrying"},
        "taking": {"take", "grabbing", "holding"},
        "grabbing": {"grab", "holding", "taking"},
        "carry": {"holding", "taking"},
        "drink": {"cup", "bottle", "wine glass"},
        "cut": {"knife", "scissors"},
        "cutting": {"knife", "scissors"},
        "slice": {"knife"},
        "slicing": {"knife"},
        "eat": {"bowl", "cup", "fork", "knife", "spoon", "sandwich", "banana", "apple", "orange", "pizza"},
        "eating": {"bowl", "cup", "fork", "knife", "spoon", "sandwich", "banana", "apple", "orange", "pizza"},
        "open": {"refrigerator", "book", "bottle"},
        "opening": {"refrigerator", "book", "bottle"},
        "typing": {"keyboard", "laptop"},
    }
    INTERACTION_TERMS = {"holding", "hold", "taking", "take", "grabbing", "grab", "carrying", "carry", "using"}
    INTERACTION_OBJECTS = {"knife", "bottle", "cup", "fork", "spoon", "book", "cell phone", "remote"}
    ACTION_TERMS = {"cutting", "cut", "slicing", "slice", "chopping", "chop", "opening", "open"}
    FOOD_PREP_TERMS = {"making", "make", "preparing", "prepare", "food", "sandwich", "cooking", "cook"}
    FOOD_PREP_OBJECTS = {"sandwich", "knife", "bowl", "cup", "fork", "spoon", "dining table", "bottle"}
    APPLIANCE_OBJECTS = {"refrigerator", "oven", "microwave", "sink", "toaster"}

    def __init__(self):
        print("[object_scorer] YOLOv8n loaded")
        self.model = YOLO("yolov8n.pt")
        self.hand_detector = HandDetector()

    def detect(self, frame, imgsz=640):
        results = self.model(frame, verbose=False, imgsz=imgsz)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        classes = results.boxes.cls.cpu().numpy() if results.boxes else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes else []

        return boxes, classes, confs

    def detect_batch(self, frames, imgsz=640):
        if not frames:
            return []

        all_results = []
        for start in range(0, len(frames), self.BATCH_SIZE):
            batch_frames = frames[start:start + self.BATCH_SIZE]
            batch_results = self.model(batch_frames, verbose=False, imgsz=imgsz)
            all_results.extend(batch_results)

        return all_results

    def _raw_query_terms(self, query):
        return [
            token for token in re.findall(r"[a-z0-9]+", query.lower())
            if token not in self.STOPWORDS
        ]

    def _query_terms(self, query):
        base_terms = self._raw_query_terms(query)
        expanded_terms = set(base_terms)
        for term in base_terms:
            expanded_terms.update(self.SYNONYMS.get(term, set()))
        normalized = set()
        for term in expanded_terms:
            normalized.update(re.findall(r"[a-z0-9]+", str(term).lower()))
        return sorted(normalized)

    def expand_query(self, query):
        terms = self._query_terms(query)
        return " ".join(dict.fromkeys([query] + terms))

    @staticmethod
    def _box_area(box):
        if box is None:
            return 0.0
        return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

    @staticmethod
    def _box_center(box):
        return np.array([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], dtype=np.float32)

    def _hand_object_proximity(self, object_box, hand_boxes):
        if object_box is None or not hand_boxes:
            return 0.0

        obj_center = self._box_center(object_box)
        obj_scale = max(20.0, np.sqrt(self._box_area(object_box)))
        best = 0.0

        for hand_box in hand_boxes:
            hand_center = self._box_center(hand_box)
            distance = float(np.linalg.norm(obj_center - hand_center))
            proximity = max(0.0, 1.0 - (distance / (2.5 * obj_scale)))
            best = max(best, proximity)

        return best

    def get_query_objects(self, query):
        targets = set()
        query_terms = self._query_terms(query)
        model_names = {str(name).lower() for name in self.model.names.values()}

        for term in query_terms:
            if term in model_names:
                targets.add(term)
            for alias in self.SYNONYMS.get(term, set()):
                if alias in model_names:
                    targets.add(alias)

        query_term_set = set(query_terms)
        for label in model_names:
            label_terms = set(label.split())
            if label_terms and label_terms.issubset(query_term_set):
                targets.add(label)

        return targets

    def get_explicit_query_objects(self, query):
        explicit_targets = set()
        query_terms = self._raw_query_terms(query)
        model_names = {str(name).lower() for name in self.model.names.values()}

        synonym_map = {
            "fridge": "refrigerator",
            "refrigerator": "refrigerator",
            "cellphone": "cell phone",
            "phone": "cell phone",
            "tv": "tv",
        }

        for term in query_terms:
            canonical = synonym_map.get(term, term)
            if canonical in model_names:
                explicit_targets.add(canonical)

        query_term_set = set(query_terms)
        for label in model_names:
            label_terms = set(label.split())
            if label_terms and label_terms.issubset(query_term_set):
                explicit_targets.add(label)

        return explicit_targets

    def is_action_only_query(self, query):
        raw_terms = set(self._raw_query_terms(query))
        explicit_targets = self.get_explicit_query_objects(query)
        return bool(raw_terms.intersection(self.ACTION_TERMS)) and not explicit_targets

    def is_food_prep_query(self, query):
        raw_terms = set(self._raw_query_terms(query))
        return bool(raw_terms.intersection(self.FOOD_PREP_TERMS))

    def _label_match_score(self, label, query_terms, targets):
        if label in targets:
            return 1.0

        label_terms = set(label.split())
        overlap = len(label_terms.intersection(query_terms))
        if overlap:
            return min(0.8, 0.4 + 0.2 * overlap)

        return 0.0

    def score_clip(self, clip, query, imgsz=640):
        best_score = 0.0
        best_box = None
        best_label = None
        best_frame_index = 0

        for frame_index, frame in enumerate(clip.frames):
            boxes, classes, confs = self.detect(frame, imgsz=imgsz)
            hand_boxes = self.hand_detector.detect(frame)
            score, box, label = self._score_detection_result(boxes, classes, confs, query, hand_boxes)
            if score > best_score:
                best_score = score
                best_box = box
                best_label = label
                best_frame_index = frame_index

        return best_score, best_box, best_label, best_frame_index

    @staticmethod
    def representative_frame_index(clip):
        return len(clip.frames) // 2

    def _score_detection_result(self, boxes, classes, confs, query, hand_boxes=None):
        class_names = [self.model.names[int(c)] for c in classes]
        targets = self.get_query_objects(query)
        query_terms = set(self._query_terms(query))
        mentions_human = bool(query_terms.intersection(self.HUMAN_TERMS))
        interaction_query = bool(query_terms.intersection(self.INTERACTION_TERMS))
        food_prep_query = self.is_food_prep_query(query)

        best_box = None
        best_label = None
        best_score = 0.0

        for i, name in enumerate(class_names):
            label = str(name).lower()
            label_score = self._label_match_score(label, query_terms, targets)
            conf_score = float(confs[i]) if len(confs) > i else 0.0
            person_bonus = 0.0

            if label == "person":
                if mentions_human:
                    person_bonus = 0.25
                elif targets:
                    person_bonus = 0.05

            interaction_bonus = 0.0
            interaction_penalty = 0.0
            if interaction_query and label in self.INTERACTION_OBJECTS:
                proximity = self._hand_object_proximity(boxes[i], hand_boxes or [])
                interaction_bonus = 0.25 * proximity
                if proximity < 0.15:
                    interaction_penalty = 0.20

            food_prep_bonus = 0.0
            food_prep_penalty = 0.0
            if food_prep_query:
                if label in self.FOOD_PREP_OBJECTS:
                    food_prep_bonus = 0.18
                if label in self.APPLIANCE_OBJECTS:
                    food_prep_penalty = 0.22

            score = min(
                1.0,
                max(
                    0.0,
                    0.70 * label_score + 0.20 * conf_score + person_bonus +
                    interaction_bonus + food_prep_bonus - interaction_penalty - food_prep_penalty,
                ),
            )

            if score > best_score:
                best_score = score
                best_box = boxes[i]
                best_label = label

        if best_box is None and len(boxes) > 0:
            best_idx = int(max(range(len(boxes)), key=lambda idx: float(confs[idx]) if len(confs) > idx else 0.0))
            best_box = boxes[best_idx]
            best_label = str(class_names[best_idx]).lower()
            if not targets:
                best_score = min(0.2, float(confs[best_idx]) * 0.2)

        return min(best_score, 1.0), best_box, best_label

    def score_all_clips(self, clips, query, imgsz=640):
        frames = [clip.frames[self.representative_frame_index(clip)] for clip in clips]
        detections = self.detect_batch(frames, imgsz=imgsz)

        scores = []
        boxes = []

        for clip, result in zip(clips, detections):
            det_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
            det_classes = result.boxes.cls.cpu().numpy() if result.boxes else []
            det_confs = result.boxes.conf.cpu().numpy() if result.boxes else []
            frame_index = self.representative_frame_index(clip)
            frame = clip.frames[frame_index]
            hand_boxes = self.hand_detector.detect(frame)
            score, box, label = self._score_detection_result(det_boxes, det_classes, det_confs, query, hand_boxes)
            scores.append(score)
            boxes.append((box, label, frame_index))

        return scores, boxes

    def refine_shortlist_clips(self, clips, query, imgsz=640):
        scores = []
        boxes = []

        for clip in clips:
            score, box, label, frame_index = self.score_clip(clip, query, imgsz=imgsz)
            scores.append(score)
            boxes.append((box, label, frame_index))

        return scores, boxes
