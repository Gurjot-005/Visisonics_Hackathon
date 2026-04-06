from dataclasses import dataclass, field
from typing import List, Optional


def format_hhmmss(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    whole_seconds = int(total_seconds)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def confidence_band(confidence: float) -> str:
    confidence = float(confidence)
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


@dataclass
class ScoreBundle:
    visual_score: float
    motion_score: float
    rotation_score: float
    object_score: float
    confidence: float

    def to_dict(self):
        return {
            "visual_score":   round(self.visual_score, 4),
            "motion_score":   round(self.motion_score, 4),
            "rotation_score": round(self.rotation_score, 4),
            "object_score":   round(self.object_score, 4),
            "confidence":     round(self.confidence, 4),
            "confidence_band": confidence_band(self.confidence),
        }


@dataclass
class ConstraintResult:
    passed: bool
    failed_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "passed":       self.passed,
            "failed_rules": self.failed_rules,
            "warnings":     self.warnings,
        }


@dataclass
class FIBAResult:
    rank: int
    clip_id: int
    start_sec: float
    end_sec: float
    frame_sec: float
    scores: ScoreBundle
    constraints: ConstraintResult
    reason: str
    rotation_inferred: bool
    bbox: Optional[List[float]]
    frame_path: Optional[str]
    gradcam_path: Optional[str]

    def to_dict(self):
        return {
            "rank":               self.rank,
            "clip_id":            self.clip_id,
            "start_sec":          round(self.start_sec, 3),
            "end_sec":            round(self.end_sec, 3),
            "frame_sec":          round(self.frame_sec, 3),
            "start_time":         format_hhmmss(self.start_sec),
            "end_time":           format_hhmmss(self.end_sec),
            "frame_time":         format_hhmmss(self.frame_sec),
            "scores":             self.scores.to_dict(),
            "constraints":        self.constraints.to_dict(),
            "reason":             self.reason,
            "rotation_inferred":  self.rotation_inferred,
            "bbox":               self.bbox,
            "frame_path":         self.frame_path,
            "gradcam_path":       self.gradcam_path,
        }
