from dataclasses import dataclass, field
from typing import List, Optional


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
    scores: ScoreBundle
    constraints: ConstraintResult
    reason: str
    rotation_inferred: bool
    bbox: Optional[List[float]]
    gradcam_path: Optional[str]

    def to_dict(self):
        return {
            "rank":               self.rank,
            "clip_id":            self.clip_id,
            "start_sec":          round(self.start_sec, 3),
            "end_sec":            round(self.end_sec, 3),
            "scores":             self.scores.to_dict(),
            "constraints":        self.constraints.to_dict(),
            "reason":             self.reason,
            "rotation_inferred":  self.rotation_inferred,
            "bbox":               self.bbox,
            "gradcam_path":       self.gradcam_path,
        }
