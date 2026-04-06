from explainability.schemas import ScoreBundle

WEIGHTS = {
    "visual":   0.40,
    "motion":   0.30,
    "rotation": 0.15,
    "object":   0.15,
}


def fuse(visual: float, motion: float, rotation: float, obj: float) -> ScoreBundle:
    confidence = (
        WEIGHTS["visual"]   * visual +
        WEIGHTS["motion"]   * motion +
        WEIGHTS["rotation"] * rotation +
        WEIGHTS["object"]   * obj
    )
    return ScoreBundle(
        visual_score=round(visual, 4),
        motion_score=round(motion, 4),
        rotation_score=round(rotation, 4),
        object_score=round(obj, 4),
        confidence=round(float(confidence), 4),
    )