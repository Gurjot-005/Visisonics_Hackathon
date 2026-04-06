from explainability.schemas import ScoreBundle


def fuse(visual, motion, rotation, obj):
    confidence = (
        0.40 * visual +
        0.30 * motion +
        0.15 * rotation +
        0.15 * obj
    )

    evidence_count = sum([
        visual >= 0.48,
        motion >= 0.45,
        rotation >= 0.60,
        obj >= 0.70,
    ])
    confidence += 0.03 * max(0, evidence_count - 1)

    if visual >= 0.50 and obj >= 0.80:
        confidence += 0.04

    if motion >= 0.55 and (rotation >= 0.70 or obj >= 0.85):
        confidence += 0.03

    return ScoreBundle(
        visual_score=round(visual, 4),
        motion_score=round(motion, 4),
        rotation_score=round(rotation, 4),
        object_score=round(obj, 4),
        confidence=round(float(min(confidence, 1.0)), 4),
    )
