from explainability.schemas import ScoreBundle, ConstraintResult

RULES = {
    "motion_floor":    lambda s: s.motion_score > 0.10,
    "visual_floor":    lambda s: s.visual_score > 0.25,
    "object_floor":    lambda s: s.object_score > 0.10,
    "confidence_floor": lambda s: s.confidence > 0.20,
}

WARNINGS = {
    "low_rotation":  lambda s: s.rotation_score < 0.15,
    "low_motion":    lambda s: s.motion_score < 0.30,
}


def check(scores: ScoreBundle) -> ConstraintResult:
    failed = [name for name, rule in RULES.items() if not rule(scores)]
    warns  = [name for name, rule in WARNINGS.items() if rule(scores)]
    return ConstraintResult(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warns,
    )
