from explainability.schemas import ConstraintResult


def check(scores):
    passed = True
    failed_rules = []
    warnings = []

    # 🔥 HARD RULE
    if scores.motion_score < 0.10:
        passed = False
        failed_rules.append("motion_floor")

    if scores.rotation_score < 0.1:
        warnings.append("low_rotation")

    if scores.motion_score < 0.2:
        warnings.append("low_motion")

    return ConstraintResult(
        passed=passed,
        failed_rules=failed_rules,
        warnings=warnings,
    )