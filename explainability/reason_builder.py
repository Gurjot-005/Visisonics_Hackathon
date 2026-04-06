from explainability.schemas import ScoreBundle, ConstraintResult, confidence_band

ROTATION_KEYWORDS = [
    "opening", "unscrewing", "turning", "rotating", "twisting",
    "screwing", "spinning", "winding", "unwinding"
]


def is_rotation_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ROTATION_KEYWORDS)


def build_reason(
    scores: ScoreBundle,
    constraints: ConstraintResult,
    query: str,
    rotation_inferred: bool,
) -> str:
    parts = []

    parts.append(
        f"Selected for query '{query}' with visual similarity score "
        f"{scores.visual_score:.2f} (alignment between visual features "
        f"and text embedding)."
    )

    if scores.motion_score > 0.60:
        parts.append(
            f"Strong motion detected (optical flow score {scores.motion_score:.2f}), "
            f"indicating active physical interaction."
        )
    elif scores.motion_score > 0.30:
        parts.append(
            f"Moderate motion detected (optical flow score {scores.motion_score:.2f})."
        )
    elif scores.motion_score <= 0.10:
        parts.append(
            f"Very low motion (score {scores.motion_score:.2f}) - this clip may be static."
        )

    if rotation_inferred:
        parts.append(
            f"Rotational motion detected (score {scores.rotation_score:.2f}), "
            f"consistent with twisting or tool usage."
        )
    elif scores.rotation_score > 0.40:
        parts.append(
            f"Partial rotational movement observed (score {scores.rotation_score:.2f})."
        )

    if scores.object_score > 0.60:
        parts.append(
            f"Strong query-relevant object evidence detected (score {scores.object_score:.2f})."
        )
    elif scores.object_score > 0.20:
        parts.append(
            f"Weak query-relevant object evidence detected (score {scores.object_score:.2f})."
        )
    else:
        parts.append(
            f"No strong query-relevant object evidence detected (score {scores.object_score:.2f})."
        )

    parts.append(
        f"Final fused confidence: {scores.confidence:.2f} "
        f"(visual x 0.40 + motion x 0.30 + rotation x 0.15 + object x 0.15)."
    )
    parts.append(f"Confidence band: {confidence_band(scores.confidence)}.")

    if not constraints.passed:
        parts.append(
            f"CONSTRAINT FAILED: {', '.join(constraints.failed_rules)}. "
            f"Result included but flagged as low quality."
        )
    if constraints.warnings:
        parts.append(f"Warnings: {', '.join(constraints.warnings)}.")

    return " ".join(parts)
