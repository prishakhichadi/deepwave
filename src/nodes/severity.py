from typing import Dict, Tuple



_SEVERITY_METRIC: Dict[str, Tuple[str, float, str]] = {
    "Memory Bandwidth Bound": ("mem_stalled", 50.0, "higher_is_worse"),
    "Compute Bound":          ("valu_util", 80.0, "higher_is_worse"),
    "Occupancy Limited":      ("max_waves_per_cu", 16.0, "lower_is_worse"),
    "Latency Bound":          (None, None, None),
}


def classify_severity(bottleneck_type: str, metrics: Dict[str, float]) -> Tuple[str, float, str]:
    """
    Returns (severity_label, severity_score, detail) where:
      severity_label: "borderline" | "moderate" | "severe" | "critical" | "unscored"
      severity_score: 0.0-1.0, how far past the threshold the driving metric is
      detail: human-readable explanation citing the actual metric value
    """
    metric_name, threshold, direction = _SEVERITY_METRIC.get(bottleneck_type, (None, None, None))

    if metric_name is None or metric_name not in metrics:
        return (
            "unscored", 0.0,
            f"No single dominant metric drives severity scoring for {bottleneck_type}; "
            "treating as a standard-severity case."
        )

    value = metrics[metric_name]

    if direction == "higher_is_worse":

        distance = max(0.0, value - threshold)
        score = min(1.0, distance / threshold)
    else: 
        distance = max(0.0, threshold - value)
        score = min(1.0, distance / threshold)

    if score < 0.15:
        label = "borderline"
    elif score < 0.45:
        label = "moderate"
    elif score < 0.75:
        label = "severe"
    else:
        label = "critical"

    detail = (
        f"{metric_name}={value:.1f} vs. threshold={threshold:.1f} "
        f"({direction.replace('_', ' ')}) — severity score {score:.2f} ({label})."
    )
    return label, score, detail