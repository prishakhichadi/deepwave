from typing import Dict, List, Optional, Tuple
from src.state import KernelAgentState
from src.nodes.reader import parse_metrics_csv



METRIC_META: Dict[str, Tuple[str, str, str]] = {
    "mem_stalled":        ("Memory Unit Stalled",   "%",  "lower_is_better"),
    "lds_bank_conflict":  ("LDS Bank Conflicts",    "%",  "lower_is_better"),
    "salu_util":          ("Scalar ALU Utilization", "%", "lower_is_better"),
    "l2_cache_hit":       ("L2 Cache Hit Rate",     "%",  "higher_is_better"),
    "max_waves_per_cu":   ("Occupancy (waves/CU)",  "",   "higher_is_better"),
    "valu_util":          ("Vector ALU Utilization", "%", "informational"),
    "fetch_size_kb":      ("Fetch Size",            "KB", "informational"),
    "write_size_kb":      ("Write Size",            "KB", "informational"),
    "vgpr_count":         ("VGPRs per Thread",      "",   "lower_is_better"),
}


_BOTTLENECK_DRIVER: Dict[str, Tuple[str, str]] = {
    "Memory Bandwidth Bound": ("mem_stalled", "lower_is_better"),
    "Compute Bound":          ("valu_util", "lower_is_better"),
    "Occupancy Limited":      ("max_waves_per_cu", "higher_is_better"),
    "Latency Bound":          ("mem_stalled", "lower_is_better"),
    "LDS Bank Conflict Bound": ("lds_bank_conflict", "lower_is_better"),
    "Register Pressure Bound": ("vgpr_count", "lower_is_better"),
}

_IMPACT_RANGE_PCT: Dict[str, Tuple[float, float]] = {
    "high":   (30.0, 55.0),
    "medium": (15.0, 30.0),
    "low":    (5.0, 15.0),
}


def _pct_change(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return ((after - before) / before) * 100.0


def _build_delta(metric: str, before: float, after: float, direction_override: Optional[str] = None) -> Dict:
    label, unit, direction = METRIC_META.get(metric, (metric, "", "informational"))
    direction = direction_override or direction

    delta = after - before
    pct = _pct_change(before, after)

    improved: Optional[bool]
    if direction == "lower_is_better":
        improved = after < before
    elif direction == "higher_is_better":
        improved = after > before
    else:
        improved = None

    return {
        "metric": metric,
        "label": label,
        "before": round(before, 4),
        "after": round(after, 4),
        "delta": round(delta, 4),
        "pct_change": round(pct, 1) if pct is not None else None,
        "unit": unit,
        "direction": direction,
        "improved": improved,
    }


def compute_measured_deltas(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
) -> List[Dict]:
    
    deltas = []
    for metric in METRIC_META:
        before = before_metrics.get(metric, 0.0)
        after = after_metrics.get(metric, 0.0)
        if before == 0.0 and after == 0.0:
            continue
        deltas.append(_build_delta(metric, before, after))
    return deltas


def summarize_measured(deltas: List[Dict], bottleneck_type: Optional[str]) -> str:
    driver_metric, _ = _BOTTLENECK_DRIVER.get(bottleneck_type, (None, None))
    driver = next((d for d in deltas if d["metric"] == driver_metric), None)

    if driver and driver["pct_change"] is not None:
        verb = "improved" if driver["improved"] else "regressed" if driver["improved"] is False else "changed"
        return (
            f"Measured re-profiling shows the primary driver of the {bottleneck_type} diagnosis, "
            f"`{driver['metric']}`, {verb} from {driver['before']:.1f}{driver['unit']} to "
            f"{driver['after']:.1f}{driver['unit']} ({driver['pct_change']:+.1f}%)."
        )
    if deltas:
        return "Measured re-profiling data was provided; see the per-metric table for before/after values."
    return "Re-profiling data was provided but contained no recognized metrics to compare."


def project_deltas(
    bottleneck_type: Optional[str],
    severity_score: Optional[float],
    expected_impact: Optional[str],
    before_metrics: Dict[str, float],
) -> Tuple[List[Dict], str]:
  
  
    if not bottleneck_type or bottleneck_type not in _BOTTLENECK_DRIVER:
        return [], "No diagnosis available — cannot project an improvement range."

    driver_metric, direction = _BOTTLENECK_DRIVER[bottleneck_type]
    before = before_metrics.get(driver_metric, 0.0)

    lo, hi = _IMPACT_RANGE_PCT.get((expected_impact or "medium").lower(), _IMPACT_RANGE_PCT["medium"])
    
    
    severity_score = severity_score if severity_score is not None else 0.5
    scale = 0.6 + 0.8 * severity_score 
    lo, hi = lo * scale, hi * scale

    if before == 0.0:
        return [], (
            f"No baseline value recorded for `{driver_metric}` — cannot project a numeric "
            f"improvement range for {bottleneck_type}."
        )

    if direction == "lower_is_better":
        projected_lo = before * (1 - hi / 100.0)
        projected_hi = before * (1 - lo / 100.0)
    else:
        projected_lo = before * (1 + lo / 100.0)
        projected_hi = before * (1 + hi / 100.0)

    delta_entry = _build_delta(driver_metric, before, projected_hi if direction == "higher_is_better" else projected_lo, direction_override=direction)
    delta_entry["projected_range_pct"] = [round(lo, 1), round(hi, 1)]
    delta_entry["projected"] = True

    summary = (
        f"No re-profiled data supplied — this is a PROJECTED estimate only. Based on a "
        f"`{(expected_impact or 'medium')}`-impact `{bottleneck_type}` fix at severity "
        f"{severity_score:.2f}, `{driver_metric}` is projected to improve by roughly "
        f"{lo:.0f}%–{hi:.0f}% ({before:.1f}{delta_entry['unit']} → "
        f"{projected_lo:.1f}–{projected_hi:.1f}{delta_entry['unit']}). "
        f"Re-profile the optimized kernel with rocprof/omniperf to replace this with a "
        f"measured value."
    )
    return [delta_entry], summary


def impact_analyzer_node(state: KernelAgentState) -> Dict:
   
   
    before_metrics = state.get("parsed_metrics") or {}
    diagnosis = state.get("diagnosis")
    bottleneck_type = diagnosis.bottleneck_type if diagnosis else None
    plan = state.get("optimization_plan")
    expected_impact = plan.expected_impact if plan else None
    severity_score = state.get("severity_score")

    raw_after = (state.get("raw_after_profiling_data") or "").strip()

    if raw_after:
        after_metrics = parse_metrics_csv(raw_after, label="impact_analyzer_node")
        deltas = compute_measured_deltas(before_metrics, after_metrics)
        summary = summarize_measured(deltas, bottleneck_type)
        mode = "measured"
    else:
        after_metrics = {}
        deltas, summary = project_deltas(bottleneck_type, severity_score, expected_impact, before_metrics)
        mode = "projected"

    print(f"[impact_analyzer] mode={mode} — {summary}")

    return {
        "after_parsed_metrics": after_metrics,
        "improvement_mode": mode,
        "improvement_metrics": deltas,
        "improvement_summary": summary,
    }