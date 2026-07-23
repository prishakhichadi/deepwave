from typing import List, Tuple
from src.state import ASTFinding, BottleneckDiagnosis



CONFIRMING_FINDINGS = {
    "Memory Bandwidth Bound": {
        "uncoalesced_memory_access", "missing_shared_memory", "redundant_global_load"
    },
    "Compute Bound": {
        "scalar_operation_in_kernel", "loop_structure"
    },
    "Occupancy Limited": {
        "poor_occupancy_structure", "high_register_pressure"
    },
    "Latency Bound": {
        "pointer_aliasing_risk"
    },
    "LDS Bank Conflict Bound": {
        "lds_bank_conflict_risk"
    },
    "Register Pressure Bound": {
        "high_register_pressure"
    },
}

CONTRADICTING_FINDINGS = {
    "Memory Bandwidth Bound": set(),
    "Compute Bound": {"uncoalesced_memory_access", "missing_shared_memory"},
    "Occupancy Limited": set(),
    "Latency Bound": {"uncoalesced_memory_access"},
    "LDS Bank Conflict Bound": set(),
    "Register Pressure Bound": set(),
}


def cross_validate_diagnosis(
    diagnosis: BottleneckDiagnosis,
    ast_findings: List[ASTFinding],
) -> Tuple[str, str]:


    bottleneck = diagnosis.bottleneck_type
    finding_types = {f.finding_type for f in ast_findings}

    confirming = finding_types & CONFIRMING_FINDINGS.get(bottleneck, set())
    contradicting = finding_types & CONTRADICTING_FINDINGS.get(bottleneck, set())

    if confirming:
        return (
            "confirmed",
            f"Structural analysis independently confirms this diagnosis: "
            f"{', '.join(sorted(confirming))} detected in the kernel AST, consistent "
            f"with {bottleneck}."
        )

    if contradicting:
        return (
            "conflicting",
            f"Caution: the hardware metrics point to {bottleneck}, but structural "
            f"analysis found {', '.join(sorted(contradicting))}, which is more "
            f"characteristic of a different bottleneck. Recommend manual review before "
            f"applying the suggested rewrite."
        )

    return (
        "metrics_only",
        f"No AST finding directly confirms or contradicts {bottleneck} — this diagnosis "
        f"rests on hardware counters alone. Structural detection for this bottleneck "
        f"type may be limited (occupancy pressure in particular often isn't visible "
        f"in the raw C++ AST); treat confidence accordingly."
    )