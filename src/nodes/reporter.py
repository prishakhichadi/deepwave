"""Generates a production-grade markdown engineering report. Includes hardware telemetry
diagnosis, AST findings summary, optimization plan rationale, annotated code diff,
and theoretical improvement estimate. Designed to be directly actionable for AMD engineers."""

import difflib
from datetime import datetime
from typing import Dict
from src.state import KernelAgentState


_THRESHOLDS = {
    "valu_util":         "> 80% = Compute Bound",
    "mem_stalled":       "> 50% = Memory Bound",
    "max_waves_per_cu":  "< 16 = Occupancy Limited",
    "l2_cache_hit":      "< 50% = Cache Miss Issue",
    "lds_bank_conflict": "< 5% target",
    "salu_util":         "< 30% target",
}

_CONSISTENCY_LABEL = {
    "confirmed":     "Confirmed by structural analysis",
    "conflicting":   "Conflicting evidence — review recommended",
    "metrics_only":  "Metrics-only — no structural corroboration",
}


def report_writer_node(state: KernelAgentState) -> Dict:
    """
    Assembles the complete DEEPWAVE optimization report from all upstream node outputs.
    Produces a markdown document an AMD engineer can read, act on, and attach to a ticket.

    Each fact is stated exactly once: there's no separate "summary" block restating
    what the sections below already say — the section tables themselves are the
    at-a-glance view.
    """
    original_code   = state.get("raw_kernel_code", "")
    optimized_code  = state.get("optimized_kernel_code", "")
    diagnosis       = state.get("diagnosis")
    plan            = state.get("optimization_plan")
    annotations     = state.get("annotations") or {}
    ast_findings    = state.get("ast_findings") or []
    parsed_metrics  = state.get("parsed_metrics") or {}
    theoretical     = state.get("theoretical_improvement", "Not estimated.")
    iteration       = state.get("iteration_count", 1)

    severity_label       = state.get("severity_label")
    severity_score       = state.get("severity_score")
    severity_detail      = state.get("severity_detail")
    consistency          = state.get("evidence_consistency")
    consistency_detail   = state.get("evidence_consistency_detail")
    improvement_mode     = state.get("improvement_mode")
    improvement_metrics  = state.get("improvement_metrics") or []
    improvement_summary  = state.get("improvement_summary")

    diff_lines = list(difflib.unified_diff(
        original_code.splitlines(keepends=True),
        optimized_code.splitlines(keepends=True),
        fromfile="before_baseline.hip",
        tofile="after_optimized.hip",
        lineterm=""
    ))
    visual_diff = "\n".join(diff_lines)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    report  = "# DEEPWAVE — GPU Kernel Optimization Report\n\n"
    report += f"Generated {timestamp} · optimization pass #{iteration}\n\n"

    # --- 1. Diagnosis ---------------------------------------------------------
    report += "---\n\n## 1. Diagnosis\n\n"
    if diagnosis:
        report += "| Field | Value |\n|---|---|\n"
        report += f"| Bottleneck | `{diagnosis.bottleneck_type}` |\n"
        report += f"| Confidence | {diagnosis.confidence_score * 100:.1f}% |\n"
        if diagnosis.secondary_bottleneck:
            report += f"| Secondary bottleneck | `{diagnosis.secondary_bottleneck}` |\n"
        if severity_label and severity_label != "unscored":
            report += f"| Severity | `{severity_label.upper()}` ({severity_score:.2f}/1.00) |\n"
        if consistency:
            report += f"| Evidence check | {_CONSISTENCY_LABEL.get(consistency, consistency)} |\n"
        report += "\n"
    else:
        report += "*No diagnosis available.*\n\n"

    if severity_detail:
        report += f"{severity_detail}\n\n"
    if consistency_detail:
        report += f"{consistency_detail}\n\n"

    if parsed_metrics:
        report += "**Raw hardware metrics**\n\n"
        report += "| Metric | Value | AMD MI300X threshold |\n|---|---|---|\n"
        for key, val in parsed_metrics.items():
            if val > 0.0:
                report += f"| `{key}` | {val:.4f} | {_THRESHOLDS.get(key, '—')} |\n"
        report += "\n"

    if diagnosis and diagnosis.evidence:
        report += "**Diagnostic evidence**\n\n"
        for item in diagnosis.evidence:
            report += f"- {item}\n"
        report += "\n"

    # --- 2. Kernel Structure Analysis ------------------------------------------
    report += "---\n\n## 2. Kernel Structure Analysis\n\n"
    if ast_findings:
        severity_order = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(ast_findings, key=lambda f: severity_order.get(f.severity, 3))
        report += "| Severity | Finding | Location | Description |\n|---|---|---|---|\n"
        for f in ordered:
            desc = f.description.replace("|", "/").replace("\n", " ")
            report += f"| {f.severity} | `{f.finding_type}` | {f.location} | {desc} |\n"
        report += "\n"
    else:
        report += "*No structural anti-patterns detected.*\n\n"

    # --- 3. Optimization Strategy -----------------------------------------------
    report += "---\n\n## 3. Optimization Strategy\n\n"
    if plan:
        report += "| Field | Value |\n|---|---|\n"
        report += f"| Strategy | `{plan.strategy_name}` |\n"
        report += f"| Target scopes | {', '.join(f'`{s}`' for s in plan.target_scopes)} |\n"
        report += f"| Expected impact | `{plan.expected_impact}` |\n"
        report += "\n"
        report += f"{plan.rationale}\n\n"
        if plan.amd_specific_hints:
            for hint in plan.amd_specific_hints:
                report += f"- {hint}\n"
            report += "\n"
    else:
        report += "*No optimization plan generated.*\n\n"

    # --- 4. Code Modifications -----------------------------------------------
    report += "---\n\n## 4. Code Modifications\n\n"
    if visual_diff.strip():
        report += "```diff\n" + visual_diff + "\n```\n\n"
    else:
        report += "*No code changes were generated in this pass.*\n\n"

    if annotations:
        for code_block, explanation in annotations.items():
            report += f"**`{code_block}`** — {explanation}\n\n"

    # --- 5. Improvement -----------------------------------------------------
    report += "---\n\n## 5. Before / After Improvement\n\n"
    if improvement_mode:
        report += f"Mode: `{improvement_mode.upper()}`\n\n"

    if improvement_summary:
        report += f"{improvement_summary}\n\n"

    if improvement_metrics:
        report += "| Metric | Before | After | Δ | % change | Verdict |\n|---|---|---|---|---|---|\n"
        for m in improvement_metrics:
            verdict = {True: "Improved", False: "Regressed", None: "Informational"}.get(m.get("improved"))
            pct = f"{m['pct_change']:+.1f}%" if m.get("pct_change") is not None else "—"
            unit = m.get("unit", "")
            if m.get("projected"):
                lo, hi = m.get("projected_range_pct", [None, None])
                pct = f"est. {lo:.0f}–{hi:.0f}%" if lo is not None else pct
            report += (
                f"| `{m['metric']}` | {m['before']:.2f}{unit} | {m['after']:.2f}{unit} | "
                f"{m['delta']:+.2f}{unit} | {pct} | {verdict} |\n"
            )
        report += "\n"
    else:
        report += "*No improvement metrics available.*\n\n"

    report += (
        f"{theoretical}\n\n"
        "*Theoretical estimates are based on architectural analysis; validate with rocprof "
        "re-profiling after applying changes.*\n"
    )

    return {"final_report": report}