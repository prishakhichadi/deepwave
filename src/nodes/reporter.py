"""Generates a production-grade markdown engineering report. Includes hardware telemetry
diagnosis, AST findings summary, optimization plan rationale, annotated code diff,
and theoretical improvement estimate. Designed to be directly actionable for AMD engineers."""

import difflib
from datetime import datetime
from typing import Dict
from src.state import KernelAgentState


def report_writer_node(state: KernelAgentState) -> Dict:
    """
    Assembles the complete DEEPWAVE optimization report from all upstream node outputs.
    Produces a markdown document an AMD engineer can read, act on, and attach to a ticket.
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

    # --- Code diff ---
    diff_lines = list(difflib.unified_diff(
        original_code.splitlines(keepends=True),
        optimized_code.splitlines(keepends=True),
        fromfile="before_baseline.hip",
        tofile="after_optimized.hip",
        lineterm=""
    ))
    visual_diff = "\n".join(diff_lines)

    # --- Build report ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    report = f"# DEEPWAVE: GPU Kernel Optimization Report\n"
    report += f"*Generated: {timestamp} | Optimization pass #{iteration}*\n\n"
    report += "---\n\n"

    # Section 1 — Hardware Telemetry
    report += "## 1. Hardware Telemetry Diagnosis\n\n"
    if diagnosis:
        confidence_pct = f"{diagnosis.confidence_score * 100:.1f}%"
        report += f"| Field | Value |\n|---|---|\n"
        report += f"| **Primary Bottleneck** | `{diagnosis.bottleneck_type}` |\n"
        report += f"| **Confidence** | {confidence_pct} |\n"
        if diagnosis.secondary_bottleneck:
            report += f"| **Secondary Bottleneck** | `{diagnosis.secondary_bottleneck}` |\n"
        report += "\n"
    else:
        report += "*No diagnosis available.*\n\n"

    # Raw metrics table
    if parsed_metrics:
        report += "### Raw Hardware Metrics\n\n"
        report += "| Metric | Value | AMD MI300X Threshold |\n|---|---|---|\n"
        thresholds = {
            "valu_util":        "> 80% = Compute Bound",
            "mem_stalled":      "> 50% = Memory Bound",
            "max_waves_per_cu": "< 16 = Occupancy Limited",
            "l2_cache_hit":     "< 50% = Cache Miss Issue",
            "lds_bank_conflict":"< 5% target",
            "salu_util":        "< 30% target",
        }
        for key, val in parsed_metrics.items():
            if val > 0.0:
                threshold = thresholds.get(key, "—")
                report += f"| `{key}` | `{val:.4f}` | {threshold} |\n"
        report += "\n"

    # Evidence list
    if diagnosis and diagnosis.evidence:
        report += "### Diagnostic Evidence\n\n"
        for item in diagnosis.evidence:
            report += f"- {item}\n"
        report += "\n"

    # Section 2 — AST Findings
    report += "---\n\n## 2. Kernel Structure Analysis (AST)\n\n"
    if ast_findings:
        high    = [f for f in ast_findings if f.severity == "high"]
        medium  = [f for f in ast_findings if f.severity == "medium"]
        low     = [f for f in ast_findings if f.severity == "low"]

        for severity_label, group in [("🔴 High", high), ("🟡 Medium", medium), ("🟢 Low", low)]:
            if group:
                report += f"### {severity_label} Severity\n\n"
                for finding in group:
                    report += f"**`{finding.finding_type}`** @ `{finding.location}`\n\n"
                    report += f"> {finding.description}\n\n"
    else:
        report += "*No structural anti-patterns detected.*\n\n"

    # Section 3 — Optimization Plan
    report += "---\n\n## 3. Optimization Strategy\n\n"
    if plan:
        report += f"**Strategy:** `{plan.strategy_name}`\n\n"
        report += f"**Target Scopes:** {', '.join(f'`{s}`' for s in plan.target_scopes)}\n\n"
        report += f"**Rationale:** {plan.rationale}\n\n"
        report += f"**Expected Impact:** `{plan.expected_impact}`\n\n"
        if plan.amd_specific_hints:
            report += "**AMD-Specific Notes:**\n"
            for hint in plan.amd_specific_hints:
                report += f"- {hint}\n"
            report += "\n"
    else:
        report += "*No optimization plan generated.*\n\n"

    # Section 4 — Code Diff
    report += "---\n\n## 4. Code Modifications\n\n"
    if visual_diff.strip():
        report += "```diff\n"
        report += visual_diff + "\n"
        report += "```\n\n"
    else:
        report += "*No code changes were generated in this pass.*\n\n"

    # Section 5 — Annotations
    report += "---\n\n## 5. Hardware-Justified Annotations\n\n"
    if annotations:
        for code_block, explanation in annotations.items():
            report += f"### `{code_block}`\n\n"
            report += f"> {explanation}\n\n"
    else:
        report += "*No annotations generated.*\n\n"

    # Section 6 — Theoretical Improvement
    report += "---\n\n## 6. Theoretical Performance Improvement\n\n"
    report += f"{theoretical}\n\n"
    report += (
        "*Note: Theoretical estimates are based on architectural analysis. "
        "Validate with rocprof re-profiling after applying changes.*\n"
    )

    return {"final_report": report}