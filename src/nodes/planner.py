from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from src.state import KernelAgentState, OptimizationStrategy
from config.settings import settings



AMD_STRATEGY_PLAYBOOK = """
AMD MI300X Optimization Strategy Playbook:
-------------------------------------------

MEMORY BANDWIDTH BOUND:
  1. Shared Memory Tiling — load global tiles into LDS, reuse across threads
  2. Memory Coalescing Fix — restructure access so warp-adjacent threads read adjacent addresses
  3. AoS → SoA Layout — transform data layout for sequential access patterns
  4. Read-Only Cache — use __ldg() or const __restrict__ to route reads through L1 texture cache
  5. Vectorized Loads — use float4 / int4 loads to maximize cache line utilization

COMPUTE BOUND:
  1. Loop Unrolling — #pragma unroll to reduce loop overhead and expose ILP
  2. Instruction Fusion — combine multiply-add into FMA instructions
  3. Reduced Precision — use fp16/bf16 where precision allows (AMD MFMAs are 2x faster)
  4. Warp-Level Primitives — use __shfl_xor_sync / DPP for register-level reductions

OCCUPANCY LIMITED:
  1. Register Pressure Reduction — reduce local variable count, use __launch_bounds__
  2. Shared Memory Reduction — decrease LDS usage per block to allow more concurrent blocks
  3. Block Size Tuning — experiment with 128/256 threads; AMD wavefront = 64 threads
  4. Kernel Splitting — break monolithic kernels into smaller focused kernels

LATENCY BOUND:
  1. Instruction-Level Parallelism — reorder independent instructions to fill pipelines
  2. Memory Prefetching — use async copies or manual prefetch to hide latency
  3. Occupancy Increase — more waves in flight hide latency through context switching
"""


def optimization_planner_node(state: KernelAgentState) -> Dict:
    """
    Selects a concrete, targeted optimization strategy based on the diagnosed
    bottleneck and the structural AST findings. Returns an OptimizationStrategy
    that the rewriter node uses to guide its code modifications.
    """
    diagnosis = state.get("diagnosis")
    if diagnosis is None:
        raise ValueError("optimization_planner_node requires a diagnosis in state. "
                         "Run bottleneck_classifier_node first.")



    llm = settings.build_llm(settings.planner_model)
    structured_llm = llm.with_structured_output(
        OptimizationStrategy, method=settings.structured_output_method
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AMD GPU optimization architect. Your job is to select the single best "
            "optimization strategy for the diagnosed hardware bottleneck.\n\n"
            "Rules:\n"
            "1. Choose the strategy most directly addressing the PRIMARY bottleneck type.\n"
            "2. Your target_scopes must name actual C++ function names or loop scopes from "
            "the kernel code — not generic descriptions.\n"
            "3. Your rationale must cite specific hardware facts: cache line sizes, wavefront "
            "width, LDS capacity, register file limits.\n"
            "4. AMD MI300X specifics: wavefront=64 threads, LDS=64KB/CU, L2=256MB, "
            "max 32 waves/CU, 512 VGPRs/wavefront.\n"
            "5. SCALE YOUR RESPONSE TO SEVERITY — do not apply the same fix regardless of how "
            "far past the threshold the metric is:\n"
            "   - 'borderline'/'moderate' severity: apply the SINGLE lowest-risk strategy from "
            "the playbook that addresses the bottleneck (e.g. just a coalescing fix). Set "
            "expected_impact to 'low' or 'medium'. Note in the rationale that this is a "
            "conservative, minimal-risk fix appropriate for a borderline case.\n"
            "   - 'severe' severity: combine 2 strategies from the playbook for a stronger fix. "
            "Set expected_impact to 'medium' or 'high'.\n"
            "   - 'critical' severity: apply the most aggressive combination available in the "
            "playbook (e.g. shared memory tiling AND vectorized loads together for memory-bound "
            "cases). Set expected_impact to 'high' and say explicitly in the rationale that the "
            "severity justifies a more invasive rewrite.\n\n"
            f"{AMD_STRATEGY_PLAYBOOK}"
        ),
        (
            "human",
            "--- BOTTLENECK DIAGNOSIS ---\n"
            "Type: {bottleneck_type}\n"
            "Confidence: {confidence}\n"
            "Evidence: {evidence}\n"
            "Secondary: {secondary}\n\n"
            "--- SEVERITY ---\n"
            "Severity: {severity_label} (score {severity_score})\n"
            "{severity_detail}\n\n"
            "--- AST STRUCTURAL FINDINGS ---\n"
            "{ast_insights}\n\n"
            "--- KERNEL SOURCE (for scope identification) ---\n"
            "{source_code}\n\n"
            "Select the optimal strategy — scaled to the severity above — and identify exactly "
            "which scopes to target."
        )
    ])

    chain = prompt | structured_llm

    ast_insights = state.get("ast_insights", ["No AST findings available."])

    plan: OptimizationStrategy = chain.invoke({
        "bottleneck_type": diagnosis.bottleneck_type,
        "confidence": f"{diagnosis.confidence_score:.2f}",
        "evidence": "\n".join(f"  - {e}" for e in diagnosis.evidence),
        "secondary": diagnosis.secondary_bottleneck or "None",
        "severity_label": state.get("severity_label", "unscored"),
        "severity_score": f"{state.get('severity_score', 0.0):.2f}",
        "severity_detail": state.get("severity_detail", ""),
        "ast_insights": "\n".join(ast_insights),
        "source_code": state.get("raw_kernel_code", ""),
    })

    print(f"[optimization_planner] Strategy: {plan.strategy_name}")
    print(f"  Targets: {plan.target_scopes}")
    print(f"  Impact:  {plan.expected_impact}")
    if plan.amd_specific_hints:
        print(f"  AMD hints: {plan.amd_specific_hints}")

    return {"optimization_plan": plan}