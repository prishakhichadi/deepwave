"""Shared LangGraph state and Pydantic schemas for the DEEPWAVE GPU kernel optimization agent.
Every node takes this state as input and returns a partial dict to merge back into it."""

from typing import Dict, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Pydantic Schemas for Structured LLM Outputs
# ---------------------------------------------------------------------------

class BottleneckDiagnosis(BaseModel):
    """Structured response schema for hardware bottleneck classification."""
    bottleneck_type: Literal[
        "Memory Bandwidth Bound",
        "Compute Bound",
        "Occupancy Limited",
        "Latency Bound"
    ]
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence value from 0.0 to 1.0"
    )
    evidence: List[str] = Field(
        description="List of specific metric values supporting this diagnosis"
    )
    secondary_bottleneck: Optional[str] = Field(
        default=None,
        description="Secondary bottleneck type if co-present (e.g. also Occupancy Limited)"
    )


class ASTFinding(BaseModel):
    """A single structured finding extracted from the kernel AST."""
    finding_type: Literal[
        "uncoalesced_memory_access",
        "thread_divergence",
        "missing_shared_memory",
        "poor_occupancy_structure",
        "redundant_global_load",
        "scalar_operation_in_kernel",
        "pointer_aliasing_risk",
        "loop_structure"
    ]
    location: str = Field(description="Function name or line-level scope where this was found")
    description: str = Field(description="Plain-English explanation of what was detected")
    severity: Literal["high", "medium", "low"] = Field(
        description="How likely this is to contribute to the diagnosed bottleneck"
    )


class OptimizationStrategy(BaseModel):
    """Structured optimization plan formulated by the planning subagent."""
    strategy_name: str = Field(description="Short name for this strategy, e.g. 'Shared Memory Tiling'")
    target_scopes: List[str] = Field(
        description="Target C++ function names or block scopes to modify"
    )
    rationale: str = Field(
        description="Hardware-grounded reason this strategy addresses the diagnosed bottleneck"
    )
    expected_impact: Literal["high", "medium", "low"] = Field(
        description="Expected performance impact of this strategy"
    )
    amd_specific_hints: List[str] = Field(
        default_factory=list,
        description="AMD/ROCm-specific hints, e.g. wavefront size, LDS bank conflicts, MI300X cache hierarchy"
    )


class OptimizedKernelOutput(BaseModel):
    """The final payload returned by the kernel rewriting subagent."""
    optimized_code: str = Field(description="The full rewritten HIP kernel source code")
    line_by_line_annotations: Dict[str, str] = Field(
        description="Key-value mapping of rewritten code blocks to hardware-justified explanations"
    )
    theoretical_improvement: str = Field(
        description="Plain-English estimate of expected improvement and why, e.g. '2-4x memory throughput due to coalescing'"
    )


# ---------------------------------------------------------------------------
# 2. Main LangGraph State Definition
# ---------------------------------------------------------------------------

class KernelAgentState(TypedDict):
    """The shared state dictionary updated dynamically across graph cycles."""

    # --- Source Inputs ---
    raw_kernel_code: str           # Raw .hip / .cu source as a string
    raw_profiling_data: str        # Raw rocprof/omniperf CSV as a string

    # --- Parsed Hardware Analytics ---
    parsed_metrics: Dict[str, float]   # Normalized metric map from reader node
    ast_insights: List[str]            # Legacy flat string list (kept for compatibility)
    ast_findings: List[ASTFinding]     # Rich structured findings from improved analyzer

    # --- LLM Inference Objects ---
    diagnosis: Optional[BottleneckDiagnosis]
    optimization_plan: Optional[OptimizationStrategy]

    # --- Output Artifacts ---
    optimized_kernel_code: Optional[str]
    annotations: Optional[Dict[str, str]]
    theoretical_improvement: Optional[str]
    final_report: Optional[str]

    # --- Cyclic Flow Control ---
    iteration_count: int
    max_iterations: int            # Guard rail to prevent infinite loops (default: 3)

    verification_status: Optional[str]   # "passed" or "failed_retry"
    critic_feedback: Optional[str]       # Human-readable fault + remedy text




class CriticVerdict(BaseModel):
    """Schema for the compiler simulator critic's structured output."""
    is_syntactically_valid: bool
    identified_flaws: List[str] = Field(
        description="List of syntax or structural issues found"
    )
    remedy_suggestions: List[str] = Field(
        description="Concrete fixes the rewriter should apply on retry"
    )
    preservation_intact: bool = Field(
        description="Whether the original kernel signature and __global__ qualifier were preserved"
    )