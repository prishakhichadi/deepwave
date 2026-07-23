from typing import Dict, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field




class BottleneckDiagnosis(BaseModel):
    """Structured response schema for hardware bottleneck classification."""
    bottleneck_type: Literal[
        "Memory Bandwidth Bound",
        "Compute Bound",
        "Occupancy Limited",
        "Latency Bound",
        "LDS Bank Conflict Bound",
        "Register Pressure Bound"
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
        "loop_structure",
        "lds_bank_conflict_risk",
        "high_register_pressure"
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



class MetricDelta(BaseModel):
    """Before/after comparison for a single hardware metric."""
    metric: str
    label: str
    before: float
    after: float
    delta: float                     # after - before
    pct_change: Optional[float]      # percent change relative to `before`, None if before == 0
    unit: str = ""
    direction: Literal["lower_is_better", "higher_is_better", "informational"]
    improved: Optional[bool]         # None when direction is "informational"


class KernelAgentState(TypedDict):
    """The shared state dictionary updated dynamically across graph cycles."""


    raw_kernel_code: str         
    raw_profiling_data: str      


    parsed_metrics: Dict[str, float]   
    ast_insights: List[str]           
    ast_findings: List[ASTFinding]   


    diagnosis: Optional[BottleneckDiagnosis]
    optimization_plan: Optional[OptimizationStrategy]

   
    severity_label: Optional[str]    # "borderline" | "moderate" | "severe" | "critical" | "unscored"
    severity_score: Optional[float]  # 0.0-1.0
    severity_detail: Optional[str]

  
    evidence_consistency: Optional[str]         
    evidence_consistency_detail: Optional[str]   


    optimized_kernel_code: Optional[str]
    annotations: Optional[Dict[str, str]]
    theoretical_improvement: Optional[str]
    final_report: Optional[str]


    raw_after_profiling_data: Optional[str]   # optional rocprof/omniperf CSV re-profiled AFTER applying the rewrite
    after_parsed_metrics: Optional[Dict[str, float]]
    improvement_mode: Optional[str]      # "measured" | "projected"
    improvement_metrics: Optional[List[Dict]]
    improvement_summary: Optional[str]


    iteration_count: int
    max_iterations: int            

    verification_status: Optional[str]   
    critic_feedback: Optional[str]      




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