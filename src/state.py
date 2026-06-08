from typing import Dict, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# 1. Pydantic Schemas for Structured LLM Outputs

class BottleneckDiagnosis(BaseModel):
    """Structured response schema for hardware bottleneck classification."""
    bottleneck_type: Literal["Memory Bandwidth Bound", "Compute Bound", "Occupancy Limited", "Latency Bound"]
    confidence_score: float = Field(description="Confidence value from 0.0 to 1.0")
    evidence: List[str] = Field(description="List of specific metric values supporting this diagnosis")

class OptimizationStrategy(BaseModel):
    """Structured optimization plan formulated by the planning subagent."""
    strategy_name: str
    target_scopes: List[str] = Field(description="Target C++ function names or block scopes to modify")
    rationale: str

class OptimizedKernelOutput(BaseModel):
    """The final payload returned by the kernel rewriting subagent."""
    optimized_code: str
    line_by_line_annotations: Dict[str, str] = Field(
        description="Key-value mapping of rewritten blocks or lines to hardware-justified explanations"
    )

# 2. Main LangGraph State Definition

class KernelAgentState(TypedDict):
    """The shared state dictionary updated dynamically across graph cycles."""
    # Source Inputs
    raw_kernel_code: str
    raw_profiling_data: str
    
    # Parsed Hardware Analytics
    parsed_metrics: Dict[str, float]
    ast_insights: List[str]
    
    # LLM Inference Objects
    diagnosis: Optional[BottleneckDiagnosis]
    optimization_plan: Optional[OptimizationStrategy]
    
    # Output Artifacts
    optimized_kernel_code: Optional[str]
    annotations: Optional[Dict[str, str]]
    final_report: Optional[str]
    
    # Cyclic Flow Control Counters
    iteration_count: int