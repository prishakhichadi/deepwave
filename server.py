import json
import asyncio
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.graph import deepwave_graph
from src.state import KernelAgentState
from config.settings import settings

app = FastAPI(
    title="DEEPWAVE GPU Kernel Optimization API",
    description="Agentic GPU kernel bottleneck diagnosis and rewriting pipeline",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _check_api_key() -> None:
    """Fail fast on startup with a clear message if OPENAI_API_KEY isn't configured,
    rather than letting every request 500 with an opaque LangChain error."""
    settings.require_api_key()


NODE_META = {
    "reader":     {"label": "Profiling Reader",       "description": "Parsing rocprof metrics"},
    "analyzer":   {"label": "Kernel Analyzer",        "description": "Running AST analysis"},
    "classifier": {"label": "Bottleneck Classifier",  "description": "Diagnosing hardware bottleneck"},
    "planner":    {"label": "Optimization Planner",   "description": "Selecting optimization strategy"},
    "rewriter":   {"label": "Kernel Rewriter",        "description": "Rewriting kernel code"},
    "critic":     {"label": "Compiler Critic",        "description": "Validating rewritten code"},
    "reporter":   {"label": "Report Writer",          "description": "Generating final report"},
}

NODE_ORDER = ["reader", "analyzer", "classifier", "planner", "rewriter", "critic", "reporter"]



def sse_event(event_type: str, data: dict) -> str:
    """Formats a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"



async def run_pipeline_stream(
    kernel_code: str,
    profiling_data: str,
) -> AsyncGenerator[str, None]:
    """
    Runs the LangGraph pipeline and yields SSE events for each node transition.
    The frontend consumes these to update the progress tracker in real time.
    """
    initial_state: KernelAgentState = {
        "raw_kernel_code":          kernel_code,
        "raw_profiling_data":       profiling_data,
        "parsed_metrics":           {},
        "ast_insights":             [],
        "ast_findings":             [],
        "diagnosis":                None,
        "optimization_plan":        None,
        "optimized_kernel_code":    None,
        "annotations":              None,
        "theoretical_improvement":  None,
        "final_report":             None,
        "iteration_count":          0,
        "max_iterations":           3,
        "verification_status":      None,
        "critic_feedback":          None,
    }


    yield sse_event("pipeline_start", {
        "nodes": [
            {"id": name, **NODE_META[name]}
            for name in NODE_ORDER
        ]
    })

    await asyncio.sleep(0.05)

    try:

        final_state = None

        for chunk in deepwave_graph.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                meta = NODE_META.get(node_name, {"label": node_name, "description": ""})

                node_output = _summarize_node_output(node_name, state_update)

                yield sse_event("node_complete", {
                    "node":        node_name,
                    "label":       meta["label"],
                    "description": meta["description"],
                    "output":      node_output,
                })

                await asyncio.sleep(0.05)  
                final_state = state_update


        if final_state is not None:

            complete_state = deepwave_graph.invoke(initial_state)
            yield sse_event("pipeline_complete", _build_result_payload(complete_state))
        else:
            yield sse_event("pipeline_error", {"message": "Pipeline produced no output."})

    except Exception as e:
        yield sse_event("pipeline_error", {"message": str(e)})


def _summarize_node_output(node_name: str, state_update: dict) -> dict:
    """Extracts a safe, serializable summary from each node's state update."""
    summary = {}

    if node_name == "reader":
        metrics = state_update.get("parsed_metrics", {})
        summary["metrics"] = {k: v for k, v in metrics.items() if v > 0.0}

    elif node_name == "analyzer":
        findings = state_update.get("ast_findings", [])
        summary["finding_count"] = len(findings)
        summary["findings"] = [
            {"type": f.finding_type, "severity": f.severity, "location": f.location}
            for f in findings
        ]

    elif node_name == "classifier":
        diagnosis = state_update.get("diagnosis")
        if diagnosis:
            summary["bottleneck_type"]    = diagnosis.bottleneck_type
            summary["confidence_score"]   = diagnosis.confidence_score
            summary["evidence"]           = diagnosis.evidence
            summary["secondary"]          = diagnosis.secondary_bottleneck
        summary["evidence_consistency"] = state_update.get("evidence_consistency")

    elif node_name == "planner":
        plan = state_update.get("optimization_plan")
        if plan:
            summary["strategy_name"]   = plan.strategy_name
            summary["target_scopes"]   = plan.target_scopes
            summary["expected_impact"] = plan.expected_impact
            summary["rationale"]       = plan.rationale

    elif node_name == "rewriter":
        summary["code_generated"]          = bool(state_update.get("optimized_kernel_code"))
        summary["annotation_count"]        = len(state_update.get("annotations") or {})
        summary["theoretical_improvement"] = state_update.get("theoretical_improvement", "")

    elif node_name == "critic":
        summary["verification_status"] = state_update.get("verification_status", "unknown")
        summary["critic_feedback"]     = state_update.get("critic_feedback", "")

    elif node_name == "reporter":
        summary["report_generated"] = bool(state_update.get("final_report"))

    return summary


def _build_result_payload(state: dict) -> dict:
    """Builds the complete result payload sent after pipeline completion."""
    diagnosis = state.get("diagnosis")
    plan      = state.get("optimization_plan")
    findings  = state.get("ast_findings") or []

    return {
        "final_report": state.get("final_report", ""),
        "original_code": state.get("raw_kernel_code", ""),
        "optimized_code": state.get("optimized_kernel_code", ""),
        "theoretical_improvement": state.get("theoretical_improvement", ""),
        "verification_status": state.get("verification_status", ""),
        "diagnosis": {
            "bottleneck_type":    diagnosis.bottleneck_type    if diagnosis else None,
            "confidence_score":   diagnosis.confidence_score   if diagnosis else None,
            "evidence":           diagnosis.evidence           if diagnosis else [],
            "secondary":          diagnosis.secondary_bottleneck if diagnosis else None,
        },
        "evidence_consistency":        state.get("evidence_consistency"),
        "evidence_consistency_detail": state.get("evidence_consistency_detail"),
        "severity_label":  state.get("severity_label"),
        "severity_score":  state.get("severity_score"),
        "severity_detail": state.get("severity_detail"),
        "optimization_plan": {
            "strategy_name":    plan.strategy_name    if plan else None,
            "target_scopes":    plan.target_scopes    if plan else [],
            "expected_impact":  plan.expected_impact  if plan else None,
            "rationale":        plan.rationale        if plan else None,
            "amd_hints":        plan.amd_specific_hints if plan else [],
        },
        "ast_findings": [
            {
                "type":        f.finding_type,
                "severity":    f.severity,
                "location":    f.location,
                "description": f.description,
            }
            for f in findings
        ],
        "parsed_metrics": state.get("parsed_metrics", {}),
        "annotations": state.get("annotations") or {},
        "iteration_count": state.get("iteration_count", 0),
    }


# Routes

@app.get("/health")
def health():
    return {"status": "ok", "service": "DEEPWAVE API"}


@app.post("/analyze")
async def analyze(
    kernel_file:   UploadFile = File(..., description=".hip or .cu kernel source file"),
    profiling_file: UploadFile = File(..., description="rocprof or omniperf CSV file"),
):
    """
    Runs the full DEEPWAVE pipeline on the uploaded kernel + profiling data.
    Returns a Server-Sent Event stream with node-by-node progress updates,
    followed by a final pipeline_complete event containing all results.
    """
    # Validate file types
    if not (kernel_file.filename.endswith(".hip") or kernel_file.filename.endswith(".cu")):
        raise HTTPException(400, "Kernel file must be a .hip or .cu file")
    if not kernel_file.filename.endswith(".csv"):
        pass  # Be lenient on CSV naming

    kernel_code    = (await kernel_file.read()).decode("utf-8")
    profiling_data = (await profiling_file.read()).decode("utf-8")

    if not kernel_code.strip():
        raise HTTPException(400, "Kernel file is empty")
    if not profiling_data.strip():
        raise HTTPException(400, "Profiling CSV file is empty")

    return StreamingResponse(
        run_pipeline_stream(kernel_code, profiling_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  
        }
    )


@app.post("/analyze/text")
async def analyze_text(
    kernel_code:    str = Form(...),
    profiling_data: str = Form(...),
):
    """
    Same as /analyze but accepts raw text instead of file uploads.
    Useful for the frontend's paste-mode input.
    """
    if not kernel_code.strip():
        raise HTTPException(400, "Kernel code is empty")
    if not profiling_data.strip():
        raise HTTPException(400, "Profiling data is empty")

    return StreamingResponse(
        run_pipeline_stream(kernel_code, profiling_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )