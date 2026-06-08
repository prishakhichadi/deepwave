"""reviews the normalized metrics and outputs a structured diagnosis classifying the primary performance limitation.
node utilizes LangChain's with_structured_output. answers in strict json."""


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from src.state import KernelAgentState, BottleneckDiagnosis

def bottleneck_classifier_node(state: KernelAgentState) -> Dict:
    """
    Ingests parsed hardware metrics and uses a structured LLM call 
    to output a concrete, schema-validated bottleneck diagnosis.
    """
    # 1. Initialize our LLM engine.
    # We use a temperature of 0.0 because hardware diagnostics require 
    # deterministic, data-driven reasoning, not creative prose.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    # 2. Bind a strict output schema to the model.
    # This guarantees the LLM returns a structured JSON payload that matches
    # our Pydantic 'BottleneckDiagnosis' model exactly, or it throws an exception.
    structured_llm = llm.with_structured_output(BottleneckDiagnosis)
    
    # 3. Create an explicit prompt engineered around GPU hardware concepts.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Senior AMD GPU Performance Engineer.\n"
            "Analyze the provided hardware telemetry metrics and diagnose the primary "
            "execution bottleneck hindering the application's speed-of-light performance.\n\n"
            "Classification Guide:\n"
            "- High mem_stalled and low valu_util -> Memory Bandwidth Bound\n"
            "- High valu_util and low mem_stalled -> Compute Bound\n"
            "- Low max_waves_per_cu -> Occupancy Limited\n"
        ),
        (
            "human",
            "Hardware Metrics to Analyze:\n{metrics}"
        )
    ])
    
    # 4. Construct our processing chain and execute it.
    classifier_chain = prompt | structured_llm
    diagnosis_output = classifier_chain.invoke({"metrics": state["parsed_metrics"]})
    
    # 5. Return the updated key to update our shared LangGraph state.
    return {"diagnosis": diagnosis_output}