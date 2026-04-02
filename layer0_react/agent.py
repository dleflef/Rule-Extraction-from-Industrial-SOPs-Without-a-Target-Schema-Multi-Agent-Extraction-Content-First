import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from state import OrchestratorState
from tools import tools, tools_by_name


# 1. LLM Configuration
llm = ChatOpenAI(
    model="qwen/qwen3-vl-4b",
    temperature=0.0,  
    api_key="lm-studio-local",
    base_url="http://127.0.0.1:1234/v1",
)
llm_with_tools = llm.bind_tools(tools)


# 2. Node: Reasoner (The "Brain")
def reasoner_node(state: OrchestratorState):
    """
    Acts as the cognitive engine for Layer 0. 
    It evaluates the current file inventory, reads the strict methodological rules,
    and decides which tools to call next to advance the pipeline.
    Critically, it delegates extraction rather than performing it directly.
    """
    
    # Expose the current status of all files so the LLM knows what still needs routing
    inventory_str = json.dumps(state.get("inventory", []), indent=2)
    
    system_prompt = SystemMessage(content=f"""You are the Layer 0 Orchestrator Agent for a Knowledge Graph construction system.
Your job is to coordinate the processing of technical documentation. Do NOT extract entities yourself.

CURRENT INVENTORY:
{inventory_str}

STRICT METHODOLOGICAL RULES:
1. STRUCTURED-FIRST POLICY: You MUST route structured sources (.csv, .json) to Layer 1 BEFORE unstructured sources (.pdf, .txt).
2. VERIFICATION: After routing a source, you MUST use `evaluate_subordinate_output` to check for stalled or poor output.
3. ESCALATION: If an output fails evaluation, you MUST use `escalate_issue`.

CRITICAL STOPPING CONDITION:
You are finished ONLY when every item in the inventory has either successfully passed evaluation OR been escalated. 
Once all items are resolved, write a final text summary and you MUST NOT invoke any more tools.
""")
    
    response = llm_with_tools.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


# 3. Node: Tool Execution & Logging
def tool_node(state: OrchestratorState):
    """
    Executes the tools chosen by the Reasoner and rigorously logs the results.
    This node intercepts all tool outputs to update the global inventory status
    and builds an immutable CSV/JSON trace for scientific evaluation metrics.
    """
    last_message = state["messages"][-1]
    outputs = []
    log_updates = []
    
    # Create a mutable copy of the inventory so to update file statuses
    # (e.g., from 'pending' to 'routed') and pass it back to the graph state.
    inventory = [dict(item) for item in state.get("inventory", [])]

    # Maps basic tool usage to their corresponding inventory state
    STATUS_MAP = {
        "route_to_layer_1": "routed",
        "escalate_issue":   "escalated",
    }

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Try to execute the requested tool
            if tool_name in tools_by_name:
                try:
                    result = tools_by_name[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
            else:
                result = f"Tool '{tool_name}' not found."

            source_id = tool_args.get("source_id") if isinstance(tool_args, dict) else None

            # Data Extraction for Scientific Logging 
            confidence_score = None
            eval_reason = None
            
            # The evaluation tool returns a JSON string containing the metrics.
            # Parse it here to extract the confidence scores and update the graph state.
            if tool_name == "evaluate_subordinate_output":
                try:
                    clean = str(result).replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    result_status = parsed.get("status", "unknown")
                    confidence_score = parsed.get("confidence_score")
                    eval_reason = parsed.get("reason")
                except (json.JSONDecodeError, AttributeError):
                    result_status = "failed"
                
                # Dynamically set the inventory status based on whether the mock evaluator passed or failed
                inv_status = "evaluation_passed" if result_status == "success" else "evaluation_failed"
                if source_id:
                    for item in inventory:
                        if item["source_id"] == source_id:
                            item["status"] = inv_status
                            break
            else:
                # Fallback logic for standard tools (like inspect_source or route_to_layer_1)
                result_status = "success" if "SUCCESS" in str(result) else "flagged"
                
                # Apply simple status updates based on the STATUS_MAP defined above
                if source_id and tool_name in STATUS_MAP:
                    for item in inventory:
                        if item["source_id"] == source_id:
                            item["status"] = STATUS_MAP[tool_name]
                            break

            # Append the tool's raw output so the LLM can "read" what happened
            outputs.append(
                ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_call.get("id")
                )
            )

            # Append the structured data to our thesis evaluation log
            log_updates.append({
                "action": tool_name,
                "arguments": tool_args,
                "result_status": result_status,
                "confidence_score": confidence_score,
                "eval_reason": eval_reason,
                "raw_result": str(result)
            })

    return {"messages": outputs, "construction_log": log_updates, "inventory": inventory}


# 4. Node: System Guardrail
def guardrail_node(state: OrchestratorState):
    """
    An automated system constraint to prevent LLM hallucinations.
    If the LLM attempts to stop processing before every item is fully evaluated
    or escalated, this node bounces the flow back to the Reasoner with a warning.
    """
    pending_items = [item["source_id"] for item in state.get("inventory", []) if item["status"] not in {"evaluation_passed", "escalated"}]
    
    warning_msg = (
        f"SYSTEM WARNING: You attempted to stop, but the inventory is not fully processed. "
        f"The following sources still need attention: {pending_items}. "
        f"You MUST continue using your tools to route, evaluate, or escalate them."
    )
    from langchain_core.messages import SystemMessage
    return {"messages": [SystemMessage(content=warning_msg)]}

# 5. Graph Routing Logic
def should_continue(state: OrchestratorState) -> str:
    """
    Determines the next edge to traverse in the ReAct loop.
    Checks structural conditions first, then defers to the Agent's tool calls.
    """
    last_message = state["messages"][-1]
    inventory = state.get("inventory", [])
    
    # Structural Stop: Check if all inventory items are in a terminal state (either passed evaluation or escalated).
    # Both "evaluation_passed" and "escalated" are considered final, terminal states.
    terminal_statuses = {"evaluation_passed", "escalated"}
    is_finished = inventory and all(item["status"] in terminal_statuses for item in inventory)

    # Agent Action: Did the LLM decide to use a tool?
    has_tools = hasattr(last_message, "tool_calls") and last_message.tool_calls

    if is_finished:
        print("[STRUCTURAL STOP]: All inventory items fully processed.")
        return "end"

    if has_tools:
        print(f"[AGENT ACTION]: Calling tools -> {[tc['name'] for tc in last_message.tool_calls]}")
        return "continue"

    # Guardrail Trigger: Agent stopped calling tools, but the inventory isn't finished.
    print("[SYSTEM GUARDRAIL]: Agent attempted premature stop. Redirecting...")
    return "guardrail"

# 6. Graph Compilation
workflow = StateGraph(OrchestratorState)

# Custom processing nodes
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("tools", tool_node)
workflow.add_node("guardrail", guardrail_node) 

# Define the entrypoint and cyclic routing
workflow.set_entry_point("reasoner")
workflow.add_conditional_edges(
    "reasoner",
    should_continue,
    {
        "continue": "tools",        # LLM wants to use a tool
        "guardrail": "guardrail",   # LLM tried to quit early
        "end": END                  # Pipeline is mathematically complete
    }
)

# After a tool executes, or the guardrail executes, always return to the Brain
workflow.add_edge("tools", "reasoner")
workflow.add_edge("guardrail", "reasoner") 

orchestrator_app = workflow.compile()