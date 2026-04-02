import re

def calculate_accuracy_metrics(construction_log: list, inventory: list):
    """
    Parses the Orchestrator's construction log to calculate quantitative evaluation metrics.
    These metrics are designed specifically for the thesis to scientifically prove that the 
    Agentic LLM is not just executing tasks randomly, but is strictly adhering to methodological 
    constraints (like the structured-first policy) and maintaining a logical reasoning loop.
    """
    total_routes = 0
    correct_routes = 0
    routed_sources = set()
    evaluated_sources = set()
    route_eval_pairs_expected = 0
    route_eval_pairs_actual = 0
    
    # 1. Establish the Ground Truth
    # Dynamically map the expected behavior based on the inventory.
    # Methodology dictates that structured files MUST get Priority 1, 
    # and unstructured files MUST get Priority 2. This creates our baseline for accuracy.
    correct_priorities = {item["source_id"]: 1 if item["is_structured"] else 2 for item in inventory}


    # 2. Parse the Construction Log
    # Iterate chronologically through the agent's actions to evaluate its behavior over time.
    for i, step in enumerate(construction_log):
        action = step.get("action", "UNKNOWN")
        raw_result = step.get("raw_result", "")
        args = step.get("arguments", {})
        
        # Robust ID Extraction: Because LLMs are non-deterministic, they sometimes pass 
        # arguments as perfect JSON dicts, and sometimes as raw strings. 
        # This fallback regex ensures capturing the Source ID regardless of LLM formatting quirks.
        src_id = args.get("source_id") if isinstance(args, dict) else None
        if not src_id:
            src_match = re.search(r'(SRC_\d+)', raw_result + str(args))
            src_id = src_match.group(1) if src_match else None
        
        # Analyze Routing Actions
        if action == "route_to_layer_1" and src_id:
            total_routes += 1
            routed_sources.add(src_id)
            route_eval_pairs_expected += 1
            
            # Metric Routing Precision 
            # Did the agent actually obey the Structured-First policy? 
            # We extract the priority it assigned and compare it to our Ground Truth.
            pri_match = re.search(r'priority\s+(\d+)', raw_result, re.IGNORECASE)
            if pri_match and correct_priorities.get(src_id) == int(pri_match.group(1)):
                correct_routes += 1
                    
            # Metric C Trajectory Sequence Alignment
            # This tests the agent's "cognitive" logic. If it routes a file, the very next 
            # logical step MUST be to evaluate that same file. We look ahead one step in the log 
            # to verify that the agent isn't getting distracted or hallucinating intermediate steps.
            if i + 1 < len(construction_log):
                next_step = construction_log[i+1]
                next_action = next_step.get("action")
                next_args = next_step.get("arguments", {})
                next_raw = next_step.get("raw_result", "")
                
                # Extract Source ID for the next step using the same robust method
                next_src_id = next_args.get("source_id") if isinstance(next_args, dict) else None
                if not next_src_id:
                    next_m = re.search(r'(SRC_\d+)', next_raw + str(next_args))
                    next_src_id = next_m.group(1) if next_m else None
                    
                # Did the agent immediately evaluate the exact same source?
                if next_action == "evaluate_subordinate_output" and next_src_id == src_id:
                    route_eval_pairs_actual += 1

        # Track successful evaluations for our Recall metric
        elif action == "evaluate_subordinate_output" and src_id:
            evaluated_sources.add(src_id)

    # 3. Calculate Final Percentages
    # Routing Precision: Tool parameter accuracy (Did it assign priorities correctly?)
    routing_precision = (correct_routes / total_routes * 100) if total_routes > 0 else 0.0
    
    # Inventory Recall: Overall task coverage (Did it fully process every file it was given?)
    fully_processed = routed_sources.intersection(evaluated_sources)
    inventory_recall = (len(fully_processed) / len(inventory) * 100) if len(inventory) > 0 else 0.0
    
    # Trajectory Alignment: Procedural logic (Did it follow the correct ReAct sequence?)
    trajectory_alignment = (route_eval_pairs_actual / route_eval_pairs_expected * 100) if route_eval_pairs_expected > 0 else 0.0

    # 4. Console Output
    print("\n" + "="*70)
    print("ACCURACY & METHODOLOGY METRICS")
    print("="*70)
    print(f"Routing Precision (Tool Accuracy) : {routing_precision:.1f}% ({correct_routes}/{total_routes} correct priorities)")
    print(f"Inventory Recall (Task Coverage)  : {inventory_recall:.1f}% ({len(fully_processed)}/{len(inventory)} files fully processed)")
    print(f"Trajectory Sequence Alignment     : {trajectory_alignment:.1f}% ({route_eval_pairs_actual}/{route_eval_pairs_expected} immediate evaluations)")
    print("="*70 + "\n")
    
    # Return the metrics payload for JSON export and statistical aggregation
    return {
        "routing_precision": routing_precision,
        "inventory_recall": inventory_recall,
        "trajectory_alignment": trajectory_alignment
    }