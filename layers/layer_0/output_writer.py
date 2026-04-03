import os
import json
import csv
from datetime import datetime

def export_construction_log(
    raw_log,
    inventory,
    run_id,
    architecture,
    output_dir,
    execution_success,
    latency_seconds,
    llm_invocations,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    final_summary,
    metrics
):
    """
    Exports the Orchestrator's construction log and metrics to JSON, CSV, and Markdown.
    Matches the data structures used by the ReAct Layer-0 experiment runner.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped base filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{run_id}_{architecture}_{timestamp}"
    
    # Define file paths
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    md_path = os.path.join(output_dir, f"{base_filename}.md")
    
    # 1. JSON Export (Full Payload)
    export_data = {
        "run_metadata": {
            "run_id": run_id,
            "architecture": architecture,
            "timestamp": timestamp,
            "execution_success": execution_success,
            "latency_seconds": round(latency_seconds, 2)
        },
        "token_usage": {
            "invocations": llm_invocations,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "metrics": metrics,
        "final_summary": final_summary,
        "inventory": inventory,
        "construction_log": raw_log
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
        
    # 2. CSV Export (Flattened Construction Log)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        headers = [
            "action", "result_status",
            "parse_success_confidence", "output_completeness", "format_validity",
            "downstream_readiness", "overall_quality_score",
            "eval_reason", "arguments", "raw_result"
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        if raw_log:
            for entry in raw_log:
                writer.writerow({
                    "action": entry.get("action", "UNKNOWN"),
                    "result_status": entry.get("result_status", "UNKNOWN"),
                    "parse_success_confidence": entry.get("parse_success_confidence", ""),
                    "output_completeness": entry.get("output_completeness", ""),
                    "format_validity": entry.get("format_validity", ""),
                    "downstream_readiness": entry.get("downstream_readiness", ""),
                    "overall_quality_score": entry.get("overall_quality_score", ""),
                    "eval_reason": entry.get("eval_reason", ""),
                    "arguments": json.dumps(entry.get("arguments", {})),
                    "raw_result": str(entry.get("raw_result", "")).replace("\n", " | ")
                })

    # 3. Markdown Export (Human-Readable Summary)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Run Summary: {run_id}\n\n")
        
        f.write("## Metadata\n")
        f.write(f"- **Architecture**: `{architecture}`\n")
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"- **Status**: {'Success' if execution_success else 'Failed'}\n")
        f.write(f"- **Latency**: {latency_seconds:.2f} seconds\n\n")
        
        f.write("## Cost & Token Usage\n")
        f.write(f"- **LLM Invocations**: {llm_invocations}\n")
        f.write(f"- **Prompt Tokens**: {prompt_tokens:,}\n")
        f.write(f"- **Completion Tokens**: {completion_tokens:,}\n")
        f.write(f"- **Total Tokens**: {total_tokens:,}\n\n")
        
        f.write("## Accuracy Metrics\n")
        if metrics:
            for metric_name, value in metrics.items():
                # Format floats cleanly
                formatted_val = f"{value:.2f}%" if isinstance(value, float) else value
                f.write(f"- **{metric_name.replace('_', ' ').title()}**: {formatted_val}\n")
        else:
            f.write("*No metrics generated for this run.*\n")
            
        f.write("\n## Orchestrator Final Summary\n")
        f.write(f"> {final_summary}\n\n")
        
        f.write("## Log Overview\n")
        f.write(f"Processed **{len(raw_log)}** total actions. See `{os.path.basename(csv_path)}` for full step-by-step trace.\n")
        
    # Return paths as expected by main.py
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path
    }