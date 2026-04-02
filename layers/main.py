import os
import zipfile
import time
import statistics

from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback

from layer_0.utils import build_source_inventory
from layer_0.metrics import calculate_accuracy_metrics
from layer_0.agent import orchestrator_app
from layer_0.output_writer import export_construction_log

OUTPUT_DIR = os.path.join("outputs", "layer_0")

# Data Preparation
def extract_source_archive(zip_path: str = "data/examples.zip", extract_to: str = "data/examples"):
    """
    Ensures a fresh, reproducible data environment for every experimental run.
    Extracting from a zipped baseline guarantees that no mutated or altered files 
    from previous runs contaminate the current trial.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[CRITICAL] Archive not found at {zip_path}.")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# Single Experimental Trial
def run_layer0_experiment(run_id: str, source_dir: str = "data/examples"):
    """
    Executes a single instantiation of the Layer 0 ReAct loop.
    Captures end-to-end telemetry including latency, token usage (for cost/scalability analysis), 
    and the step-by-step construction log.
    """
    print(f"\n[Layer 0 — ReAct] Starting run: {run_id}")

    # Build the agnostic initial inventory representing the raw technical documentation
    inventory = build_source_inventory(source_dir)
    if not inventory:
        print("[WARN] Inventory is empty. Check zip contents.")
        return
    print(f"[Layer 0] Discovered {len(inventory)} sources.")

    # Initialize the Graph State with our explicit methodological constraints
    initial_state = {
        "messages":         [HumanMessage(content="Begin orchestration of the current inventory. Remember the structured-first policy.")],
        "inventory":        inventory,
        "construction_log": [],
    }

    final_state       = {k: v for k, v in initial_state.items()}
    execution_success = False
    final_summary     = ""
    start_time        = time.time()

    # Context manager to track API calls, tokens, and potential costs
    # This is critical for evaluating the "scalability" and "costs" mentioned in the literature gap
    with get_openai_callback() as cb:
        try:
            # Invoke the Agentic Graph. A recursion limit prevents infinite ReAct loops 
            # if the LLM gets trapped in an unexpected behavioral loop.
            final_state   = orchestrator_app.invoke(initial_state, {"recursion_limit": 100})
            execution_success = True
        except Exception as e:
            final_summary = f"Execution failed: {e}"
            print(f"[ERROR] {e}")
            final_state = initial_state

    latency_seconds = time.time() - start_time

    # Extract the Agent's final textual conclusion
    messages = final_state.get("messages", [])
    if execution_success and messages:
        final_summary = messages[-1].content

    # Calculate accuracy metrics based on the strict routing constraints
    raw_log = final_state.get("construction_log", [])
    metrics = calculate_accuracy_metrics(raw_log, inventory)

    # Export all telemetry to disk for subsequent statistical analysis and charting
    paths = export_construction_log(
        raw_log=raw_log,
        inventory=final_state.get("inventory", inventory), # Pass the dynamically updated final inventory
        run_id=run_id,
        architecture="react",
        output_dir=OUTPUT_DIR,
        execution_success=execution_success,
        latency_seconds=latency_seconds,
        llm_invocations=cb.successful_requests,
        prompt_tokens=cb.prompt_tokens,
        completion_tokens=cb.completion_tokens,
        total_tokens=cb.total_tokens,
        final_summary=final_summary,
        metrics=metrics,
    )

    # Just to confirm the output paths in the console for quick access during development and debugging 
    print(f"\n[Layer 0 — ReAct] Run complete.")
    print(f"  JSON     → {paths['json']}")
    print(f"  CSV      → {paths['csv']}")
    print(f"  Markdown → {paths['markdown']}")

    return metrics


# Statistical Aggregation (N-Runs)
def run_multi_experiment(n_runs: int, source_dir: str = "data/examples"):
    """
    Statistically validates the framework against LLM non-determinism.
    Executes the pipeline N times and calculates the mean, standard deviation, 
    and 95% Confidence Interval for all accuracy metrics.
    """
    print(f"\n[Layer 0] Starting {n_runs}-run experiment series.")
    all_metrics = []

    # Execute N independent iterations
    for i in range(1, n_runs + 1):
        metrics = run_layer0_experiment(run_id=f"_AGENTIC_R{i:02d}", source_dir=source_dir)
        if metrics:
            all_metrics.append(metrics)

    if len(all_metrics) < 2:
        print("[WARN] Not enough successful runs to compute statistics. <2 runs required.")
        return

    # Compute and output the required statistical aggregations 
    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(all_metrics)}/{n_runs} successful runs)")
    print("=" * 70)
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        mean = statistics.mean(values)
        std  = statistics.stdev(values)
        ci   = 1.96 * std / (len(values) ** 0.5)   # 95% CI (normal approximation)
        print(f"  {key:35s}: {mean:6.1f}%  ±{std:.1f}%  (95% CI ±{ci:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    # NSet N_RUNS = 10 for the final 
    N_RUNS         = 2   
    ZIP_FILE_PATH  = os.path.join("data", "examples.zip")
    EXTRACTION_DIR = os.path.join("data", "examples")

    try:
        extract_source_archive(zip_path=ZIP_FILE_PATH, extract_to=EXTRACTION_DIR)
        run_multi_experiment(n_runs=N_RUNS, source_dir=EXTRACTION_DIR)
    except Exception as e:
        print(f"[FATAL] {e}")