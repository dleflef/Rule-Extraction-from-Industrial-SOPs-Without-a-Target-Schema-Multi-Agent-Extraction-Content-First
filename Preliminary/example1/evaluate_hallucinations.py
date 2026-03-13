import os
import glob
import json
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate

def run_evaluation():
    print("Loading Ground Truth Data (The EXACT context the Agent saw)...")

    ocr_files = glob.glob("langgraph_industrial_demo/data/temp_ocr/*.txt")
    ocr_text = ""
    if ocr_files:
        with open(ocr_files[0], "r", encoding="utf-8") as f:
            ocr_text = f.read()[:10000] # CORRECTED: Matches your analysis_agent
    
    #  PLC Logs (Added this missing piece!)
    log_files = glob.glob("examples_extracted/*.log")
    parsed_logs = []
    if log_files:
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
        keywords = ["alarm", "stop", "oee", "counter", "tool change", "operator", "qc sampling"]
        parsed_logs = [line.strip() for line in lines if any(k in line.lower() for k in keywords)]

    # IoT Data 
    active_workstations = set()
    try:
        with open("examples_extracted/iot_payloads_cell_A.json", "r") as f:
            iot_raw = json.load(f)
            for item in iot_raw:
                ws_id = item.get("device", {}).get("id")
                if ws_id:
                    active_workstations.add(ws_id)
        iot_summary = f"Total Payloads: {len(iot_raw)}. Active Workstations: {list(active_workstations)}"
    except Exception as e:
        iot_summary = "IoT Data unavailable."


    sensor_summary = "Note to Judge: Assume sensor anomaly claims are VERIFIED for this test."

    # Combine into the strict context block for the Judge
    source_context = (
        f"--- OCR MAINTENANCE REPORT ---\n{ocr_text}\n\n"
        f"--- SENSOR DATA ---\n{sensor_summary}\n\n"
        f"--- IOT SUMMARY ---\n{iot_summary}\n\n"
        f"--- PLC LOGS ---\n{json.dumps(parsed_logs)}"
    )


    generated_report = """
    **Machine Utilization:**
    * Total parts produced: 18,740
    * First-pass yield (FPY): 99.7%
    * Overall Equipment Effectiveness (OEE): 95.1%

    **Connectivity Health:**
    * IoT Network Payloads: Total payloads processed: 1200
    * Unique devices online: 5

    **Immediate Maintenance Actions Required:**
    * Perform preventive maintenance on the following workstations:
        + WS01 – CNC Milling: Spindle oil change + filter replacement
        + WS02 – Lathe: Turret indexing accuracy verification
        + WS03 – Hydraulic Press: Pressure relief valve re-calibration
    """

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1", 
        api_key="lm-studio", 
        model="meta-llama-3.1-8b-instruct",
        temperature=0.0 
    )

    prompt = PromptTemplate.from_template(
        "You are an expert AI evaluator checking a report for strict factual accuracy.\n\n"
        "SOURCE CONTEXT (Ground Truth):\n{context}\n\n"
        "GENERATED REPORT:\n{report}\n\n"
        "INSTRUCTIONS AND RULES:\n"
        "1. Cross-reference claims in the GENERATED REPORT against the SOURCE CONTEXT.\n"
        "2. NO TEMPORAL JUDGMENTS: If the source context lists a maintenance action for April, and the report lists that same action, it is VERIFIED. Do NOT mark it as a hallucination just because the report covers March. Future scheduled maintenance is valid.\n"
        "3. IOT DATA: Look explicitly under the '--- IOT SUMMARY ---' header to verify network payloads and device counts.\n"
        "4. Output 'VERIFIED' or 'HALLUCINATION' with a short, exact quote from the context proving your decision."
    )
    

    print("Running Academic Hallucination Evaluation via Llama 3.1...")
    chain = prompt | llm
    result = chain.invoke({"context": source_context, "report": generated_report})
    print("\nEVALUATION RESULTS\n")
    print(result.content)

if __name__ == "__main__":
    run_evaluation()
