import os
import json
import zipfile
import pandas as pd
from neo4j import GraphDatabase 
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from utils.metrics_tracker import track_performance
from utils.ocr_utils import extract_text_from_pdf

# --- NEO4J CONFIGURATION ---
NEO4J_URI = "bolt://54.86.186.129:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "REDACTED-NEO4J-PASSWORD"

# Returns file paths for CSV, JSON, LOG, and PDF files extracted from a zip archive.
@track_performance("Data Ingestion Agent")
def ingestion_agent(state: dict, **kwargs) -> dict:
    zip_path = "examples.zip"
    extract_dir = "examples_extracted"
    
    found_files = {"csv": None, "json": None, "log": None, "pdf": None}

    if not os.path.exists(zip_path):
        return {"file_paths": found_files}

    # Extract files
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        # If the zip file is corrupted, return the current dictionary
        return {"file_paths": found_files}

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            ext = file.lower().split('.')[-1]
            if ext in found_files and found_files[ext] is None:
                found_files[ext] = os.path.join(root, file)

    return {"file_paths": found_files}

# Agent to process sensor data, compute statistics, and detect anomalies
@track_performance("Sensor Data Agent")
def sensor_agent(state: dict, **kwargs) -> dict:
    csv_path = state.get("file_paths", {}).get("csv")
    if not csv_path or not os.path.exists(csv_path):
        return {"sensor_summary": {"error": "CSV file missing."}}

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Column Detection, avoiding station_id/sensor_id collision
    id_col = next((c for c in df.columns if any(k in c for k in ['workstation', 'station', 'machine'])), None)
    # Ensure name_col is not the same as id_col even if it contains 'id'
    name_col = next((c for c in df.columns if c != id_col and any(k in c for k in ['sensor', 'type', 'id'])), None)
    val_col = next((c for c in df.columns if any(k in c for k in ['value', 'reading'])), None)

    if not all([id_col, name_col, val_col]):
        return {"sensor_summary": {"error": f"Column mismatch. Found: {list(df.columns)}"}}

    # Aggregation with manual column naming to prevent reset_index errors
    grouped = df.groupby([id_col, name_col])[val_col].agg(['min', 'max', 'mean', 'std'])
    grouped.columns = ['min_val', 'max_val', 'mean_val', 'std_val']
    # Transform back to a flat structure for easier anomaly detection for the agent
    summary_df = grouped.reset_index()
    # Convert summary to records for easier processing in the agent
    summary_records = summary_df.to_dict(orient='records')
    
    anomalies = {}
    for row in summary_records:
        if pd.notna(row['std_val']) and row['std_val'] > 0:
            # Anomaly Detection: Max value exceeding Mean + 3*Std can be a threshold for critical anomalies in sensor data
            three_sigma_limit = row['mean_val'] + (3 * row['std_val'])
            if row['max_val'] > three_sigma_limit:
                key = f"{row[id_col]}_{row[name_col]}"
                anomalies[key] = f"Critical Anomaly: Max {row['max_val']:.2f} > Mean+3Std ({three_sigma_limit:.2f})"
                
    return {"sensor_summary": {"stats_count": len(summary_records), "anomalies": anomalies}}
    
@track_performance("IoT Payload Agent")
def iot_agent(state: dict, **kwargs) -> dict:
    json_path = state.get("file_paths", {}).get("json")
    if not json_path or not os.path.exists(json_path):
        return {"iot_payloads": []}

    # Load the JSON data 
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if not data:
        return {"iot_payloads": []}

    total_payloads = len(data)
    workstations_active = set() # Unique workstation IDs observed in the payloads
    
    for item in data:
        # Look inside the nested "device" block for the "id"
        device_info = item.get("device", {})
        workstation_id = device_info.get("id") 
        
        if workstation_id: 
            workstations_active.add(workstation_id)

    summary = [{
        "total_payloads_processed": total_payloads,
        "unique_devices_online": len(workstations_active),
        # Convert back to list for easier readability
        "active_workstations": list(workstations_active),
        "network_status": f"Stable - {total_payloads} packets received"
    }]
    
    print(f"IoT Agent found {len(workstations_active)} active workstations: {list(workstations_active)}")
    return {"iot_payloads": summary}

@track_performance("PLC Log Agent")
def log_agent(state: dict, **kwargs) -> dict:
    log_path = state.get("file_paths", {}).get("log")
    if not log_path or not os.path.exists(log_path):
        return {"parsed_logs": []}

    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    keywords = ["alarm", "stop", "oee", "counter", "tool change", "operator", "qc sampling"]

    parsed = [
        line.strip() for line in lines 
        if any(k in line.lower() for k in keywords)
    ]
    return {"parsed_logs": parsed}

@track_performance("OCR Document Agent")
def ocr_agent(state: dict, **kwargs) -> dict:
    pdf_path = state.get("file_paths", {}).get("pdf")
    full_text = extract_text_from_pdf(pdf_path)
    
    # Create a storage directory
    storage_dir = "langgraph_industrial_demo/data/temp_ocr"
    os.makedirs(storage_dir, exist_ok=True)
    
    # Save full text to a local file
    file_name = f"ocr_output_{os.path.basename(pdf_path)}.txt"
    storage_path = os.path.join(storage_dir, file_name)
    with open(storage_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Return only the path and a tiny snippet to the State
    return {
        "ocr_result": {
            "path": storage_path,
            "snippet": full_text[:500], # Small preview for the LLM
            "word_count": len(full_text.split())
        }
    }


# Structure: The data is organized exactly like a factory (Cell -> Machine -> Sensor).
# Cross-Reference: The Neo4j agent creates a graph structure that mirrors the physical layout of the manufacturing environment, allowing for intuitive querying and analysis in subsequent agents. Anomalies detected in the Sensor Data Agent are directly linked to their respective machines and sensors in the graph, enabling a holistic view of operational issues without needing to cross-reference multiple data sources manually.
@track_performance("Neo4j Knowledge Graph Agent")
def neo4j_agent(state: dict, **kwargs) -> dict:
    anomalies = state.get("sensor_summary", {}).get("anomalies", {})
    iot_summary = state.get("iot_payloads", [])
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        return {"neo4j_status": f"Connection Error: {str(e)}"}
        
    def create_graph_data(tx):
        # Extract active workstations safely
        active_workstations = []
        if iot_summary:
            active_workstations = iot_summary[0].get("active_workstations", [])
            
            # Create a central anchor node to tie the graph together
            tx.run("MERGE (c:Cell {id: 'Cell_A', name: 'Manufacturing Cell A'})")
            
            for ws_id in active_workstations:
                tx.run("""
                    MERGE (c:Cell {id: 'Cell_A'})
                    MERGE (w:Workstation {id: $ws_id})
                    MERGE (c)-[:CONTAINS]->(w)
                    MERGE (s:Sensor {uid: $ws_id + '_HEALTH', name: 'OPERATIONAL_STATUS', type: 'Health'})
                    MERGE (w)-[:HAS_SENSOR]->(s)
                """, ws_id=ws_id)
                
        # Map actual Anomalies correctly
        for anomaly_key, description in anomalies.items():
   
            ws_id = next((ws for ws in active_workstations if anomaly_key.startswith(ws)), None)
            
            if ws_id:
                # Extract the rest of the string as the sensor name
                # +1 removes the joining underscore (e.g. 'WS01_CNC_MILLING' + '_' -> 'SPINDLE_SPEED')
                sensor_name = anomaly_key[len(ws_id)+1:] 
                
                tx.run("""
                    MERGE (w:Workstation {id: $ws_id})
                    MERGE (s:Sensor {uid: $ws_id + '_' + $sensor_name, name: $sensor_name})
                    MERGE (w)-[:HAS_SENSOR]->(s)
                  
                    MERGE (a:Anomaly {message: $desc, time_logged: timestamp()})
                    MERGE (s)-[:LOGGED_ERROR]->(a)
                """, ws_id=ws_id, sensor_name=sensor_name, desc=description)

    try:
        with driver.session() as session:
            session.execute_write(create_graph_data)
        status = "Successfully updated Neo4j Knowledge Graph with connected Hierarchical Structure."
    except Exception as e:
        status = f"Neo4j Syntax/Transaction Error: {str(e)}"
    finally:
        driver.close()
        
    return {"neo4j_status": status}
    
@track_performance("Industrial Analysis Agent")
def analysis_agent(state: dict, **kwargs) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    import json
    
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1", 
        api_key="lm-studio", 
        model="meta-llama-3.1-8b-instruct",
        temperature=0.0
    )
    
    prompt = PromptTemplate.from_template(
        "You are an Industrial AI Architect. Generate a brief Manufacturing Cell Operational Summary.\n"
        "Sensor Anomalies: {sensors}\n"
        "IoT Network Payloads: {iot_data}\n"
        "Critical PLC Logs: {logs}\n"
        "Maintenance OCR Notes: {ocr}\n"
        "Knowledge Graph Status: {neo4j_status}\n\n"
        "Provide a structured summary of machine utilization, connectivity health, and immediate maintenance actions required."
    )
    
    chain = prompt | llm
    
    ocr_meta = state.get("ocr_result", {})
    ocr_context = ""
    if ocr_meta.get("path") and os.path.exists(ocr_meta["path"]):
        with open(ocr_meta["path"], "r", encoding="utf-8") as f:
            # We can now selectively read or truncate without bloating the graph state
            ocr_context = f.read()[:10000] 
    
    input_data = {
        "sensors": json.dumps(state.get("sensor_summary", {}).get("anomalies", {})),
        "iot_data": json.dumps(state.get("iot_payloads", [])),
        "logs": json.dumps(state.get("parsed_logs", [])),
        "ocr": ocr_context, # Sent to LLM, but never stored in permanent Graph State
        "neo4j_status": state.get("neo4j_status", "Not run")
    }
    
    response = chain.invoke(input_data)
    return {"final_report": response.content}
