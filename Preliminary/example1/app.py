import streamlit as st
import os
import sys
import json
import pandas as pd
import asyncio

# 1. ADD THE PROJECT ROOT TO SYSTEM PATH
current_dir = os.getcwd()
project_root = os.path.join(current_dir, "langgraph_industrial_demo")
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. IMPORTS
from workflows.langgraph_pipeline import build_workflow
from utils.metrics_tracker import tracker

st.set_page_config(page_title="Industrial AI Architect", layout="wide")

st.title("Industrial Multi-Agent Analysis Dashboard")
st.markdown("---")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    st.info("Model: Llama 3.1 (Local via LM Studio)")
    st.write("Status: Connected to http://localhost:1234/v1")
    
    # NEW: File uploader logic
    uploaded_file = st.file_uploader("Upload Industrial Data (zip)", type="zip")
    
    if uploaded_file is not None:
        # Save the uploaded file locally so the Ingestion Agent can find it
        with open("examples.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded and ready for processing!")

# ASYNC WRAPPER FUNCTION
async def process_pipeline_async(state):
    """Wraps the LangGraph build and execution in an async definition."""
    app = build_workflow()
    # Using ainvoke() allows non-blocking execution of I/O heavy nodes
    return await app.ainvoke(state)

if st.button("Run Full Diagnostic Pipeline"):
    # Check if we have the data file needed
    if not os.path.exists("examples.zip"):
        st.error("Please upload a 'examples.zip' file in the sidebar first!")
    else:
        with st.spinner("Executing Multi-Agent DAG (Asynchronously)..."):
            
            # Initial State
            initial_state = {
                "file_paths": {},
                "sensor_summary": {},
                "iot_payloads": [],
                "parsed_logs": [],
                "ocr_result": {}, 
                "final_report": "",
                "neo4j_status": ""
            }
            
            # Safely execute the async graph
            # Note: We clear metrics here to ensure fresh data per run
            tracker.metrics = [] 
            final_state = asyncio.run(process_pipeline_async(initial_state))
            
            # Layout: Report and Metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("Final Manufacturing Report")
                st.markdown(final_state.get("final_report", "Analysis failed."))
                
            with col2:
                st.header("Agent Performance")
                metrics_df = pd.DataFrame(tracker.metrics)
                if not metrics_df.empty:
                    st.dataframe(metrics_df[['agent_name', 'execution_time_seconds', 'total_tokens']])
                    
                    # Visualizing Token Usage
                    st.bar_chart(metrics_df.set_index('agent_name')['total_tokens'])

            # Show raw data extracted during the process
            with st.expander("View Extracted Agent Data"):
                st.subheader("Sensor Summary & Anomalies")
                st.json(final_state.get("sensor_summary", {}))
                
                st.subheader("PLC Logs (Filtered)")
                st.write(final_state.get("parsed_logs", []))
                
                st.subheader("IoT Network Summary")
                st.json(final_state.get("iot_payloads", []))
                
                st.subheader("Neo4j Status")
                st.info(final_state.get("neo4j_status", "Not attempted"))
