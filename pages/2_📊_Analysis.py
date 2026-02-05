import streamlit as st
import os
import glob
import pandas as pd
from traffic_rl.config import AppConfig
from traffic_rl.tools.analyzer import run_analysis_suite
from traffic_rl.gui_utils import render_sidebar, get_experiment_folders

# PAGE SETUP
st.set_page_config(page_title="Analysis Dashboard", layout="wide", page_icon="📊")
render_sidebar()

st.header("📊 Experiment Analysis Dashboard")

# Retrieves list of folders.
available_folders = get_experiment_folders()

if not available_folders:
    st.warning("No experiment folders found. Run a training batch first!")
else:
    # Dropdown to select which batch to analyze
    folder = st.selectbox("Select Experiment Folder", available_folders)
    
    if folder:
        col1, col2 = st.columns([1, 4])
        

        with col1:
            st.write("Actions:")
            # Trigs the processing of training logs
            if st.button("Run Analysis Suite", type="primary"):
                with st.spinner(f"Analyzing {folder}..."):
                    conf = AppConfig.load("configs/default.yaml")
                    # runs analysis aka pars logs, calc metrics, and gens summaries CSVs/Plots
                    run_analysis_suite(folder, conf)
                    st.success("Complete!")
        
        # Define expect path
        csv_path = os.path.join(folder, "analysis_data.csv")
        
        # We mostly try to find our analysis [mostly there because we went through several code versions for this project] 
        if not os.path.exists(csv_path):
            res_dirs = sorted(glob.glob("results/analysis_*"), reverse=True)
            if res_dirs:
                # Use most recent analysis folder
                csv_path = os.path.join(res_dirs[0], "analysis_data.csv")
                heatmap_path = os.path.join(res_dirs[0], "heatmap_improvement.png")
            else:
                heatmap_path = None
        else:
            heatmap_path = os.path.join(folder, "heatmap_improvement.png")

        # TABLE
        with col2:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Low wait times Good = Green / High wait times Bad = Red.
                st.dataframe(
                    df.style.background_gradient(subset=['Overall_Avg_Wait'], cmap="RdYlGn_r"), 
                    width="stretch"
                )
            else:
                st.info("No analysis CSV found. Click 'Run Analysis Suite'.")

        # visual
        if os.path.exists(csv_path):
            st.markdown("### Visualization")
            c1, c2 = st.columns(2)
            
            # Heatmap
            if heatmap_path and os.path.exists(heatmap_path):
                c1.image(heatmap_path, caption="Improvement Heatmap", width="stretch")
            
            # Dynamic Bar Chart
            with c2:
                df = pd.read_csv(csv_path)
                # Filter columns that look like scenarios
                scenarios = [c for c in df.columns if "x)" in c]
                
                if scenarios:
                    melted = df.melt(
                        id_vars=["Model", "Exp"], 
                        value_vars=scenarios, 
                        var_name="Scenario", 
                        value_name="Wait Time"
                    )
                    #Bar charts comp wait times across models
                    st.bar_chart(melted, x="Scenario", y="Wait Time", color="Model")