import streamlit as st
import os
import glob
import pandas as pd
from traffic_rl.config import AppConfig
from traffic_rl.tools.analyzer import run_analysis_suite
from traffic_rl.gui_utils import render_sidebar, get_experiment_folders

st.set_page_config(page_title="Analysis Dashboard", layout="wide", page_icon="📊")
render_sidebar()

st.header("📊 Experiment Analysis Dashboard")

available_folders = get_experiment_folders()
if not available_folders:
    st.warning("No experiment folders found. Run a training batch first!")
else:
    folder = st.selectbox("Select Experiment Folder", available_folders)
    
    if folder:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write("Actions:")
            if st.button("Run Analysis Suite", type="primary"):
                with st.spinner(f"Analyzing {folder}..."):
                    conf = AppConfig.load("configs/default.yaml")
                    run_analysis_suite(folder, conf)
                    st.success("Complete!")
        
        csv_path = os.path.join(folder, "analysis_data.csv")
        # Fallback search
        if not os.path.exists(csv_path):
            res_dirs = sorted(glob.glob("results/analysis_*"), reverse=True)
            if res_dirs:
                csv_path = os.path.join(res_dirs[0], "analysis_data.csv")
                heatmap_path = os.path.join(res_dirs[0], "heatmap_improvement.png")
            else:
                heatmap_path = None
        else:
            heatmap_path = os.path.join(folder, "heatmap_improvement.png")

        with col2:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.dataframe(
                    df.style.background_gradient(subset=['Overall_Avg_Wait'], cmap="RdYlGn_r"), 
                    width="stretch"
                )
            else:
                st.info("No analysis CSV found. Click 'Run Analysis Suite'.")

        if os.path.exists(csv_path):
            st.markdown("### Visualization")
            c1, c2 = st.columns(2)
            if heatmap_path and os.path.exists(heatmap_path):
                c1.image(heatmap_path, caption="Improvement Heatmap", width="stretch")
            with c2:
                df = pd.read_csv(csv_path)
                scenarios = [c for c in df.columns if "x)" in c]
                if scenarios:
                    melted = df.melt(id_vars=["Model", "Exp"], value_vars=scenarios, var_name="Scenario", value_name="Wait Time")
                    st.bar_chart(melted, x="Scenario", y="Wait Time", color="Model")