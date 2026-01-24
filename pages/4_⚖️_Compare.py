import streamlit as st
import pandas as pd
import os
from traffic_rl.config import AppConfig
from traffic_rl.tools.comparator import run_comparison_suite
from traffic_rl.gui_utils import render_sidebar, get_experiment_folders

st.set_page_config(page_title="Compare Experiments", layout="wide", page_icon="⚖️")
render_sidebar()

st.header("⚖️ Compare Experiments")

folders = get_experiment_folders()
if len(folders) < 2:
    st.warning("Need at least 2 experiment folders to compare.")
else:
    c1, c2 = st.columns(2)
    with c1: 
        folder_a = st.selectbox("Baseline (Old)", folders, index=1 if len(folders)>1 else 0)
    with c2: 
        folder_b = st.selectbox("Candidate (New)", folders, index=0)
    
    if st.button("Run Comparison", type="primary"):
        with st.spinner("Comparing..."):
            conf = AppConfig.load("configs/default.yaml")
            run_comparison_suite(folder_a, folder_b, conf)
        
        if os.path.exists("comparison_dashboard.png"): 
            st.image("comparison_dashboard.png", caption="Wait Time Comparison (Lower is Better)")
        
        if os.path.exists("full_comparison_results.csv"):
            df = pd.read_csv("full_comparison_results.csv")
            pivot = df.pivot_table(index=["Scenario"], columns="Version", values="Wait_Time").reset_index()
            
            if "Baseline (Old)" in pivot and "Optimized (New)" in pivot:
                pivot["Improvement %"] = ((pivot["Baseline (Old)"] - pivot["Optimized (New)"]) / pivot["Baseline (Old)"]) * 100
                st.subheader("Improvement Statistics")
                st.dataframe(
                    pivot.style.format({"Baseline (Old)": "{:.2f}", "Optimized (New)": "{:.2f}", "Improvement %": "{:.2f}%"})
                         .background_gradient(subset=["Improvement %"], cmap="RdYlGn", vmin=-10, vmax=50),
                    width="stretch"
                )