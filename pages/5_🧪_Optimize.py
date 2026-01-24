import streamlit as st
import os
import yaml
from traffic_rl.tools.optimizer import run_hyperparameter_optimization
from traffic_rl.gui_utils import render_sidebar

st.set_page_config(page_title="Hyperparameter Optimization", layout="wide", page_icon="🧪")
render_sidebar()

st.header("🧪 Hyperparameter Optimization (Optuna)")

search_space = {}
col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    st.subheader("1. Learning Dynamics")
    tune_lr = st.checkbox("Tune Learning Rate", value=True)
    if tune_lr:
        c1, c2 = st.columns(2)
        lr_min = c1.number_input("Min LR", value=1e-5, format="%.6f")
        lr_max = c2.number_input("Max LR", value=1e-2, format="%.6f")
        search_space['lr'] = {'enabled': True, 'min': lr_min, 'max': lr_max}
    else: search_space['lr'] = {'enabled': False}

    tune_gamma = st.checkbox("Tune Gamma", value=False)
    if tune_gamma:
        gamma_opts = st.multiselect("Gamma Candidates", [0.9, 0.95, 0.98, 0.99, 0.995], default=[0.95, 0.99])
        search_space['gamma'] = {'enabled': True, 'choices': gamma_opts}
    else: search_space['gamma'] = {'enabled': False}

    tune_decay = st.checkbox("Tune Epsilon Decay", value=True)
    if tune_decay:
        c1, c2 = st.columns(2)
        d_min = c1.number_input("Min Decay", value=0.9000, step=0.001, format="%.4f")
        d_max = c2.number_input("Max Decay", value=0.9995, step=0.001, format="%.4f")
        search_space['epsilon_decay'] = {'enabled': True, 'min': d_min, 'max': d_max}
    else: search_space['epsilon_decay'] = {'enabled': False}

with col_opt2:
    st.subheader("2. Architecture")
    tune_batch = st.checkbox("Tune Batch Size", value=False)
    if tune_batch:
        batch_opts = st.multiselect("Batch Size Candidates", [32, 64, 128, 256], default=[32, 64, 128])
        search_space['batch_size'] = {'enabled': True, 'choices': batch_opts}
    else: search_space['batch_size'] = {'enabled': False}

    tune_dim = st.checkbox("Tune Hidden Dimension", value=True)
    if tune_dim:
        dim_opts = st.multiselect("Hidden Dim Candidates", [64, 128, 256, 512], default=[64, 128])
        search_space['hidden_dim'] = {'enabled': True, 'choices': dim_opts}
    else: search_space['hidden_dim'] = {'enabled': False}
    
    tune_target = st.checkbox("Tune Target Update Freq", value=False)
    if tune_target:
        target_opts = st.multiselect("Freq Candidates", [100, 500, 1000, 2000], default=[100, 1000])
        search_space['target_update'] = {'enabled': True, 'choices': target_opts}
    else: search_space['target_update'] = {'enabled': False}

st.markdown("---")
trials = st.slider("Number of Optuna Trials", 5, 100, 20)

if st.button("🚀 Start Optimization Loop", type="primary"):
    with st.spinner(f"Running {trials} trials..."):
        try:
            if not os.path.exists("configs/default.yaml"):
                st.error("configs/default.yaml not found.")
            else:
                run_hyperparameter_optimization("configs/default.yaml", trials, search_space)
                st.success("Optimization Complete!")
                if os.path.exists("configs/best_params_found.yaml"):
                    with open("configs/best_params_found.yaml") as f: best = yaml.safe_load(f)
                    st.subheader("🏆 Best Parameters Found")
                    st.json(best)
        except Exception as e: 
            st.error(f"Optimization Failed: {e}")