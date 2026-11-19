import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import keras

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from datasets.loader import get_har_data, ACTIVITY_LABELS
from models.custom_layers import TemporalAttentionLayer

st.set_page_config(
    page_title="Motion Mindset — Wearable HAR Live System",
    page_icon="🏃",
    layout="wide",
)

# Custom clean styling for professional UI
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .stAlert {padding: 0.5rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    data_dir = os.path.join(PROJECT_ROOT, "data", "UCI HAR Dataset")
    _, _, _, (_, _, _, _, X_test, y_test), _ = get_har_data(data_dir)

    bilstm_model = keras.models.load_model(
        os.path.join(PROJECT_ROOT, "saved_models", "bilstm_har.keras")
    )
    trans_model = keras.models.load_model(
        os.path.join(PROJECT_ROOT, "saved_models", "transformer_har.keras")
    )
    adv_model = keras.models.load_model(
        os.path.join(PROJECT_ROOT, "saved_models", "cnn_bilstm_attn_har.keras"),
        custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer},
    )

    # Precompute batch predictions to easily find comparison cases
    bilstm_all = np.argmax(bilstm_model.predict(X_test, verbose=0), axis=1)
    adv_all = np.argmax(adv_model.predict(X_test, verbose=0), axis=1)
    hybrid_wins = np.where((adv_all == y_test) & (bilstm_all != y_test))[0]

    return X_test, y_test, bilstm_model, trans_model, adv_model, hybrid_wins


# Header Section
st.title("🏃 Motion Mindset — Wearable HAR Live System")
st.caption(
    "Deep Learning Final Project (Fall 2026)[cite: 3] | Real-time inference on smartphone inertial sensors (50Hz, 2.56s window)."
)

X_test, y_test, bilstm_model, trans_model, adv_model, hybrid_wins = load_assets()

# Navigation Tabs matching project requirements
tab_live, tab_bench, tab_info = st.tabs([
    "⚡ Live Activity Inference", 
    "📊 4-Model Benchmark & Computational Cost",
    "ℹ️ Project, Architecture & Team"
])

# ==============================================================================
# TAB 1: LIVE INFERENCE
# ==============================================================================
with tab_live:
    st.sidebar.header("🎛️ Sample Control Panel")
    filter_act = st.sidebar.selectbox("1. Filter by True Activity:", ["ALL ACTIVITIES"] + ACTIVITY_LABELS)

    if filter_act == "ALL ACTIVITIES":
        valid_indices = np.arange(len(y_test))
    else:
        target_id = ACTIVITY_LABELS.index(filter_act)
        valid_indices = np.where(y_test == target_id)[0]

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = int(valid_indices[0])

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("🎲 Random Sample", use_container_width=True):
        st.session_state.sample_idx = int(np.random.choice(valid_indices))
    if col_btn2.button("🏆 Hybrid Win Case", use_container_width=True, help="Pick a hard case where Bi-LSTM failed but Advanced Hybrid succeeded"):
        if len(hybrid_wins) > 0:
            st.session_state.sample_idx = int(np.random.choice(hybrid_wins))

    selected_idx = st.sidebar.number_input(
        f"2. Test Window ID (0 – {len(y_test)-1}):",
        min_value=0,
        max_value=len(y_test) - 1,
        value=st.session_state.sample_idx,
        step=1,
    )
    st.session_state.sample_idx = int(selected_idx)

    st.sidebar.divider()
    sensor_view = st.sidebar.radio(
        "3. Sensor Channels to Plot:",
        ["Body Accelerometer (X, Y, Z)", "Body Gyroscope (X, Y, Z)", "Total Accelerometer (X, Y, Z)"],
    )

    # Extract sample & run timed inference across models
    sample = X_test[selected_idx : selected_idx + 1]
    true_label = ACTIVITY_LABELS[y_test[selected_idx]]

    t0 = time.perf_counter()
    adv_probs = adv_model.predict(sample, verbose=0)[0]
    adv_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bilstm_probs = bilstm_model.predict(sample, verbose=0)[0]
    bilstm_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    trans_probs = trans_model.predict(sample, verbose=0)[0]
    trans_ms = (time.perf_counter() - t0) * 1000

    adv_pred = ACTIVITY_LABELS[int(np.argmax(adv_probs))]
    bilstm_pred = ACTIVITY_LABELS[int(np.argmax(bilstm_probs))]
    trans_pred = ACTIVITY_LABELS[int(np.argmax(trans_probs))]

    # Top KPI Cards
    st.subheader(f"Window #{selected_idx} — Model Inference Results")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("🎯 Ground Truth", true_label, "Unseen Test Subject")

    adv_status = "✅ Correct" if adv_pred == true_label else "❌ Error"
    k2.metric(
        "🥇 Advanced Hybrid",
        adv_pred,
        f"{adv_status} | {np.max(adv_probs)*100:.1f}% ({adv_ms:.1f}ms)",
    )

    bilstm_status = "✅ Correct" if bilstm_pred == true_label else "❌ Error"
    k3.metric(
        "🥈 Classic Bi-LSTM",
        bilstm_pred,
        f"{bilstm_status} | {np.max(bilstm_probs)*100:.1f}% ({bilstm_ms:.1f}ms)",
    )

    trans_status = "✅ Correct" if trans_pred == true_label else "❌ Error"
    k4.metric(
        "🥉 Transformer",
        trans_pred,
        f"{trans_status} | {np.max(trans_probs)*100:.1f}% ({trans_ms:.1f}ms)",
    )

    st.divider()

    # Visualizations: Signal Waveform & Confidence Distribution
    left_col, right_col = st.columns([1.25, 1])

    with left_col:
        st.subheader(f"📈 {sensor_view} (2.56s Window)")
        t = np.arange(128) / 50.0
        ch_offset = 0 if "Total" in sensor_view else (3 if "Body Acc" in sensor_view else 6)

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=t, y=sample[0, :, ch_offset], name="X-Axis", line=dict(width=2)))
        fig_sig.add_trace(go.Scatter(x=t, y=sample[0, :, ch_offset + 1], name="Y-Axis", line=dict(width=2)))
        fig_sig.add_trace(go.Scatter(x=t, y=sample[0, :, ch_offset + 2], name="Z-Axis", line=dict(width=2)))
        fig_sig.update_layout(
            xaxis_title="Time (seconds @ 50Hz)",
            yaxis_title="Z-Score Normalized",
            height=370,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with right_col:
        st.subheader("📊 Class Confidence Distribution")
        fig_prob = go.Figure(
            data=[
                go.Bar(name="Advanced Hybrid", x=ACTIVITY_LABELS, y=adv_probs * 100),
                go.Bar(name="Classic Bi-LSTM", x=ACTIVITY_LABELS, y=bilstm_probs * 100),
                go.Bar(name="Modern Transformer", x=ACTIVITY_LABELS, y=trans_probs * 100),
            ]
        )
        fig_prob.update_layout(
            barmode="group",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 105],
            height=370,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ==============================================================================
# TAB 2: BENCHMARK & COMPUTATIONAL COST
# ==============================================================================
with tab_bench:
    st.subheader("📊 Unified 4-Model Benchmark Comparison (Mean ± Std)")
    csv_path = os.path.join(PROJECT_ROOT, "results", "final_4models_comparison.csv")
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)
        
        st.download_button(
            label="📥 Download Benchmark CSV Results",
            data=df_bench.to_csv(index=False).encode('utf-8'),
            file_name="final_4models_comparison.csv",
            mime="text/csv",
        )

    img_path = os.path.join(PROJECT_ROOT, "results", "image4_advanced_model_results.png")
    if os.path.exists(img_path):
        st.divider()
        st.subheader("📈 Advanced Hybrid Model Performance & Confusion Matrix")
        st.image(img_path, use_container_width=True)

# ==============================================================================
# ==============================================================================
# TAB 3: PROJECT, ARCHITECTURE & TEAM INFO
# ==============================================================================
with tab_info:
    st.subheader("ℹ️ Project Metadata & Academic Context")
    st.markdown("""
    * **Course:** Deep Learning (A1403) — Fall 2026[cite: 3]
    * **Institution:** Pharos University in Alexandria (PUA) — Faculty of Computer Science & Artificial Intelligence
    * **Supervisor:** Dr. Sahar Ghanem
    * **Teaching Assistant:** Philopater Hany
    * **Team Name:** Motion Mindset
    * **Team Members:** 
        * Engy Khaled Mostafa
        * Habiba Abdulhameed
        * Rahaf Mohamed
        * Hana Yasser
    """)
    
    st.divider()
    st.subheader("⚙️ Technical Architecture Overview")
    st.markdown("""
    * **Input Data:** UCI HAR Dataset (9-channel inertial sensor time series, sampled at 50Hz with 2.56s sliding windows).
    * **Advanced Hybrid Model:** Combines 1D Spatial Convolutions (feature extraction), Bidirectional LSTM layers (temporal dependency modeling), and a Custom Temporal Attention Layer (focusing on critical motion segments).
    * **Evaluation Protocol:** Leakage-free train/val/test splits, 3 random seeds evaluation, per-class metrics, error analysis, and hardware performance tracking.
    """)
