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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_utils import (parse_recording, make_windows, normalize, denormalize,
                         to_csv_bytes, STRIDE, FS, WINDOW)

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
    _, _, _, (_, _, _, _, X_test, y_test), stats = get_har_data(data_dir)

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

    return X_test, y_test, bilstm_model, trans_model, adv_model, hybrid_wins, stats


# Header Section
st.title("🏃 Motion Mindset — Wearable HAR Live System")
st.caption(
    "Deep Learning Final Project (Fall 2026) | Real-time inference on smartphone inertial sensors (50Hz, 2.56s window)."
)

X_test, y_test, bilstm_model, trans_model, adv_model, hybrid_wins, stats = load_assets()

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
    input_mode = st.sidebar.radio(
        "Input source:", ["Test-set window", "Upload my own sensor CSV"],
        help="Test-set windows come from the 9 held-out subjects. "
             "Upload lets you run the models on any recording.",
    )
    sample, true_label, window_title, timeline = None, None, "", None

    if input_mode == "Test-set window":
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

    sensor_view = st.sidebar.radio(
        "3. Sensor Channels to Plot:",
        ["Body Accelerometer (X, Y, Z)", "Body Gyroscope (X, Y, Z)", "Total Accelerometer (X, Y, Z)"],
    )


    if input_mode == "Test-set window":
        sample = X_test[selected_idx : selected_idx + 1]
        true_label = ACTIVITY_LABELS[y_test[selected_idx]]
        window_title = f"Window #{selected_idx}"
    else:
        st.sidebar.header("📤 Upload sensor data")
        layout_label = st.sidebar.radio(
            "CSV layout:",
            ["9 columns (UCI HAR order)", "6 columns (phone: total_acc xyz + gyro xyz)"],
            help="UCI order = body_acc xyz, body_gyro xyz, total_acc xyz. "
                 "With 6 columns, body_acc is derived by removing gravity (0.3 Hz low-pass).",
        )
        layout = "uci9" if layout_label.startswith("9") else "phone6"
        acc_ms2 = st.sidebar.radio("Accelerometer unit:", ["g", "m/s²"]) == "m/s²"
        up = st.sidebar.file_uploader(f"Sensor CSV (≥ {WINDOW} rows, {FS} Hz)", type=["csv", "txt"])

        # ready-made files so the demo works without a real sensor
        single = denormalize(X_test[0], stats).reshape(-1, 9)
        multi = np.concatenate([denormalize(X_test[int(np.where(y_test == k)[0][0])], stats).reshape(-1, 9) for k in (0, 3, 5)])
        st.sidebar.download_button("⬇️ Sample: 1 window (walking)", to_csv_bytes(single), "sample_window.csv", "text/csv")
        st.sidebar.download_button("⬇️ Sample: walking → sitting → laying", to_csv_bytes(multi), "sample_recording.csv", "text/csv")

        if up is None:
            st.info("Upload a CSV from the sidebar (or download one of the sample files, then upload it back).")
        else:
            try:
                rec = parse_recording(up, layout=layout, acc_in_ms2=acc_ms2)
                windows = normalize(make_windows(rec), stats)
                n_win = len(windows)
                i = st.sidebar.slider("Window", 1, n_win, 1) - 1 if n_win > 1 else 0
                sample = windows[i : i + 1]
                window_title = f"Uploaded recording — window {i + 1}/{n_win} ({i * STRIDE / FS:.2f}s)"
                if n_win > 1:
                    pred_all = np.argmax(adv_model.predict(windows, verbose=0), axis=1)
                    timeline = {"t": np.arange(n_win) * STRIDE / FS, "pred": pred_all}
            except ValueError as err:
                st.error(f"Could not read the file: {err}")

    def _status(pred):
        if true_label is None:
            return ""
        return "✅ Correct | " if pred == true_label else "❌ Error | "

    if sample is not None:
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
        st.subheader(f"{window_title} — Model Inference Results")
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("🎯 Ground Truth", true_label if true_label else "Unknown", "Unseen Test Subject" if true_label else "Your uploaded data (no label)")

        adv_status = _status(adv_pred)
        k2.metric(
            "🥇 Advanced Hybrid",
            adv_pred,
            f"{adv_status}{np.max(adv_probs)*100:.1f}% ({adv_ms:.1f}ms)",
        )

        bilstm_status = _status(bilstm_pred)
        k3.metric(
            "🥈 Classic Bi-LSTM",
            bilstm_pred,
            f"{bilstm_status}{np.max(bilstm_probs)*100:.1f}% ({bilstm_ms:.1f}ms)",
        )

        trans_status = _status(trans_pred)
        k4.metric(
            "🥉 Transformer",
            trans_pred,
            f"{trans_status}{np.max(trans_probs)*100:.1f}% ({trans_ms:.1f}ms)",
        )

        st.divider()

        # Visualizations: Signal Waveform & Confidence Distribution
        left_col, right_col = st.columns([1.25, 1])

        with left_col:
            st.subheader(f"📈 {sensor_view} (2.56s Window)")
            t = np.arange(128) / 50.0
            ch_offset = 6 if "Total" in sensor_view else (0 if "Body Acc" in sensor_view else 3)

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

        # Timeline over a whole uploaded recording (Advanced Hybrid)
        if timeline is not None:
            st.divider()
            st.subheader("🕒 Activity timeline over the whole recording (Advanced Hybrid)")
            fig_tl = go.Figure(go.Scatter(
                x=timeline["t"], y=[ACTIVITY_LABELS[i] for i in timeline["pred"]],
                mode="lines+markers", line=dict(shape="hv"),
            ))
            fig_tl.update_layout(xaxis_title="Window start (s)", height=320,
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 yaxis=dict(categoryorder="array", categoryarray=ACTIVITY_LABELS))
            st.plotly_chart(fig_tl, use_container_width=True)


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
    * **Course:** Deep Learning (AI403) — Fall 2026
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
