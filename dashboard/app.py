"""Streamlit dashboard - Phase 6 of the project plan.

Upload a vibration signal, get a fault prediction + confidence + anomaly
score from the FastAPI service (src/api/main.py). Run:

    uv run streamlit run dashboard/app.py

Set API_URL to point somewhere other than localhost:8000 (e.g. inside
Docker Compose, see docker/docker-compose.yml).
"""

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fault Detection in Engine Parts", page_icon="⚙️", layout="centered")

st.title("⚙️ Fault Detection in Engine Parts")
st.caption(
    "Upload a bearing vibration signal (CWRU-style `.mat` with a `*_DE_time` channel, or a "
    "plain `.csv`/`.txt` of one reading per line) to get a fault classification, confidence, "
    "and an unsupervised anomaly score."
)


@st.cache_data(show_spinner=False)
def check_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


health = check_health()
if health is None:
    st.error(
        f"Can't reach the inference API at `{API_URL}`. Make sure it's running "
        "(`uv run uvicorn src.api.main:app`) or set the `API_URL` environment variable."
    )
    st.stop()

with st.sidebar:
    st.subheader("About")
    st.write(
        "Classifier: 1D CNN tuned via Bayesian optimization (Phase 4), retrained on all "
        "CWRU loads. Anomaly score: convolutional autoencoder trained on healthy-bearing "
        "signal only (Phase 2)."
    )
    st.write(f"Known fault classes: {', '.join(health['classes'])}")
    st.write(f"Window size: {health['window_size']} samples")

uploaded_file = st.file_uploader("Signal file", type=["mat", "csv", "txt"])
rpm = st.number_input(
    "Shaft speed (RPM) - optional",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="If given, shows the physics-predicted defect frequency for the detected fault "
    "(Phase 5's bearing kinematics) alongside the model's own prediction.",
)

if uploaded_file is not None:
    if st.button("Analyze", type="primary"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"rpm": rpm} if rpm > 0 else {}
            try:
                resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
            except requests.RequestException as exc:
                st.error(f"Request to the API failed: {exc}")
                st.stop()

        if resp.status_code != 200:
            st.error(resp.json().get("detail", f"API returned HTTP {resp.status_code}"))
            st.stop()

        result = resp.json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted class", result["predicted_class"])
        col2.metric("Confidence", f"{result['confidence']:.1%}")
        col3.metric(
            "Anomaly",
            "Yes" if result["is_anomaly"] else "No",
            delta=f"{result['anomaly_rate']:.0%} of windows flagged",
            delta_color="inverse",
        )

        windows = result["windows"]
        physics = next((w["physics_context"] for w in windows if w["physics_context"]), None)
        if physics:
            st.info(
                f"At {physics['rpm']:.0f} RPM, the physics-predicted defect frequency for "
                f"**{result['predicted_class']}** is **{physics['expected_defect_frequency_hz']:.1f} Hz** "
                "(src/physics/bearing.py - the bearing kinematics from Phase 5)."
            )

        st.subheader("Average class probabilities across all windows")
        avg_probs = pd.DataFrame(
            [w["class_probabilities"] for w in windows]
        ).mean().sort_values(ascending=False)
        st.bar_chart(avg_probs)

        st.subheader(f"Per-window detail ({result['n_windows']} windows)")
        detail_df = pd.DataFrame(
            [
                {
                    "window": w["window_index"],
                    "predicted_class": w["predicted_class"],
                    "confidence": w["confidence"],
                    "reconstruction_error": w["reconstruction_error"],
                    "is_anomaly": w["is_anomaly"],
                }
                for w in windows
            ]
        )
        st.dataframe(detail_df, hide_index=True, use_container_width=True)
