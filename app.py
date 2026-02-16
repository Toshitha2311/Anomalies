import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Traffic Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= TITLE =================
st.title("🚗 Traffic Flow Anomaly Detection")
st.markdown("Detect anomalies in network traffic using Isolation Forest algorithm")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.04, 0.01)
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100, 10)

st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
use_sample = st.sidebar.checkbox("Use sample dataset", value=False)

# ================= LOAD DATA =================
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✓ File uploaded successfully")

elif use_sample:
    sample_path = "embedded_system_network_security_dataset.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.sidebar.success("✓ Sample dataset loaded")
    else:
        st.warning("Sample dataset not found. Please upload a CSV file.")
        st.stop()

else:
    st.info("Please upload a CSV file or enable sample dataset.")
    st.stop()

# ================= DATA PROCESSING =================
st.sidebar.header("⚙️ Data Processing")
with st.sidebar.expander("Processing Steps"):
    st.write("1. Drop label column (if exists)")
    st.write("2. Convert bool to int")
    st.write("3. Scale features")
    st.write("4. Train Isolation Forest")
    st.write("5. Predict anomalies")

# Drop label safely
features = df.drop(columns=['label'], errors='ignore')

# Convert bool to int
for col in features.columns:
    if features[col].dtype == 'bool':
        features[col] = features[col].astype(int)

# Keep numeric only (prevents crashes)
features = features.select_dtypes(include=[np.number])

# Fill missing values
features = features.fillna(features.mean())

# Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# ================= MODEL =================
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=256,
    random_state=42
)

model.fit(scaled_df)
anomaly_labels = model.predict(scaled_df)
scaled_df['anomaly'] = anomaly_labels

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Visualizations", "📋 Details", "📥 Export"]
)

# ================= TAB 1 =================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    normal = scaled_df[scaled_df['anomaly'] == 1]
    anomaly = scaled_df[scaled_df['anomaly'] == -1]

    normal_count = len(normal)
    anomaly_count = len(anomaly)
    total_count = len(scaled_df)

    col1.metric("Total Records", total_count)
    col2.metric("Normal", normal_count, f"{normal_count/total_count*100:.1f}%")
    col3.metric("Anomalies", anomaly_count, f"{anomaly_count/total_count*100:.1f}%")
    col4.metric("Anomaly Rate", f"{contamination*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Data**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**With Anomaly Labels**")
        st.dataframe(scaled_df.head(), use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("Anomaly Detection Visualizations")

    numeric_cols = scaled_df.columns.drop('anomaly').tolist()

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            feat1 = st.selectbox("X-axis:", numeric_cols, key="feat1")
            feat2 = st.selectbox("Y-axis:", numeric_cols, key="feat2", index=1)

        with col2:
            plot_type = st.radio("Plot Type:", ["Scatter", "Density"], horizontal=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "Scatter":
            ax.scatter(normal[feat1], normal[feat2],
                       c='blue', label='Normal', alpha=0.6, s=20)
            ax.scatter(anomaly[feat1], anomaly[feat2],
                       c='red', label='Anomaly', alpha=0.9, s=80, marker='x')

        else:
            ax.hist2d(normal[feat1], normal[feat2], bins=30, cmap='Blues')
            ax.scatter(anomaly[feat1], anomaly[feat2],
                       c='red', s=80, marker='x')

        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f"{feat1} vs {feat2}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)

    # 3D Plot
    if len(numeric_cols) >= 3:
        st.subheader("3D Interactive Visualization")

        col1, col2, col3 = st.columns(3)
        feat_x = col1.selectbox("X-axis:", numeric_cols, key="3dx")
        feat_y = col2.selectbox("Y-axis:", numeric_cols, key="3dy", index=1)
        feat_z = col3.selectbox("Z-axis:", numeric_cols, key="3dz", index=2)

        fig3d = go.Figure()

        fig3d.add_trace(go.Scatter3d(
            x=normal[feat_x], y=normal[feat_y], z=normal[feat_z],
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Normal'
        ))

        fig3d.add_trace(go.Scatter3d(
            x=anomaly[feat_x], y=anomaly[feat_y], z=anomaly[feat_z],
            mode='markers',
            marker=dict(size=6, color='red', symbol='diamond'),
            name='Anomaly'
        ))

        fig3d.update_layout(height=600)
        st.plotly_chart(fig3d, use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("Anomaly Statistics")

    col1, col2 = st.columns(2)

    col1.write("**Normal Data Statistics**")
    col1.dataframe(normal.describe(), use_container_width=True)

    col2.write("**Anomaly Data Statistics**")
    col2.dataframe(anomaly.describe(), use_container_width=True)

# ================= TAB 4 =================
with tab4:
    st.subheader("Export Results")

    results_df = df.copy()
    results_df['anomaly_prediction'] = anomaly_labels
    results_df['anomaly_type'] = results_df['anomaly_prediction'].map(
        {1: 'Normal', -1: 'Anomaly'}
    )

    csv = results_df.to_csv(index=False)

    st.download_button(
        "📥 Download Full Results (CSV)",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )

    anomaly_only = results_df[results_df['anomaly_type'] == 'Anomaly']
    csv2 = anomaly_only.to_csv(index=False)

    st.download_button(
        "📥 Download Anomalies Only (CSV)",
        data=csv2,
        file_name="detected_anomalies.csv",
        mime="text/csv"
    )

# ================= FOOTER =================
st.divider()
st.caption("🔬 Traffic Flow Anomaly Detection System | Powered by Streamlit")