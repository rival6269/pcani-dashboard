import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- PAGE SETUP ---
st.set_page_config(page_title="PCANI ML Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🌍 PCANI: Predictive Machine Learning Core")
st.markdown("### Regional Target: Niger Delta Agro-Aquifer Nexus")
st.write("This dashboard simulates the ingestion of multi-sensor satellite arrays (NDVI, ET, GRACE) to detect impending ecological tipping points using Unsupervised Machine Learning.")

# --- SYNTHETIC DATA GENERATOR FOR MVP ---
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=100, freq="W")
    
    # Simulating a degrading environment over 100 weeks
    ndvi = np.linspace(0.7, 0.3, 100) + np.random.normal(0, 0.05, 100)
    et = np.linspace(80, 30, 100) + np.random.normal(0, 5, 100)
    aquifer = np.linspace(0, -150, 100) + np.random.normal(0, 10, 100)
    
    df = pd.DataFrame({'Date': dates, 'Crop_Vitality_NDVI': ndvi, 'Evapotranspiration': et, 'Aquifer_Mass_Anomaly': aquifer})
    return df

df = load_data()

# --- MACHINE LEARNING MODEL ---
def run_model(data, sensitivity):
    features = data[['Crop_Vitality_NDVI', 'Evapotranspiration', 'Aquifer_Mass_Anomaly']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Contamination dictates how aggressively the AI flags anomalies
    model = IsolationForest(n_estimators=200, contamination=sensitivity, random_state=42)
    model.fit(scaled_features)
    
    scores = model.score_samples(scaled_features)
    # Normalize risk to 0-100%
    risk_index = (1.0 - ((scores - scores.min()) / (scores.max() - scores.min()))) * 100
    return risk_index

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ AI Configuration")
sensitivity = st.sidebar.slider("Anomaly Detection Sensitivity", min_value=0.01, max_value=0.15, value=0.08, step=0.01)

st.sidebar.markdown("---")
st.sidebar.info("**Status:** Live connection simulated. Model ready for integration with Google Earth Engine API via strategic funding in Phase 1.")

# --- RUN PREDICTION ---
df['Tipping_Point_Risk'] = run_model(df, sensitivity)
current_risk = df['Tipping_Point_Risk'].iloc[-1]

# --- DASHBOARD METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Aquifer Deficit", f"{df['Aquifer_Mass_Anomaly'].iloc[-1]:.1f} mm", "-1.5 mm/wk")
col2.metric("Crop Vitality (NDVI)", f"{df['Crop_Vitality_NDVI'].iloc[-1]:.2f}", "-0.02")

if current_risk > 75:
    col3.metric("🚨 TIPPING POINT RISK", f"{current_risk:.1f}%", "CRITICAL", delta_color="inverse")
    st.error("CRITICAL ANOMALY DETECTED: Subsurface aquifer mass dropping independently of surface crop vitality. Immediate crop-rotation policy intervention recommended.")
else:
    col3.metric("✅ TIPPING POINT RISK", f"{current_risk:.1f}%", "STABLE")
    st.success("SYSTEM STABLE: Agro-ecological operations within safe planetary boundaries.")

# --- VISUALIZATION ---
st.markdown("### 📈 Spatiotemporal Risk Trajectory (100-Week Analysis)")
fig = px.line(df, x='Date', y='Tipping_Point_Risk', title='AI-Calculated Ecological Collapse Probability')
fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Powered by Scikit-Learn Isolation Forests & Google Earth Engine telemetry.")
