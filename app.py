import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="PCANI Predictive Engine", page_icon="🌍", layout="wide")

st.markdown("<h1 style='color: #002B5B;'>🌍 PCANI: Predictive Machine Learning Core</h1>", unsafe_allow_html=True)
st.markdown("<b>Target Deployment Area:</b> Niger Delta Region (5.05°E - 8.68°E, 4.15°N - 7.17°N)", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Predictive Simulation")
scenario = st.sidebar.selectbox("Select Strategy (2026-2036)", ["Business as Usual (Unmitigated Extraction)", "PCANI Managed Stabilization Model"])

# --- DATA GENERATOR ---
@st.cache_data
def generate_world_class_data(selected_scenario):
    hist_dates = pd.date_range(start="2022-01-01", end="2026-06-01", freq="W")
    future_dates = pd.date_range(start="2026-06-08", end="2036-06-01", freq="W")
    
    h_weeks = len(hist_dates)
    f_weeks = len(future_dates)
    
    ndvi = np.concatenate([np.linspace(0.72, 0.34, h_weeks), np.linspace(0.34, 0.08, f_weeks) if "Business" in selected_scenario else np.linspace(0.34, 0.65, f_weeks)])
    aquifer = np.concatenate([np.linspace(0, -138.5, h_weeks), np.linspace(-138.5, -340, f_weeks) if "Business" in selected_scenario else np.linspace(-138.5, -75, f_weeks)])
    risk = np.concatenate([np.linspace(12, 70.8, h_weeks), np.linspace(70.8, 98.5, f_weeks) if "Business" in selected_scenario else np.linspace(70.8, 24.5, f_weeks)])
    
    df = pd.DataFrame({'Date': hist_dates.append(future_dates), 'Risk': np.clip(risk, 0, 100)})
    df['Data_Type'] = ['Historical'] * h_weeks + ['Forecast'] * f_weeks
    df['Upper'] = df['Risk'] + 10; df['Lower'] = df['Risk'] - 10
    return df, h_weeks

df, hist_cutoff = generate_world_class_data(scenario)

# --- METRICS & ALERTS ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Aquifer Delta", "-138.0 mm", "-1.8 mm/wk")
col2.metric("Surface Canopy (NDVI)", "0.35", "-0.03")
col3.metric("2036 Risk Outlook", f"{df['Risk'].iloc[-1]:.1f}%", "Critical" if "Business" in scenario else "Stable")

if "Business" in scenario:
    st.error("🚨 CRITICAL ALERT: Systemic collapse imminent.")
else:
    st.success("✅ STABILIZATION CHANNELS ACTIVE: Recovery trajectory confirmed.")

# --- CHART ---
fig = go.Figure()
hist_df = df[df['Data_Type'] == 'Historical']
fore_df = df[df['Data_Type'] == 'Forecast']

# Shaded Area
fig.add_trace(go.Scatter(x=pd.concat([fore_df['Date'], fore_df['Date'][::-1]]), y=pd.concat([fore_df['Upper'], fore_df['Lower'][::-1]]), fill='toself', fillcolor='rgba(200, 200, 200, 0.2)', name="Confidence Interval"))
# Lines
fig.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="Observed", line=dict(color='#002B5B', width=3)))
fig.add_trace(go.Scatter(x=fore_df['Date'], y=fore_df['Risk'], name="Forecast", line=dict(color='red' if "Business" in scenario else 'green', width=3, dash='dash')))

# FIXED LINE (Using x parameter instead of x0)
fig.add_vline(x=pd.to_datetime("2026-06-01"), line_dash="solid", line_color="black", line_width=2, annotation_text="Present Day")

fig.update_layout(title="Spatiotemporal Risk Projection (2022-2036)", yaxis_title="Collapse Probability (%)", height=500)
st.plotly_chart(fig, use_container_width=True)

if st.button("Download Data"):
    st.download_button("Download CSV", df.to_csv(), "data.csv", "text/csv")
