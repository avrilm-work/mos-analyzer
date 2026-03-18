import streamlit as st
import pandas as pd
import numpy as np
from mos_core import calc_mos, plot_mos_quintiles, mos_sensitivity

st.set_page_config(layout="wide", page_title="MOS Analyzer", page_icon="📈")

st.title("Mover Opportunity Score (MOS) Analyzer")

@st.cache_data
def load_dummy_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'Market': np.random.choice(['NYC', 'LA', 'CHI', 'MIA', 'DAL'], n),
        'Zip': [f'{sz:05d}' for sz in np.random.randint(10000, 99999, n)],
        'Housing Starts': np.random.uniform(10, 500, n),
        'Job Growth %': np.random.uniform(-2, 8, n),
        'Migration Net': np.random.uniform(-1000, 5000, n),
        'Search Volume': np.random.uniform(500, 20000, n)
    })
    return df

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using built-in mock data for demonstration.")
    df = load_dummy_data()

st.sidebar.markdown("---")
st.sidebar.header("2. Identifier Columns")

default_id_cols = []
if 'Market' in df.columns: default_id_cols.append('Market')
if 'Zip' in df.columns: default_id_cols.append('Zip')
id_cols = st.sidebar.multiselect("Select Identifier Columns", df.columns.tolist(), default=default_id_cols)

st.sidebar.markdown("---")
st.sidebar.header("3. MOS Inputs & Weights")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in id_cols]

if len(numeric_cols) == 0:
    st.warning("No numeric columns found for MOS calculation. Please upload a valid CSV.")
    st.stop()

default_metrics = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
selected_metrics = st.sidebar.multiselect("Select Metrics", numeric_cols, default=default_metrics)

mos_inputs = {}
if len(selected_metrics) > 0:
    st.sidebar.write("Set Weights:")
    sum_w = 0.0
    for i, metric in enumerate(selected_metrics):
        default_w = round(1.0 / len(selected_metrics), 4)
        v = st.sidebar.number_input(f"{metric} Weight", min_value=0.0, max_value=1.0, value=default_w, step=0.05, key=f"w_{metric}")
        mos_inputs[metric] = v
        sum_w += v
        
    st.sidebar.write(f"**Current Weight Sum:** {sum_w:.4f}")
    if round(sum_w, 4) != 1.0:
        st.sidebar.error("Weights must sum exactly to 1.0 to calculate MOS.")
        st.stop()
else:
    st.warning("Please select at least one metric.")
    st.stop()

st.header("Results")
try:
    mos_df = calc_mos(df, mos_inputs, id_cols=id_cols)
except Exception as e:
    st.error(f"Error computing MOS: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Data Table", "🔍 Sensitivity Analysis"])

with tab1:
    st.subheader("MOS by Quintile")
    try:
        chart = plot_mos_quintiles(mos_df)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting data: {e}")

with tab2:
    st.subheader("Calculated MOS Results")
    st.dataframe(mos_df, use_container_width=True)
    csv = mos_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download MOS Results as CSV",
        data=csv,
        file_name='mos_results.csv',
        mime='text/csv',
    )

with tab3:
    st.subheader("Sensitivity Analysis")
    if len(selected_metrics) > 1:
        with st.spinner("Running sensitivity analysis..."):
            sens_res = mos_sensitivity(mos_df, mos_inputs)
            
            st.markdown("#### Leave-One-Out Sensitivity")
            st.caption("→ **Higher rank correlation** = rankings more stable without that input.  \n→ **Higher avg rank shift** = that input was driving more differentiation.")
            st.dataframe(sens_res['loo'], use_container_width=True)
            
            st.markdown("#### Weight Perturbation (±5% and ±10%)")
            st.dataframe(sens_res['perturb'], use_container_width=True)
            
            st.markdown("#### Rank Stability (Top 25 ZIPs in Base)")
            st.caption("→ **Small rank range + low std dev** = stable, high-confidence ZIP.  \n→ **Large rank range** = ZIP ranking is sensitive to weight choices.")
            st.dataframe(sens_res['stability'], use_container_width=True)
    else:
        st.info("Sensitivity analysis requires at least 2 metrics.")
