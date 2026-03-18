import streamlit as st
import pandas as pd
import numpy as np
from mos_core import calc_mos, plot_mos_quintiles, mos_sensitivity

st.set_page_config(layout="wide", page_title="MOS Analyzer", page_icon="📈")

st.title("Mover Opportunity Score (MOS) Analyzer")

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {
        "MOS Base": {
            "inputs": {
                'Renter Occupancy Share': 0.3,
                'Mover Rate 2019-25': 0.3,
                'HU Growth Rate 2020-25': 0.2,
                'Avg_Monthly Move Rate': 0.2
            },
            "df": None
        }
    }

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

halt = False

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in id_cols]

if len(numeric_cols) == 0:
    st.sidebar.warning("No numeric columns found for MOS calculation. Please upload a valid CSV.")
    halt = True

default_metrics = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
selected_metrics = st.sidebar.multiselect("Select Metrics", numeric_cols, default=default_metrics, key="selected_metrics_list")

mos_inputs = {}
if not halt:
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
            halt = True
    else:
        st.sidebar.warning("Please select at least one metric to calculate MOS.")
        halt = True

st.sidebar.markdown("---")
st.sidebar.header("📁 Manage Scenarios")

if 'scenarios' in st.session_state and len(st.session_state.scenarios) > 0:
    st.sidebar.subheader("Load Settings")
    saved_names = list(st.session_state.scenarios.keys())
    load_name = st.sidebar.selectbox("Load a saved scenario:", ["-- Select --"] + saved_names, key="load_scenario_box")
    
    def apply_scenario(name):
        st.session_state.pop('scenario_error', None)
        if name != "-- Select --":
            saved_inputs = st.session_state.scenarios[name]['inputs']
            missing = [k for k in saved_inputs.keys() if k not in numeric_cols]
            if missing:
                st.session_state['scenario_error'] = f"The active CSV doesn't have these columns: {', '.join(missing)}"
            else:
                st.session_state["selected_metrics_list"] = list(saved_inputs.keys())
                for metric, weight in saved_inputs.items():
                    st.session_state[f"w_{metric}"] = weight
                
    st.sidebar.button("Load Settings", on_click=apply_scenario, args=(st.session_state.get('load_scenario_box', '-- Select --'),))
    
    if 'scenario_error' in st.session_state:
        st.sidebar.error(st.session_state['scenario_error'])

st.sidebar.markdown("---")
st.sidebar.subheader("Save Current Setup")
if halt:
    st.sidebar.info("Resolve input warnings above to enable scenario saving.")
else:
    scenario_name = st.sidebar.text_input("Name a new scenario to save:")
    if st.sidebar.button("Save Current Results"):
        if scenario_name:
            st.session_state.save_trigger = scenario_name
        else:
            st.sidebar.error("Provide a scenario name first.")

if not halt:
    st.header("Results")
    try:
        mos_df = calc_mos(df, mos_inputs, id_cols=id_cols)
    except Exception as e:
        st.error(f"Error computing MOS: {e}")
        halt = True

if not halt:
    if st.session_state.get('save_trigger'):
        s_name = st.session_state.pop('save_trigger')
        st.session_state.scenarios[s_name] = {
            'inputs': mos_inputs,
            'df': mos_df.copy()
        }
        st.sidebar.success(f"Scenario '{s_name}' saved! Check the Compare tab.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📋 Data Table", "🔍 Sensitivity Analysis", "🔄 Compare Scenarios"])

    with tab1:
        st.subheader("MOS by Quintile")
        try:
            chart = plot_mos_quintiles(mos_df)
            st.altair_chart(chart, width="stretch")
        except Exception as e:
            st.error(f"Error plotting data: {e}")

    with tab2:
        st.subheader("Calculated MOS Results")
        st.dataframe(mos_df, width="stretch")
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
                st.dataframe(sens_res['loo'], width="stretch")
                
                st.markdown("#### Weight Perturbation (±5% and ±10%)")
                st.dataframe(sens_res['perturb'], width="stretch")
                
                st.markdown("#### Rank Stability (Top 25 ZIPs in Base)")
                st.caption("→ **Small rank range + low std dev** = stable, high-confidence ZIP.  \n→ **Large rank range** = ZIP ranking is sensitive to weight choices.")
                st.dataframe(sens_res['stability'], width="stretch")
        else:
            st.info("Sensitivity analysis requires at least 2 metrics.")

    with tab4:
        st.subheader("Compare Saved Scenarios")
        if len(st.session_state.scenarios) < 2:
            st.info("Save at least 2 scenarios using the sidebar to compare them here.")
        else:
            saved_names = list(st.session_state.scenarios.keys())
            selected_for_comp = st.multiselect("Select Scenarios to Compare", saved_names, default=saved_names)
            
            if len(selected_for_comp) > 0:
                comp_metrics = []
                example_df = None
                for name in selected_for_comp:
                    if st.session_state.scenarios[name].get('df') is not None:
                        df_s = st.session_state.scenarios[name]['df']
                        s = df_s.set_index('Zip')['MOS Rank'].rename(f"{name} (Rank)")
                        comp_metrics.append(s)
                        if example_df is None:
                            example_df = df_s
                
                if len(comp_metrics) > 0:
                    comp_df = pd.concat(comp_metrics, axis=1).reset_index()
                    
                    if example_df is not None and 'Market' in example_df.columns:
                        comp_df = comp_df.merge(example_df[['Zip', 'Market']], on='Zip', how='left')
                        cols = ['Zip', 'Market'] + [c for c in comp_df.columns if c not in ['Zip', 'Market']]
                        comp_df = comp_df[cols]
                        
                    st.dataframe(comp_df, use_container_width=True)
                else:
                    st.info("No calculated data available for the selected scenarios. (Save scenarios with loaded data first)")
else:
    st.info("Awaiting valid MOS inputs from the sidebar.")
