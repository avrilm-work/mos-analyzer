import streamlit as st
import pandas as pd
import numpy as np
from score_core import calc_score, plot_score_quintiles, score_sensitivity, plot_baseline_comparison

st.set_page_config(layout="wide", page_title="Composite Score Analyzer", page_icon="📈")

st.title("Composite Score Analyzer")

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {
        "Score Base": {
            "inputs": {
                'Renter Occupancy Share': 0.3,
                'Mover Rate 2019-25': 0.3,
                'HU Growth Rate 2020-25': 0.2,
                'Avg_Monthly Move Rate': 0.2
            },
            "df": None
        },
        "Score Core 7": {
            "inputs": {
                'Renter Occupancy Share': 0.2,
                'Mover Rate 2019-25': 0.2,
                'HU Growth Rate 2020-25': 0.2,
                'Avg_Monthly Move Rate': 0.15,
                '1-2 Person HH Share': 0.05,
                '15-34 Age Householder Share': 0.05,
                'Log Pop Density': 0.15
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
        'Search Volume': np.random.uniform(500, 20000, n),
        'Renter Occupancy Share': np.random.uniform(0.1, 0.9, n),
        'Mover Rate 2019-25': np.random.uniform(0.05, 0.5, n),
        'HU Growth Rate 2020-25': np.random.uniform(-0.05, 0.3, n),
        'Avg_Monthly Move Rate': np.random.uniform(0.01, 0.1, n),
        '1-2 Person HH Share': np.random.uniform(0.2, 0.8, n),
        '15-34 Age Householder Share': np.random.uniform(0.1, 0.6, n),
        'Log Pop Density': np.random.uniform(2.0, 10.0, n),
        'Mover Churn Rate': np.random.uniform(0.01, 0.20, n),
        'Customers': np.random.randint(10, 5000, n)
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
st.sidebar.header("3. Score Inputs & Weights")

halt = False

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in id_cols]

if len(numeric_cols) == 0:
    st.sidebar.warning("No numeric columns found for Score calculation. Please upload a valid CSV.")
    halt = True

default_metrics = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
selected_metrics = st.sidebar.multiselect("Select Metrics", numeric_cols, default=default_metrics, key="selected_metrics_list")

score_inputs = {}
if not halt:
    if len(selected_metrics) > 0:
        st.sidebar.write("Set Weights:")
        sum_w = 0.0
        for i, metric in enumerate(selected_metrics):
            default_w = round(1.0 / len(selected_metrics), 4)
            v = st.sidebar.number_input(f"{metric} Weight", min_value=0.0, max_value=1.0, value=default_w, step=0.05, key=f"w_{metric}")
            score_inputs[metric] = v
            sum_w += v
            
        st.sidebar.write(f"**Current Weight Sum:** {sum_w:.4f}")
        if round(sum_w, 4) != 1.0:
            st.sidebar.error("Weights must sum exactly to 1.0 to calculate Score.")
            halt = True
    else:
        st.sidebar.warning("Please select at least one metric to calculate Score.")
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
        score_df = calc_score(df, score_inputs, id_cols=id_cols)
    except Exception as e:
        st.error(f"Error computing Score: {e}")
        halt = True

if not halt:
    if st.session_state.get('save_trigger'):
        s_name = st.session_state.pop('save_trigger')
        st.session_state.scenarios[s_name] = {
            'inputs': score_inputs,
            'df': score_df.copy()
        }
        st.sidebar.success(f"Scenario '{s_name}' saved! Check the Compare tab.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualization", "📋 Data Table", "🔍 Sensitivity Analysis", "🔄 Compare Scenarios", "📈 Baseline Comparison"])

    with tab1:
        st.subheader("Score by Quintile")
        try:
            chart = plot_score_quintiles(score_df)
            st.altair_chart(chart, width="stretch")
        except Exception as e:
            st.error(f"Error plotting data: {e}")

    with tab2:
        st.subheader("Calculated Score Results")
        st.dataframe(score_df, width="stretch")
        csv = score_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Score Results as CSV",
            data=csv,
            file_name='score_results.csv',
            mime='text/csv',
        )

    with tab3:
        st.subheader("Sensitivity Analysis")
        if len(selected_metrics) > 1:
            with st.spinner("Running sensitivity analysis..."):
                sens_res = score_sensitivity(score_df, score_inputs)
                
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
                score_metrics = []
                rank_metrics = []
                example_df = None
                for name in selected_for_comp:
                    if st.session_state.scenarios[name].get('df') is not None:
                        df_s = st.session_state.scenarios[name]['df']
                        primary_id = 'Zip' if 'Zip' in df_s.columns else ('Market' if 'Market' in df_s.columns else df_s.columns[0])
                        score_s = df_s.set_index(primary_id)['Score'].rename(f"{name} (Score)")
                        rank_s = df_s.set_index(primary_id)['Score Rank'].rename(f"{name} (Rank)")
                        score_metrics.append(score_s)
                        rank_metrics.append(rank_s)
                        if example_df is None:
                            example_df = df_s
                
                if len(score_metrics) > 0:
                    st.markdown("#### Comparison by Score")
                    score_compare_df = pd.concat(score_metrics, axis=1).reset_index()
                    if example_df is not None and 'Zip' in example_df.columns and 'Market' in example_df.columns:
                        score_compare_df = score_compare_df.merge(example_df[['Zip', 'Market']], on='Zip', how='left')
                        cols = ['Zip', 'Market'] + [c for c in score_compare_df.columns if c not in ['Zip', 'Market']]
                        score_compare_df = score_compare_df[cols]
                    st.dataframe(score_compare_df, use_container_width=True)
                    
                    st.markdown("#### Comparison by Score Rank")
                    rank_df = pd.concat(rank_metrics, axis=1).reset_index()
                    if example_df is not None and 'Zip' in example_df.columns and 'Market' in example_df.columns:
                        rank_df = rank_df.merge(example_df[['Zip', 'Market']], on='Zip', how='left')
                        cols = ['Zip', 'Market'] + [c for c in rank_df.columns if c not in ['Zip', 'Market']]
                        rank_df = rank_df[cols]
                    st.dataframe(rank_df, use_container_width=True)
                else:
                    st.info("No calculated data available for the selected scenarios. (Save scenarios with loaded data first)")

    with tab5:
        st.subheader("Baseline vs Score Comparison")
        st.caption("Use this chart to visualize how the newly calculated Composite Score correlates dynamically against actual baseline metrics.")
        
        # Merge score_df with original df to get all potential baseline columns
        merge_on = [c for c in id_cols if c in score_df.columns and c in df.columns]
        if not merge_on:
            merge_on = [score_df.columns[0]] # Fallback
            
        cols_to_add = [c for c in df.columns if c not in score_df.columns]
        merged_df = score_df.merge(df[merge_on + cols_to_add], on=merge_on, how='left')
        
        # Backend modification: append 'None' column for uniform mapping
        merged_df['None'] = 1
        
        all_cols = ["None"] + [c for c in merged_df.columns if c != "None"]
        numeric_only = ["None"] + [c for c in merged_df.select_dtypes(include=[np.number]).columns if c != "None"]
        baseline_options = [c for c in numeric_only if c not in ['Score', 'Score Rank', 'None']]
        
        col_controls, col_plot = st.columns([1, 3])
        
        with col_controls:
            baseline_y = st.selectbox(
                "Y-Axis Baseline Metric", 
                baseline_options, 
                index=baseline_options.index('Mover Churn Rate') if 'Mover Churn Rate' in baseline_options else 0
            )
            
            color_var = st.selectbox(
                "Marker Color", 
                all_cols, 
                index=0
            )
            
            size_var = st.selectbox(
                "Marker Size", 
                numeric_only, 
                index=0
            )
            
            st.markdown("---")
            st.markdown("#### Custom Filtering")
            filter_var = st.selectbox(
                "Filter By", 
                numeric_only, 
                index=0
            )
            
            if filter_var != "None":
                min_raw = float(merged_df[filter_var].min())
                max_raw = float(merged_df[filter_var].max())
                
                if min_raw == max_raw:
                    custom_filter_min, custom_filter_max = min_raw, max_raw
                    st.info(f"'{filter_var}' has no variance.")
                else:
                    custom_filter_min = st.number_input(
                        f"Min {filter_var}", 
                        min_value=min_raw, max_value=max_raw, 
                        value=min_raw, step=(max_raw-min_raw)/100.0
                    )
                    custom_filter_max = st.number_input(
                        f"Max {filter_var}", 
                        min_value=min_raw, max_value=max_raw, 
                        value=max_raw, step=(max_raw-min_raw)/100.0
                    )
            else:
                custom_filter_min, custom_filter_max = None, None
        
        with col_plot:
            # Build filtered dataframe
            plot_df = merged_df.copy()
            
            if filter_var != "None" and custom_filter_min is not None:
                plot_df = plot_df[(plot_df[filter_var] >= custom_filter_min) & (plot_df[filter_var] <= custom_filter_max)]
                               
            if len(plot_df) > 1:
                try:
                    chart = plot_baseline_comparison(plot_df, x_col='Score', y_col=baseline_y, size_col=size_var, color_col=color_var)
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"Error plotting data: {e} \n\n {error_details}")
            elif len(plot_df) == 1:
                st.info("Only 1 data point passes the filters. Not enough to plot a distribution/trendline.")
            else:
                st.info("No data available with the current filters.")

else:
    st.info("Awaiting valid Score inputs from the sidebar.")