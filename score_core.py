import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

def calc_score(df, score_inputs: dict, id_cols=['Market', 'Zip']):
    score_fields = list(score_inputs.keys())
    weights      = list(score_inputs.values())

    if round(sum(weights), 6) != 1.0:
        raise ValueError(f"Weights must sum to 1.0 — current sum: {sum(weights):.4f}")

    id_cols = [c for c in id_cols if c in df.columns]

    score_df = df[id_cols + score_fields].copy()

    null_check = score_df[score_fields].isnull().sum()
    if null_check.any():
        score_df = score_df.dropna(subset=score_fields)

    scaler      = MinMaxScaler()
    scaled_cols = [f'{f} (Scaled)' for f in score_fields]
    score_df[scaled_cols] = scaler.fit_transform(score_df[score_fields])

    score_df['Score'] = sum(
        score_df[col] * w for col, w in zip(scaled_cols, weights)
    )

    score_df = score_df.sort_values('Score', ascending=False).reset_index(drop=True)
    score_df['Score Rank'] = score_df['Score'].rank(ascending=False).astype(int)

    return score_df

def plot_score_quintiles(score_df, title='Score by ZIP — By Quintile'):
    quintile_labels = ['Q1 — Lowest', 'Q2', 'Q3', 'Q4', 'Q5 — Highest']
    quintile_colors = {
        'Q1 — Lowest':  '#d7191c',
        'Q2':           '#fdae61',
        'Q3':           '#f4e84a',
        'Q4':           '#a6d96a',
        'Q5 — Highest': '#1a9641',
    }

    primary_id = 'Zip' if 'Zip' in score_df.columns else 'Market'
    all_zips = score_df.sort_values('Score Rank').copy()
    all_zips[primary_id] = all_zips[primary_id].astype(str)
    all_zips['Score Quintile'] = pd.qcut(
        all_zips['Score'],
        q=5,
        labels=quintile_labels
    )

    charts = []
    for label in quintile_labels:
        subset = all_zips[all_zips['Score Quintile'] == label].sort_values('Score')
        color  = quintile_colors[label]

        y_enc = alt.Y(f'{primary_id}:N', sort=alt.SortField('Score', order='descending'), title=None)
        tooltip_enc = [
            alt.Tooltip(f'{primary_id}:N', title=primary_id),
            alt.Tooltip('Score:Q', title='Score', format='.3f'),
        ]
        if 'Market' in subset.columns and primary_id != 'Market':
            tooltip_enc.insert(1, alt.Tooltip('Market:N', title='Market'))

        base = alt.Chart(subset).encode(
            y=y_enc,
            tooltip=tooltip_enc
        )

        bars = base.mark_bar(color=color).encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1]), title='Score'),
        )

        text_enc = alt.Text('Market:N') if 'Market' in subset.columns else alt.Text(f'{primary_id}:N')
        labels = base.mark_text(align='left', dx=3, fontSize=9, color='#444').encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
            text=text_enc
        )

        range_label = f'Score: {subset["Score"].min():.2f} – {subset["Score"].max():.2f}'

        chart = (bars + labels).properties(
            title=alt.TitleParams(
                text=label,
                subtitle=range_label,
                subtitleFontSize=10,
                subtitleColor='#666'
            ),
            width=160,
            height=max(200, len(subset) * 16)
        )
        charts.append(chart)

    return (
        alt.hconcat(*charts)
        .properties(
            title=alt.TitleParams(text=title, fontSize=16, anchor='middle')
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=10, titleFontSize=12, grid=False)
    )

def plot_baseline_comparison(score_df, x_col='Score', y_col='Mover Churn Rate', size_col='None', color_col='None'):
    primary_id = 'Zip' if 'Zip' in score_df.columns else ('Market' if 'Market' in score_df.columns else score_df.columns[0])
    
    tooltip_enc = [
        alt.Tooltip(f'{primary_id}:N', title=primary_id),
        alt.Tooltip(f'{x_col}:Q', title=x_col, format='.3f'),
        alt.Tooltip(f'{y_col}:Q', title=y_col, format='.3f'),
        alt.Tooltip(f'{size_col}:Q', title=size_col),
        alt.Tooltip(f'{color_col}:N', title=color_col),
    ]

    base = alt.Chart(score_df).encode(
        x=alt.X(f'{x_col}:Q', title=x_col, scale=alt.Scale(zero=False)),
        y=alt.Y(f'{y_col}:Q', title=y_col, scale=alt.Scale(zero=False)),
        tooltip=tooltip_enc
    )

    color_type = 'N' if score_df[color_col].dtype == 'object' else 'Q'
    
    size_range = [40, 40] if size_col == 'None' else [60, 1500]

    bubbles = base.mark_circle(opacity=0.6).encode(
        size=alt.Size(f'{size_col}:Q', title=size_col, scale=alt.Scale(range=size_range)),
        color=alt.Color(f'{color_col}:{color_type}', title=color_col)
    )

    trend = base.transform_regression(f'{x_col}', f'{y_col}').mark_line(color='black', strokeDash=[4, 4], strokeWidth=2)

    chart = (bubbles + trend).properties(
        title=alt.TitleParams(
            text=f'{y_col} vs {x_col}',
            subtitle=f'Sized by {size_col}, Colored by {color_col}',
            fontSize=16,
            anchor='middle'
        ),
        height=550
    )

    return chart

def score_sensitivity(score_df, score_inputs: dict, perturbations=[-0.10, -0.05, 0.05, 0.10], top_n=25):
    score_fields = list(score_inputs.keys())
    base_weights = list(score_inputs.values())
    scaled_cols  = [f'{f} (Scaled)' for f in score_fields]
    base_rank    = score_df['Score Rank']

    def _calc_score(weights):
        return sum(score_df[col] * w for col, w in zip(scaled_cols, weights))

    loo_results = []
    for i, field in enumerate(score_fields):
        remaining   = [j for j in range(len(score_fields)) if j != i]
        new_w       = [0.0] * len(score_fields)
        if len(remaining) > 0:
            redistribute = base_weights[i] / len(remaining)
            for j in remaining:
                new_w[j] = base_weights[j] + redistribute
        else:
            new_w[i] = 1.0

        rank_loo      = _calc_score(new_w).rank(ascending=False).astype(int)
        rank_corr, _  = spearmanr(base_rank, rank_loo)
        loo_results.append({
            'Dropped Input':    field,
            'Rank Correlation': round(rank_corr, 4),
            'Avg Rank Shift':   round((rank_loo - base_rank).abs().mean(), 2),
            'Max Rank Shift':   int((rank_loo - base_rank).abs().max()),
        })

    perturb_results = []
    for i, field in enumerate(score_fields):
        for delta in perturbations:
            new_w    = base_weights.copy()
            new_w[i] = round(base_weights[i] + delta, 2)

            if new_w[i] < 0 or new_w[i] > 1:
                continue

            remaining = [j for j in range(len(score_fields)) if j != i]
            if len(remaining) > 0:
                for j in remaining:
                    new_w[j] = round(base_weights[j] - delta / len(remaining), 4)

            rank_p       = _calc_score(new_w).rank(ascending=False).astype(int)
            rank_corr, _ = spearmanr(base_rank, rank_p)
            perturb_results.append({
                'Input':            field,
                'Weight Change':    f'{delta:+.0%}',
                'New Weight':       round(new_w[i], 2),
                'Rank Correlation': round(rank_corr, 4),
                'Avg Rank Shift':   round((rank_p - base_rank).abs().mean(), 2),
            })

    cols_to_keep = [col for col in ['Zip', 'Market'] if col in score_df.columns]
    top_n_df   = score_df.nsmallest(top_n, 'Score Rank')[cols_to_keep]
    primary_id = 'Zip' if 'Zip' in score_df.columns else 'Market'
    top_n_ids  = top_n_df[primary_id].values

    if len(score_fields) > 1:
        all_scenarios = [base_weights] + [
            [base_weights[j] + (delta if j == i else -delta / (len(score_fields) - 1))
             for j in range(len(score_fields))]
            for i in range(len(score_fields))
            for delta in perturbations
        ]
    else:
        all_scenarios = [base_weights]

    rank_matrix = pd.DataFrame(index=top_n_ids)
    for idx, w in enumerate(all_scenarios):
        w_sum = sum(w)
        if w_sum > 0:
            w = [val / w_sum for val in w]
        rank_s = _calc_score(w).rank(ascending=False).astype(int)
        rank_matrix[f's{idx}'] = rank_s.values[:len(top_n_ids)]

    stability_data = {}
    for col in cols_to_keep:
        stability_data[col] = top_n_df[col].values
        
    stability_data.update({
        'Base Rank':  score_df.nsmallest(top_n, 'Score Rank')['Score Rank'].values,
        'Min Rank':   rank_matrix.min(axis=1).values,
        'Max Rank':   rank_matrix.max(axis=1).values,
        'Rank Range': rank_matrix.max(axis=1).values - rank_matrix.min(axis=1).values,
        'Std Dev':    rank_matrix.std(axis=1).round(2).values,
    })
        
    stability = pd.DataFrame(stability_data)

    return {
        'loo': pd.DataFrame(loo_results),
        'perturb': pd.DataFrame(perturb_results),
        'stability': stability
    }