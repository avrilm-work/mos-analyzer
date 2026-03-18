import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

def calc_mos(df, mos_inputs: dict, id_cols=['Market', 'Zip']):
    mos_fields = list(mos_inputs.keys())
    weights    = list(mos_inputs.values())

    if round(sum(weights), 6) != 1.0:
        raise ValueError(f"Weights must sum to 1.0 — current sum: {sum(weights):.4f}")

    id_cols = [c for c in id_cols if c in df.columns]

    mos_df = df[id_cols + mos_fields].copy()

    null_check = mos_df[mos_fields].isnull().sum()
    if null_check.any():
        mos_df = mos_df.dropna(subset=mos_fields)

    scaler     = MinMaxScaler()
    scaled_cols = [f'{f} (Scaled)' for f in mos_fields]
    mos_df[scaled_cols] = scaler.fit_transform(mos_df[mos_fields])

    mos_df['MOS'] = sum(
        mos_df[col] * w for col, w in zip(scaled_cols, weights)
    )

    mos_df = mos_df.sort_values('MOS', ascending=False).reset_index(drop=True)
    mos_df['MOS Rank'] = mos_df['MOS'].rank(ascending=False).astype(int)

    return mos_df

def plot_mos_quintiles(mos_df, title='MOS by ZIP — By Quintile'):
    quintile_labels = ['Q1 — Lowest', 'Q2', 'Q3', 'Q4', 'Q5 — Highest']
    quintile_colors = {
        'Q1 — Lowest':  '#d7191c',
        'Q2':           '#fdae61',
        'Q3':           '#f4e84a',
        'Q4':           '#a6d96a',
        'Q5 — Highest': '#1a9641',
    }

    all_zips = mos_df.sort_values('MOS Rank').copy()
    all_zips['Zip'] = all_zips['Zip'].astype(str)
    all_zips['MOS Quintile'] = pd.qcut(
        all_zips['MOS'],
        q=5,
        labels=quintile_labels
    )

    charts = []
    for label in quintile_labels:
        subset = all_zips[all_zips['MOS Quintile'] == label].sort_values('MOS')
        color  = quintile_colors[label]

        y_enc = alt.Y('Zip:N', sort=alt.SortField('MOS', order='descending'), title=None)
        tooltip_enc = [
            alt.Tooltip('Zip:N', title='ZIP'),
            alt.Tooltip('MOS:Q', title='MOS', format='.3f'),
        ]
        if 'Market' in subset.columns:
            tooltip_enc.insert(1, alt.Tooltip('Market:N', title='Market'))

        base = alt.Chart(subset).encode(
            y=y_enc,
            tooltip=tooltip_enc
        )

        bars = base.mark_bar(color=color).encode(
            x=alt.X('MOS:Q', scale=alt.Scale(domain=[0, 1]), title='MOS'),
        )

        text_enc = alt.Text('Market:N') if 'Market' in subset.columns else alt.Text('Zip:N')
        labels = base.mark_text(align='left', dx=3, fontSize=9, color='#444').encode(
            x=alt.X('MOS:Q', scale=alt.Scale(domain=[0, 1])),
            text=text_enc
        )

        range_label = f'MOS: {subset["MOS"].min():.2f} – {subset["MOS"].max():.2f}'

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

def mos_sensitivity(mos_df, mos_inputs: dict, perturbations=[-0.10, -0.05, 0.05, 0.10], top_n=25):
    mos_fields   = list(mos_inputs.keys())
    base_weights = list(mos_inputs.values())
    scaled_cols  = [f'{f} (Scaled)' for f in mos_fields]
    base_rank    = mos_df['MOS Rank']

    def _calc_mos(weights):
        return sum(mos_df[col] * w for col, w in zip(scaled_cols, weights))

    loo_results = []
    for i, field in enumerate(mos_fields):
        remaining   = [j for j in range(len(mos_fields)) if j != i]
        new_w       = [0.0] * len(mos_fields)
        if len(remaining) > 0:
            redistribute = base_weights[i] / len(remaining)
            for j in remaining:
                new_w[j] = base_weights[j] + redistribute
        else:
            new_w[i] = 1.0

        rank_loo      = _calc_mos(new_w).rank(ascending=False).astype(int)
        rank_corr, _  = spearmanr(base_rank, rank_loo)
        loo_results.append({
            'Dropped Input':    field,
            'Rank Correlation': round(rank_corr, 4),
            'Avg Rank Shift':   round((rank_loo - base_rank).abs().mean(), 2),
            'Max Rank Shift':   int((rank_loo - base_rank).abs().max()),
        })

    perturb_results = []
    for i, field in enumerate(mos_fields):
        for delta in perturbations:
            new_w    = base_weights.copy()
            new_w[i] = round(base_weights[i] + delta, 2)

            if new_w[i] < 0 or new_w[i] > 1:
                continue

            remaining = [j for j in range(len(mos_fields)) if j != i]
            if len(remaining) > 0:
                for j in remaining:
                    new_w[j] = round(base_weights[j] - delta / len(remaining), 4)

            rank_p       = _calc_mos(new_w).rank(ascending=False).astype(int)
            rank_corr, _ = spearmanr(base_rank, rank_p)
            perturb_results.append({
                'Input':            field,
                'Weight Change':    f'{delta:+.0%}',
                'New Weight':       round(new_w[i], 2),
                'Rank Correlation': round(rank_corr, 4),
                'Avg Rank Shift':   round((rank_p - base_rank).abs().mean(), 2),
            })

    top_n_df   = mos_df.nsmallest(top_n, 'MOS Rank')[['Zip'] + (['Market'] if 'Market' in mos_df.columns else [])]
    top_n_zips = top_n_df['Zip'].values

    if len(mos_fields) > 1:
        all_scenarios = [base_weights] + [
            [base_weights[j] + (delta if j == i else -delta / (len(mos_fields) - 1))
             for j in range(len(mos_fields))]
            for i in range(len(mos_fields))
            for delta in perturbations
        ]
    else:
        all_scenarios = [base_weights]

    rank_matrix = pd.DataFrame(index=top_n_zips)
    for idx, w in enumerate(all_scenarios):
        w_sum = sum(w)
        if w_sum > 0:
            w = [val / w_sum for val in w]
        rank_s = _calc_mos(w).rank(ascending=False).astype(int)
        rank_matrix[f's{idx}'] = rank_s.values[:len(top_n_zips)]

    stability_data = {
        'Zip':        top_n_zips,
    }
    if 'Market' in top_n_df.columns:
        stability_data['Market'] = top_n_df['Market'].values
        
    stability_data.update({
        'Base Rank':  mos_df.nsmallest(top_n, 'MOS Rank')['MOS Rank'].values,
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
