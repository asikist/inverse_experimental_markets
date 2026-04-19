"""Regenerate bar chart PDFs from pre-computed result .ft files.

This script reproduces the bar charts from the result_analysis notebooks
without requiring torch (which is an optional dependency).
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
PLOT_DIR_AE = os.path.join(DATA_ROOT, 'results', 'plots', 'ae')
PLOT_DIR_CE = os.path.join(DATA_ROOT, 'results', 'plots', 'ce')
PREPROCESSED = os.path.join(DATA_ROOT, 'preprocessed')

os.makedirs(PLOT_DIR_AE, exist_ok=True)
os.makedirs(PLOT_DIR_CE, exist_ok=True)

# Colors (same as notebooks)
emh_color = '#AF8D86'
cemh_color = '#F6E27F'
ob_rlm_color = '#473BF0'
gbt_color = '#8C271E'
color_map = [emh_color, cemh_color, ob_rlm_color, gbt_color]


def load_models(result_dir, ape_col):
    """Load all four model results and concatenate."""
    emh = pd.read_feather(os.path.join(result_dir, 'emh.ft'))
    cemh = pd.read_feather(os.path.join(result_dir, 'cemh.ft'))
    obrlm = pd.read_feather(os.path.join(result_dir, 'ob_rlm.ft'))
    gbt = pd.read_feather(os.path.join(result_dir, 'gbt.ft'))
    df = pd.concat([emh, cemh, obrlm, gbt])
    df['model'] = df['model'].map(lambda x: x.upper().replace('_', '-'))
    models = df['model'].unique()
    df['model'] = pd.Categorical(df['model'], models)
    return df, models


def make_bar_chart(sample_df, fs_pr_df, models, query, output_path):
    """Generate a grouped bar chart of Median APE by feedback setting."""
    a = sample_df.join(fs_pr_df).query(query).groupby(['feedback_setting'])[models].median()
    a = a.stack()
    a.index.set_names('model', level=1, inplace=True)
    a.name = 'ape'
    a = a.unstack('model')

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_bar(
            x=a.index.get_level_values('feedback_setting'),
            y=a[model],
            name=model,
            marker_color=color_map[i],
            text=['{:,.3f}'.format(x) for x in a[model].tolist()],
            textposition='outside',
            textangle=90,
        )

    fig.update_layout(barmode="group")
    fig.layout.template = 'simple_white'

    for trace in fig.data:
        trace.textposition = ['inside' if y > 0.7 else 'outside' for y in trace.y]

    fig.layout.xaxis.mirror = True
    fig.layout.yaxis.mirror = True
    fig.update_layout(legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.2,
        bgcolor='rgba(0,0,0,0)', orientation="h",
    ))
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
    fig.layout.width = 300
    fig.layout.height = 300
    fig.layout.yaxis.tickformat = ',.1f'
    fig.layout.xaxis.title.text = 'Feedback Setting'
    fig.layout.yaxis.title.text = 'Median APE'
    fig.layout.margin.t = 10
    fig.layout.margin.r = 10
    fig.layout.margin.b = 10
    fig.layout.margin.l = 10
    fig.layout.yaxis.range = [0, 1.1]
    fig.layout.font.family = 'Arial'

    fig.write_image(output_path, scale=10)
    print(f"  Written: {output_path}")


def generate_charts(result_dir, plot_dir, ape_col, label):
    """Generate nodeal and deal bar charts for one prediction target."""
    print(f"\n=== {label} ===")
    df, models = load_models(result_dir, ape_col)

    # Build sample orientation (pivot by model)
    sample_df = df.set_index([
        'sample_id', 'treatment', 'game', 'round', 'time',
        'n_unique_deals_round', 'model'
    ])[ape_col].unstack('model')

    # Load feedback/price rule mapping
    original_df = pd.read_feather(os.path.join(PREPROCESSED, 'original_df.ft'))
    fs_pr_df = original_df.groupby('treatment')[['price_rule', 'feedback_setting']].first()

    # Round 1, no deals
    make_bar_chart(
        sample_df, fs_pr_df, models,
        'round==1 and n_unique_deals_round==0',
        os.path.join(plot_dir, 'bar_round1_nodeal.pdf'),
    )

    # Round 1, with deals
    make_bar_chart(
        sample_df, fs_pr_df, models,
        'round==1 and n_unique_deals_round>0',
        os.path.join(plot_dir, 'bar_round1_deal.pdf'),
    )


if __name__ == '__main__':
    # AE charts
    generate_charts(
        os.path.join(DATA_ROOT, 'results', 'allocative_efficiency'),
        PLOT_DIR_AE,
        'ae_ape',
        'Allocative Efficiency',
    )

    # CEP charts
    generate_charts(
        os.path.join(DATA_ROOT, 'results', 'ce_price'),
        PLOT_DIR_CE,
        'ce_ape',
        'CE Price',
    )

    print("\nDone. All bar charts regenerated.")
