from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd


def calc_quantile_feats(row: pd.Series, quants=np.linspace(0, 1, 11).round(1)):
    quantile_values = []
    quantile_index = []
    for i in range(len(row.values)):
        dist = row.values[i]
        col_name = row.index[i]
        if isinstance(dist, dict):
            dist = list(dist.values())
        if len(dist) > 0:
            quantile_values += np.quantile(dist, quants).tolist()
        else:
            quantile_values += np.zeros_like(quants).tolist()
        assert np.isfinite(quantile_values).all()
        quantile_index += [col_name + '_quant_' + str(quant).replace('.', '') for quant in quants]

    return pd.Series(index=quantile_index, data=quantile_values)


def accumulate_dictionary(group, col_name, sorting_col='time'):
    group = group.sort_values(sorting_col)
    for i in range(group.shape[0]):
        row = group.iloc[i]
        if i > 0:
            prev_row = group.iloc[i - 1]
            prev_row_dict = prev_row[col_name][0]
            prev_dict = deepcopy(prev_row_dict)
            if isinstance(prev_dict, dict):
                prev_dict.update(row[col_name][0])
                group.iloc[i, group.columns.get_loc(col_name)]= [prev_dict]
    return group


# TODO: recheck and explain
def calculate_running_distribution(df, distribution_col: str = 'bid', prefix: str = 'running'):
    """ Calculates the cumulative sample distribution per round for a given column."""
    resulting_name = distribution_col
    id_col_dicts = df.groupby(['treatment', 'game', 'round', 'time', 'side'])[['id', distribution_col]].apply(
        lambda g: [dict(zip(g['id'], g[distribution_col]))])
    id_col_dicts.name = resulting_name
    accumulated_round_dictionaries = id_col_dicts.reset_index().groupby(['treatment', 'game', 'round', 'side'], group_keys=False).apply(
        lambda g: accumulate_dictionary(g, col_name=resulting_name))
    side_pivot_df = accumulated_round_dictionaries.pivot(index=['treatment', 'game', 'round', 'time'], columns='side')

    side_pivot_df.columns = [prefix + "_" + '_'.join(reversed(col)).strip().lower() for col in side_pivot_df.columns]
    side_pivot_df = side_pivot_df.reset_index().groupby(['treatment', 'game', 'round'], group_keys=False).apply(
        lambda r: r.sort_values('time').ffill()).reset_index(drop=True).set_index(['treatment', 'game',  'round', 'time'])
    side_pivot_df = side_pivot_df.map(lambda r: r if not isinstance(r, float) and not pd.isna(r) else [{}])
    side_pivot_df = side_pivot_df.map(lambda r: r[0])
    return side_pivot_df


def calc_histograms(row: pd.Series, all_values: Sequence, normalize: bool = False):
    hist_values = []
    hist_index = []
    for i in range(len(row.values)):
        dist = row.values[i]
        col_name = row.index[i]
        if isinstance(dist, dict):
            dist = list(dist.values())
        value_counts_series = pd.Series(dist).value_counts().reindex(all_values).fillna(0)
        if normalize:
            value_counts_series /= value_counts_series.sum()
        hist_values += value_counts_series.values.tolist()
        hist_index += [col_name + '_cat_' + str(cat) for cat in value_counts_series.index]

    return pd.Series(index=hist_index, data=hist_values)
