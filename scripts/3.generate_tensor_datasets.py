import os

import numpy as np
import torch
import pandas as pd
from data_helpers.comp_equilibrium import dynamic_treatments
from data_helpers.preprocessors.torch_preprocessors import TensorPreprocessor, TensorDataSet

if __name__ == '__main__':
    dataset_pth = '../data/preprocessed/time_aggregate_dataset.ft'
    dataset = pd.read_feather(dataset_pth)
    dataset = dataset[dataset['round'] <= 5]
    dataset = dataset[~dataset['treatment'].isin(dynamic_treatments)].reset_index(drop=True)

    data_pth = '../data/preprocessed/original_df.ft'
    original_data = pd.read_feather(data_pth)
    original_data = original_data.drop_duplicates().reset_index(drop=True)

    tdl = TensorPreprocessor(
        deal_prices_file='../data/preprocessed/torch/raw/deal_prices.pt',
        bids_file='../data/preprocessed/torch/raw/running_buyer_bid.pt',
        asks_file='../data/preprocessed/torch/raw/running_seller_ask.pt',
        buyer_valuations_file='../data/preprocessed/torch/raw/running_buyer_valuation.pt',
        seller_valuations_file='../data/preprocessed/torch/raw/running_seller_valuation.pt',
        user_index_file='../data/preprocessed/torch/raw/user_index.ft'
    )

    tdl.validate_ce_metrics(dataset)
    tdl.validate_ae_metrics(dataset)

    train_test_ratio = 0.5
    n_samples = 50

    split_dataframe = tdl.get_train_test_splits(ratio=train_test_ratio, n_samples=n_samples)
    split_dataframe.to_feather('../data/preprocessed/train_test_split.ft')
    tds = tdl.generate_tensor_dataset(max_rounds=5, max_time=120, name='original_dataset')

    os.makedirs('../data/preprocessed/torch/', exist_ok=True)
    tds.serialize('../data/preprocessed/torch/')
    print('done')

    tds_new = TensorDataSet.deserialize('../data/preprocessed/torch/original_dataset')
    tds_vars = vars(tds)
    tds_new_vars = vars(tds_new)

    assert set(tds_vars.keys()) == set(tds_new_vars.keys())
    for k, v in tds_vars.items():
        if isinstance(v, int):
            assert v == tds_new_vars[k]
        elif isinstance(v, np.ndarray):
            assert np.all(v == tds_new_vars[k])
        elif isinstance(v, torch.Tensor):
            assert torch.allclose(v, tds_new_vars[k])
    print('checks done')
