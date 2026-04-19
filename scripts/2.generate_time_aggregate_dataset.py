import logging

import pandas as pd

from data_helpers.preprocessors.pandas_preprocessors import TimeBasedDataSet

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    original_df = pd.read_feather('../data/preprocessed/original_df.ft')

    logging.info('Preprocessing original dataset to time-aggregate values.')
    tbds = TimeBasedDataSet(df=original_df)

    logging.info('Successful preprocessing, persisting file to: '
                 'project_root/data/preprocessed/time_aggregate_dataset.ft')
    tbds.dataset.to_feather('../data/preprocessed/time_aggregate_dataset.ft')

    estimated_max_players = 50

    logging.info('Preprocessing dataframes and persisting safetensors for torch models to: '
                 'project_root/data/preprocessed/torch/raw/')
    tbds.torch_data_preparation('../data/preprocessed/torch/raw/',
                                max_num_players=estimated_max_players
                                )
    logging.info('Successfuly perrsisted all preprocessed data.')
    logging.info(f'Dataset columns: {tbds.dataset.columns.tolist()}')