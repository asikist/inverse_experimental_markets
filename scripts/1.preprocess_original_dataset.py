from data_helpers.preprocessors.pandas_preprocessors import OriginalDataPreprocessor
import logging
import pandas as pd

if __name__ == '__main__':

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    odp = OriginalDataPreprocessor(original_file_pth='../data/Data.csv')
    odp.df.to_feather('../data/preprocessed/original_df.ft')
    logging.info(f'Preprocessed dataframe shape: {odp.df.shape}')

