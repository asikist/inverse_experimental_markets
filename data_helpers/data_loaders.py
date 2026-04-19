from collections import namedtuple

import torch
import pandas as pd

from data_helpers.preprocessors.torch_preprocessors import TensorDataSet


TrainTestSplit = namedtuple("TrainTestSplit", "train test")
class DataLoader:
    def __init__(self, sample_splits_df: pd.DataFrame):
        self.sample_splits_df = sample_splits_df
        self.max_samples = sample_splits_df['sample_id'].max() + 1
    def get_sample_split_dfs(self, i: int):
        assert (i >= 0) and i <= self.sample_splits_df['sample_id'].max() + 1, 'Value of `i` is invalid.'
        current_sample_split_df = self.sample_splits_df.query('sample_id == ' + str(i))
        train_ids_df = current_sample_split_df[current_sample_split_df['dataset_type']=='train']
        test_ids_df = current_sample_split_df[current_sample_split_df['dataset_type']=='test']
        return train_ids_df, test_ids_df

class PandasDataLoader(DataLoader):
    def __init__(self, sample_splits_df: pd.DataFrame, time_aggregate_dataset: pd.DataFrame):
        super().__init__(sample_splits_df)
        self.sample_splits_df = sample_splits_df
        self.time_aggregate_dataset = time_aggregate_dataset.reset_index(drop=True).sort_values(['treatment','game','round','time'])

    def get_sample_split_dataset(self, i: int):
        train_ids_df, test_ids_df = self.get_sample_split_dfs(i)
        train_df = pd.merge(self.time_aggregate_dataset, train_ids_df[['treatment', 'game']], on=['treatment','game'])
        test_df = pd.merge(self.time_aggregate_dataset, test_ids_df[['treatment', 'game']], on=['treatment','game'])

        return train_df, test_df


class TorchDataLoader(DataLoader):
    def __init__(self, sample_splits_df: pd.DataFrame, tensor_dataset: TensorDataSet):
        super().__init__(sample_splits_df)
        self.sample_splits_df = sample_splits_df
        self.tensor_dataset = tensor_dataset


    def get_sample_split_tnsr_dataset(self, i: int):
        train_ids_df, test_ids_df = self.get_sample_split_dfs(i)

        train_treatment_tensor_index = train_ids_df['treatment_tnsr_idx']
        train_game_tensor_index = train_ids_df['game_tnsr_idx']

        test_treatment_tensor_index = test_ids_df['treatment_tnsr_idx']
        test_game_tensor_index = test_ids_df['game_tnsr_idx']

        train_tensor_set = self.tensor_dataset.subset('train_'+str(i),
                                                      treatment_idx=train_treatment_tensor_index.tolist(),
                                                      game_idx = train_game_tensor_index.tolist()
                                                      )
        test_tensor_set = self.tensor_dataset.subset('test_'+str(i),
                                                     treatment_idx=test_treatment_tensor_index.tolist(),
                                                     game_idx = test_game_tensor_index.tolist()
                                                     )

        return TrainTestSplit(train_tensor_set, test_tensor_set)

    def tensor_output_to_df(self, tnsr: torch.Tensor, ):
        """
        Takes a 5-D tensor (treatment, game, round, time, n_players) and converts it to dataframe.
        Drops nan values in dataframe.

        Parameters
        ----------
        tnsr

        Returns
        -------

        """
        #TODO: add code
        pass


