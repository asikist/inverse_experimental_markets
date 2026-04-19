import os
from ast import literal_eval
from pathlib import Path
from typing import Union, Collection

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from data_helpers.comp_equilibrium import dynamic_treatments
from data_helpers.treatment import treatment_feats
from models.sim.operators import calculate_ce_metrics

import itertools

def padded_stack(tensors, dim=0, stack_to_start: Union[bool, Collection[bool]] = True, default_element=0.0):
    # stack to start refers to the direction axis. E.g. for a matix, if first stack element is true, then null rows will be append to the end of rows of each sequence.
    max_shape = None

    device = tensors[0].device
    dtype = tensors[0].dtype

    for tensor in tensors:
        if max_shape is None:
            max_shape = torch.tensor(tensor.shape)
        if len(max_shape) != len(tensor.shape):
            raise ValueError('All tensors must have the same len of shapes or number of dimensions!')
        max_shape = torch.stack([max_shape, torch.tensor(tensor.shape)], dim=0).max(dim=0).values

    if isinstance(stack_to_start, bool):
        stack_directions = [stack_to_start] * len(max_shape)
    else:
        stack_directions = stack_to_start
        assert len(stack_directions) == len(
            max_shape), "Invalid list of stack_to_start length, should be equal to tensor directions."

    stack_tnsr_dims = max_shape.cpu().long().numpy().tolist()
    stack_tnsr_dims.insert(dim, len(tensors))
    stacked_tensor = torch.zeros(stack_tnsr_dims, device=device, dtype=dtype)

    for i, tensor in enumerate(tensors):
        shape_diff = max_shape - torch.tensor(tensor.shape)
        shape_list = shape_diff.cpu().long().numpy().tolist()
        slice_list = []
        for j, s in enumerate(shape_list):
            current_slice = slice(None)
            if s > 0:
                if stack_directions[j]:
                    current_slice = slice(0, -s)
                else:
                    current_slice = slice(s, None)
            slice_list.append(current_slice)
        slice_list.insert(dim, i)

        stacked_tensor[slice_list] = tensor
    return stacked_tensor


def prepare_tensor_series(pth: str) -> pd.Series:
    """
    Loads a pandas series with tensor elements from path.

    Parameters
    ----------
    pth: str
        The path to read from.

    Returns
    -------
    tensor: pd.Series
        The series with tensors.

    """
    series = pd.Series(torch.load(pth))
    series_name = Path(pth).stem
    series.name = series_name
    series.index.names = ['treatment', 'game', 'round', 'time']
    series = series.reset_index().sort_values(['treatment', 'game', 'round', 'time']).set_index(
        ['treatment', 'game', 'round', 'time']).iloc[:, 0]
    return series


def forward_fill(t: torch.Tensor, dim: int) -> torch.Tensor:
    # nice code found here:
    # https://stackoverflow.com/questions/77202743/how-to-efficiently-implement-forward-fill-in-pytorch
    t_dim = t.shape[dim]
    # Generate indices range
    rng = torch.arange(t_dim).to(device=t.device)
    singular_shape = torch.ones(len(t.shape), device=t.device).int()
    singular_shape[dim] = t_dim
    inverse_singular_shape = torch.tensor(t.shape, device=t.device)
    inverse_singular_shape[dim] = 1
    rng_nd = rng.view(*singular_shape.cpu().numpy().tolist()).repeat(*inverse_singular_shape.cpu().numpy().tolist())
    rng_nd[t == 0] = 0

    # Forward fill of indices range so all zero elements will be replaced with previous non-zero index.
    idx = rng_nd.cummax(dim).values
    fft = t.gather(dim=dim, index=idx)

    return fft


class TensorDataSet:
    def __init__(self,
                 name: str,
                 treatments: np.ndarray,
                 games: np.ndarray,
                 rounds: np.ndarray,
                 times: np.ndarray,
                 feedback_settings: np.ndarray,
                 price_rules: np.ndarray,
                 treatment_dim: int,
                 game_dim: int,
                 round_dim: int,
                 time_dim: int,
                 player_dim: int,
                 deal_prices_tnsr: torch.Tensor,
                 bids_tnsr: torch.Tensor,
                 asks_tnsr: torch.Tensor,
                 buyer_vals_tnsr: torch.Tensor,
                 seller_vals_tnsr: torch.Tensor,
                 times_tnsr: torch.Tensor,
                 feedback_setting_tnsr: torch.Tensor,
                 price_rule_tnsr: torch.Tensor,
                 cumulative_max_bids_time_tnsr: torch.Tensor,
                 cumulative_max_bids_time_round_tnsr: torch.Tensor,
                 cumulative_min_asks_time_tnsr: torch.Tensor,
                 cumulative_min_asks_time_round_tnsr: torch.Tensor,
                 time_bid_diff_tnsr: torch.Tensor,
                 time_ask_diff_tnsr: torch.Tensor,
                 round_time_bid_diff_tnsr: torch.Tensor,
                 round_time_ask_diff_tnsr: torch.Tensor,
                 cum_min_deal_prices_tnsr: torch.Tensor,
                 cum_max_deal_prices_tnsr: torch.Tensor,
                 eq_quantity_tnsr: torch.Tensor,
                 got_tnsr: torch.Tensor,
                 ce_low_tnsr: torch.Tensor,
                 ce_high_tnsr: torch.Tensor,
                 ce_tnsr: torch.Tensor,
                 matched_valuations_mask: torch.Tensor,
                 sorted_buyer_idx: torch.Tensor,
                 sorted_seller_idx: torch.Tensor,
                 sorted_buyer_vals_tnsr: torch.Tensor,
                 sorted_seller_asks_tnsr: torch.Tensor,
                 allocative_efficiency_tnsr: torch.Tensor,
                 user_index_df: pd.DataFrame
                 ):
        self.name = name
        self.treatments = treatments
        self.games = games
        self.rounds = rounds
        self.times = times
        self.feedback_settings = feedback_settings
        self.price_rules = price_rules
        self.treatment_dim = treatment_dim
        self.game_dim = game_dim
        self.round_dim = round_dim
        self.time_dim = time_dim
        self.player_dim = player_dim
        self.deal_prices_tnsr = deal_prices_tnsr
        self.bids_tnsr = bids_tnsr
        self.asks_tnsr = asks_tnsr
        self.buyer_vals_tnsr = buyer_vals_tnsr
        self.seller_vals_tnsr = seller_vals_tnsr
        self.times_tnsr = times_tnsr
        self.feedback_setting_tnsr = feedback_setting_tnsr
        self.price_rule_tnsr = price_rule_tnsr
        self.cumulative_max_bids_time_tnsr = cumulative_max_bids_time_tnsr
        self.cumulative_max_bids_time_round_tnsr = cumulative_max_bids_time_round_tnsr
        self.cumulative_min_asks_time_tnsr = cumulative_min_asks_time_tnsr
        self.cumulative_min_asks_time_round_tnsr = cumulative_min_asks_time_round_tnsr
        self.time_bid_diff_tnsr = time_bid_diff_tnsr
        self.time_ask_diff_tnsr = time_ask_diff_tnsr
        self.round_time_bid_diff_tnsr = round_time_bid_diff_tnsr
        self.round_time_ask_diff_tnsr = round_time_ask_diff_tnsr
        self.cum_min_deal_prices_tnsr = cum_min_deal_prices_tnsr
        self.cum_max_deal_prices_tnsr = cum_max_deal_prices_tnsr
        self.eq_quantity_tnsr = eq_quantity_tnsr
        self.got_tnsr = got_tnsr
        self.ce_low_tnsr = ce_low_tnsr
        self.ce_high_tnsr = ce_high_tnsr
        self.ce_tnsr = ce_tnsr
        self.matched_valuations_mask = matched_valuations_mask
        self.sorted_buyer_idx = sorted_buyer_idx
        self.sorted_seller_idx = sorted_seller_idx
        self.sorted_buyer_vals_tnsr = sorted_buyer_vals_tnsr
        self.sorted_seller_asks_tnsr = sorted_seller_asks_tnsr
        self.allocative_efficiency_tnsr = allocative_efficiency_tnsr
        self.user_index_df = user_index_df

    def serialize(self, pth: str):
        parent_path = os.path.join(pth, self.name)
        os.makedirs(parent_path, exist_ok=True)

        self.__to_safetensors(os.path.join(parent_path, "tensor_dataset.safetensors"))
        self.user_index_df.to_feather(os.path.join(parent_path, "user_index_df.ft"))

    @staticmethod
    def deserialize(parent_path: str, device: str = 'cpu'):
        object_params = TensorDataSet.__from_safetensors(os.path.join(parent_path, "tensor_dataset.safetensors"),
                                                         device=device)
        user_index_df = pd.read_feather(os.path.join(parent_path, "user_index_df.ft"))
        object_params['user_index_df'] = user_index_df
        return TensorDataSet(**object_params)

    def __to_safetensors(self, pth: str):
        tensor_dict = dict()
        metadata_dict = dict()
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v.contiguous()
            elif not (isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)):
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                metadata_dict[k] = str(v) if not isinstance(v, str) else v
        save_file(tensors=tensor_dict, metadata=metadata_dict, filename=pth)

    @staticmethod
    def __from_safetensors(file_pth: str, device='cpu'):
        object_params = {}
        with  safe_open(file_pth, framework="pt", device=device) as f:
            for k in f.keys():
                object_params[k] = f.get_tensor(k)
            for k, v in f.metadata().items():
                if k == 'name':
                    parsed_v = v
                else:
                    parsed_v = literal_eval(v)
                if isinstance(parsed_v, list):
                    parsed_v = np.array(parsed_v)
                object_params[k] = parsed_v
        return object_params

    def subset(self, name: str, treatment_idx: list, game_idx: list):
        tdsss = TensorDataSetSubsetSelector(self, treatment_idx, game_idx)

        new_tensors_params = dict()
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                new_tensors_params[k] = tdsss.subset_tnsr(v)

        new_user_index_df = self.user_index_df[self.user_index_df['treatment'].isin(tdsss.subset_treatments) &
                                               self.user_index_df['game'].isin(tdsss.subset_games)
                                               ]
        params_dict = dict(
            name=name,
            treatments=tdsss.subset_treatments,
            games=tdsss.subset_games,
            rounds=self.rounds,
            times=self.times,
            feedback_settings=self.feedback_settings,
            price_rules=self.price_rules,
            treatment_dim=self.treatment_dim,
            game_dim=self.game_dim,
            round_dim=self.round_dim,
            time_dim=self.time_dim,
            player_dim=self.player_dim,
            user_index_df=new_user_index_df

        )
        params_dict.update(new_tensors_params)
        tds = TensorDataSet(**params_dict)
        return tds

    def reconstruct_player_dataframe(self, tnsr):
        idx = itertools.product(self.treatments, self.games, self.rounds, self.times,
                                list(range(self.deal_prices_tnsr.shape[-1]))
                                )
        idx_df = pd.DataFrame(data=idx, columns=['treatment', 'game', 'round', 'time', 'player_tnsr_id'])
        idx_df['value'] = tnsr.view(-1, 1).detach().cpu().numpy()
        return idx_df


class TensorDataSetSubsetSelector:
    def __init__(self, tnsr_dataset: TensorDataSet, treatment_idx: list, game_idx: list):
        self.tnsr_dataset = tnsr_dataset
        self.treatment_idx = treatment_idx
        self.game_idx = game_idx
        # TODO: kind of redudant to convert arrays to lists and back, check if future when having time.
        self.unique_treatments_idx = pd.unique(
            np.array(treatment_idx))  # pandas brings unique elements in order of appearance.
        self.unique_games_idx = pd.unique(np.array(game_idx))  # pandas brings unique elements in order of appearance.
        self.subset_treatments = self.tnsr_dataset.treatments[self.unique_treatments_idx]
        self.subset_games = self.tnsr_dataset.games[self.unique_games_idx]
        self.max_treatments = len(self.subset_treatments)
        self.max_games = len(self.subset_games)

        self.new_treatment_idx = list(map(lambda x: np.where(self.unique_treatments_idx == x)[0].item(), treatment_idx))
        self.new_game_idx = list(map(lambda x: np.where(self.unique_games_idx == x)[0].item(), game_idx))

    def subset_tnsr(self, tnsr):
        # Create a new tensor to put the data inside.
        new_shape = list(tnsr.shape)
        new_shape[self.tnsr_dataset.treatment_dim] = self.max_treatments
        new_shape[self.tnsr_dataset.game_dim] = self.max_games
        new_tnsr = torch.zeros(new_shape, device=tnsr.device, dtype=tnsr.dtype)

        # create an advanced index to subset the orginial tensor.
        adv_idx = [slice(None)] * len(tnsr.shape)
        adv_idx[self.tnsr_dataset.treatment_dim] = self.treatment_idx
        adv_idx[self.tnsr_dataset.game_dim] = self.game_idx

        new_adv_idx = [slice(None)] * len(tnsr.shape)
        new_adv_idx[self.tnsr_dataset.treatment_dim] = self.new_treatment_idx
        new_adv_idx[self.tnsr_dataset.game_dim] = self.new_game_idx

        # assign the sub-selected values to the new tensor.
        new_tnsr[new_adv_idx] = tnsr[adv_idx]
        return new_tnsr


class TensorPreprocessor:

    def __load_tensor_dataframe(self,
                                deal_prices_file,
                                bids_file,
                                asks_file,
                                buyer_valuations_file,
                                seller_valuations_file):
        # load from disk
        deal_prices_tnsr_series = prepare_tensor_series(deal_prices_file)
        buyer_bid_tsnr_series = prepare_tensor_series(bids_file)
        seller_ask_tnsr_series = prepare_tensor_series(asks_file)
        buyer_valuation_tnsr_series = prepare_tensor_series(buyer_valuations_file)
        seller_valuation_tnsr_series = prepare_tensor_series(seller_valuations_file)

        # combine to dataframe
        df = pd.concat([deal_prices_tnsr_series, buyer_bid_tsnr_series, seller_ask_tnsr_series,
                        buyer_valuation_tnsr_series, seller_valuation_tnsr_series], axis=1)
        df.rename(columns={'deal_prices': 'deal_price', 'running_seller_bid': 'running_seller_ask'})
        df = df[~df.index.get_level_values('treatment').isin(dynamic_treatments)]
        df.columns = list(map(lambda x: x.replace('running_', ''), df.columns))
        return df

    def __get_metadata_tensors(self):
        # cumulative max bids over time and rounds
        cumulative_max_bids_time = self.bids_tensor.cummax(dim=self.time_dim).values
        bid_group_dim = list(self.bids_tensor.shape[self.treatment_dim:self.round_dim]) + [1, -1] + [
            self.bids_tensor.shape[self.player_dim]]
        cumulative_max_bids_time_round = cumulative_max_bids_time.view(*bid_group_dim).cummax(
            dim=self.time_dim).values.view(self.bids_tensor.shape)

        self.cumulative_max_bids_time = cumulative_max_bids_time
        self.cumulative_max_bids_time_round = cumulative_max_bids_time_round

        # cumulative min asks over time and rounds
        cumulative_min_asks_time = self.asks_tensor.clone()
        cumulative_min_asks_time[cumulative_min_asks_time == 0] = torch.inf
        cumulative_min_asks_time = cumulative_min_asks_time.cummin(dim=self.time_dim).values
        ask_group_dim = list(self.asks_tensor.shape[self.treatment_dim:self.round_dim]) + [1, -1] + [
            self.asks_tensor.shape[self.player_dim]]
        cumulative_min_asks_time_round = cumulative_min_asks_time.view(*ask_group_dim).cummin(
            dim=self.time_dim).values.view(self.asks_tensor.shape)
        cumulative_min_asks_time[torch.isinf(cumulative_min_asks_time)] = 0
        cumulative_min_asks_time_round[torch.isinf(cumulative_min_asks_time_round)] = 0

        self.cumulative_min_asks_time = cumulative_min_asks_time
        self.cumulative_min_asks_time_round = cumulative_min_asks_time_round

        # bid and ask changes (differences) over time and rounds
        time_bid_diff = forward_fill(self.bids_tensor, dim=self.time_dim).diff(dim=self.time_dim,
                                                                               prepend=torch.zeros_like(
                                                                                   self.bids_tensor[:, :, :, 0:1,
                                                                                   :])) * (self.bids_tensor != 0)
        time_ask_diff = forward_fill(self.asks_tensor, dim=self.time_dim).diff(dim=self.time_dim,
                                                                               prepend=torch.zeros_like(
                                                                                   self.asks_tensor[:, :, :, 0:1,
                                                                                   :])) * (self.asks_tensor != 0)
        self.time_bid_diff = time_bid_diff
        self.time_ask_diff = time_ask_diff

        round_time_bid_diff = time_bid_diff.clone()
        round_time_bid_diff[:, :, 1:, 0, :] = time_bid_diff[:, :, :-1, -1, :]
        self.round_time_bid_diff = round_time_bid_diff

        round_time_ask_diff = time_ask_diff.clone()
        round_time_ask_diff[:, :, 1:, 0, :] = time_ask_diff[:, :, :-1, -1, :]
        self.round_time_ask_diff = round_time_ask_diff

        # cumulative min and max deal prices for sellers and buyers
        cum_min_deal_prices = self.deal_prices_tensor.clone()
        cum_min_deal_prices[cum_min_deal_prices == 0] = torch.inf
        cum_min_deal_prices = cum_min_deal_prices.cummin(dim=self.time_dim).values
        deal_price_group_dim = list(self.bids_tensor.shape[self.treatment_dim:self.round_dim]) + [1, -1] + [
            self.bids_tensor.shape[self.player_dim]]
        cum_min_deal_prices = cum_min_deal_prices.view(*deal_price_group_dim).cummin(dim=self.time_dim).values.view(
            self.deal_prices_tensor.shape)
        cum_min_deal_prices[torch.isinf(cum_min_deal_prices)] = 0
        self.cum_min_deal_prices = cum_min_deal_prices

        cum_max_deal_prices = self.deal_prices_tensor.clone().cummax(dim=self.time_dim).values
        self.cum_max_deal_prices = cum_max_deal_prices.view(deal_price_group_dim).cummax(dim=self.time_dim).values.view(
            self.deal_prices_tensor.shape)

    def __validate_tensor_assignments(self):
        # validate tensors against df
        for ind in self.df.index:
            treat_j = np.where(self.treatments == ind[0])[0].item()
            game_j = np.where(self.games == ind[1])[0].item()
            round_j = np.where(self.rounds == ind[2])[0].item()
            time_j = np.where(self.times == ind[3])[0].item()
            assert (self.deal_prices_tensor[treat_j, game_j, round_j, time_j, :] == self.df.loc[
                ind, 'deal_prices']).all()
            assert (self.bids_tensor[treat_j, game_j, round_j, time_j, :] == self.df.loc[ind, 'buyer_bid']).all()
            assert (self.asks_tensor[treat_j, game_j, round_j, time_j, :] == self.df.loc[ind, 'seller_ask']).all()
            assert (self.buyer_vals[treat_j, game_j, round_j, time_j, :] == self.df.loc[ind, 'buyer_valuation']).all()
            assert (self.seller_vals[treat_j, game_j, round_j, time_j, :] == self.df.loc[ind, 'seller_valuation']).all()
            assert self.times_tnsr[treat_j, game_j, round_j, time_j, 0].item() == ind[3]

        assert 2 * ((self.deal_prices_tensor != 0) & (self.seller_vals != 0)).sum() == (
                self.deal_prices_tensor != 0).sum()
        assert 2 * ((self.deal_prices_tensor != 0) & (self.buyer_vals != 0)).sum() == (
                self.deal_prices_tensor != 0).sum()

    def __init__(self,
                 deal_prices_file,
                 bids_file,
                 asks_file,
                 buyer_valuations_file,
                 seller_valuations_file,
                 user_index_file: pd.DataFrame
                 ):

        self.user_index_df = pd.read_feather(user_index_file)
        self.df = self.__load_tensor_dataframe(deal_prices_file,
                                               bids_file,
                                               asks_file,
                                               buyer_valuations_file,
                                               seller_valuations_file)

        # prepare tensors
        self.treatments = self.df.index.get_level_values('treatment').unique().values
        max_treatments = len(self.treatments)
        self.games = self.df.index.get_level_values('game').unique().values
        max_games = len(self.games)
        self.rounds = np.array(sorted(self.df.index.get_level_values('round').unique().tolist()))
        max_rounds = len(self.rounds)
        self.times = np.array(sorted(self.df.index.get_level_values('time').unique().tolist()))
        max_times = len(self.times)

        self.treatment_dim = 0
        self.game_dim = 1
        self.round_dim = 2
        self.time_dim = 3
        self.player_dim = 4

        # check if timesteps and rounds are strictly increasing.
        assert (np.diff(self.times) > 0).all()
        assert (np.diff(self.rounds) > 0).all()

        # create tensor indices
        ti = self.df.index.get_level_values('treatment').map(
            lambda x: np.where(self.treatments == x)[0].item()).values
        gi = self.df.index.get_level_values('game').map(lambda x: np.where(self.games == x)[0].item()).values
        ri = self.df.index.get_level_values('round').map(lambda x: np.where(self.rounds == x)[0].item()).values
        tti = self.df.index.get_level_values('time').map(lambda x: np.where(self.times == x)[0].item()).values

        # populate tensors
        self.deal_prices_tensor = torch.zeros(
            [max_treatments, max_games, max_rounds, max_times, self.df.iloc[0, 0].shape[0]])
        self.deal_prices_tensor[ti, gi, ri, tti, :] = torch.stack(self.df['deal_prices'].values.tolist())

        self.bids_tensor = torch.zeros([max_treatments, max_games, max_rounds, max_times, self.df.iloc[0, 1].shape[0]])
        self.bids_tensor[ti, gi, ri, tti, :] = torch.stack(self.df['buyer_bid'].values.tolist())

        self.asks_tensor = torch.zeros([max_treatments, max_games, max_rounds, max_times, self.df.iloc[0, 1].shape[0]])
        self.asks_tensor[ti, gi, ri, tti, :] = torch.stack(self.df['seller_ask'].values.tolist())

        self.buyer_vals = torch.zeros([max_treatments, max_games, max_rounds, max_times, self.df.iloc[0, 1].shape[0]])
        self.buyer_vals[ti, gi, ri, tti, :] = torch.stack(self.df['buyer_valuation'].values.tolist())

        self.seller_vals = torch.zeros([max_treatments, max_games, max_rounds, max_times, self.df.iloc[0, 1].shape[0]])
        self.seller_vals[ti, gi, ri, tti, :] = torch.stack(self.df['seller_valuation'].values.tolist())

        self.times_tnsr = torch.zeros([max_treatments, max_games, max_rounds, max_times, 1])
        self.times_tnsr[ti, gi, ri, tti, :] = torch.tensor(
            self.df.index.get_level_values('time').values.tolist()).unsqueeze(-1).to(self.times_tnsr.dtype)

        self.__validate_tensor_assignments()

        self.ce_res = self.__get_ce_metrics()

        categorical_feats_df = self.df.groupby('treatment')['deal_prices'].first()
        categorical_feats_df = treatment_feats(categorical_feats_df)

        assert (self.treatments == categorical_feats_df.index).all()

        drop_columns = ['deal_prices'] + list(
            filter(lambda x: 'market_structure' in x, categorical_feats_df.columns.values))
        categorical_feats_df.drop(columns=drop_columns, inplace=True)
        self.feedback_settings = sorted(
            list(filter(lambda x: 'feedback_setting' in x, categorical_feats_df.columns.tolist())))
        self.price_rules = sorted(list(filter(lambda x: 'price_rule' in x, categorical_feats_df.columns.tolist())))
        label_feedback_setting_tensor = torch.tensor(categorical_feats_df[self.feedback_settings].values)
        label_price_rule_tensor = torch.tensor(categorical_feats_df[self.price_rules].values)
        self.feedback_setting_tnsr = label_feedback_setting_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(
            [1] + list(self.deal_prices_tensor.shape)[1:-1] + [1]).to(self.deal_prices_tensor.dtype)
        self.price_rule_tnsr = label_price_rule_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(
            [1] + list(self.deal_prices_tensor.shape)[1:-1] + [1]).to(self.deal_prices_tensor.dtype)

        self.__get_metadata_tensors()

        self.realized_buyer_gains = ((self.deal_prices_tensor != 0) * self.buyer_vals).sum(dim=-1).sum(dim=-1)
        self.realized_seller_gains = ((self.deal_prices_tensor != 0) * self.seller_vals).sum(dim=-1).sum(dim=-1)
        self.realized_got = self.realized_buyer_gains - self.realized_seller_gains
        self.allocative_efficiency = (self.realized_got / self.ce_res['got'].unsqueeze(-1)).nan_to_num(0)

    def __get_ce_metrics(self):
        return calculate_ce_metrics(self.buyer_vals.amax([-2, -3]), self.seller_vals.amax([-2, -3]))

    def validate_ce_metrics(self, dataset):
        dataset_filt = dataset[~(dataset['treatment'].isin(dynamic_treatments))]
        dataset_eq_values = dataset_filt.groupby(['treatment', 'game'])[['EQ', 'GOT', 'ce', 'ce_low', 'ce_high']].last()
        # validate results against df
        for ind, g in dataset_eq_values.groupby(['treatment', 'game']):
            treatment_index = np.where(self.treatments == ind[0])
            game_index = np.where(self.games == ind[1])
            assert g[['ce_low']].values == self.ce_res['ce_low'][treatment_index, game_index].squeeze().item()

            assert np.isclose(g[['ce_high']].values,
                              self.ce_res['ce_high'][treatment_index, game_index].squeeze().item())
            assert g[['ce']].values == self.ce_res['ce'][treatment_index, game_index].squeeze().item()

            assert g[['EQ']].values == self.ce_res['eq_quantity'][treatment_index, game_index].squeeze().item()


        # be carefult to include all rounds required for retrieving the valuaion prices from the df.

    def validate_ae_metrics(self, dataset):
        dataset_filt = dataset[~(dataset['treatment'].isin(dynamic_treatments))]
        dataset_eq_values = dataset_filt.groupby(['treatment', 'game', 'round'])[
            ['realized_got', 'allocative_efficiency']].max()
        # validate results against df
        for ind, g in dataset_eq_values.groupby(['treatment', 'game', 'round']):
            treatment_index = np.where(self.treatments == ind[0])
            game_index = np.where(self.games == ind[1])
            round_index = np.where(self.rounds == ind[2])
            check = np.allclose(g[['allocative_efficiency']].values.flatten()[0],
                                self.allocative_efficiency[treatment_index, game_index, round_index].squeeze().item())
            try:
                assert check
            except AssertionError:
                expected = g[['allocative_efficiency']].values.flatten()[0]
                actual = self.allocative_efficiency[treatment_index, game_index, round_index].squeeze().item()
                print(f'AE mismatch: expected={expected}, actual={actual}')

    def get_tensor_index(self, treatment: str, games: np.ndarray):
        treatment_tensor_index = np.where(self.treatments == treatment)[0].item()
        game_tensor_index = list(
            map(lambda x: np.where(self.games == int(x))[0].item(), games))
        return treatment_tensor_index, game_tensor_index

    def get_result_df(self, sample_id, dataset_type, treatment, games):
        treatment_tensor_index, game_tensor_index = self.get_tensor_index(treatment, games)
        result_df = pd.DataFrame(data=np.zeros([len(games), 4]),
                                 columns=['treatment', 'game', 'treatment_tnsr_idx', 'game_tnsr_idx'])
        result_df['treatment'] = treatment
        result_df['game'] = games
        result_df['treatment_tnsr_idx'] = treatment_tensor_index
        result_df['game_tnsr_idx'] = game_tensor_index
        result_df['dataset_type'] = dataset_type
        result_df['sample_id'] = sample_id
        return result_df

    def get_train_test_splits(self, ratio, n_samples):
        frames = []
        for sample_id in range(n_samples):
            gg = self.df.reset_index().groupby('treatment')
            for g in gg:
                np.random.seed(sample_id)
                treatment = g[0]
                available_games = g[1]['game'].unique()
                train_games = np.random.choice(available_games, size=round(len(available_games) * ratio + 0.00000001),
                                               replace=False)
                train_games.sort()
                test_games = np.array(sorted(set(available_games) - set(train_games)))
                train_result = self.get_result_df(sample_id, 'train', treatment, train_games)
                test_result = self.get_result_df(sample_id, 'test', treatment, test_games)

                frames.append(train_result)
                frames.append(test_result)
        return pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)

    def generate_tensor_dataset(self, max_rounds: int, max_time: int, name: str) -> TensorDataSet:

        max_time_idx = np.where(self.times == max_time)[0].item()
        max_rounds_idx = np.where(self.rounds == max_rounds)[0].item()
        tds = TensorDataSet(
            name=name,
            treatments=self.treatments,
            games=self.games,
            rounds=self.rounds[:max_rounds_idx],
            times=self.times[:max_time_idx],
            feedback_settings=np.array(self.feedback_settings),
            price_rules=np.array(self.price_rules),
            treatment_dim=self.treatment_dim,
            game_dim=self.game_dim,
            round_dim=self.round_dim,
            time_dim=self.time_dim,
            player_dim=self.player_dim,
            deal_prices_tnsr=self.deal_prices_tensor[:, :, :max_rounds_idx, :max_time_idx, :],
            bids_tnsr=self.bids_tensor[:, :, :max_rounds_idx, :max_time_idx, :],
            asks_tnsr=self.asks_tensor[:, :, :max_rounds_idx, :max_time_idx, :],
            buyer_vals_tnsr=self.buyer_vals[:, :, :max_rounds_idx, :max_time_idx, :],
            seller_vals_tnsr=self.seller_vals[:, :, :max_rounds_idx, :max_time_idx, :],
            times_tnsr=self.times_tnsr[:, :, :max_rounds_idx, :max_time_idx, :],
            feedback_setting_tnsr=self.feedback_setting_tnsr[:, :, :max_rounds_idx, :max_time_idx, :],
            price_rule_tnsr=self.price_rule_tnsr[:, :, :max_rounds_idx, :max_time_idx, :],
            cumulative_max_bids_time_tnsr=self.cumulative_max_bids_time[:, :, :max_rounds_idx, :max_time_idx, :],
            cumulative_max_bids_time_round_tnsr=self.cumulative_max_bids_time_round[:, :, :max_rounds_idx,
                                                :max_time_idx, :],
            cumulative_min_asks_time_tnsr=self.cumulative_min_asks_time[:, :, :max_rounds_idx, :max_time_idx, :],
            cumulative_min_asks_time_round_tnsr=self.cumulative_min_asks_time_round[:, :, :max_rounds_idx,
                                                :max_time_idx, :],
            time_bid_diff_tnsr=self.time_bid_diff[:, :, :max_rounds_idx, :max_time_idx, :],
            time_ask_diff_tnsr=self.time_ask_diff[:, :, :max_rounds_idx, :max_time_idx, :],
            round_time_bid_diff_tnsr=self.round_time_bid_diff[:, :, :max_rounds_idx, :max_time_idx, :],
            round_time_ask_diff_tnsr=self.round_time_ask_diff[:, :, :max_rounds_idx, :max_time_idx, :],
            cum_min_deal_prices_tnsr=self.cum_min_deal_prices[:, :, :max_rounds_idx, :max_time_idx, :],
            cum_max_deal_prices_tnsr=self.cum_max_deal_prices[:, :, :max_rounds_idx, :max_time_idx, :],
            eq_quantity_tnsr=self.ce_res['eq_quantity'],
            got_tnsr=self.ce_res['got'],
            ce_low_tnsr=self.ce_res['ce_low'].squeeze(-1),
            ce_high_tnsr=self.ce_res['ce_high'].squeeze(-1),
            ce_tnsr=self.ce_res['ce'].squeeze(-1),
            matched_valuations_mask=self.ce_res['matching_mask'],  # Treats x Games x Players
            sorted_buyer_idx=self.ce_res['sorted_buyer_idx'],  # Treats x Games x Players
            sorted_seller_idx=self.ce_res['sorted_seller_idx'],  # Treats x Games x Players
            sorted_buyer_vals_tnsr=self.ce_res['sorted_buyer_vals'],
            sorted_seller_asks_tnsr=self.ce_res['sorted_seller_asks'],
            allocative_efficiency_tnsr=self.allocative_efficiency[:, :, :max_rounds_idx],
            user_index_df=self.user_index_df  # filtering by rounds, might exclude some users that appear in the index.
        )

        return tds
