import os
from logging import info, debug
from typing import Collection

import numpy as np
import pandas as pd
import torch

from data_helpers.comp_equilibrium import dynamic_treatments, calc_bid_feats, calc_time_feats, calc_ce_feats_game, \
    calc_realized_price_feats, calc_ce_feats_round
from data_helpers.distribution_features import calculate_running_distribution, calc_quantile_feats
from data_helpers.treatment import treatment_feats, extract_setting, feedback_settings, market_structures, price_rules

def find_missaligned_match_time_indeces(merged_df: pd.DataFrame) -> np.ndarray:
    """
    Finds all missaligned time instances in the provided dataframe.

    Parameters
    ----------
    merged_df: pandas.DataFrame
        The provided dataframe to look missaligned times.

    Returns
    -------
    resulting_index: numpy.ndarray
        The resulting indices after fixing bad alignments.

    """
    a = (merged_df['match_time_y'] == merged_df['time_x'])
    b = (merged_df['match_time_y'] == merged_df['time_y'])
    c = (merged_df['match_time_x'] == merged_df['time_x'])
    d = (merged_df['match_time_x'] == merged_df['time_y'])
    e = (merged_df['match_time_x'] == merged_df['match_time_y'])

    match_time_filter = a | b | c | d  # at least one needs to be true
    match_time_filter = ~match_time_filter | ~e  # both match times needs to be equal
    filtered_result = merged_df.loc[
        match_time_filter, ['match_time_y', 'match_time_x', 'time_y', 'time_x', 'index_x', 'index_y']]
    debug('Missaligned match times detected: ' + str(filtered_result.shape[0]))
    all_indices = filtered_result[['index_x', 'index_y']].values.flatten()
    return all_indices


class OriginalDataPreprocessor:
    def __init__(self, original_file_pth: str = '../data/Data.csv'):
        """
        This class preprocesses the data from Ikica et al.

        Parameters
        ----------
        original_file_pth: str
            The path to the original data file.

        """
        info('Reading original dataframe.')
        self.df = pd.read_csv(original_file_pth)

        # Fixes on original data file, due to potential sync challenges on the web platform
        info('Removing duplicate records.')
        self.__remove_duplicate_records()

        info('Aligning invalid match times.')
        self.__align_match_times()

        info('Calculating market features related to bids and asks, e.g. quantiles of their running distributions.')
        # Calculating relevant features based on literature and model requirements.
        self.__calculate_market_features()

        info('Extracting dummy variables on experimental settings, such as feedback setting, price mechanism and market'
             'structure.')
        # Add dummy variables based on treatment feedback setting, price mechanism and market structure.
        self.__extract_experimental_setting_features()

    def __remove_duplicate_records(self):
        """
        Removes duplicate records due to double action registration.
        This method is called in the constructor of the class and is in place in `self.df`
        - the original dataframe.

        Returns
        -------
        None
        """

        # drop duplicate actions
        debug('Dropping duplicate records.')
        self.df = self.df.drop_duplicates().reset_index(drop=True)

    def __align_match_times(self):
        """
        Fix missaligned times when deal prices occur.
        Since the provided data are based on online systems, they may contain missalign-timestamps, most
        probably due to race conditions and network delays. Some of these alignments are detectable
        and can be easily be fixed. Typical example, a bid is recorded a second later than submitted.

        Returns
        -------
        None
        """
        debug('Self-merging dataframe to align matching times.')
        # merge the dataframe with itself based on user id matches for deals.
        merged_df = pd.merge(self.df.reset_index(), self.df.reset_index(), left_on=['game', 'round', 'id'],
                             right_on=['game', 'round', 'match_id']).dropna(subset=['match_id_x', 'match_id_x'])

        debug('Detecting miss-aligned matching times.')
        all_indices = find_missaligned_match_time_indeces(merged_df)

        debug('Fixing miss-aligned matching times by subtracting a second.')
        self.df.loc[all_indices, 'match_time'] = self.df.loc[all_indices, 'match_time'] - 1

        # second of the merging operator after changes to see if any erros occur.
        debug('Remerging dataframe to recheck for missalignments.')
        merged_df = pd.merge(self.df.reset_index(), self.df.reset_index(), left_on=['game', 'round', 'id'],
                             right_on=['game', 'round', 'match_id']).dropna(subset=['match_id_x', 'match_id_x'])
        all_indices = find_missaligned_match_time_indeces(merged_df)
        assert len(all_indices) == 0, 'Found non-aligned match time.'

    def __calculate_market_features(self):
        """
        Collect and combine values from different rounds and games, in place to original dataframe.
        Uses the code from Ikica et al. to calculate variables relevant to competitive equilibrium,
        such as gains of trade, allocate efficiency, range of competitive equilibrium prices etc.

        Returns
        -------
        None
        """

        debug('Calculating changes on bids/asks in time.')
        self.df = calc_bid_feats(self.df)
        self.df = self.df.reset_index(drop=True)

        debug('Calculating time related feats, such as normalized times to total duration, time discounts and constants'
              'for regresison.')
        self.df = calc_time_feats(self.df)
        self.df = self.df.reset_index(drop=True)

        debug('Calculating competitive equillibrium feats per game, such as gains of trade, allocative efficiency etc.')
        self.df = calc_ce_feats_game(self.df)
        self.df = self.df.reset_index(drop=True)

        debug('Calculating competitive equillibrium feats per round, such as gains of trade, allocative efficiency etc.')
        self.df = calc_ce_feats_round(self.df)
        self.df = self.df.reset_index(drop=True)


        debug('Calculating realized/deal price feats such as realized price distance and relative demand.')
        self.df = calc_realized_price_feats(self.df)

        debug('Sorting dataframe after operations, according to treatment, game, round, time and user id.')
        self.df.sort_values(['treatment', 'game', 'round', 'time', 'id'], inplace=True)
        self.df = self.df.reset_index(drop=True)

    def __extract_experimental_setting_features(self):
        """
        Extract features based on feedback setting, pricing mechanism and market structure of a game.
        In place operation on orignal dataframe.

        Returns
        -------
        None
        """

        unique_treatments = self.df['treatment'].unique()
        treatment_to_feedback_dict = {}
        for treat in unique_treatments:
            for setting in feedback_settings:
                if setting in treat:
                    treatment_to_feedback_dict[treat] = setting
                    break
                treatment_to_feedback_dict[treat] = feedback_settings[-1]

        treatment_to_price_rule = {}
        for treat in unique_treatments:
            for rule in price_rules:
                if rule in treat:
                    treatment_to_price_rule[treat] = rule
                    break
                treatment_to_price_rule[treat] = price_rules[-1]

        treatment_to_market_structure = {}
        for treat in unique_treatments:
            for structure in market_structures:
                if structure in treat:
                    treatment_to_market_structure[treat] = structure
                    break
                treatment_to_market_structure[treat] = market_structures[-1]

        self.df['feedback_setting'] = self.df['treatment'].map(lambda val: treatment_to_feedback_dict[val])
        self.df = self.df.reset_index(drop=True)

        self.df['price_rule'] = self.df['treatment'].map(lambda val: treatment_to_price_rule[val])
        self.df = self.df.reset_index(drop=True)

        self.df['market_structure'] = self.df['treatment'].map(lambda val: treatment_to_market_structure[val])
        self.df = self.df.reset_index(drop=True)


def unique_deal_pairs(row: pd.Series) -> pd.Series:
    """
    Find a deal pair based on matching times and adds to the pandas series (row) a list with the pair as a `deal_pair`.

    Parameters
    ----------
    row: pandas.Series
        The pandas series (row), often the result of an `apply` operation on the original dataframe.

    Returns
    -------
    row: pands.Series
        The resulting row, with an extra `deal_pair` column.
    """
    if row['match_time'] <= row['time']:
        row['deal_pair'] = [tuple(sorted([row['id'], int(row['match_id'])]))]
    return row


def gather_unique_pairs(group, last_grouper: str) -> pd.DataFrame:
    """
    Gather all the unique pairs in the gived group.

    Parameters
    ----------
    group: pandas.DataFrame
        The group of the original dataframe to find unique pairs. Usually this is a group per time, round,
        game and treatment.
    last_grouper: the name of the last grouping column, to use for column name generation after grouping.

    Returns
    -------
    group: pandas.DataFrame
        The resulting group with extra columns added for deal pairs, number of unique deals at tiem, and the unique
        deals set.

    """
    group = group.sort_values('time')
    unique_deals = set()
    group['n_unique_deals_' + last_grouper] = np.nan
    group['unique_deals_' + last_grouper] = None
    group['unique_deals_' + last_grouper].astype(object)
    for i in group.index.values:
        unique_deals.add(group.loc[i, 'deal_pair'][0])
        group.loc[i, 'n_unique_deals_' + last_grouper] = int(len(unique_deals))
        group.loc[i, 'unique_deals_' + last_grouper] = [tuple(sorted(unique_deals))]

    return group


def realized_gains_group_oprtr(group):
    """
    Calculating different quantities of realized gains at time.

    Parameters
    ----------
    group: pandas.Dataframe
        The group dataframe, usually a group on treatment, game, round and time.

    Returns
    -------
    resulting_row: pandas.Series
        The resulting row (group) that contains all GOT related measures.
    """
    # summing all buyer and seller prices appearing on same match time, in the original dataframe.
    total_buyer_valuations = group.loc[group['side'] == 'Buyer', 'valuation'].sum()
    total_seller_valuations = group.loc[group['side'] == 'Seller', 'valuation'].sum()

    # calculating realized gains by subtracting total valuations of buyers and sellers.
    realized_gains = total_buyer_valuations - total_seller_valuations

    # calculating minimum seller valuation and maximum buyer valuation.
    min_seller_valuation = group.loc[group['side'] == 'Seller', 'valuation'].min()
    max_buyer_valuation = group.loc[group['side'] == 'Buyer', 'valuation'].max()

    return pd.Series(dict(current_realized_gains=realized_gains,
                          min_seller_valuation=min_seller_valuation,
                          max_buyer_valuation=max_buyer_valuation,
                          total_seller_valuation=total_seller_valuations,
                          total_buyer_valuation=total_buyer_valuations
                          )
                     )


def gains_of_trade_group_oprtr(group):
    """
    Calculates theoretical gains of trade and related features.

    Parameters
    ----------
    group: pandas.DataFrame
        The original data grouped on 'treatment', 'game', 'round'.

    Returns
    -------
    group: pandas.DataFrame
        The original group with the following columns added:
        'min_seller_valuation', 'max_buyer_valuation', 'total_seller_valuation',
        'total_buyer_valuation', 'cum_realized_got', 'realized_got'
    """
    group = group.sort_values('time')
    group['min_seller_valuation'] = group['min_seller_valuation'].ffill().cummin()
    group['max_buyer_valuation'] = group['max_buyer_valuation'].ffill().cummax()
    group['total_seller_valuation'] = group['total_seller_valuation'].fillna(0).cumsum()
    group['total_buyer_valuation'] = group['total_buyer_valuation'].fillna(0).cumsum()
    group['cum_realized_got'] = group['current_realized_gains'].fillna(0).cumsum()
    group['realized_got'] = group['cum_realized_got'].max()
    return group


class TimeBasedDataSet:

    def __init__(self, df: pd.DataFrame):
        """
        Use the preprocessed data from the original dataset and applies aggregations on time level,
        essentially removing the record granularity on user level.

        Parameters
        ----------
        df: pandas.DataFrame
            The original dataframe after data preprocessing.
        """
        self.df = df
        self.round_cols = ['max_time', 'linear_time_discount', 'norm_time', 'const', 'ce_low', 'ce_high', 'ce', 'GOT',
                           'EQ', 'realized_price', 'realized_price_distance', 'feedback_setting', 'price_rule',
                           'market_structure', 'ce_low_round', 'ce_high_round', 'ce_round', 'GOT_round',
                           'EQ_round', 'relative_demand']
        info('Generating the cumulative dataset on time, by keeping first appearing aggregate values.')
        self.dataset = self.__generate_dataset()

        info('Calculating quantiles of distributions of bids and asks, persisting in time until replaced.')
        self.__calculate_bid_ask_quantile_features()

        info('Calculating cumulative number of deals in time for each round.')
        self.__calculate_unique_deals_features()

        info('Calculating quantiles of valuation price distributions per game.')
        self.__calculate_valuation_quantiles()

        info('Calculating features of realized gains.')
        self.__calculate_realized_gains_features()

        info('Calculating allocative efficiency features.')
        self.__calculate_allocative_efficiency_features()

    def __generate_dataset(self):
        """
        Generate dataset, by aggregating the original dataframe in time.

        Returns
        -------
        dataset: pd.DataFrame
            The resulting time aggretated dataset.

        """
        debug('Grouping original data by time and keeping first time')
        feats_targets_df_first = self.df.groupby(['treatment', 'game', 'round', 'time'])[self.round_cols].first()
        feats_targets_df_last = self.df.groupby(['treatment', 'game', 'round', 'time'])[self.round_cols].last()
        debug('Checking if grouping by time changes features during a round, '
              'when such features are not expected to change.')
        # Exclude `relative_demand` column, as it changes within a round.
        assert (feats_targets_df_first.iloc[:, :-1] == feats_targets_df_last.iloc[:, :-1]).all().all(), \
            'Invalid last and first calculations for columns. Some columns are unequal, but are expected to.'
        return feats_targets_df_first

    def __calculate_bid_ask_quantile_features(self):
        """
        Calculate bid and ask quantile features per time.
        Bid are fill forwarded to future, until replaced.

        Returns
        -------

        """
        # Group and calculate user - bid/ask dictionaries per time
        # In round where a user bids multiple times, keep the latest
        debug('Calculating bid/ask distrbutions over time and users.')
        self.latest_same_user_bids_df = self.df.groupby(
            ['treatment', 'game', 'round', 'time', 'id']).last().reset_index()
        self.bid_dist_feats = calculate_running_distribution(self.latest_same_user_bids_df, 'bid')
        debug("Calculating quantiles over bid distributions.")
        bid_run_quants = self.bid_dist_feats.apply(calc_quantile_feats, 1)

        debug('Checking if number of samples changes in dataset after previous quantile tranformers. It should not!')
        rows_before = self.dataset.shape[0]
        self.dataset = self.dataset.join(bid_run_quants)
        assert rows_before == self.dataset.shape[0], 'Operation changed number of rows. This should not happen!'

        debug(
            'Checking that accumulated dictionaries at final time per round '
            'match total dictionaries at final time per round.')

        # Test that accumulated dictionaries at final time per round match total dictionaries at final time per round
        for k, g in self.latest_same_user_bids_df.groupby(['treatment', 'game', 'round'], group_keys=False):
            g = g.sort_values('time')
            g_seller = g[g.side == 'Seller']
            g_buyer = g[g.side == 'Buyer']
            dseller = dict(zip(g_seller['id'].tolist(), g_seller['bid'].tolist()))
            dbuyer = dict(zip(g_buyer['id'].tolist(), g_buyer['bid'].tolist()))
            key = tuple(*g.tail(1)[['treatment', 'game', 'round', 'time']].values.tolist())
            assert self.bid_dist_feats.loc[key, 'running_buyer_bid'] == dbuyer, 'Buyer accumulation error at ' + str(k)
            assert self.bid_dist_feats.loc[key, 'running_seller_bid'] == dseller, 'Seller accumulation error at ' + str(
                k)
        debug('All bid/ask dictionary accumulations seem valid!')

    def __calculate_unique_deals_features(self):
        """
        Add in-place columns to the dataset, like number of unique deals so far in the round.

        Returns
        -------
        None
        """

        debug('Gathering the unique deal pairs per timestep, based on matches.')
        self.unique_deals = self.df[['treatment',
                                     'game',
                                     'round',
                                     'time',
                                     'match_time',
                                     'id',
                                     'match_id',
                                     'price',
                                     'bid'
                                     ]
        ].dropna().apply(unique_deal_pairs,
                         axis=1
                         )

        self.unique_deals = self.unique_deals.dropna().sort_values(['treatment', 'game', 'round', 'id']).reset_index(
            drop=True)
        debug("Accumulating unique deal pairs per round.")
        unique_round_deals = self.unique_deals.groupby(['treatment', 'game', 'round'], group_keys=True).apply(
            lambda g: gather_unique_pairs(g, last_grouper='round')).reset_index(drop=True)

        debug("Keeping max number numbers of deals at a time")
        unique_round_deals = unique_round_deals.groupby(['treatment', 'game', 'round', 'time'])[
            'n_unique_deals_round'].max().to_frame()

        debug('Checking if unique deals operation invalidly changed the number of rows in the dataset after merging.')
        rows_before = self.dataset.shape[0]

        self.dataset = pd.merge(self.dataset, unique_round_deals.reset_index(),
                                on=['treatment', 'game', 'round', 'time'],
                                how='left')
        self.dataset['n_unique_deals_round'] = self.dataset.groupby(['treatment', 'game', 'round'])[
            'n_unique_deals_round'].ffill()
        debug('Filling missing value on `n_deal_prices_round` column with zeros.')
        self.dataset['n_unique_deals_round'] = self.dataset['n_unique_deals_round'].fillna(0)
        assert rows_before == self.dataset.shape[0], 'Operation changed number of rows. This should not happen!'

        debug('Checking if number of deal prices per round are valid.')
        assert (self.dataset[(self.dataset.n_unique_deals_round == 0) & (self.dataset.realized_price > 0)]).shape[
                   0] == 0, ('Number of deals per round is greater than 0 - when no realized prices exists yet. '
                             'Please recheck dataset.')

        debug('Sorting dataset based on treatment, game, round and time.')
        self.dataset = self.dataset.sort_values(['treatment', 'game', 'round', 'time'])
        assert not (self.dataset.groupby(['treatment', 'game', 'round'])[
                        'n_unique_deals_round'].diff().dropna() < 0).any(), \
            'Detected decreasing unique deals per round in single round'
        debug('All deal price checks passed successfully!')

    def __calculate_valuation_quantiles(self):
        """
        Add valuation price targets (in-place) quantiles to the dataset.

        Returns
        -------
        None
        """
        # Buyer Valuations
        vdf_buyers = self.df.query("side=='Buyer'")
        debug('Finding all buyer valuations per game, and keeping the first occurence in dynamic treatments.')
        vdf_buyers = vdf_buyers.groupby(['treatment', 'game', 'id'])['valuation'] \
            .first().reset_index()
        debug('Persisting all buyer valuations per game in a list.')
        vdf_buyers = vdf_buyers.groupby(['treatment', 'game'])['valuation'].apply(list)
        debug('Calculating all eleven buyer valuation quantiles.')
        vdf_buyers = vdf_buyers.to_frame().apply(calc_quantile_feats, 1)

        # Seller Valuations
        vdf_sellers = self.df.query("side=='Seller'")
        debug('Finding all seller valuations per game, and keeping the first occurence in dynamic treatments.')
        vdf_sellers = vdf_sellers.groupby(['treatment', 'game', 'id'])['valuation'] \
            .first().reset_index()
        debug('Persisting all seller valuations per game in a list.')
        vdf_sellers = vdf_sellers.groupby(['treatment', 'game'])['valuation'].apply(list)
        debug('Calculating all eleven seller valuation quantiles.')
        vdf_sellers = vdf_sellers.to_frame().apply(calc_quantile_feats, 1)

        debug('Merging buyer and seller valuations.')
        valuation_quantiles = vdf_sellers.join(vdf_buyers, lsuffix='_seller', rsuffix='_buyer')
        self.dataset = pd.merge(self.dataset, valuation_quantiles, on=['treatment', 'game'])

    def __calculate_realized_gains_features(self):
        """
        Calculate realized gains of trade, based on valuation of values of players trading at a time.

        Returns
        -------
        None
        """
        debug("Grouping original dataframe on match_times.")
        realized_gains_series = self.df.groupby(['treatment', 'game', 'round', 'match_time']).apply(
            realized_gains_group_oprtr)
        debug('Adding realized gain features to dataset.')
        realized_gains_series.index = realized_gains_series.index.rename('time', level='match_time')
        self.dataset = pd.merge(self.dataset, realized_gains_series.reset_index(),
                                on=['treatment', 'game', 'round', 'time'],
                                how='left')

    def __calculate_allocative_efficiency_features(self):
        """
        Use GOT and realized GOT features and calculate allcoative efficiency features.

        Returns
        -------
        None
        """
        self.dataset = self.dataset.groupby(['treatment', 'game', 'round'], group_keys=True).apply(
            gains_of_trade_group_oprtr)
        self.dataset['cum_allocative_efficiency'] = self.dataset['cum_realized_got'] / self.dataset['GOT']
        self.dataset['allocative_efficiency'] = self.dataset['realized_got'] / self.dataset['GOT']
        self.dataset['allocative_efficiency'] = self.dataset['allocative_efficiency'].fillna(0)

        self.dataset['cum_allocative_efficiency_round'] = self.dataset['cum_realized_got'] / self.dataset['GOT_round']
        self.dataset['allocative_efficiency_round'] = self.dataset['realized_got'] / self.dataset['GOT_round']
        self.dataset['allocative_efficiency_round'] = self.dataset['allocative_efficiency_round'].fillna(0)

    def __map_to_tensor(self, user_index_game: pd.DataFrame, result_dictionary: dict, treatment: str, game: int,
                        max_tensor_len: int) -> torch.Tensor:
        r"""
        Converts a result dictionary to a dataframe with tensors as cell values.
        Uses the provided user index to assign each players id to a given position in a `max_tensor_len`-dimensional
        vector.
        The resulting dataframe has the same index as the provided user_index_game grouped dataframe.


        Parameters
        ----------
        user_index_game: pandas.DataFrame
            A grouped dataframe on `treatment` and `game` aggregating on unique `id` (player ids).
        result_dictionary: dict
            The dictionary containint `id` as keys and bids/asks/valuations as values.
        treatment: str
            The current treatment.
        game: int
            The current game
        max_tensor_len: int
            The max tensor length, which should be a number greater or equal to max total players in any game.

        Returns
        -------
        price_vector: torch.Tensor
            The resulting vector with bids/asks/valuations as elements. If a player did not have an assigned price
            the default value is zero. Price values are considered strictly positive.
        """
        x = torch.zeros(max_tensor_len)
        if len(result_dictionary) == 0:
            return x
        else:
            # print(user_index_game.loc[(treatment, game)])
            # print(list( result_dictionary.keys()))
            indices = list(
                map(lambda k: np.where(user_index_game.loc[(treatment, game)] == k)[0], result_dictionary.keys()))
            # print(indices)
            # print(str(treatment) + ' ' + str(game))
            indices = np.array(indices).flatten()
            x[indices] = torch.tensor(list(result_dictionary.values())).to(torch.float)
            return x

    def __map_row_to_tensor(self, row: pd.Series, dict_cols: Collection, max_elements: int, user_index_game) -> pd.Series:
        """
        maps a dataframe row to a tensor.

        Parameters
        ----------
        row: pd.Series
            A dataframe row, with the desired player price columns (bids/asks/valuations) to be converted to a tensor.
        dict_cols: Collection
            A list of columns to retrieve the relevant dictionary from the given row.
        max_elements: int
            The max number of elements/players to expect. All resulting tensors are padded to this number with 0s.

        Returns
        -------
        row: pd.Series
            The resulting row with added tensors as columns.
        """
        for col in dict_cols:
            result_dict = row[col]
            treatment = row['treatment']
            game = row['game']
            col_tensor = self.__map_to_tensor(user_index_game, result_dict, treatment, game, max_elements)
            row[col] = col_tensor
        return row

    def torch_data_preparation(self, output_folder_pth: str, max_num_players: int = 50):
        """
        Processes the dataframes of this class to produce safetensors files with tensor dictionaries to be used later
        for pytorch models. Dictionary keys are a tuple: `(treatment, game, round, time)`. The values are tensors.
        The following files are persisted in the path:
        - `running_buyer_bid.safetensors`: Contains buyer bids at a time values. If no bid is present, these are zeros.
        - `running_seller_ask.safetensors`: Contains seller asks at a time values. If no bid is present, these are zeros.
        - `running_buyer_valuation.safetensors`: Contains participating (non-zero bid) buyer valuations at a time
           values. If no bid is present, these are zeros.
        - `running_seller_valuation.safetensors`: Contains participating (non-zero ask) seller valuations at a time
           values. If no bid is present, these are zeros.
        -  `deal_prices.safetensors`: Contains deal prices that happened at time assigned to respective player indices.

        Parameters
        ----------
        output_folder_pth: str
            The output folder path
        max_num_players: int
            The max number of players to consider for all games. If this is higher than the actual max number of
            players, empty slots are padded with 0s.

        Returns
        -------
        None
        """
        debug('Preparing base dictionaries for pytorch models.')
        user_index_game = self.latest_same_user_bids_df.groupby(['treatment', 'game'])['id'].unique()
        debug('Generating distributions for valuations of participating (biding/asking) players over time.')
        val_dist_feats = calculate_running_distribution(self.latest_same_user_bids_df, 'valuation')
        user_dicts_df = self.bid_dist_feats.join(val_dist_feats, how='outer')
        debug('Creating respective user bid/ask/valuation tensor valued dataframes.')
        user_tnsr_df = user_dicts_df.reset_index().apply(
            lambda r: self.__map_row_to_tensor(r, user_dicts_df.columns.values.tolist(), max_num_players, user_index_game),
            axis=1)
        user_tnsr_df = user_tnsr_df.set_index(['treatment', 'game', 'round', 'time'])

        debug('Persisting safetensors dictionaries for bid/ask/valuations.')
        os.makedirs(output_folder_pth, exist_ok=True)
        for col in user_tnsr_df.columns:
            res_dict = user_tnsr_df[col].to_dict()
            if 'seller_bid' in col:
                # patching to respect nomeclature.
                # TODO: fix future code base to avoid this patching.
                col = col.replace('seller_bid', 'seller_ask')
            torch.save(res_dict, os.path.join(output_folder_pth, col + '.pt'))

        debug('Preparing deal prices tensors.')
        deal_prices = dict()
        for index, row in self.unique_deals.iterrows():
            key = tuple(row[['treatment', 'game', 'round', 'time']])
            if key not in deal_prices:
                deal_prices[key] = torch.zeros(max_num_players)
            for idd in row['deal_pair'][0]:
                id_index = np.where(user_index_game.loc[key[:2]] == idd)
                deal_prices[key][id_index] = row['price']
        for i in user_tnsr_df.index:
            if i not in deal_prices:
                deal_prices[i] = torch.zeros(max_num_players)
        debug('Persisting deal prices tensors dictionaries.')

        torch.save(deal_prices, os.path.join(output_folder_pth, 'deal_prices.pt'))

        debug('Persisting user index in game dataframe, to reuse for mapping tensors back to original index.')
        user_index_df = user_index_game.apply(pd.Series)
        user_index_df = user_index_df
        user_index_df.reset_index().to_feather(os.path.join(output_folder_pth, 'user_index.ft'))



