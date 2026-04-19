import pandas as pd

treatment_dimensions = {'structure': (['BBLimS', 'BBMoreB', 'BBMoreS', 'BBLargeCE', 'BBMoreRounds', 'FullLimS', 'FullMoreB', 'FullMoreS', 'FullLargeCE'], 'Varying the market structure', 0),
                        'dynamics': (['FullSShift','FullFromBtoS','FullFromStoB'], 'Dynamic environments', 0),
                        'feedback': (['BB', 'Same', 'Other', 'Full'], 'Varying the amount of feedback', 1),
                        'price': (['BBRandom','BBMMK','SameMMK'], 'Varying the price rule', 2)}

dynamic_treatments = treatment_dimensions['dynamics'][0]

def get_valuations(df):
    '''Extract buyer and seller valuations.'''
    valuations = {}

    for side, df_side in df.groupby('side'):
        side_valuations = df_side[['id', 'valuation']].drop_duplicates()['valuation'].tolist()

        if side == 'Buyer':
            side_valuations.sort(reverse=True)
        else:
            side_valuations.sort()

        valuations[side] = side_valuations

    return valuations


def get_equilibrium(df):
    '''Extract the equilibrium price range, the gains of trade,
    and the equilibrium quantity corresponding to a given experimental market.'''

    valuations = get_valuations(df)
    buyer_valuations = valuations['Buyer']
    seller_valuations = valuations['Seller']

    GOT = 0
    eq_quantity = min(len(buyer_valuations), len(seller_valuations))

    # Find the intersection of supply and demand schedules and compute the corresponding gains of trade
    for i in range(eq_quantity):
        if seller_valuations[i] > buyer_valuations[i]:
            eq_quantity = i
            break
        else:
            GOT += buyer_valuations[i] - seller_valuations[i]

    if len(buyer_valuations) == eq_quantity:
        EPR0 = seller_valuations[eq_quantity - 1]
    else:
        EPR0 = max(seller_valuations[eq_quantity - 1], buyer_valuations[eq_quantity])

    if len(seller_valuations) == eq_quantity:
        EPR1 = buyer_valuations[eq_quantity - 1]
    else:
        EPR1 = min(buyer_valuations[eq_quantity - 1], seller_valuations[eq_quantity])

    return [EPR0, EPR1], GOT, eq_quantity


def calc_ce_feats_round(df):
    for (treatment, game, rnd), df_round in df.groupby(['treatment', 'game', 'round']):
        EPR, GOT, eq_quantity = get_equilibrium(df_round)
        valid_ids_original = df['treatment'] == treatment
        valid_ids_original &= df['game'] == game
        valid_ids_original &= df['round'] == rnd

        df.loc[valid_ids_original, 'ce_low_round'] = EPR[0]
        df.loc[valid_ids_original, 'ce_high_round'] = EPR[1]
        df.loc[valid_ids_original, 'ce_round'] = sum(EPR) / len(EPR)
        df.loc[valid_ids_original, 'GOT_round'] = GOT
        df.loc[valid_ids_original, 'EQ_round'] = eq_quantity
    return df


def set_eq_values_to_df(df, valid_ids_original, EPR, GOT, eq_quantity):
    df.loc[valid_ids_original, 'ce_low'] = EPR[0]
    df.loc[valid_ids_original, 'ce_high'] = EPR[1]
    df.loc[valid_ids_original, 'ce'] = sum(EPR) / len(EPR)
    df.loc[valid_ids_original, 'GOT'] = GOT
    df.loc[valid_ids_original, 'EQ'] = eq_quantity
def calc_ce_feats_game(df):
    for (treatment, game), df_game in df.groupby(['treatment', 'game']):
        valid_ids_original = df['treatment'] == treatment
        valid_ids_original &= df['game'] == game

        if treatment in dynamic_treatments:
            # The structure of these markets changes after round 5
            is_early_round = (df_game['round'].isin(range(1, 6)))
            is_late_round = (df_game['round'].isin(range(6, 11)))
            df_game1 = df_game[is_early_round]
            df_game2 = df_game[is_late_round]

            EPR1, GOT1, eq_quantity1 = get_equilibrium(df_game1)
            EPR2, GOT2, eq_quantity2 = get_equilibrium(df_game2)

            is_early_round_all = valid_ids_original & df['round'].isin(range(1, 6))
            is_late_round_all = valid_ids_original & df['round'].isin(range(6, 11))

            set_eq_values_to_df(df, is_early_round_all, EPR1, GOT1, eq_quantity1)
            set_eq_values_to_df(df, is_late_round_all, EPR2, GOT2, eq_quantity2)
        else:
            EPR, GOT, eq_quantity = get_equilibrium(df_game)
            set_eq_values_to_df(df, valid_ids_original, EPR, GOT, eq_quantity)

    return df

def calc_deal_price(group):
    deal_prices = group.loc[group.status == 'Accepted', ['match_time', 'price'] + ['treatment', 'game', 'round']] \
        .rename(columns={'match_time': 'time', 'price': 'realized_price'}).groupby(
        ['treatment', 'game', 'round', 'time']).median().reset_index()
    if deal_prices.shape[0] == 0:
        group['realized_price'] = 0
        return group
    else:
        group = pd.merge(group, deal_prices, on=['treatment', 'game', 'round', 'time'], how='left')
    group['realized_price'] = group.sort_values('time')['realized_price'].ffill().fillna(0)
    return group


def calc_realized_price_feats(df):
    df = df.groupby(['treatment', 'game', 'round'], group_keys=False).apply(calc_deal_price)
    df['realized_price_distance'] = ((df.realized_price - df.ce).abs() / df.ce)
    df['relative_demand'] = 1 * (df.side == 'Buyer') + (1 - 2 * (df.side == 'Buyer')) * df.bid / df.ce - 1 * (
            df.side == 'Seller')

    return df


def calc_bid_change(group):
    group = group.sort_values('time')
    group['bid_change'] = group['bid'].ffill().diff().fillna(0)
    return group


def calc_bid_feats(df):
    df = df.groupby(['treatment', 'game', 'round', 'id'], group_keys=False).apply(calc_bid_change).reset_index(
        drop=True)
    return df


def calc_max_time(group):
    group['max_time'] = group.time.max()
    return group


def calc_time_feats(df):
    df = df.groupby(['treatment', 'game', 'round'], group_keys=False).apply(calc_max_time)
    df['linear_time_discount'] = (df['max_time'] - df['time']) / df['max_time']

    df['norm_time'] = df['time'] / df['max_time']

    df['const'] = 1
    return df


# code from ikica repo.
def get_rounds_dicts(df):
    EPR_rounds = {}
    GOT_rounds = {}
    eq_quantity_rounds = {}

    for (treatment, game, rnd), df_round in df.groupby(['treatment', 'game', 'round']):
        EPR, GOT, eq_quantity = get_equilibrium(df_round)

        EPR_rounds[treatment, game, rnd] = EPR
        GOT_rounds[treatment, game, rnd] = GOT
        eq_quantity_rounds[treatment, game, rnd] = eq_quantity
    return EPR_rounds, GOT_rounds, eq_quantity_rounds


def allocative_efficiency(df, GOT_rounds):
    for (treatment, game, rnd), df_round in df.groupby(['treatment', 'game', 'round']):
        GOT = GOT_rounds[treatment, game, rnd]
        realised_vals = get_valuations(df_round[df_round['price'].notna()])
        allocative_eff = (sum(realised_vals.get('Buyer', [])) - sum(realised_vals.get('Seller', []))) / GOT
