import pandas as pd
import torch


def calculate_supply_demand_line_coeffs(dff, is_desc: bool = True, aggfunc: str = 'mean', use_round_level=True):
    cols = ['treatment', 'game']
    if use_round_level:
        cols.append('round')
    y = pd.pivot_table(dff, index=cols, columns=['id'], values='valuation', aggfunc='mean')
    y_tnsr = torch.tensor(y.values).float()
    y_tnsr_desc = y_tnsr.sort(-1, descending=is_desc).values.nan_to_num(0)
    y_mask = (y_tnsr_desc != 0)
    x_tnsr = (y_mask * y_mask.cumsum(1)).float().unsqueeze(-1)
    y_tnsr_desc = y_tnsr_desc.unsqueeze(-1)
    x_tnsr = torch.cat([x_tnsr, torch.ones_like(x_tnsr)], dim=-1)
    x_tnsr[:, :, 1] = x_tnsr[:, :, 1] * (x_tnsr[:, :, 0] != 0)
    coeffs = torch.linalg.lstsq(x_tnsr, y_tnsr_desc).solution.squeeze(-1)

    coeff_df = pd.DataFrame(index=y.index, data=coeffs.numpy(), columns=['slope', 'intercept'])
    # coeff_df['side'] = dff['side'].iloc[0]
    return coeff_df.reset_index()


if __name__ == '__main__':
    dff = pd.read_feather('../data/preprocessed/original_df.ft')
    dff_b = dff.query('side=="Buyer"')
    dff_s = dff.query('side=="Seller"')

    coeffs_b = calculate_supply_demand_line_coeffs(dff_b, is_desc=True, aggfunc='first')
    coeffs_s = calculate_supply_demand_line_coeffs(dff_s, is_desc=False, aggfunc='first')
    df_round_coeffs = pd.merge(coeffs_b, coeffs_s, on=['treatment', 'game', 'round'], suffixes=['_buyer', '_seller'])
    df_round_coeffs['linear_eq'] = (df_round_coeffs['intercept_buyer'] - df_round_coeffs['intercept_seller']) / (
            df_round_coeffs['slope_seller'] - df_round_coeffs['slope_buyer'])
    df_round_coeffs['linear_cep'] = df_round_coeffs['linear_eq'] * df_round_coeffs['slope_seller'] + df_round_coeffs[
        'intercept_seller']
    df_round_coeffs['linear_got'] = df_round_coeffs['linear_eq'] * (
            df_round_coeffs['intercept_buyer'] + df_round_coeffs['slope_buyer'] - df_round_coeffs['intercept_seller'] -
            df_round_coeffs['slope_seller']) / 2
    df_round_coeffs.to_feather('../data/preprocessed/supply_demand_lines_round.ft')

    coeffs_b_g = calculate_supply_demand_line_coeffs(dff_b, is_desc=True, aggfunc='first', use_round_level=False)
    coeffs_s_g = calculate_supply_demand_line_coeffs(dff_s, is_desc=False, aggfunc='first', use_round_level=False)
    df_game_coeffs = pd.merge(coeffs_b_g, coeffs_s_g, on=['treatment', 'game'], suffixes=['_buyer', '_seller'])
    df_game_coeffs['linear_eq'] = (df_game_coeffs['intercept_buyer'] - df_game_coeffs['intercept_seller']) / (
            df_game_coeffs['slope_seller'] - df_game_coeffs['slope_buyer'])
    df_game_coeffs['linear_cep'] = df_game_coeffs['linear_eq'] * df_game_coeffs['slope_seller'] + df_game_coeffs[
        'intercept_seller']
    df_game_coeffs['linear_got'] = df_game_coeffs['linear_eq'] * (
            df_game_coeffs['intercept_buyer'] + df_game_coeffs['slope_buyer'] - df_game_coeffs['intercept_seller'] -
            df_game_coeffs['slope_seller']) / 2
    df_game_coeffs.to_feather('../data/preprocessed/supply_demand_lines_game.ft')
