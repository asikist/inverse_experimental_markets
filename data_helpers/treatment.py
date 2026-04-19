import pandas as pd
import numpy as np
import torch

feedback_settings = ['Full', 'BB', 'Other', 'Same', 'None']
price_rules = ['MMK', 'Random', 'First']
market_structures = ['LimS', 'FromStoB', 'FromBtoS', 'SShift', 'MoreRounds', 'LargeCE', 'MoreB', 'MoreS', 'Regular']

treatment_feature_columns = {'feedback_setting' : feedback_settings,
                              'price_rule' : price_rules, 
                              'market_structure' : market_structures
                            }



def extract_feedback_setting(treatment: str):
    for setting in feedback_settings:
        if setting in treatment:
            return setting


def extract_price_rule(treatment: str):
    for rule in price_rules:
        if rule in treatment:
            return rule


def convert_to_tensor(element, element_list):
    res_tensor = torch.zeros(len(element_list))
    res_tensor[np.where(element == np.array(element_list))] = 1.0
    return res_tensor

def extract_setting(row: pd.Series, column_name: str, setting_list: list):
    treatment = row['treatment']
    row[column_name] = setting_list[-1]
    for setting in setting_list:
        if setting in treatment:
            row[column_name] = setting
            break
    return row


def __treatment_to_features(row: pd.Series) -> pd.Series:
    for k,v in treatment_feature_columns.items():
        row = extract_setting(row, k, v)
    return row

def treatment_feats(df: pd.DataFrame):
    original_grouping = df.index.names
    treatment_feature_cols = list(treatment_feature_columns.keys())
    df = df.reset_index().apply(__treatment_to_features, axis=1).groupby(original_grouping).first()
    df = pd.get_dummies(df, columns=treatment_feature_cols)
    # df.drop(columns=treatment_feature_cols, inplace=True)
    # df.join(df_treatment_dummies)
    return df