from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm.auto import trange

import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy import stats


class TrainTestSplit:
    def __init__(self, df, train_query: str = None, test_query: str = None, split_keys=['treatment', 'game']):
        self.split_keys = split_keys
        self.filtered_df = df.copy()
        self.unique_sample_keys = self.filtered_df[self.split_keys].drop_duplicates().reset_index(
            drop=True)
        self.train_query = train_query
        self.test_query = test_query

    def get_datasets(self, split_rate=0.5, random_state=None):
        train_set_keys = self.unique_sample_keys.sample(frac=split_rate, replace=False, random_state=random_state)
        test_set_keys = self.unique_sample_keys[~self.unique_sample_keys.index.isin(train_set_keys.index)]
        train_set = pd.merge(self.filtered_df, train_set_keys, on=self.split_keys)
        if self.train_query is not None:
            train_set = train_set.query(self.train_query)

        test_set = pd.merge(self.filtered_df, test_set_keys, on=self.split_keys)
        if self.test_query is not None:
            test_set = test_set.query(self.test_query)

        return train_set, test_set


class ModelFit:
    def __init__(self, name:str):
        self.name = name

    def fit(self, df: pd.DataFrame):
        raise ValueError('Not Implemented yet for abstract class!')

    def predict(self, df: pd.DataFrame):
        raise ValueError('Not Implemented yet for abstract class!')


class SMFormulaFit(ModelFit):
    def __init__(self,
                 name: str,
                 formula: str,
                 model_method: Callable = smf.ols,
                 model_method_params: dict = None,
                 ):
        super().__init__(name)
        self.formula = formula
        self.model_method = model_method
        self.model_method_params = model_method_params
        if self.model_method_params is None:
            self.model_method_params = {}
        self.fitted_model = None

    def fit(self, df: pd.DataFrame):
        model_desc = self.model_method(self.formula, df, **self.model_method_params)
        self.fitted_model = model_desc.fit()
        return self.fitted_model

    def predict(self, df: pd.DataFrame):
        if self.fitted_model is None:
            raise ValueError('Fit needs to be called first to fit the model.')
        #TODO: keep a counter for a warning on async calls of fit, i.e. fit calls < predict cals
        return self.fitted_model.predict(df)


class MAPEFitEval:
    def __init__(self,
                 dataset,
                 models: List[ModelFit],
                 n_samples: int = 100,
                 split_ratio: float = 0.5,
                 train_filter_query: str = None,
                 test_filter_query: str = None
                 ):
        self.dataset = dataset
        self.n_samples = n_samples
        self.split_ratio = split_ratio
        self.train_filter_query = train_filter_query
        self.test_filter_query = test_filter_query
        self.models = models


    def single_fit_mape(self, model, sample_split):
        try:
            # LOO should be calculated analytically for OLS, so we can use this instead.
            sample_train, sample_test = sample_split.get_datasets()
            fitted_model = model.fit(sample_train)
            yhat = fitted_model.predict(sample_test)
            y = sample_test.ce
            return mape(y, yhat)
        except Exception:
            return np.nan

    def test_formulas(self, error_statistic: Callable = np.nanmedian, use_progress=True):
        progress = trange(self.n_samples,
                          desc='Fitting on sample: ') if use_progress else range(self.n_samples)

        test_errors = dict()
        for model in self.models:
            test_errors[model.name] = []

        for i in progress:
            sample_split = TrainTestSplit(self.dataset,
                                          train_query=self.train_filter_query,
                                          test_query=self.test_filter_query,
                                          )
            for model in self.models:
                mape = self.single_fit_mape(model, sample_split)
                test_errors[model.name].append(mape)

        model_rows = []
        for model_name, test_error_values in test_errors.items():
            test_error_array= np.array(test_error_values).reshape([1, -1])
            confidence_interval = stats.bootstrap(test_error_array,
                                                  statistic=error_statistic,
                                                  method='basic'
                                                  ).confidence_interval
            confidence_interval = (confidence_interval.low, confidence_interval.high)

            model_row = pd.Series({
                'name' : model_name,
                error_statistic.__name__ : error_statistic(test_error_array),
                error_statistic.__name__ + '_ci' : confidence_interval,
                'test_errors' : test_error_array

            })
            model_rows.append(model_row)
        return pd.DataFrame(model_rows)


class MultipeTesting:
    def __init__(self, dataset, train_filter_query: str = None, test_filter_query: str = None):
        self.dataset = dataset
        self.train_filter_query = train_filter_query
        self.test_filter_query = test_filter_query

    def single_fit(self, formula, sample_split):
        try:
            # LOO should be calculated analytcally for OLS, so we can use this instead.
            sample_train, sample_test = sample_split.get_datasets()
            model = smf.ols(formula, sample_train)
            fitted_model = model.fit()
            yhat = fitted_model.predict(sample_test)
            y = sample_test.ce
            return mape(y, yhat)
        except Exception:
            return np.nan

    def get_train_test_split(self):
        sample_split = TrainTestSplit(self.dataset, train_query=self.train_filter_query,
                                      test_query=self.test_filter_query)
        return sample_split

    def test_formula(self, formula: str,
                     n_samples: int = 100,
                     formula_name: str = None,
                     ci_statistic=np.nanmedian,
                     use_progress=True,
                     sample_split: TrainTestSplit = None):
        if formula_name is None:
            formula_name = formula
        if sample_split is None:
            sample_split = self.get_train_test_split()
        progress = trange(n_samples, desc='Fitting formula: ' + formula_name) if use_progress else range(n_samples)
        test_errors = np.array(list(map(lambda i: self.single_fit(formula, sample_split),
                                        progress))).reshape([1, -1])
        confidence_interval = stats.bootstrap(test_errors, statistic=ci_statistic, method='basic').confidence_interval
        confidence_interval = (confidence_interval.low, confidence_interval.high)
        return test_errors, confidence_interval
