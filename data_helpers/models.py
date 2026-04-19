from copy import deepcopy

import pandas as pd
import torch
from abc import abstractmethod
import numpy as np
import statsmodels.formula.api as smf


from torch.optim.adagrad import Adagrad
from torch.optim.swa_utils import SWALR


class Model:
    """
    Abstract NN model class to imitate the SKLearn API.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplemented()

    @abstractmethod
    def transform(self, X):
        raise NotImplemented

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class FormulaModel(Model):

    def __init__(self, formula: str):
        self.formula = formula
        self.fitted_model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        sample_train = X.join(y)

        self.fitted_model = smf.rlm(self.formula, sample_train)
        return self

    def transform(self, X: pd.DataFrame):
        if self.fitted_model is None:
            raise  ValueError("Model not fitted yet!")
        yhat = self.fitted_model.predict(X)
        return yhat



class TransformerModel(Model):
    """
    A torch transformer model that follows SKLearn API conventions.
    The provided torch NN models is assumed to calculate self-attention on inputs.
    """
    def __init__(self,
                 transformer: torch.nn.Module,
                 loss: callable,
                 optimizer: torch.optim.Optimizer,
                 ):
        self.transformer = transformer
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int = 100) -> Model:
        """
         Fit a transformer NN model with self-attention on :param:`X`.
        The best fitting model is returned.

        Parameters
        ----------
        X: np.ndarray
            The input features for the model.
        y: np.ndarray
            The target labels for the model.
        n_epochs: int
            The number of epochs to fit the model.

        Returns
        -------
        fitted_model: TransformerModel
            The fitted model.

        """


        best_model = None
        best_loss = np.infty

        for i in range(n_epochs):
            self.optimizer.zero_grad()
            def closure():
                y_hat = self.transformer(X, X)
                loss_val = self.loss(y_hat, y)
                loss_val.backward()
                return loss_val.item()
            training_loss = self.optimizer.step(closure)

            if training_loss <= best_loss:
                best_loss = training_loss
                best_model = deepcopy(self.transformer.state_dict())
        #TODO: This is super not thread safe...
        self.transformer.load_state_dict(best_model)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply self attention to input features.

        Parameters
        ----------
        X: np.ndarray
            The input features.

        Returns
        -------
        y_pred: np.ndarray
            The transformed input.
        """
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            return self.transformer(X, X).cpu().numpy()


class NNModel(Model):
    """
    A torch transformer model that follows SKLearn API conventions.
    The provided torch NN models is assumed to calculate self-attention on inputs.
    """
    def __init__(self,
                 nn: torch.nn.Module,
                 loss: callable,
                 optimizer: torch.optim.Optimizer,
                 ):
        self.nn = nn
        self.loss = loss
        self.optimizer = optimizer
        self.training_losses = []

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int = 100) -> Model:
        """
         Fit a transformer NN model with self-attention on :param:`X`.
        The best fitting model is returned.

        Parameters
        ----------
        X: np.ndarray
            The input features for the model.
        y: np.ndarray
            The target labels for the model.
        n_epochs: int
            The number of epochs to fit the model.

        Returns
        -------
        fitted_model: TransformerModel
            The fitted model.

        """

        best_model = None
        best_loss = np.infty

        for i in range(n_epochs):
            self.optimizer.zero_grad()
            def closure():
                y_hat = self.nn(X)
                loss_val = self.loss(y_hat, y)
                loss_val.backward()
                return loss_val.item()

            training_loss = self.optimizer.step(closure)
            self.training_losses.append(training_loss)
            if training_loss <= best_loss:
                best_loss = training_loss
                best_model = deepcopy(self.nn.state_dict())
        #TODO: This is super not thread safe...
        self.nn.load_state_dict(best_model)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply self attention to input features.

        Parameters
        ----------
        X: np.ndarray
            The input features.

        Returns
        -------
        y_pred: np.ndarray
            The transformed input.
        """
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            return self.nn(X).cpu().numpy()

class NNModelSWA(Model):
    """
    A torch transformer model that follows SKLearn API conventions.
    The provided torch NN models is assumed to calculate self-attention on inputs.
    """
    def __init__(self,
                 nn: torch.nn.Module,
                 loss: callable,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 swa_start:int=160,
                 ):
        self.nn = nn
        self.loss = loss
        self.optimizer = optimizer
        self.training_losses = []
        self.swa_start = swa_start
        self.scheduler = scheduler
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.nn)


    def fit(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int = 100) -> Model:
        """
         Fit a transformer NN model with self-attention on :param:`X`.
        The best fitting model is returned.

        Parameters
        ----------
        X: np.ndarray
            The input features for the model.
        y: np.ndarray
            The target labels for the model.
        n_epochs: int
            The number of epochs to fit the model.

        Returns
        -------
        fitted_model: TransformerModel
            The fitted model.

        """

        swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)

        best_model = None
        best_loss = np.infty

        for i in range(n_epochs):
            self.optimizer.zero_grad()
            def closure():
                y_hat = self.nn(X)
                loss_val = self.loss(y_hat, y)
                loss_val.backward()
                return loss_val.item()

            training_loss = self.optimizer.step(closure)
            self.training_losses.append(training_loss)
            if i > self.swa_start:
                self.swa_model.update_parameters(self.nn)
                swa_scheduler.step()
            else:
                self.scheduler.step()
            if training_loss <= best_loss:
                best_loss = training_loss
                best_model = deepcopy(self.nn.state_dict())
        #TODO: This is super not thread safe...
        torch.optim.swa_utils.update_bn(X, self.swa_model)
        self.nn.load_state_dict(best_model)
        return self

    def transform(self, X: np.ndarray, use_stochastic=False) -> np.ndarray:
        """
        Apply self attention to input features.

        Parameters
        ----------
        X: np.ndarray
            The input features.

        Returns
        -------
        y_pred: np.ndarray
            The transformed input.
        """
        with torch.no_grad():
            if use_stochastic:
                self.swa_model(X)
            else:
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X)
                return self.nn(X).cpu().numpy()


if __name__ == '__main__':
    #TODO: move to unit tests
    N = 32
    T = 10
    K = 5
    X = torch.rand([N, T, K])
    y = torch.rand([N, T, 1])
    class TNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Transformer(d_model=K, nhead=1,
                                               dim_feedforward=8,
                                               batch_first=True,
                                               num_encoder_layers=1,
                                               num_decoder_layers=1
                                               )
            self.transformer_out = torch.nn.Linear(5, 1)

        def forward(self, x, y):
            h = self.transformer(x, y)
            h = self.transformer_out(h)
            return h
    #TODO: create self attention module as unary operator to respect sequential.

    nn_module = TNN()
    nn_optimizer = torch.optim.Adam(nn_module.parameters(), lr=1e-3)
    model = TransformerModel(transformer=nn_module,
                             loss=torch.nn.functional.mse_loss,
                             optimizer=nn_optimizer)

    best_model = model.fit(X, y)
