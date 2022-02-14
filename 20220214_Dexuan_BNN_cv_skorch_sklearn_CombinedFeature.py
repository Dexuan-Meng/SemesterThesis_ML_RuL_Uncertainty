import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.modules.loss import _Loss
from torch import Tensor
import numpy as np
import pandas as pd
import os
import sys

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import _scorer

from scipy.stats import uniform, loguniform, randint

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt

import skorch
from skorch.regressor_for_BNN import NeuralNetRegressor

from blitz.modules.base_bayesian_module import BayesianModule

from typing import Union, Tuple
import matplotlib


def plot_xy(
        y_pred: np.ndarray,
        y_std: np.ndarray,
        y_true: np.ndarray,
        x: np.ndarray,
        n_subset: Union[int, None] = None,
        ylims: Union[Tuple[float, float], None] = None,
        xlims: Union[Tuple[float, float], None] = None,
        num_stds_confidence_bound: int = 2,
        leg_loc: Union[int, str] = 3,
        ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing x
    order = np.argsort(y_true)
    y_pred, y_std, y_true, x = (
        y_pred[order],
        y_std[order],
        y_true[order],
        x[order],
    )

    # Optionally select a subset
    # if n_subset is not None:
        # [y_pred, y_std, y_true, x] = filter_subset([y_pred, y_std, y_true, x], n_subset)

    intervals = num_stds_confidence_bound * y_std

    h1 = ax.plot(x, y_true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(x, y_pred, "-.", marker='o', c="#1f77b4", linewidth=2)
    h3 = ax.fill_between(
        x,
        y_pred - intervals,
        y_pred + intervals,
        color="lightsteelblue",
        alpha=0.4,
    )
    ax.legend(
        [h1[0], h2[0], h3],
        ["Observations", "Predictions", "$95\%$ Interval"],
        loc=leg_loc,
    )

    # Format plot
    if ylims is not None:
        ax.set_ylim(ylims)

    if xlims is not None:
        ax.set_xlim(xlims)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Confidence Band")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return


def plot_1_2_3(pred_mean, pred_std, te_y, x, save=None):
    # No.1
    uct.viz.plot_intervals_ordered(pred_mean, pred_std, te_y)
    plt.gcf().set_size_inches(4, 4)
    plt.show()
    if save:
        string = "plt.savefig('data/20211110_Dexuan_RepresentativeSamples/Ordered Prediction Intervals_" + str(
            save) + ".jpg')"
        exec(string)
    # No.2
    uct.viz.plot_calibration(pred_mean, pred_std, te_y)
    plt.gcf().set_size_inches(4, 4)
    plt.show()
    if save:
        string = "plt.savefig('data/20211101_Dexuan_Test_UncertaintyToolbox/Average Calibration_" + str(save) + ".jpg')"
        exec(string)
    # No.3
    plot_xy(pred_mean, pred_std, te_y, x, leg_loc=2)
    plt.show()


def plot_1_2(pred_mean, pred_std, te_y, save=None):
    # Warning: problem with the string, need to be correct

    uct.viz.plot_intervals_ordered(pred_mean, pred_std, te_y)
    # plt.gca().set_ylim([1.0,1.1])
    # string = "mu mean_" + str(mu_mean) + "_mu std_" + str(mu_std) + "_label_" + str(label) + ""
    # plt.title(string)
    plt.gcf().set_size_inches(4, 4)
    plt.show()
    if save:
        string = "plt.savefig('data/20211110_Dexuan_RepresentativeSamples/" + str(
            save) + "_Ordered Prediction Intervals.jpg')"
        exec(string)
    uct.viz.plot_calibration(pred_mean, pred_std, te_y)
    # string = "mu mean_" + str(mu_mean) + "_mu std_" + str(mu_std) + "_"
    # plt.title(string)
    plt.gcf().set_size_inches(4, 4)
    plt.show()
    if save:
        string = "plt.savefig('data/20211101_Dexuan_Test_UncertaintyToolbox/" + str(save) + "_Average Calibration.jpg')"
        exec(string)


class MyDataset(Dataset):
    def __init__(self, df, MinMaxScaler):
        self.x_data = torch.tensor(MinMaxScaler.transform(df[['F8', 'F9', 'F17']].values)).float()
        self.y_data = torch.tensor(df[['rul_80%']].values).float()
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def load_data(standard_scaling=False):
    df_train = pd.read_csv(r"D:\MasterStudiengang\21WiSe\Semesterarbeit\Git_repository_13102021\batterytesting_eol_prediction-main/data/MIT/wichtig/train_battery.csv")
    df_test = pd.read_csv(r"D:\MasterStudiengang\21WiSe\Semesterarbeit\Git_repository_13102021\batterytesting_eol_prediction-main/data/MIT/wichtig/test_battery.csv")
    StandardScaler = preprocessing.StandardScaler()
    MinMaxScaler = preprocessing.MinMaxScaler()
    MinMaxScaler.fit(df_train.loc[:, ['F8', 'F9', 'F17']].values)
    StandardScaler.fit(df_train.loc[:, ['F8', 'F9', 'F17']].values)
    if standard_scaling:
        scaler = StandardScaler
    else:
        scaler = MinMaxScaler

    train_dataset = MyDataset(df_train, MinMaxScaler)
    test_dataset = MyDataset(df_test, MinMaxScaler)

    return train_dataset, test_dataset

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, size_layer1=50, size_layer2=30, size_layer3=10, l1_rho=-3, l1_sigma_1=1e2, l1_sigma_2=1e-2, l1_pi=0.5, l4_rho=-3, l4_sigma_1=1e2, l4_sigma_2=1e-2, l4_pi=0.5):
        super(BayesianRegressor, self).__init__()

        self.size1 = size_layer1
        self.size2 = size_layer2
        self.size3 = size_layer3
        self.l1_posterior_rho = l1_rho
        self.l1_prior_sigma_1 = l1_sigma_1
        self.l1_prior_sigma_2 = l1_sigma_2
        self.l1_prior_pi = l1_pi
        self.l4_posterior_rho = l4_rho
        self.l4_prior_sigma_1 = l4_sigma_1
        self.l4_prior_sigma_2 = l4_sigma_2
        self.l4_prior_pi = l4_pi

        self.blinear1 = BayesianLinear(9, self.size1,
                                       posterior_rho_init=self.l1_posterior_rho,
                                       prior_sigma_1=self.l1_prior_sigma_1,
                                       prior_sigma_2=self.l1_prior_sigma_2,
                                       prior_pi=self.l1_prior_pi)
        self.linear2 = nn.Linear(self.size1, self.size2)
        self.linear3 = nn.Linear(self.size2, self.size3)
        self.blinear4 = BayesianLinear(self.size3, 1,
                                       posterior_rho_init=self.l4_posterior_rho,
                                       prior_sigma_1=self.l4_prior_sigma_1,
                                       prior_sigma_2=self.l4_prior_sigma_2,
                                       prior_pi=self.l4_prior_pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.blinear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.blinear4(out)
        out = self.relu(out)
        # out = self.blinear5(out)
        # out = self.relu(out)
        return out


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


class _PredictScorer(_scorer._BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_preds = torch.stack([method_caller(estimator, "forward", X, training=True).cpu() for i in range(20)])
        y_pred = y_preds.mean(axis=0).reshape(-1).detach().numpy()
        y_std = y_preds.std(axis=0).reshape(-1).detach().numpy() + 1e-6
        y_true = y_true.reshape(-1).detach().numpy()
        # print(y_pred, y_std, y_true)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y_pred, y_std, y_true, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y_pred, y_std, y_true, **self._kwargs)


def make_scorer(
        score_func,
        *,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
        **kwargs,
):
    sign = 1 if greater_is_better else -1
    cls = _PredictScorer
    return cls(score_func, sign, kwargs)


def get_scalers():

    df_combined = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined.csv')
    StandardScaler = preprocessing.StandardScaler()
    MinMaxScaler = preprocessing.MinMaxScaler()
    MinMaxScaler.fit(df_combined.loc[:, ['slope_2_100', 'intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values)
    StandardScaler.fit(df_combined.loc[:, ['slope_2_100', 'intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values)

    return MinMaxScaler, StandardScaler


def kl_divergence_from_nn(model):

    """
    Gathers the KL Divergence from a nn.Module object
    Works by gathering each Bayesian layer kl divergence and summing it, doing nothing with the non Bayesian ones
    """
    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence


class sample_elbo_class(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', complexity_cost_weight=1) -> None:
        super(sample_elbo_class, self).__init__(size_average, reduce, reduction)

        self.complexity_cost_weight = complexity_cost_weight

    def forward(self, input: Tensor, target: Tensor, model) -> Tensor:

        return self.sample_elbo(input, target, model=model, complexity_cost_weight=self.complexity_cost_weight)

    def nn_kl_divergence(self, model):
        return kl_divergence_from_nn(model)

    def sample_elbo(self,
                    inputs,
                    labels,
                    model=None,
                    sample_nbr=20,
                    criterion=nn.MSELoss(),
                    complexity_cost_weight=1):

        loss = 0
        for _ in range(sample_nbr):
            # regressor = self.model
            # outputs = regressor.forward(inputs)
            loss += criterion(inputs, labels)
            loss += self.nn_kl_divergence(model) * complexity_cost_weight
        return loss / sample_nbr


if __name__ == '__main__':

    param_distributions = {
        'lr': loguniform(1e-3, 5e-2),
        'module__size_layer1': randint(20, 60),
        'module__size_layer2': randint(20, 50),
        'module__size_layer3': randint(20, 40),
        'criterion__complexity_cost_weight': loguniform(0.0001, 0.05),
        # 'criterion__complexity_cost_weight': [0.01, 10],
        'module__l1_rho': uniform(-6.0, 6),
        # 'module__l1_sigma_1': loguniform(1e-2, 1e3),
        # 'module__l1_sigma_2': loguniform(1e-5, 1e-2),
        'module__l1_pi': uniform(0, 1),
        'module__l4_rho': uniform(-6.0, 6),
        # 'module__l4_sigma_1': loguniform(1e-2, 1e3),
        # 'module__l4_sigma_2': loguniform(1e-5, 1e-2),
        'module__l4_pi': uniform(0, 1),
    }

    regr = NeuralNetRegressor(
        BayesianRegressor,
        criterion=sample_elbo_class,
        optimizer=torch.optim.RMSprop,
        max_epochs=1000,
        batch_size=40,
        train_split=skorch.dataset.ValidSplit(cv=5),
        device='cuda',
        callbacks=[skorch.callbacks.EarlyStopping(patience=10)],
        verbose=0,
        lr=0.005,
        criterion__complexity_cost_weight=1/3,
        module__l4_rho=-2
    )

    CRPS = make_scorer(uct.crps_gaussian, greater_is_better=False)

    random_search = RandomizedSearchCV(
        regr,
        param_distributions,
        n_iter=100,
        n_jobs=1,
        cv=5,
        verbose=3,
        random_state=1,
        scoring=CRPS
    )

    df_combined_train = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined_train.csv')
    df_combined_test = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined_test.csv')
    MinMaxScaler, StandardScaler = get_scalers()
    X_train = torch.Tensor(StandardScaler.transform(df_combined_train.loc[:, ['slope_2_100', 'intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values))
    y_train = torch.Tensor(df_combined_train[['RuL_0.88Ah']].values)
    X_test = torch.Tensor(StandardScaler.transform(df_combined_test.loc[:, ['slope_2_100', 'intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values))
    y_test = df_combined_test[['RuL_0.88Ah']].values.reshape(-1)

    random_search.fit(X_train, y_train)

    # preds = torch.stack([regr.forward(X_test, training=True).cpu() for i in range(20)])
    preds = torch.stack([random_search.best_estimator_.forward(X_test, training=True).cpu() for i in range(20)])
    y_pred = preds.mean(axis=0).reshape(-1).detach().numpy()
    y_std = preds.std(axis=0).reshape(-1).detach().numpy() + 1e-6
    battery = df_combined_test['battery'].values
    plot_1_2_3(y_pred, y_std, y_test, battery)
    # report(random_search.cv_results_, n_top=5)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    crps = uct.crps_gaussian(y_pred, y_std, y_test)
    cal_error = uct.root_mean_squared_calibration_error(y_pred, y_std, y_test)
    mean_std = np.mean(y_std)
    print('mse: %.2F, \nmean of std: %.2f, \ncrps: %.2f,\ncal_error: %.2f, \nmape: %.2f' % (
        mse, mean_std, crps, cal_error, mape))
    report(random_search.cv_results_, n_top=5)



