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


from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import _scorer

from scipy.stats import loguniform, randint

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt

import skorch
from skorch import NeuralNetRegressor

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
    def __init__(self, df, scaler):
        self.x_data = torch.tensor(scaler.transform(df[['intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values)).float()
        self.y_data = torch.tensor(df[['RuL_0.88Ah']].values).float()
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def load_data(standard_scaling=True):
    df_combined_train = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined_train.csv')
    df_combined_test = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined_test.csv')
    df_combined = pd.read_csv('data/MIT/wichtig/df_severson_fei_combined.csv')
    X = df_combined.loc[:,
        ['intercept_2_100', 'slope_91_100', 'disC_2', 'disC_max_2', 'disC_100', 'F8', 'F9', 'F17']].values

    StandardScaler = preprocessing.StandardScaler()
    MinMaxScaler = preprocessing.MinMaxScaler()
    MinMaxScaler.fit(X)
    StandardScaler.fit(X)
    if standard_scaling:
        scaler = StandardScaler
    else:
        scaler = MinMaxScaler

    train_dataset = MyDataset(df_combined_train, scaler)
    test_dataset = MyDataset(df_combined_test, scaler)

    return train_dataset, test_dataset


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MCDRegressor(nn.Module):
    def __init__(self, layer_1_size, layer_2_size, layer_3_size, layer_4_size, dropout_p):
        super(MCDRegressor, self).__init__()

        self.l1 = layer_1_size
        self.l2 = layer_2_size
        self.l3 = layer_3_size
        self.l4 = layer_4_size
        self.dropout_p = dropout_p

        # self.l1 = 50
        # self.l2 = 30
        # self.l3 = 20
        # self.l4 = 5
        # self.dropout_p = 0.1

        self.fc = nn.Sequential(
            nn.Linear(9, self.l1),
            nn.Dropout(self.dropout_p),
            nn.LeakyReLU(),
            nn.Linear(self.l1, self.l2),
            nn.Dropout(self.dropout_p),
            nn.LeakyReLU(),
            nn.Linear(self.l2, self.l3),
            nn.Dropout(self.dropout_p),
            nn.LeakyReLU(),
            nn.Linear(self.l3, self.l4),
            nn.Dropout(self.dropout_p),
            nn.LeakyReLU(),
            nn.Linear(self.l4, 1),
            nn.LeakyReLU()
        )

        self.fc.apply(init_weights)

    def forward(self, x):
        out = self.fc(x)
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

if __name__ == '__main__':

    param_distributions = {
        'lr': loguniform(1e-3, 5e-2),
        'module__layer_1_size': randint(45, 65),
        'module__layer_2_size': randint(20, 40),
        'module__layer_3_size': randint(10, 25),
        'module__layer_4_size': randint(2, 10),
        'module__dropout_p': [0.02, 0.05, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]
    }


    regr = NeuralNetRegressor(
        MCDRegressor,
        module__layer_1_size=50,
        module__layer_2_size=30,
        module__layer_3_size=20,
        module__layer_4_size=5,
        module__dropout_p=0.1,
        criterion=nn.MSELoss,
        optimizer=torch.optim.RMSprop,
        max_epochs=1000,
        batch_size=8,
        device='cuda',
        callbacks=[skorch.callbacks.EarlyStopping(patience=20)],
        verbose=0
    )

    CRPS = make_scorer(uct.crps_gaussian, greater_is_better=False)

    random_search = RandomizedSearchCV(
        regr,
        param_distributions,
        n_iter=100,
        n_jobs=1,
        cv=5,
        verbose=3,
        random_state=2,
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
    report(random_search.cv_results_, n_top=5)

    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    crps = uct.crps_gaussian(y_pred, y_std, y_test)
    cal_error = uct.root_mean_squared_calibration_error(y_pred, y_std, y_test)
    mean_std = np.mean(y_std)
    print('mse: %.2F, \nmean of std: %.2f, \ncrps: %.2f,\ncal_error: %.2f, \nmape: %.2f' % (
        mse, mean_std, crps, cal_error, mape))